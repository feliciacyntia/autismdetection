"""v4 — eksperimen yang ditunda dari v3: morfometri FreeSurfer PCP (E1 + tuning + LOSO). E2 fusion DL ditunda (lihat config).
Config dibaca dari locked_config_v4.json; hash harus sama dengan locked_config_v4.sha256. Hanya 853 dev; 149 test tidak dibaca.

    .venv-model\\Scripts\\python.exe src\\run_v4.py [--resume] [--stop-at 07:00]     # ±3 jam
    .venv-model\\Scripts\\python.exe src\\run_v4.py --test                           # uji singkat -> results\\v4_test

Berhenti rapi: file STOP di folder hasil / --stop-at HH:MM / Ctrl+C sekali. Unit gagal dicatat lalu runner lanjut.
"""
import argparse
import datetime
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc  # noqa: E402
import run_pilot as rp  # noqa: E402
import run_v3_tabular as rv  # noqa: E402

LOG = logging.getLogger("pilot")
PROJECT = Path(__file__).resolve().parents[1]
CFG_PATH = pc.RESULTS_ROOT / "v4" / "locked_config_v4.json"
MORPH = ["morph_raw", "morph_icv", "morph_cb"]
DL_MODES = ["image", "concat", "film", "daft"]
UNIT_RE = re.compile(r"^e[13]_(morph_[a-z]+)_(logreg_cv|logreg|tabm|catboost|tabpfn)$")


def morph_variants():
    """Fitur tanpa parameter yang dipelajari. Relatif ICV: volume / ICV, luas / ICV^(2/3), ketebalan tetap, ICV dipertahankan."""
    X = pd.read_csv(PROJECT / "data" / "freesurfer_pcp" / "morfometri.csv").set_index("SUB_ID")
    R = X.copy()
    vol = [c for c in X if c.endswith("_gmvol") or c.startswith("vol_") or c in ("CortexVol", "CorticalWhiteMatterVol", "SubCortGrayVol",
                                                                                "TotalGrayVol", "SupraTentorialVol")]
    area = [c for c in X if c.endswith("_area") or c.endswith("_WhiteSurfArea")]
    R[vol] = X[vol].div(X["ICV"], axis=0)
    R[area] = X[area].div(X["ICV"] ** (2 / 3), axis=0)
    return {"morph_raw": X, "morph_icv": R, "morph_cb": R}


class V4(rv.V3):
    def __init__(self, a):
        self.args = a
        raw = CFG_PATH.read_bytes()
        self.cfg = json.loads(raw)
        self.file_hash = hashlib.sha256(raw).hexdigest()
        lock = CFG_PATH.with_suffix(".sha256")
        if not lock.exists() or lock.read_text(encoding="utf-8").split()[0] != self.file_hash:
            raise SystemExit(f"Config BELUM dikunci / berubah setelah dikunci: sha256 {self.file_hash} tidak cocok dengan {lock}")
        self.hash = self.file_hash[:16] + ("-test" if a.test else "")
        self.rdir = Path(a.results_dir) if a.results_dir else pc.RESULTS_ROOT / ("v4_test" if a.test else "v4")
        self.rdir.mkdir(parents=True, exist_ok=True)
        self.folds = [0] if a.test else [0, 1, 2, 3, 4]
        self.seeds = [0] if a.test else self.cfg["seed"]["neural_multi_seed"]
        self.trials = 3 if a.test else self.cfg["E3_tuning"]["trial"]
        self.stop_at = None
        self.vol = None
        if a.stop_at:
            t = datetime.datetime.combine(datetime.date.today(), datetime.datetime.strptime(a.stop_at, "%H:%M").time())
            self.stop_at = t if t > datetime.datetime.now() else t + datetime.timedelta(days=1)

    def data(self):
        super().data()
        ids = self.cohort.SUB_ID
        self.morph = {k: v.loc[ids] for k, v in morph_variants().items()}
        cols = list(self.morph["morph_raw"].columns)
        bad = [c for c in cols if c.upper().startswith(("SITE", "DX_", "ADOS", "ADI", "SRS", "SCQ", "AQ", "DSM"))]
        assert not bad, f"kolom terlarang di fitur: {bad}"
        LOG.info("Morfometri: %d fitur | subjek dev tanpa fitur (QC/tidak tersedia): %d", len(cols), int(self.morph["morph_raw"].isna().all(axis=1).sum()))

    def X(self, feat):
        return self.morph[feat] if feat in MORPH else super().X(feat)

    def prep(self, feat, tr, others):
        """Fit di tr saja: median -> [ComBat usia+SEX untuk morph_cb] -> scaler."""
        import pilot_models as pm
        if feat not in MORPH:
            return super().prep(feat, tr, others)
        X = self.morph[feat]
        p = pm.TabPre("roi", list(X.columns), feat == "morph_cb")
        cv = lambda ids: self.cov.loc[ids].reset_index(drop=True)
        p.fit(X.loc[tr], self.site.loc[tr].to_numpy(), cv(tr))
        tf = lambda ids: p.transform(X.loc[ids], self.site.loc[ids].to_numpy(), cv(ids))
        Ztr = tf(tr)
        outs, unseen = [], 0
        for o in others:
            outs.append(tf(o)); unseen += getattr(p, "unseen", 0)
        return Ztr, outs, unseen

    # ------------------------------------------------------------------ E2
    def volumes(self):
        if self.vol is None:
            t = time.time()
            ids = self.cohort.SUB_ID.to_numpy()
            v, m = pc.load_volumes(ids)
            self.vol = {"idx": {int(s): i for i, s in enumerate(ids)}, "v": v, "m": m}
            LOG.info("Volume dimuat ke RAM: %d subjek (%.1f s)", len(ids), time.time() - t)
        return self.vol

    def run_dl_unit(self, unit, mode, fold, s):
        import torch
        import fusion3d
        import pilot_models as pm
        hp = dict(self.cfg["E2"]["hp"])
        if self.args.test:
            hp["epochs"] = 2
        sp_ = self.splits[fold]
        seed = self.cfg["seed"]["basis"] + fold + 1000 * s
        y = {k: torch.tensor(self.lab.loc[v].to_numpy(), dtype=torch.float32) for k, v in sp_.items()}
        pre = pm.TabPre("pheno", self.features).fit(self.pheno.loc[sp_["inner_train"]])     # imputasi + scaling fit di inner_train
        tab = {k: torch.tensor(np.asarray(pre.transform(self.pheno.loc[v]), float), dtype=torch.float32) for k, v in sp_.items()}
        V = self.volumes()
        data = {k: np.asarray([V["idx"][int(i)] for i in v]) for k, v in sp_.items()}
        pm.seed_all(seed)
        model = fusion3d.Fusion3D(hp["channels"], hp["se_ratio"], hp["dropout"], mode, tab["inner_train"].shape[1], hp["daft_bottleneck"]).to(pm.DEV)
        ptot, _ = pm.count_params(model)
        B, gens = hp["batch"], {}

        def prep(part, idx, train):
            rows = data[part][np.asarray(idx)]
            v, m = torch.from_numpy(V["v"][rows]).to(pm.DEV).float(), torch.from_numpy(V["m"][rows]).to(pm.DEV)
            if train:
                v, m = pm.augment3d(v, m, gens["aug"], self.fv2["augment"])
            return torch.where(m, v, torch.zeros_like(v))[:, None], tab[part][idx].to(pm.DEV)

        def batches(ep):
            g = torch.Generator().manual_seed(seed * 1000 + ep)
            gens["aug"] = torch.Generator(device=pm.DEV).manual_seed(seed * 1000 + ep)
            perm = torch.randperm(len(sp_["inner_train"]), generator=g)
            for i in range(0, len(perm), B):
                b = perm[i:i + B]
                yield prep("inner_train", b, True), y["inner_train"][b].to(pm.DEV)

        def predict(part):
            model.eval()
            P = []
            with torch.no_grad(), torch.autocast(pm.DEV.type, dtype=torch.float16):
                for i in range(0, len(sp_[part]), B):
                    lg_, _ = model(prep(part, torch.arange(i, min(i + B, len(sp_[part]))), False))
                    P.append(lg_.float().cpu())
            return torch.cat(P)

        ytr = y["inner_train"].numpy()
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

        def evaluate(_m):
            from sklearn.metrics import roc_auc_score
            lg_ = predict("inner_val")
            return (float(torch.nn.functional.binary_cross_entropy_with_logits(lg_, y["inner_val"], pos_weight=torch.tensor(pw))),
                    float(roc_auc_score(y["inner_val"].numpy(), lg_.numpy())))

        def on_epoch(ep, dt, vl, vram, st):
            h = st["history"][-1]
            LOG.info("%s | epoch %d/%d | %.1f s | train %.4f | val %.4f (halus %.4f) | AUC val %.3f | best %d | VRAM %.0f MiB",
                     unit, ep + 1, hp["epochs"], dt, h["train_loss"], h["val_loss"], h["val_crit"], h["val_auc"], st["best_epoch"] + 1, vram)
            pc.update_status(self.rdir, unit, last_epoch=ep + 1, max_epochs=hp["epochs"], best_epoch=st["best_epoch"] + 1,
                             best_val_loss=st["best"], seconds=st["elapsed"], vram_peak_mib=vram, params_total=ptot)

        def recal(m):                               # bugfix BN: statistik dari inner_train tanpa augmentasi
            n = len(sp_["inner_train"])
            pm.recalibrate_bn(m, lambda: (prep("inner_train", torch.arange(i, min(i + B, n)), False) for i in range(0, n, B)))

        try:
            st = pm.train_loop(model, hp, batches, evaluate, pc.unit_dir(self.rdir, unit) / "ckpt", seed, self.stop, hp["budget_min"] * 60,
                               on_epoch, ytr, bn_recal_fn=recal)
        except pm.StopRequested:
            raise rv._Stop()
        recal(model)
        probs = {k: torch.sigmoid(predict(k)).numpy() for k in ("inner_val", "test")}
        h = st["history"]
        info = {"mode": mode, "seed": seed, "hp": hp, "best_epoch": st["best_epoch"] + 1, "epochs_run": st["epoch"] + 1, "best_val_loss": st["best"],
                "val_auc_best_epoch": h[st["best_epoch"]]["val_auc"], "seconds": st["elapsed"], "params_total": ptot,
                "vram_peak_mib": max(x["vram_mib"] for x in h), "s_per_epoch": float(np.mean([x["seconds"] for x in h])), "history": h}
        rp.save_unit(self.rdir, unit, self.cohort, fold, probs, sp_, {"info": info})
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ daftar unit
    def units_tabel(self):
        U = []
        for feat in MORPH:
            for m in rv.E1_MODELS:
                for s in (self.seeds if m in rv.NEURAL else [0]):
                    U += [dict(unit=f"e1_{feat}_{m}_s{s}:{k}", kind="cv", feat=feat, model=m, fold=k, s=s) for k in self.folds]
        for feat in MORPH:
            for m in rv.E3_MODELS:
                U += [dict(unit=f"e3tune_{feat}_{m}:{k}", kind="tune", feat=feat, model=m, fold=k) for k in self.folds]
                for s in (self.seeds if m in rv.NEURAL else [0]):
                    U += [dict(unit=f"e3_{feat}_{m}_s{s}:{k}", kind="cv_tuned", feat=feat, model=m, fold=k, s=s) for k in self.folds]
        return U

    def morph_selection(self):
        """Skor = AUC pooled inner_oof, multi-seed dirata-rata. Tidak memakai test."""
        best = None
        for feat in MORPH:
            for pre, ms in (("e1", rv.E1_MODELS), ("e3", rv.E3_MODELS)):
                for m in ms:
                    p = rv.model_preds(f"v3:{pre}_{feat}_{m}", self.rdir)
                    if p is None:
                        continue
                    q = p[p.part == "inner_oof"]
                    if q.fold.nunique() < len(self.folds):
                        continue
                    sc = rv.auc(q.label, q.prob)
                    if best is None or sc > best[1]:
                        best = (f"{pre}_{feat}_{m}", sc)
        return best

    def units_loso(self):
        best = self.morph_selection()
        if best is None:
            LOG.warning("LOSO dilewati: morfometri belum lengkap.")
            return []
        pc.sp.write_json_atomic(self.rdir / "selection_morfometri.json", {"model": best[0], "skor_inner_oof": best[1]})
        feat, m = UNIT_RE.match(best[0]).groups()
        m = "logreg_cv" if m == "logreg" else m
        LOG.info("LOSO morfometri: terpilih %s (AUC inner %.3f) -> %s, setting E1", best[0], best[1], m)
        sites = sorted(self.loso.unique())
        return [dict(unit=f"loso_{feat}_{m}_s{s}:{si}", kind="loso", feat=feat, model=m, fold=si, site=site, s=s)
                for s in (self.seeds if m in rv.NEURAL else [0]) for si, site in enumerate(sites)]

    def units_dl(self):
        return [dict(unit=f"dl_{mode}_s{s}:{k}", kind="dl", mode=mode, fold=k, s=s) for s in self.seeds for mode in DL_MODES for k in self.folds]

    def exec_unit(self, u):
        if u["kind"] == "dl":
            return self.run_dl_unit(u["unit"], u["mode"], u["fold"], u["s"])
        return super().exec_unit(u)

    def run(self):
        if self.args.bagian in ("tabel", "semua"):
            rc = self.run_list(self.units_tabel(), "E1+E3 morfometri")
            if rc:
                return rc
            if not self.args.no_loso:
                rc = self.run_list(self.units_loso(), "LOSO morfometri")
                if rc:
                    return rc
        if self.args.bagian in ("dl", "semua"):
            rc = self.run_list(self.units_dl(), "E2 fusion DL")
            if rc:
                return rc
        LOG.info("RINGKASAN: %s", pc.read_status(self.rdir).status.value_counts().to_dict())
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bagian", choices=["tabel", "dl", "semua"], default="tabel")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at", default=None)
    ap.add_argument("--test", action="store_true", help="uji singkat: fold 0, seed 0, 3 trial, 2 epoch -> results\\v4_test")
    ap.add_argument("--only", default=None, help="prefix unit, dipisah koma (uji)")
    ap.add_argument("--no-loso", action="store_true")
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args(argv)
    r = V4(a)
    if a.bagian != "tabel" and r.cfg["E2"].get("status") == "ditunda":
        raise SystemExit("E2 fusion DL ditunda di locked_config_v4.json: --bagian dl/semua ditolak.")
    rp.setup_logging(r.rdir)
    LOG.info("v4 | bagian %s | hasil %s | config_hash %s | sha256 config %s", a.bagian, r.rdir, r.hash, r.file_hash)
    cj = r.rdir / "config.json"
    if cj.exists():
        old = json.loads(cj.read_text(encoding="utf-8"))["config_hash"]
        if old != r.hash:
            LOG.error("config_hash berbeda (%s di folder vs %s): ditolak agar hasil tidak tercampur.", old, r.hash); return 2
        if not a.resume:
            LOG.error("Sudah ada hasil di %s. Gunakan --resume (tidak ada yang ditimpa).", r.rdir); return 2
    if (r.rdir / "STOP").exists():
        LOG.error("File STOP ada di %s. Hapus dulu.", r.rdir / "STOP"); return 2
    import psutil
    ram = psutil.virtual_memory().available / 2**30
    if ram < 4.0:
        LOG.error("RAM tersedia %.2f GiB < 4 GiB: tidak dimulai.", ram); return 2
    if a.bagian != "dl" and not rv.TABPFN_PY.exists():
        LOG.error("%s tidak ada (venv TabPFN).", rv.TABPFN_PY); return 2
    if not cj.exists():                             # ditulis setelah semua pengecekan: run yang ditolak tidak meninggalkan jejak
        pc.sp.write_json_atomic(cj, {"config_hash": r.hash, "config_sha256": r.file_hash, "config": r.cfg, "test": a.test,
                                     "created": datetime.datetime.now().isoformat(timespec="seconds")})
    shim = rp.Shim(r.rdir)
    try:
        lock = pc.sp.RunLock(shim); lock.__enter__()
    except RuntimeError as e:
        LOG.error("%s", e); return 2
    try:
        with pc.sp.KeepAwake():
            r.stopper = pc.sp.StopController(shim).install()
            try:
                r.data()
                return r.run()
            finally:
                r.stopper.uninstall()
    finally:
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    sys.exit(main())
