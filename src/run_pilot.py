"""06 — Runner eksperimen modeling (pilot subset 200 atau semua dev), proses mandiri yang bisa di-resume.

    .venv-model\\Scripts\\python.exe src\\run_pilot.py --subset pilot200 --resume [--stop-at 07:00]

Unit kerja = (model, fold). Status: <results>/<subset>/status.csv. Progres: progress.json / pilot_progress.py.
Berhenti rapi: file STOP di folder hasil (atau --stop-at HH:MM / Ctrl+C sekali) -> epoch berjalan diselesaikan,
checkpoint disimpan, keluar (kode 3). Lanjutkan dengan --resume (unit running dilanjutkan dari checkpoint epoch).
Kode keluar: 0 = semua unit terjadwal selesai, 1 = ada unit gagal/dilewati, 2 = tidak dimulai, 3 = dihentikan.
"""
import argparse
import datetime
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import os

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc  # noqa: E402

LOG = logging.getLogger("pilot")


class Shim:
    def __init__(self, rdir):
        self.paths = {"lock": rdir / "run.lock", "stop": rdir / "STOP"}


def setup_logging(rdir):
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger(); root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); root.addHandler(ch)
    (rdir / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(rdir / "logs" / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8")
    fh.setFormatter(fmt); root.addHandler(fh)
    for noisy in ("httpx", "huggingface_hub", "timm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return fmt


def write_progress(rdir, **kw):
    p = rdir / "progress.json"
    try:
        cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        cur = {}
    cur.update(kw); cur["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    pc.sp.write_json_atomic(p, cur)


def save_unit(rdir, unit, cohort, fold, probs, ids, extra, emb=None, method=None):
    """preds.csv (inner_val + test) + metrics.json + emb.npz, ditulis atomik."""
    d = pc.unit_dir(rdir, unit); d.mkdir(parents=True, exist_ok=True)
    c = cohort.set_index("SUB_ID")
    rows = []
    for part in [q for q in ("inner_val", "inner_oof", "test") if q in probs]:
        for s, p in zip(ids[part], probs[part]):
            rows.append({"SUB_ID": int(s), "SITE_ID": c.at[s, "SITE_ID"], "fold": fold, "part": part,
                         "label": int(c.at[s, "label"]), "prob": float(p)})
    df = pd.DataFrame(rows)
    if emb is not None:
        tmp = d / ".emb.npz.tmp.npz"
        np.savez(tmp, **{f"ids_{k}": np.asarray(ids[k]) for k in emb}, **{f"emb_{k}": v for k, v in emb.items()})
        pc.sp._replace_retry(tmp, d / "emb.npz")
    pc.sp.write_csv_atomic(d / "preds.csv", df)
    m = {"val": pc.metrics(df.loc[df.part == "inner_val", "label"], df.loc[df.part == "inner_val", "prob"]),
         "test": pc.metrics(df.loc[df.part == "test", "label"], df.loc[df.part == "test", "prob"]), **extra}
    pc.sp.write_json_atomic(d / "metrics.json", m)
    return m


# ==================================================================================================
class Runner:
    VERSION = "v1"

    def __init__(self, args):
        self.args = args
        self.subset = args.subset
        self.cfg, self.hash = pc.run_config(self.subset, self.VERSION)
        if getattr(args, "locked", None):          # cek kunci SEBELUM modifikasi uji apa pun
            self.apply_locked(Path(args.locked), getattr(args, "budget_scale", None))
        if args.max_epochs:                        # hanya untuk uji (mengubah hash)
            for hp in self.cfg["models"].values():
                if "epochs" in hp:
                    hp["epochs"] = min(hp["epochs"], args.max_epochs)
            self.hash = self.hash + f"-ep{args.max_epochs}"
        self.rdir = Path(args.results_dir) if args.results_dir else pc.results_dir(self.subset, self.VERSION)
        self.rdir.mkdir(parents=True, exist_ok=True)
        self.vol = None
        self.stop_at = None
        if args.stop_at:
            t = datetime.datetime.combine(datetime.date.today(), datetime.datetime.strptime(args.stop_at, "%H:%M").time())
            self.stop_at = t if t > datetime.datetime.now() else t + datetime.timedelta(days=1)

    def apply_locked(self, path, budget_scale=None):
        """Data penuh: batasi config ke kandidat terkunci. Hyperparameter WAJIB sama persis dengan file terkunci;
        budget_scale = pengali batas waktu pengaman (operasional, bukan hyperparameter model). config_hash dihitung ulang."""
        L = json.loads(path.read_text(encoding="utf-8"))
        keep = [m for v in L["kandidat_data_penuh"].values() for m in v]
        M = self.cfg["models"]
        for n in keep:
            assert n in M, f"kandidat {n} tidak ada di config"
            a, b = json.dumps(M[n], sort_keys=True, default=str), json.dumps(L["hyperparameter"][n], sort_keys=True, default=str)
            assert a == b, f"hyperparameter {n} berbeda dari file terkunci:\n{a}\n{b}"
        self.cfg["models"] = {n: hp for n, hp in M.items() if n in keep}
        self.cfg["modality"] = {mod: [n for n in ns if n in keep] for mod, ns in self.cfg["modality"].items()}
        need_cache = {"cache_roi": any(n.startswith("roi_") for n in keep)}
        self.cfg["order"] = [o for o in self.cfg["order"] if o in keep or o in ("EMBCOMBAT", "FUSION", "ENSEMBLE")
                             or need_cache.get(o, False)]
        self.cfg["locked_from"] = {"file": str(path), "sha256": pc.sha_file(path), "kandidat": keep,
                                   "fusion": L["fusion"], "threshold": L.get("threshold"), "bugfix": L.get("bugfix")}
        if budget_scale:
            self.cfg["budget_scale"] = float(budget_scale)
        self.hash = hashlib.sha256(json.dumps(self.cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # ---------------- infrastruktur ----------------
    def stop(self):
        if self.stop_at and datetime.datetime.now() >= self.stop_at and not self.stopper.requested:
            self.stopper.requested, self.stopper.reason = True, f"--stop-at {self.args.stop_at}"
            LOG.warning("Waktu --stop-at tercapai: berhenti setelah epoch/unit berjalan.")
        return self.stopper.should_stop()

    def check_config(self):
        p = self.rdir / "config.json"
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            if old.get("config_hash") != self.hash:
                LOG.error("config_hash berbeda (%s di folder vs %s sekarang): resume DITOLAK agar hasil tidak tercampur. "
                          "Pakai --results-dir baru.", old.get("config_hash"), self.hash)
                return False
        else:
            pc.sp.write_json_atomic(p, {"config_hash": self.hash, "config": self.cfg,
                                        "created": datetime.datetime.now().isoformat(timespec="seconds")})
        return True

    def data(self):
        self.cohort, self.excluded = pc.load_cohort(self.subset)
        self.pheno, self.overrides = pc.load_pheno(self.cohort)
        mr = pc.missing_report(self.cohort, self.pheno)
        pc.sp.write_csv_atomic(self.rdir / "missing_report.csv", mr)
        self.features = mr.loc[mr["used"], "feature"].tolist()
        self.splits = {int(k): pc.make_split(self.cohort, k, self.cfg["seed"]) for k in sorted(self.cohort["fold_strat"].unique())}
        pc.sp.write_json_atomic(self.rdir / "splits.json", {str(k): {p: v.tolist() for p, v in s.items()}
                                                             for k, s in self.splits.items()})
        LOG.info("Kohort %s: %d subjek (ASD %d) | fold %s | exclude QC %s | override %s | fitur fenotipik %s",
                 self.subset, len(self.cohort), int(self.cohort["label"].sum()), list(self.splits),
                 self.excluded, self.overrides, self.features)
        for r in mr.itertuples():
            LOG.info("Missing %-15s ASD %.1f%% | TD %.1f%% | selisih %.1f pp | dipakai %s", r.feature,
                     r.missing_pct_ASD, r.missing_pct_TD, r.gap_pp, r.used)

    def volumes(self):
        if self.vol is None:
            t = time.time()
            ids = self.cohort["SUB_ID"].to_numpy()
            v, m = pc.load_volumes(ids)
            self.vol = {"ids": ids, "idx": {int(s): i for i, s in enumerate(ids)}, "v": v, "m": m}
            self.pos = pc.slice_positions()
            LOG.info("Volume dimuat ke RAM: %d subjek (%.1f s) | posisi slice %s", len(ids), time.time() - t, self.pos)
        return self.vol

    def units(self):
        out = []
        folds = sorted(self.splits)
        for name in self.cfg["order"]:
            if name == "FUSION":
                out += [f"fusion_{s}:{k}" for s in ("S4", "S5") for k in folds]
                continue
            hp = self.cfg["models"][name]
            if hp["kind"] in ("sfcn", "dino"):
                out.append(f"cache_{name}")
            out += [f"{name}:{k}" for k in folds]
        if self.args.only:
            keep = self.args.only.split(",")
            out = [u for u in out if any(u == k or u.startswith(k.rstrip("*")) for k in keep)]
        return out

    # ---------------- unit fenotipik ----------------
    def run_pheno_unit(self, unit, name, fold):
        import pilot_models as pm
        hp = self.cfg["models"][name]
        t = time.time()
        probs, info = pm.run_pheno(name, hp, self.pheno, self.features, self.splits[fold], self.cohort,
                                   self.cfg["seed"] + fold)
        info.update(seconds=time.time() - t, features=self.features)
        m = save_unit(self.rdir, unit, self.cohort, fold, probs, self.splits[fold], {"info": info})
        return {"seconds": info["seconds"], "params_total": info.get("params_total"), "best_val_loss": info.get("best_val_loss"),
                "best_epoch": info.get("best_epoch"), "auc_val": m["val"]["auc"]}

    # ---------------- unit citra ----------------
    def _tensors(self, ids, source):
        import torch
        V = self.volumes()
        idx = [V["idx"][int(s)] for s in ids]
        if source == "vol":
            return torch.from_numpy(V["v"][idx]), torch.from_numpy(V["m"][idx])
        return torch.from_numpy(np.stack([np.load(self.cache_dir(source) / f"{int(s)}.npy") for s in ids]))

    def cache_dir(self, name):
        return self.rdir / "cache" / name

    def run_cache_unit(self, unit, name):
        import torch
        import pilot_models as pm
        hp = self.cfg["models"][name]
        V = self.volumes()
        d = self.cache_dir(name); d.mkdir(parents=True, exist_ok=True)
        todo = [s for s in V["ids"] if not (d / f"{int(s)}.npy").exists()]
        LOG.info("Cache %s: %d/%d subjek perlu dihitung", name, len(todo), len(V["ids"]))
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        if name == "sfcn":
            net = pm.sfcn_pretrained(hp).to(pm.DEV)
            frozen = torch.nn.Sequential(*[net.feature_extractor[i] for i in range(4)]).eval()
        else:
            import timm
            net = timm.create_model(hp["timm"], pretrained=True, num_classes=0, img_size=hp["img_size"]).to(pm.DEV).eval()
        for i, s in enumerate(todo):
            v, m = self._tensors([s], "vol")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                v, m = v.to(pm.DEV), m.to(pm.DEV)
                if name == "sfcn":
                    f = frozen(pm.sfcn_input(v, m))[0]                        # (256,10,12,10)
                else:
                    imgs = pm.make_slices(v, m, self.pos, self.cfg["slices"]["size"])[0]   # (15,3,224,224)
                    tok = net.forward_features(imgs)                            # (15,1+N,384)
                    f = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], 1)           # (15,768)
            pc.sp.save_npy_atomic(d / f"{int(s)}.npy", f.float().cpu().numpy().astype(np.float16))
            if (i + 1) % 25 == 0:
                LOG.info("Cache %s: %d/%d (%.0f s)", name, i + 1, len(todo), time.time() - t)
                write_progress(self.rdir, unit=unit, epoch=None, last_message=f"cache {i + 1}/{len(todo)}")
            if self.stop():
                raise pm.StopRequested()
        vram = torch.cuda.max_memory_allocated() / 2**20
        pc.sp.write_json_atomic(d.parent.parent / "units" / f"cache_{name}" / "_" / "cache_done.json",
                                {"n": len(V["ids"]), "seconds": time.time() - t, "vram_peak_mib": vram})
        del net
        torch.cuda.empty_cache()
        return {"seconds": time.time() - t, "vram_peak_mib": vram}

    def run_image_unit(self, unit, name, fold):
        import torch
        import pilot_models as pm
        hp = self.cfg["models"][name]
        sp_ = self.splits[fold]
        lab = self.cohort.set_index("SUB_ID")["label"]
        y = {k: torch.tensor(lab.loc[v].to_numpy(), dtype=torch.float32) for k, v in sp_.items()}
        seed = self.cfg["seed"] + fold
        pm.seed_all(seed)
        kind = hp["kind"]
        if kind == "2d":
            model = pm.Slice25D(hp["timm"], hp["train_modules"])
            data = {k: self._tensors(v, "vol") for k, v in sp_.items()}
        elif kind == "sfcn":
            model = pm.SFCNTail(pm.sfcn_pretrained(hp))
            data = {k: (self._tensors(v, "sfcn"),) for k, v in sp_.items()}
        else:
            model = pm.SliceFeatHead(768)
            data = {k: (self._tensors(v, "dinov2_s"),) for k, v in sp_.items()}
        model = model.to(pm.DEV)
        ptot, ptr = pm.count_params(model)
        B = hp["batch"]

        def prep(part, idx, train):
            if kind == "2d":
                v, m = data[part][0][idx].to(pm.DEV), data[part][1][idx].to(pm.DEV)
                if train:
                    v, m = pm.augment3d(v.float(), m, gen_aug, self.cfg["augment"])
                return pm.make_slices(v, m, self.pos, self.cfg["slices"]["size"])
            return data[part][0][idx].to(pm.DEV).float()

        gen_aug = None

        def batches(ep):
            nonlocal gen_aug
            g = torch.Generator().manual_seed(seed * 1000 + ep)
            gen_aug = torch.Generator(device=pm.DEV).manual_seed(seed * 1000 + ep)
            perm = torch.randperm(len(sp_["inner_train"]), generator=g)
            for i in range(0, len(perm), B):
                b = perm[i:i + B]
                yield prep("inner_train", b, True), y["inner_train"][b].to(pm.DEV)

        def predict(part):
            model.eval()
            P, E = [], []
            with torch.no_grad(), torch.autocast(pm.DEV.type, dtype=torch.float16):
                for i in range(0, len(sp_[part]), B):
                    b = torch.arange(i, min(i + B, len(sp_[part])))
                    lg, e = model(prep(part, b, False))
                    P.append(lg.float().cpu()); E.append(e.float().cpu())
            return torch.cat(P), torch.cat(E)

        pw = float((y["inner_train"] == 0).sum() / max((y["inner_train"] == 1).sum(), 1))

        def evaluate(_m):
            lg, _ = predict("inner_val")
            return float(torch.nn.functional.binary_cross_entropy_with_logits(
                lg, y["inner_val"], pos_weight=torch.tensor(pw)))

        first = {}

        def on_epoch(ep, dt, vl, vram, st):
            LOG.info("%s | epoch %d/%d | %.1f s | val loss %.4f | best epoch %d | VRAM puncak %.0f MiB",
                     unit, ep + 1, hp["epochs"], dt, vl, st["best_epoch"] + 1, vram)
            if not first and ep == 0:
                first["s"] = dt
                if fold == min(self.splits):
                    est = dt * hp["epochs"] * len(self.splits) / 60
                    LOG.info("ESTIMASI %s: ±%.1f s/epoch -> maks ±%.0f menit untuk %d fold (early stopping biasanya "
                             "lebih cepat)", name, dt, est, len(self.splits))
            pc.update_status(self.rdir, unit, last_epoch=ep + 1, max_epochs=hp["epochs"], best_epoch=st["best_epoch"] + 1,
                             best_val_loss=st["best"], seconds=st["elapsed"], vram_peak_mib=vram,
                             params_total=ptot, params_trainable=ptr)
            write_progress(self.rdir, unit=unit, epoch=ep + 1, max_epochs=hp["epochs"], epoch_s=round(dt, 1),
                           eta_unit_min=round(dt * (hp["epochs"] - ep - 1) / 60, 1))

        ckpt = pc.unit_dir(self.rdir, unit) / "ckpt"
        budget = hp.get("budget_min", 60) * 60 / len(self.splits)
        st = pm.train_loop(model, hp, batches, evaluate, ckpt, seed, self.stop, budget, on_epoch,
                           y["inner_train"].numpy())
        probs, embs = {}, {}
        for part in ("inner_train", "inner_val", "test"):
            lg, e = predict(part)
            probs[part] = torch.sigmoid(lg).numpy(); embs[part] = e.numpy().astype(np.float32)
        vram = max(h["vram_mib"] for h in st["history"])
        info = {"best_epoch": st["best_epoch"] + 1, "epochs_run": st["epoch"] + 1, "best_val_loss": st["best"],
                "seconds": st["elapsed"], "vram_peak_mib": vram, "params_total": ptot, "params_trainable": ptr,
                "s_per_epoch": float(np.mean([h["seconds"] for h in st["history"]])), "history": st["history"],
                "trainable_modules": hp.get("train_modules"), "frozen": hp.get("frozen", "selain train_modules")}
        save_unit(self.rdir, unit, self.cohort, fold, probs, sp_, {"info": info}, emb=embs)
        del model
        torch.cuda.empty_cache()
        return {"seconds": st["elapsed"], "vram_peak_mib": vram, "params_total": ptot, "params_trainable": ptr,
                "best_epoch": st["best_epoch"] + 1, "best_val_loss": st["best"]}

    # ---------------- fusion ----------------
    def selection(self):
        """Model terbaik per modalitas = AUC pooled inner val lima fold, hanya model yang SEMUA fold-nya done."""
        sel = {}
        for mod, names in self.cfg["modality"].items():
            cand = {}
            for n in names:
                us = [f"{n}:{k}" for k in self.splits]
                if all(pc.unit_done_valid(self.rdir, u) for u in us):
                    p = pd.concat([pd.read_csv(pc.unit_dir(self.rdir, u) / "preds.csv") for u in us])
                    v = p[p.part == "inner_val"]
                    cand[n] = pc.metrics(v.label, v.prob)["auc"]
            sel[mod] = {"best": max(cand, key=cand.get) if cand else None, "auc_inner_val": cand}
        return sel

    def run_fusion_unit(self, unit, scen, fold):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from lightgbm import LGBMClassifier
        import pilot_models as pm
        sel = self.selection()
        img_mod = "img25d" if scen == "S4" else "img3d"
        img, ph = sel[img_mod]["best"], sel["pheno"]["best"]
        if img is None or ph is None:
            raise RuntimeError(f"SKIP: model {img_mod}/pheno belum lengkap 5 fold")
        sp_ = self.splits[fold]
        seed = self.cfg["seed"] + fold
        lab = self.cohort.set_index("SUB_ID")["label"]
        yv = lab.loc[sp_["inner_val"]].to_numpy()
        P = {}
        for n in (img, ph):
            p = pd.read_csv(pc.unit_dir(self.rdir, f"{n}:{fold}") / "preds.csv").set_index("SUB_ID")
            P[n] = {part: p.loc[sp_[part], "prob"].to_numpy() for part in ("inner_val", "test")}
        lg = lambda a: np.log(np.clip(a, 1e-6, 1 - 1e-6) / (1 - np.clip(a, 1e-6, 1 - 1e-6)))
        out = {}
        # F1 late fusion
        Xv = np.c_[lg(P[img]["inner_val"]), lg(P[ph]["inner_val"])]
        Xt = np.c_[lg(P[img]["test"]), lg(P[ph]["test"])]
        out["F1_stack_logreg"] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xv, yv).predict_proba(Xt)[:, 1]
        out["F1_mean_prob"] = (P[img]["test"] + P[ph]["test"]) / 2
        # F2 intermediate fusion
        e = np.load(pc.unit_dir(self.rdir, f"{img}:{fold}") / "emb.npz")
        sc = StandardScaler().fit(e["emb_inner_train"])
        pre = pm.pheno_preprocessor(self.features).fit(self.pheno.loc[sp_["inner_train"], self.features])
        ph_v = pre.transform(self.pheno.loc[sp_["inner_val"], self.features])
        ph_t = pre.transform(self.pheno.loc[sp_["test"], self.features])
        variants = {"raw128": (sc.transform(e["emb_inner_val"]), sc.transform(e["emb_test"]))}
        for k in self.cfg["fusion"]["pca_dims"]:
            pca = PCA(n_components=k, random_state=seed).fit(sc.transform(e["emb_inner_train"]))
            variants[f"pca{k}"] = (pca.transform(sc.transform(e["emb_inner_val"])), pca.transform(sc.transform(e["emb_test"])))
        for vn, (ev, et) in variants.items():
            Zv, Zt = np.c_[ev, ph_v], np.c_[et, ph_t]
            heads = {"logreg": LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
                     "mlp": MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=3000, random_state=seed),
                     "lgbm": LGBMClassifier(n_estimators=100, num_leaves=4, min_child_samples=5, learning_rate=0.05,
                                            class_weight="balanced", random_state=seed, verbose=-1)}
            for hn, h in heads.items():
                out[f"F2_{vn}_{hn}"] = h.fit(Zv, yv).predict_proba(Zt)[:, 1]
        d = pc.unit_dir(self.rdir, unit); d.mkdir(parents=True, exist_ok=True)
        c = self.cohort.set_index("SUB_ID")
        rows = [{"SUB_ID": int(s), "SITE_ID": c.at[s, "SITE_ID"], "fold": fold, "part": "test", "label": int(c.at[s, "label"]),
                 "method": mth, "prob": float(p)} for mth, ps in out.items() for s, p in zip(sp_["test"], ps)]
        df = pd.DataFrame(rows)
        pc.sp.write_csv_atomic(d / "preds.csv", df)
        mets = {mth: pc.metrics(g.label, g.prob) for mth, g in df.groupby("method")}
        pc.sp.write_json_atomic(d / "metrics.json", {"test": mets, "selection": {"img": img, "pheno": ph},
                                                     "note": self.cfg["fusion"]["note"]})
        return {"seconds": 0}

    def fusion_current(self, unit):
        try:
            m = json.loads((pc.unit_dir(self.rdir, unit) / "metrics.json").read_text(encoding="utf-8"))
            sel = self.selection()
            img_mod = "img25d" if unit.startswith("fusion_S4") else "img3d"
            return m["selection"] == {"img": sel[img_mod]["best"], "pheno": sel["pheno"]["best"]}
        except Exception:
            return False

    # ---------------- laporan Tabel A begitu fenotipik selesai ----------------
    def maybe_report_A(self, units):
        """Sekali saja: semua unit fenotipik done -> make_tables, cetak Tabel A, salin ke folder proyek."""
        ph = [u for u in units if u.split(":")[0] in self.cfg["modality"]["pheno"]]
        flag = self.rdir / "tables" / ".tabel_A_dicetak"
        if not ph or flag.exists() or not all(pc.unit_done_valid(self.rdir, u) for u in ph):
            return
        try:
            import make_tables as mt
            mt.build(self.rdir, self.subset)
            txt = mt.console_A(self.rdir)
            print(txt, flush=True)
            (self.rdir / "tables" / "tabel_A_layar.txt").write_text(txt, encoding="utf-8")
            where = mt.mirror(self.rdir, self.subset) if not self.args.results_dir else None
            flag.write_text(datetime.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
            LOG.info("Tabel A disimpan: %s%s | lanjut ke model citra.", self.rdir / "tables" / "tabel_A_fenotipik.csv",
                     f" | salinan: {where}" if where else "")
        except Exception as e:                     # pelaporan tidak boleh menghentikan training
            LOG.warning("Tabel A gagal dibuat (training tetap lanjut): %s", e)

    def dispatch(self, u, model, fold, is_fusion):
        if u.startswith("cache_"):
            return self.run_cache_unit(u, u[len("cache_"):])
        if is_fusion:
            return self.run_fusion_unit(u, model.split("_")[1], fold)
        if self.cfg["models"][model]["kind"] == "pheno":
            return self.run_pheno_unit(u, model, fold)
        return self.run_image_unit(u, model, fold)

    # ---------------- loop utama ----------------
    def run(self):
        import pilot_models as pm
        units = self.units()
        st = pc.read_status(self.rdir).set_index("unit") if (self.rdir / "status.csv").exists() else None
        for u in units:
            if st is None or u not in st.index:
                pc.update_status(self.rdir, u, model=u.split(":")[0], fold=u.split(":")[1] if ":" in u else "",
                                 status="pending", config_hash=self.hash)
        running = [u for u in units if st is not None and u in st.index and st.at[u, "status"] == "running"]
        if running:
            LOG.warning("Unit berstatus running dari run sebelumnya (crash/putus): %s -> dilanjutkan dari checkpoint.", running)
        n_done = 0
        exit_code = 0
        t_run = time.time()
        self.maybe_report_A(units)                 # bila fenotipik sudah selesai di run sebelumnya
        queue, rechecked, i = list(units), False, -1
        while True:
            i += 1
            if i >= len(queue):
                # seleksi model bisa berubah setelah model terakhir (mis. ConvNeXt) selesai -> fusion dihitung ulang
                if rechecked or exit_code == 3:
                    break
                rechecked = True
                stale = [u for u in units if u.startswith("fusion_") and pc.unit_done_valid(self.rdir, u)
                         and not self.fusion_current(u)]
                if not stale:
                    break
                LOG.info("Seleksi model berubah -> fusion dihitung ulang: %s", stale)
                n_done -= len(stale)
                queue += stale
                continue
            u = queue[i]
            if self.stop():
                exit_code = 3
                break
            stt = pc.read_status(self.rdir).set_index("unit").at[u, "status"]
            is_fusion = u.startswith("fusion_")
            if pc.unit_done_valid(self.rdir, u) and (not is_fusion or self.fusion_current(u)):
                if stt != "done":
                    pc.update_status(self.rdir, u, status="done")
                n_done += 1
                continue
            err = str(pc.read_status(self.rdir).set_index("unit").at[u, "error"] or "")
            if not self.args.retry_failed and (stt == "failed" or (stt == "skipped" and err.startswith("budget"))):
                LOG.info("Lewati %s (%s; pakai --retry-failed)", u, stt)
                continue
            model, _, fold = u.partition(":")
            fold = int(fold) if fold else None
            udir = pc.unit_dir(self.rdir, u)
            udir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(udir / "unit.log", encoding="utf-8")
            fh.setFormatter(logging.getLogger().handlers[0].formatter); logging.getLogger().addHandler(fh)
            pc.update_status(self.rdir, u, status="running", error=None, config_hash=self.hash)
            write_progress(self.rdir, subset=self.subset, unit=u, unit_index=i + 1, units_total=len(units),
                           units_done=n_done, epoch=None, last_message="mulai")
            LOG.info("=== Unit %d/%d: %s ===", i + 1, len(queue), u)
            try:
                info = self.dispatch(u, model, fold, is_fusion)
                ok = pc.unit_done_valid(self.rdir, u)
                pc.update_status(self.rdir, u, status="done" if ok else "failed",
                                 error=None if ok else "output tidak valid", **{k: v for k, v in info.items()
                                                                                if k in pc.STATUS_COLS})
                n_done += ok
                LOG.info("Unit %s selesai (%s) | progres %d/%d unit selesai | %.1f menit sejak mulai", u,
                         "done" if ok else "failed", n_done, len(units), (time.time() - t_run) / 60)
                if ok:
                    self.maybe_report_A(units)
            except pm.StopRequested:
                pc.update_status(self.rdir, u, status="running", error="dihentikan (checkpoint tersimpan)")
                LOG.warning("Berhenti rapi di %s (%s); lanjutkan dengan --resume.", u, self.stopper.reason)
                exit_code = 3
                logging.getLogger().removeHandler(fh); fh.close()
                break
            except pm.BudgetExceeded as e:
                pc.update_status(self.rdir, u, status="skipped", error=f"budget: {e}")
                LOG.warning("Unit %s dilewati: %s", u, e)
                exit_code = max(exit_code, 1)
            except Exception as e:
                tb = traceback.format_exc()
                skip = str(e).startswith("SKIP")
                pc.update_status(self.rdir, u, status="skipped" if skip else "failed", error=f"{type(e).__name__}: {e}"[:500])
                (LOG.warning if skip else LOG.error)("Unit %s %s: %s", u, "dilewati" if skip else "GAGAL", e)
                if not skip:
                    LOG.error(tb)
                exit_code = max(exit_code, 1)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            logging.getLogger().removeHandler(fh); fh.close()
            write_progress(self.rdir, units_done=n_done, last_message=f"selesai {u}")
        write_progress(self.rdir, unit=None, epoch=None, units_done=n_done, units_total=len(units),
                       last_message="berhenti" if exit_code == 3 else "run selesai")
        LOG.info("RINGKASAN: %d/%d unit done | %.1f menit | %s", n_done, len(units), (time.time() - t_run) / 60,
                 "dihentikan" if exit_code == 3 else "selesai")
        return exit_code


def main(argv=None, cls=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["pilot200", "all"], default="pilot200")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--stop-at", default=None, help="HH:MM, berhenti rapi pada jam ini")
    ap.add_argument("--only", default=None, help="uji: daftar unit/prefix, mis. effnet_b0:0")
    ap.add_argument("--max-epochs", type=int, default=None, help="uji saja (mengubah config_hash)")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--locked", default=None, help="data penuh: locked_config_kandidat.json (hanya kandidat terkunci)")
    ap.add_argument("--budget-scale", type=float, default=None, help="pengali batas waktu pengaman per model (data penuh)")
    args = ap.parse_args(argv)
    r = (cls or Runner)(args)
    setup_logging(r.rdir)
    LOG.info("06 %s | subset %s | hasil %s | config_hash %s", r.cfg["pilot_version"], r.subset, r.rdir, r.hash)
    if not r.check_config():
        return 2
    if (r.rdir / "STOP").exists():
        LOG.error("File STOP ada di %s. Hapus dulu (atau pakai resume_pilot.ps1).", r.rdir / "STOP")
        return 2
    if (r.rdir / "status.csv").exists() and not args.resume:
        LOG.error("Sudah ada hasil di %s. Gunakan --resume (tidak ada yang ditimpa).", r.rdir)
        return 2
    import psutil
    ram = psutil.virtual_memory().available / 2**30
    if ram < 4.0:
        LOG.error("RAM tersedia %.2f GiB < 4 GiB: tidak dimulai.", ram)
        return 2
    shim = Shim(r.rdir)
    try:
        lock = pc.sp.RunLock(shim)
        lock.__enter__()
    except RuntimeError as e:
        LOG.error("%s", e)
        return 2
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
