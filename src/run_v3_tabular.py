"""v3 — model tabel (fenotipik & ROI): E1 (LogReg-CV, TabM, CatBoost, TabPFN v2), E3 tuning Optuna, LOSO.
Config dibaca dari locked_config_v3.json; hash file harus sama dengan locked_config_v3.sha256 (dikunci sebelum run).

    .venv-model\\Scripts\\python.exe src\\run_v3_tabular.py [--resume] [--stop-at 07:00]
    .venv-model\\Scripts\\python.exe src\\run_v3_tabular.py --test            # uji singkat: 1 fold, 1 seed, 3 trial -> results\\v3_test

Berhenti rapi: file STOP di folder hasil / --stop-at HH:MM / Ctrl+C sekali -> unit berjalan diselesaikan dulu.
Unit gagal dicatat (status failed + error) lalu runner lanjut ke unit berikutnya.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc  # noqa: E402
import run_pilot as rp  # noqa: E402

LOG = logging.getLogger("pilot")
PROJECT = Path(__file__).resolve().parents[1]
CFG_PATH = pc.RESULTS_ROOT / "v3" / "locked_config_v3.json"     # RESULTS_ROOT = env TA06_RESULTS_ROOT (default C:\ta_local\results)
FULL = pc.RESULTS_ROOT / "full_v2"
TABPFN_PY = Path(os.environ.get("TA06_TABPFN_PY", PROJECT / ".venv-tabpfn" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
FEATS = ["pheno", "roi"]
E1_MODELS = ["logreg_cv", "tabm", "catboost", "tabpfn"]
E3_MODELS = ["logreg", "tabm", "catboost"]
NEURAL = {"tabm", "mlp_tab"}
warnings.filterwarnings("ignore", category=FutureWarning)


def lg(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


# ================================================================================================ seleksi (dipakai juga analisis)
def model_preds(key, rdir=None, full=FULL):
    """key = 'v3:<prefix>' (unit v3, seed dirata-rata) atau 'full:<model>' -> DataFrame part/SUB_ID/fold/label/SITE_ID/prob."""
    src, name = key.split(":", 1)
    base = (rdir if src == "v3" else full) / "units"
    dirs = sorted(d for d in base.glob(f"{name}_s*") if d.name[len(name) + 2:].isdigit()) if src == "v3" else [base / name]
    if not dirs or not all(d.exists() for d in dirs):
        return None
    ps = []
    for d in dirs:
        fs = sorted(d.glob("fold*/preds.csv"))
        if not fs:
            return None
        ps.append(pd.concat([pd.read_csv(f) for f in fs], ignore_index=True))
    keys = ["part", "SUB_ID", "fold", "label", "SITE_ID"]
    return pd.concat(ps).groupby(keys, as_index=False)["prob"].mean()


def candidates(cfg):
    c = {"fenotipik": [f"v3:e1_pheno_{m}" for m in E1_MODELS] + [f"v3:e3_pheno_{m}" for m in E3_MODELS] +
                      ["full:logreg", "full:mlp_tab", "full:tabm"],
         "roi": [f"v3:e1_roi_{m}" for m in E1_MODELS] + [f"v3:e3_roi_{m}" for m in E3_MODELS] + ["full:roi_logreg_cb", "full:roi_tabm_cb"],
         "citra_2_5d": ["full:effnet_b0", "full:resnet18", "full:scratch2d"],
         "citra_3d": ["full:scratch3d"]}
    return c


def selection(rdir, cfg, n_folds=5):
    """Skor seleksi = AUC pooled inner (tabel: inner_oof; citra: inner_val). Tidak memakai test."""
    out, scores = {}, []
    for mod, keys in candidates(cfg).items():
        best = None
        for k in keys:
            p = model_preds(k, rdir)
            if p is None:
                scores.append({"modalitas": mod, "model": k, "skor_seleksi": np.nan, "bagian": "belum ada"}); continue
            part = "inner_oof" if (p.part == "inner_oof").any() else "inner_val"
            q = p[p.part == part]
            if q.fold.nunique() < n_folds:
                scores.append({"modalitas": mod, "model": k, "skor_seleksi": np.nan, "bagian": "fold belum lengkap"}); continue
            s = auc(q.label, q.prob)
            scores.append({"modalitas": mod, "model": k, "skor_seleksi": s, "bagian": part})
            if best is None or s > best[1]:
                best = (k, s, part)
        if best:
            out[mod] = {"model": best[0], "skor": best[1], "bagian": best[2], "masuk": bool(best[1] >= 0.55)}
    return out, pd.DataFrame(scores)


# ================================================================================================ runner
class V3:
    def __init__(self, a):
        self.args = a
        raw = CFG_PATH.read_bytes()
        self.cfg = json.loads(raw)
        self.file_hash = hashlib.sha256(raw).hexdigest()
        lock = CFG_PATH.with_suffix(".sha256")
        if not lock.exists() or lock.read_text(encoding="utf-8").split()[0] != self.file_hash:
            raise SystemExit(f"Config BELUM dikunci / berubah setelah dikunci: sha256 {self.file_hash} tidak cocok dengan {lock}")
        self.hash = self.file_hash[:16] + ("-test" if a.test else "")
        self.rdir = Path(a.results_dir) if a.results_dir else pc.RESULTS_ROOT / ("v3_test" if a.test else "v3")
        self.rdir.mkdir(parents=True, exist_ok=True)
        self.folds = [0] if a.test else [0, 1, 2, 3, 4]
        self.seeds = [0] if a.test else self.cfg["seed"]["neural_multi_seed"]
        self.trials = 3 if a.test else self.cfg["E3_tuning"]["trial"]
        self.stop_at = None
        if a.stop_at:
            t = datetime.datetime.combine(datetime.date.today(), datetime.datetime.strptime(a.stop_at, "%H:%M").time())
            self.stop_at = t if t > datetime.datetime.now() else t + datetime.timedelta(days=1)

    # ------------------------------------------------------------------ data
    def data(self):
        fcfg = json.loads((FULL / "config.json").read_text(encoding="utf-8"))
        self.fv2 = fcfg["config"]
        assert fcfg["config_hash"] == "560c89845312e371", "full_v2 config_hash berubah"
        self.cohort, _ = pc.load_cohort("all")
        self.pheno, _ = pc.load_pheno(self.cohort)
        self.features = self.fv2["pheno_features"]
        self.roi = pd.read_csv(FULL / "cache" / "roi" / "roi_features.csv").set_index("SUB_ID").loc[self.cohort.SUB_ID]
        bad = [c for c in list(self.roi.columns) + self.features if c.upper().startswith(("SITE", "DX_", "ADOS", "ADI", "SRS", "SCQ", "AQ", "DSM"))]
        assert not bad, f"kolom terlarang di fitur: {bad}"
        self.cov = self.pheno[["AGE_AT_SCAN", "SEX"]]
        c = self.cohort.set_index("SUB_ID")
        self.lab, self.site = c["label"], c["SITE_ID"]
        raw = json.loads((FULL / "splits.json").read_text(encoding="utf-8"))
        self.splits = {int(k): {p: np.asarray(v) for p, v in s.items()} for k, s in raw.items()}
        for k, s in self.splits.items():
            m = pc.make_split(self.cohort, k, self.fv2["seed"])
            assert all(np.array_equal(np.sort(m[p]), np.sort(s[p])) for p in s), f"split fold {k} beda dengan full_v2"
        heldout = set(pd.read_csv(PROJECT / "data" / "folds.csv").query("split == 'test'").SUB_ID)
        assert not heldout & set(self.cohort.SUB_ID), "subjek test held-out masuk kohort!"
        f = pd.read_csv(PROJECT / "data" / "folds.csv").set_index("SUB_ID").loc[self.cohort.SUB_ID]
        self.loso = f[f.loso_eligible.astype(bool)]["loso_site"]
        LOG.info("Kohort %d subjek | ROI %d fitur | fenotipik %s | LOSO %d subjek / %d site",
                 len(self.cohort), self.roi.shape[1], self.features, len(self.loso), self.loso.nunique())

    def X(self, feat):
        return self.pheno if feat == "pheno" else self.roi

    def prep(self, feat, tr, others):
        """Fit preprocessing di tr saja; kembalikan Z_tr dan Z untuk tiap array id lain."""
        import pilot_models as pm
        X = self.X(feat)
        cols = self.features if feat == "pheno" else list(X.columns)
        p = pm.TabPre("pheno" if feat == "pheno" else "roi", cols, feat == "roi")
        cv = lambda ids: self.cov.loc[ids].reset_index(drop=True)
        p.fit(X.loc[tr], self.site.loc[tr].to_numpy(), cv(tr))
        tf = lambda ids: p.transform(X.loc[ids], self.site.loc[ids].to_numpy(), cv(ids))
        Ztr = tf(tr)
        outs, unseen = [], 0
        for o in others:
            outs.append(tf(o)); unseen += getattr(p, "unseen", 0)
        return Ztr, outs, unseen

    # ------------------------------------------------------------------ model
    def hp(self, model, params=None):
        if model == "tabm":
            h = dict(self.fv2["models"]["tabm"]); h.update(params or {}); return h
        if model == "mlp_tab":
            return dict(self.fv2["models"]["mlp_tab"])
        if model == "catboost":
            e = self.cfg["E1"]["catboost"]
            h = {k: e[k] for k in ("iterations", "depth", "learning_rate", "l2_leaf_reg", "early_stopping_rounds")}
            h.update(params or {}); return h
        return dict(params or {})

    def fit_predict(self, model, hp, Z, y, Zs, seed):
        import pilot_models as pm
        if model == "logreg":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=5000).fit(Z, y)
            return [m.predict_proba(A)[:, 1] for A in Zs]
        if model == "catboost":
            from catboost import CatBoostClassifier
            from sklearn.model_selection import train_test_split
            tr, es = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
            m = CatBoostClassifier(iterations=int(hp["iterations"]), depth=int(hp["depth"]), learning_rate=hp["learning_rate"],
                                   l2_leaf_reg=hp["l2_leaf_reg"], auto_class_weights="Balanced", random_seed=seed,
                                   thread_count=self.cfg["E1"]["catboost"]["thread_count"], verbose=0,
                                   early_stopping_rounds=int(hp["early_stopping_rounds"]), allow_writing_files=False)
            m.fit(Z[tr], y[tr], eval_set=(Z[es], y[es]))
            return [m.predict_proba(A)[:, 1] for A in Zs]
        f = pm.fit_tab_model("tabm" if model == "tabm" else "mlp_tab", hp, Z, y, seed)
        return [f(A) for A in Zs]

    def select_C(self, feat, tr, seed):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        grid = self.cfg["E1"]["logreg_cv"]["C_grid"]
        y = self.lab.loc[tr].to_numpy()
        sc = np.zeros(len(grid))
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(tr, y):
            Za, (Zb,), _ = self.prep(feat, tr[a], [tr[b]])
            for i, C in enumerate(grid):
                sc[i] += auc(y[b], LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Za, y[a]).predict_proba(Zb)[:, 1])
        return float(grid[int(np.flatnonzero(np.isclose(sc, sc.max()))[0])])

    def tabpfn_batch(self, jobs):
        e = self.cfg["E1"]["tabpfn"]
        with tempfile.TemporaryDirectory(dir=self.rdir) as td:
            src, dst = Path(td) / "in.npz", Path(td) / "out.npz"
            arr = {"n_jobs": len(jobs)}
            for i, (Z, y, Zt) in enumerate(jobs):
                arr[f"Xtr_{i}"], arr[f"ytr_{i}"], arr[f"Xte_{i}"] = Z.astype(np.float32), y, Zt.astype(np.float32)
            np.savez(src, **arr)
            r = subprocess.run([str(TABPFN_PY), str(PROJECT / "src" / "tabpfn_worker.py"), str(src), str(dst),
                                str(e["n_estimators"]), str(e["random_state"])], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("tabpfn_worker gagal: " + r.stderr[-1500:])
            self.tabpfn_info = json.loads(r.stdout.strip().splitlines()[-1])
            with np.load(dst) as d:
                return [np.array(d[f"p_{i}"]) for i in range(len(jobs))]

    def jobs_fit(self, feat, model, hp, jobs, seed):
        """jobs: list (train_ids, [pred_ids...]) -> list of list prob. Preprocessing fit di train tiap job."""
        prepared, info = [], {"C": [], "combat_unseen": 0}
        for tr, preds in jobs:
            Z, Zs, un = self.prep(feat, tr, preds)
            info["combat_unseen"] += un
            prepared.append((tr, Z, Zs))
        if model == "tabpfn":
            flat = [(Z, self.lab.loc[tr].to_numpy(), np.vstack(Zs)) for tr, Z, Zs in prepared]
            outs = self.tabpfn_batch(flat)
            res = []
            for (tr, Z, Zs), p in zip(prepared, outs):
                cut = np.cumsum([len(z) for z in Zs])[:-1]
                res.append(np.split(p, cut))
            return res, info
        res = []
        for tr, Z, Zs in prepared:
            h = hp
            if model == "logreg_cv":
                h = {"C": self.select_C(feat, tr, seed)}; info["C"].append(h["C"])
            res.append(self.fit_predict("logreg" if model == "logreg_cv" else model, h, Z, self.lab.loc[tr].to_numpy(), Zs, seed))
        return res, info

    # ------------------------------------------------------------------ unit: protokol 5-fold (OOF inner CV + fit akhir)
    def run_cv_unit(self, unit, feat, model, hp, fold, s):
        from sklearn.model_selection import StratifiedKFold
        sp_ = self.splits[fold]
        T = np.concatenate([sp_["inner_train"], sp_["inner_val"]])
        yT = self.lab.loc[T].to_numpy()
        seed = self.cfg["seed"]["basis"] + fold + 1000 * s
        cv = list(StratifiedKFold(5, shuffle=True, random_state=self.cfg["seed"]["basis"] + fold).split(T, yT))
        jobs = [(T[a], [T[b]]) for a, b in cv] + [(T, [sp_["test"]])]
        res, info = self.jobs_fit(feat, model, hp, jobs, seed)
        oof = np.full(len(T), np.nan)
        for (a, b), r in zip(cv, res[:-1]):
            oof[b] = r[0]
        probs = {"inner_oof": oof, "inner_val": pd.Series(oof, index=T).loc[sp_["inner_val"]].to_numpy(), "test": res[-1][0]}
        ids = {"inner_oof": T, "inner_val": sp_["inner_val"], "test": sp_["test"]}
        extra = {"info": {"feat": feat, "model": model, "seed": seed, "hp": hp, **info,
                          **({"tabpfn": self.tabpfn_info} if model == "tabpfn" else {})}}
        rp.save_unit(self.rdir, unit, self.cohort, fold, probs, ids, extra)

    def run_tune_unit(self, unit, feat, model, fold):
        import optuna
        from sklearn.model_selection import StratifiedKFold
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sp_ = self.splits[fold]
        T = np.concatenate([sp_["inner_train"], sp_["inner_val"]])
        yT = self.lab.loc[T].to_numpy()
        base = self.cfg["seed"]["basis"] + fold
        splits = []
        for a, b in StratifiedKFold(3, shuffle=True, random_state=base).split(T, yT):
            Za, (Zb,), _ = self.prep(feat, T[a], [T[b]])
            splits.append((Za, yT[a], Zb, yT[b]))

        def objective(tr):
            if model == "logreg":
                p = {"C": tr.suggest_float("C", 1e-4, 1e2, log=True)}
            elif model == "tabm":
                p = {"d_block": tr.suggest_categorical("d_block", [64, 128, 256]), "n_blocks": tr.suggest_categorical("n_blocks", [1, 2, 3]),
                     "dropout": tr.suggest_float("dropout", 0.0, 0.3), "lr": tr.suggest_float("lr", 3e-4, 3e-3, log=True),
                     "wd": tr.suggest_float("wd", 1e-5, 1e-3, log=True)}
            else:
                p = {"depth": tr.suggest_categorical("depth", [3, 4, 5, 6]), "learning_rate": tr.suggest_float("learning_rate", 0.01, 0.1, log=True),
                     "l2_leaf_reg": tr.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True), "iterations": tr.suggest_int("iterations", 300, 1500)}
            h = self.hp(model, p) if model != "logreg" else p
            sc = [auc(yb, self.fit_predict(model, h, Za, ya, [Zb], base)[0]) for Za, ya, Zb, yb in splits]
            if self.stopper.should_stop():
                tr.study.stop()
            return float(np.mean(sc))

        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=base))
        st.optimize(objective, n_trials=self.trials)
        if len(st.trials) < self.trials:
            raise _Stop()
        d = pc.unit_dir(self.rdir, unit); d.mkdir(parents=True, exist_ok=True)
        pc.sp.write_csv_atomic(d / "trials.csv", st.trials_dataframe())
        pc.sp.write_json_atomic(d / "best.json", {"best_params": st.best_params, "best_inner_auc": st.best_value,
                                                  "n_trials": len(st.trials), "fold": fold, "feat": feat, "model": model})

    # ------------------------------------------------------------------ unit LOSO
    def run_loso_unit(self, unit, feat, model, hp, si, site, s):
        ids = self.loso.index.to_numpy()
        te = ids[self.loso.to_numpy() == site]
        tr = ids[self.loso.to_numpy() != site]
        seed = self.cfg["seed"]["basis"] + si + 1000 * s
        res, info = self.jobs_fit(feat, model, hp, [(tr, [te])], seed)
        rp.save_unit(self.rdir, unit, self.cohort, si, {"test": res[0][0]}, {"test": te},
                     {"info": {"feat": feat, "model": model, "site": site, "seed": seed, "hp": hp, **info}})

    # ------------------------------------------------------------------ daftar unit
    def units(self):
        U = []
        for feat in FEATS:
            for m in E1_MODELS:
                for s in (self.seeds if m in NEURAL else [0]):
                    for k in self.folds:
                        U.append(dict(unit=f"e1_{feat}_{m}_s{s}:{k}", kind="cv", feat=feat, model=m, fold=k, s=s))
        for feat in FEATS:
            for m in E3_MODELS:
                for k in self.folds:
                    U.append(dict(unit=f"e3tune_{feat}_{m}:{k}", kind="tune", feat=feat, model=m, fold=k))
                for s in (self.seeds if m in NEURAL else [0]):
                    for k in self.folds:
                        U.append(dict(unit=f"e3_{feat}_{m}_s{s}:{k}", kind="cv_tuned", feat=feat, model=m, fold=k, s=s))
        return U

    def loso_units(self):
        sel, _ = selection(self.rdir, self.cfg, len(self.folds))
        pc.sp.write_json_atomic(self.rdir / "selection.json", sel)
        sites = sorted(self.loso.unique())
        U = []
        for mod, feat in (("fenotipik", "pheno"), ("roi", "roi")):
            if mod not in sel:
                continue
            src, name = sel[mod]["model"].split(":")
            m = self.loso_model(src, name)
            for s in (self.seeds if m in NEURAL else [0]):
                for si, site in enumerate(sites):
                    U.append(dict(unit=f"loso_{feat}_{m}_s{s}:{si}", kind="loso", feat=feat, model=m, fold=si, site=site, s=s))
            LOG.info("LOSO %s: terpilih %s -> tipe %s (setting E1)", mod, sel[mod]["model"], m)
        return U

    @staticmethod
    def loso_model(src, name):
        if src == "full":
            return {"logreg": "logreg_cv", "mlp_tab": "mlp_tab", "tabm": "tabm", "roi_logreg_cb": "logreg_cv", "roi_tabm_cb": "tabm"}[name]
        m = name.split("_", 2)[2]
        return "logreg_cv" if m == "logreg" else m

    def done(self, u):
        if u["kind"] == "tune":
            return (pc.unit_dir(self.rdir, u["unit"]) / "best.json").exists()
        return pc.unit_done_valid(self.rdir, u["unit"])

    def stop(self):
        if self.stop_at and datetime.datetime.now() >= self.stop_at and not self.stopper.requested:
            self.stopper.requested, self.stopper.reason = True, f"--stop-at {self.args.stop_at}"
        return self.stopper.should_stop()

    def exec_unit(self, u):
        if u["kind"] == "cv":
            self.run_cv_unit(u["unit"], u["feat"], u["model"], self.hp(u["model"]), u["fold"], u["s"])
        elif u["kind"] == "tune":
            self.run_tune_unit(u["unit"], u["feat"], u["model"], u["fold"])
        elif u["kind"] == "cv_tuned":
            best = json.loads((pc.unit_dir(self.rdir, f"e3tune_{u['feat']}_{u['model']}:{u['fold']}") / "best.json").read_text(encoding="utf-8"))
            self.run_cv_unit(u["unit"], u["feat"], u["model"], self.hp(u["model"], best["best_params"]), u["fold"], u["s"])
        else:
            self.run_loso_unit(u["unit"], u["feat"], u["model"], self.hp(u["model"]), u["fold"], u["site"], u["s"])

    def run_list(self, U, label):
        todo = [u for u in U if not self.done(u)]
        LOG.info("%s: %d unit, %d sudah selesai, %d dijalankan", label, len(U), len(U) - len(todo), len(todo))
        t0 = time.time()
        for i, u in enumerate(todo):
            if self.stop():
                LOG.warning("Berhenti rapi sebelum %s (%s); lanjutkan dengan --resume.", u["unit"], self.stopper.reason)
                return 3
            if self.args.only and not any(u["unit"].startswith(o) for o in self.args.only.split(",")):
                continue
            if u["kind"] == "cv_tuned" and not (pc.unit_dir(self.rdir, f"e3tune_{u['feat']}_{u['model']}:{u['fold']}") / "best.json").exists():
                LOG.warning("Lewati %s: tuning belum ada", u["unit"]); continue
            t = time.time()
            pc.update_status(self.rdir, u["unit"], model=u["unit"].split(":")[0], fold=u["fold"], status="running", config_hash=self.hash)
            try:
                self.exec_unit(u)
                dt = time.time() - t
                pc.update_status(self.rdir, u["unit"], status="done", seconds=round(dt, 1), error="")
                el = time.time() - t0
                LOG.info("[%d/%d] %s selesai %.1f s | %.1f menit berjalan | sisa ±%.0f menit", i + 1, len(todo), u["unit"], dt,
                         el / 60, el / (i + 1) * (len(todo) - i - 1) / 60)
            except _Stop:
                pc.update_status(self.rdir, u["unit"], status="stopped")
                LOG.warning("Berhenti rapi di %s; lanjutkan dengan --resume.", u["unit"])
                return 3
            except Exception as e:
                pc.update_status(self.rdir, u["unit"], status="failed", seconds=round(time.time() - t, 1), error=repr(e)[:500])
                LOG.exception("Unit %s GAGAL (dicatat, lanjut)", u["unit"])
            rp.write_progress(self.rdir, unit=u["unit"], selesai=i + 1, total=len(todo), tahap=label)
        return 0

    def run(self):
        rc = self.run_list(self.units(), "E1+E3")
        if rc:
            return rc
        if self.args.no_loso:
            return 0
        rc = self.run_list(self.loso_units(), "LOSO")
        st = pc.read_status(self.rdir)
        LOG.info("RINGKASAN: %s", st.status.value_counts().to_dict())
        return rc


class _Stop(Exception):
    pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at", default=None)
    ap.add_argument("--test", action="store_true", help="uji singkat: fold 0, seed 0, 3 trial -> results\\v3_test")
    ap.add_argument("--only", default=None, help="prefix unit, dipisah koma (uji)")
    ap.add_argument("--no-loso", action="store_true")
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args(argv)
    r = V3(a)
    rp.setup_logging(r.rdir)
    LOG.info("v3 tabel | hasil %s | config_hash %s | sha256 config %s", r.rdir, r.hash, r.file_hash)
    cj = r.rdir / "config.json"
    if cj.exists():
        old = json.loads(cj.read_text(encoding="utf-8"))["config_hash"]
        if old != r.hash:
            LOG.error("config_hash berbeda (%s di folder vs %s): ditolak agar hasil tidak tercampur.", old, r.hash); return 2
        if not a.resume:
            LOG.error("Sudah ada hasil di %s. Gunakan --resume (tidak ada yang ditimpa).", r.rdir); return 2
    else:
        pc.sp.write_json_atomic(cj, {"config_hash": r.hash, "config_sha256": r.file_hash, "config": r.cfg, "test": a.test,
                                     "created": datetime.datetime.now().isoformat(timespec="seconds")})
    if (r.rdir / "STOP").exists():
        LOG.error("File STOP ada di %s. Hapus dulu.", r.rdir / "STOP"); return 2
    import psutil
    ram = psutil.virtual_memory().available / 2**30
    if ram < 4.0:
        LOG.error("RAM tersedia %.2f GiB < 4 GiB: tidak dimulai.", ram); return 2
    if not TABPFN_PY.exists():
        LOG.error("%s tidak ada (venv TabPFN).", TABPFN_PY); return 2
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
