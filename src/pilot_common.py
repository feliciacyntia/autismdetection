"""06 — Utilitas eksperimen modeling (pilot subset 200 / data penuh): konfigurasi, data, split, metrik, status.

Dipakai oleh run_pilot.py, make_tables.py, pilot_progress.py. Dijalankan dengan .venv-model (paket model terisolasi
dari Python utama agar fingerprint preprocessing 05c tidak berubah).
"""
import copy
import datetime
import hashlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parent
sys.path.insert(0, str(SRC))
import smri_preprocessing as sp  # noqa: E402  (tulis atomik, RunLock, KeepAwake, StopController)

PILOT_VERSION = "06-pilot-v1"
IMAGE_ROOT = Path(os.environ.get("TA06_IMAGE_ROOT", r"C:\ta_local\data\smri_preproc"))
RESULTS_ROOT = Path(os.environ.get("TA06_RESULTS_ROOT", r"C:\ta_local\results"))
PRETRAINED = Path(os.environ.get("TA06_PRETRAINED", r"C:\ta_local\pretrained"))

FORBIDDEN_PREFIX = ("DX_GROUP", "ADOS", "ADI_R", "SRS", "SCQ", "AQ", "DSM_IV_TR", "SITE_ID", "label")

# --------------------------------------------------------------------------------------------------
# Konfigurasi (hyperparameter TETAP, tanpa tuning). Masuk config_hash.
# --------------------------------------------------------------------------------------------------
CONFIG = {
    "pilot_version": PILOT_VERSION,
    "seed": 42,
    "inner_val_frac": 0.2,                      # inner holdout tunggal dari train tiap fold (penyederhanaan pilot)
    "preproc_config_fp": "682404ae0f5d2624",
    "pheno_features": ["AGE_AT_SCAN", "SEX", "FIQ", "volume_mask_ml"],
    "missing_max_gap_pp": 10.0,                 # kolom dibuang bila |%missing ASD - %missing TD| > 10 poin persen
    "threshold": 0.5,
    "bootstrap_n": 2000,
    "slices": {"fractions": [0.3, 0.4, 0.5, 0.6, 0.7], "planes": ["sag", "cor", "ax"], "size": 224,
               "channels": "3 slice bersebelahan (p-1, p, p+1) sebagai RGB",
               "intensity": "x01 = clip((z+3)/6, 0, 1) di dalam mask, 0 di luar; lalu normalisasi mean/std ImageNet"},
    "augment": {"rot_deg": 5.0, "shift_vox": 3.0, "int_scale": 0.10, "int_shift": 0.10},
    "models": {
        # ---- fenotipik ----
        "logreg": {"kind": "pheno", "group": "CNN/ML klasik", "C": 1.0},
        "xgb": {"kind": "pheno", "group": "CNN/ML klasik", "n_estimators": 500, "max_depth": 3,
                "learning_rate": 0.05, "subsample": 0.8, "min_child_weight": 1, "early_stopping_rounds": 30},
        "mlp_tab": {"kind": "pheno", "group": "CNN/ML klasik", "hidden": [64, 32], "dropout": 0.2, "lr": 1e-3,
                    "wd": 1e-4, "epochs": 400, "patience": 40, "batch": 32},
        "tabm": {"kind": "pheno", "group": "arsitektur modern", "d_block": 128, "n_blocks": 2, "k": 32,
                 "dropout": 0.1, "lr": 1e-3, "wd": 1e-4, "epochs": 400, "patience": 40, "batch": 32},
        # ---- citra 2.5D (fine-tune blok akhir) ----
        "effnet_b0": {"kind": "2d", "group": "CNN klasik", "timm": "efficientnet_b0.ra_in1k",
                      "train_modules": ["blocks.5", "blocks.6", "conv_head", "bn2"],
                      "epochs": 30, "patience": 6, "batch": 4, "lr_head": 1e-3, "lr_backbone": 1e-4, "wd": 1e-4,
                      "budget_min": 90},
        "resnet18": {"kind": "2d", "group": "CNN klasik", "timm": "resnet18.a1_in1k",
                     "train_modules": ["layer3", "layer4"],
                     "epochs": 30, "patience": 6, "batch": 4, "lr_head": 1e-3, "lr_backbone": 1e-4, "wd": 1e-4,
                     "budget_min": 75},
        # ---- citra 3D: SFCN pretrained (UK Biobank), blok 0-3 beku + cache, blok 4-5 + head dilatih ----
        "sfcn": {"kind": "sfcn", "group": "CNN klasik (pretrained otak)",
                 "weights": "sfcn/run_20190719_00_epoch_best_mae.p",
                 "weights_sha256": "f9197c92486a8dba7fd87bb9f2ecc4f42f7c9628b9150d1b95c8a5ac10721ad3",
                 "frozen": ["conv_0", "conv_1", "conv_2", "conv_3"], "train_modules": ["conv_4", "conv_5", "head"],
                 "input": "upsample x2 (trilinear) -> crop MNI 160x192x160 (offset UKB 11,13,11) -> flip sumbu x "
                          "(RAS->LAS UKB) -> intensitas (z - min_mask) di mask, dibagi rerata volume",
                 "bn": "BatchNorm blok dilatih tetap eval (statistik UKB)",
                 "epochs": 60, "patience": 10, "batch": 8, "lr_head": 1e-3, "lr_backbone": 1e-4, "wd": 1e-4,
                 "budget_min": 60},
        # ---- foundation model 2.5D frozen: fitur per slice di-cache, head kecil dilatih ----
        "dinov2_s": {"kind": "dino", "group": "foundation model", "timm": "vit_small_patch14_dinov2.lvd142m",
                     "img_size": 224, "feature": "concat(CLS, mean patch) = 768-d per slice",
                     "epochs": 300, "patience": 30, "batch": 16, "lr": 1e-3, "wd": 1e-4, "budget_min": 30},
        # ---- opsional (hanya jika waktu cukup) ----
        "convnextv2_t": {"kind": "2d", "group": "arsitektur modern", "timm": "convnextv2_tiny.fcmae_ft_in22k_in1k",
                         "train_modules": ["stages.3", "head"], "optional": True,
                         "epochs": 30, "patience": 6, "batch": 2, "lr_head": 1e-3, "lr_backbone": 5e-5, "wd": 1e-4,
                         "budget_min": 120},
    },
    "deferred": {
        "tabpfn_v2": "semua versi tabpfn v2 (2.0.x-2.2.x) menuntut scikit-learn<1.7; env dikunci 1.7.0 -> env terpisah",
        "dinov3_vits16": "repo HF facebook/dinov3-vits16-pretrain-lvd1689m gated (manual, butuh login + persetujuan)",
        "brainiac": "bobot ada (Dropbox, lisensi akademik) tetapi daftar data pretraining (ABIDE?) belum terverifikasi",
        "cnn3d_se_scratch": "ditunda (rencana)", "S6": "ditunda (rencana)", "inner_cv_full": "untuk run data penuh",
    },
    "modality": {"pheno": ["logreg", "xgb", "mlp_tab", "tabm"],
                 "img25d": ["effnet_b0", "resnet18", "dinov2_s", "convnextv2_t"],
                 "img3d": ["sfcn"]},
    "fusion": {"selection": "AUC pooled inner validation lima fold", "pca_dims": [8, 16],
               "heads": ["logreg", "mlp", "lgbm"],
               "note": "head fusion dilatih pada inner holdout (20% train) tiap fold; penyederhanaan pilot"},
    "order": ["logreg", "xgb", "mlp_tab", "tabm", "effnet_b0", "resnet18", "sfcn", "dinov2_s", "FUSION",
              "convnextv2_t"],
}


def lib_versions():
    out = {"python": sys.version.split()[0]}
    for n in ["numpy", "pandas", "scikit-learn", "torch", "torchvision", "timm", "xgboost", "lightgbm", "tabm"]:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def sha_file(p):
    return sp.sha256_file(p) if Path(p).exists() else None


def run_config(subset, version="v1"):
    """Config lengkap + hash (parameter, versi library, hash data). v1 tidak berubah (hash v1 tetap)."""
    cfg = copy.deepcopy(CONFIG if version == "v1" else CONFIG_V2)
    cfg["subset"] = subset
    cfg["data_hashes"] = {"folds.csv": sha_file(PROJECT / "data" / "folds.csv"),
                          "subset_20pct.csv": sha_file(PROJECT / "data" / "subset_20pct.csv") if subset == "pilot200" else None,
                          "qc_decisions.csv": sha_file(IMAGE_ROOT / "qc_decisions.csv")}
    cfg["versions"] = lib_versions()
    if version != "v1":
        for n in ("neuroHarmonize", "neuroCombat"):
            try:
                cfg["versions"][n] = md.version(n)
            except md.PackageNotFoundError:
                cfg["versions"][n] = None
        cfg["data_hashes"]["atlas"] = sha_file(PRETRAINED / "atlas" / "harvard_oxford_final_grid.npy")
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return cfg, h


def results_dir(subset, version="v1"):
    return RESULTS_ROOT / (subset if version == "v1" else f"{subset}_{version}")


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
def load_cohort(subset):
    """Subjek dev (subset200 atau semua dev) dengan output 05c valid; ikuti exclude di qc_decisions."""
    folds = pd.read_csv(PROJECT / "data" / "folds.csv")
    dev = folds[folds["split"] == "dev"][["SUB_ID", "SITE_ID", "label", "fold_strat"]].copy()
    if subset == "pilot200":
        ids = set(pd.read_csv(PROJECT / "data" / "subset_20pct.csv")["SUB_ID"].astype(int))
        dev = dev[dev["SUB_ID"].isin(ids)]
        assert len(dev) == len(ids), "subset_20pct memuat subjek non-dev"
    elif subset != "all":
        raise ValueError(subset)
    qd = IMAGE_ROOT / "qc_decisions.csv"
    excl = set()
    if qd.exists():
        q = pd.read_csv(qd)
        excl = set(q.loc[q["decision"].astype(str).str.lower() == "exclude", "subject_id"].astype(int))
    dev = dev[~dev["SUB_ID"].isin(excl)]
    st = pd.read_csv(IMAGE_ROOT / "status.csv")
    ok = set(st.loc[st["status"].isin(["success", "flagged"]) & (st["config_hash"] == CONFIG["preproc_config_fp"]),
                    "subject_id"].astype(int))
    missing = sorted(set(dev["SUB_ID"]) - ok)
    assert not missing, f"{len(missing)} subjek belum punya output 05c valid: {missing[:10]}"
    dev["fold_strat"] = dev["fold_strat"].astype(int)
    return dev.sort_values("SUB_ID").reset_index(drop=True), sorted(excl)


def load_pheno(cohort):
    """AGE_AT_SCAN, SEX, FIQ (master) + volume_mask_ml (05c), dengan feature_overrides qc_decisions."""
    m = pd.read_csv(PROJECT / "dataset" / "cleaned_dataset" / "cleaned_master_dataset.csv",
                    usecols=["SUB_ID", "AGE_AT_SCAN", "SEX", "FIQ"])
    q = pd.read_csv(IMAGE_ROOT / "qc_report.csv", usecols=["SUB_ID", "volume_mask_ml"])
    df = cohort[["SUB_ID"]].merge(m, on="SUB_ID", how="left").merge(q, on="SUB_ID", how="left")
    qd = IMAGE_ROOT / "qc_decisions.csv"
    overrides = []
    if qd.exists():
        for r in pd.read_csv(qd).itertuples():
            for item in str(getattr(r, "feature_overrides", "") or "").split(";"):
                if "=" in item:
                    col, val = item.split("=", 1)
                    if col.strip() in df.columns and val.strip() == "missing":
                        df.loc[df["SUB_ID"] == int(r.subject_id), col.strip()] = np.nan
                        overrides.append((int(r.subject_id), col.strip()))
    bad = [c for c in df.columns if c.upper().startswith(tuple(p.upper() for p in FORBIDDEN_PREFIX))]
    assert not bad, f"kolom terlarang di fitur fenotipik: {bad}"
    return df.set_index("SUB_ID"), overrides


def missing_report(cohort, pheno):
    d = pheno.join(cohort.set_index("SUB_ID")["label"])
    rows = []
    for c in CONFIG["pheno_features"]:
        asd = float(d.loc[d["label"] == 1, c].isna().mean() * 100)
        td = float(d.loc[d["label"] == 0, c].isna().mean() * 100)
        rows.append({"feature": c, "missing_pct_ASD": round(asd, 2), "missing_pct_TD": round(td, 2),
                     "gap_pp": round(abs(asd - td), 2), "used": abs(asd - td) <= CONFIG["missing_max_gap_pp"]})
    return pd.DataFrame(rows)


def make_split(cohort, fold, seed):
    """test = fold_strat==fold; train sisanya; inner_val = 20% train (stratifikasi label), seed+fold."""
    from sklearn.model_selection import train_test_split
    test = cohort[cohort["fold_strat"] == fold]
    train = cohort[cohort["fold_strat"] != fold]
    tr, va = train_test_split(train["SUB_ID"].to_numpy(), test_size=CONFIG["inner_val_frac"],
                              stratify=train["label"].to_numpy(), random_state=seed + fold)
    return {"inner_train": np.sort(tr), "inner_val": np.sort(va), "test": np.sort(test["SUB_ID"].to_numpy())}


def load_volumes(ids):
    """Volume (96,112,96) float16 dan mask bool, urut ids."""
    v = np.stack([np.load(IMAGE_ROOT / "vol" / f"{int(s)}.npy")[0].astype(np.float16) for s in ids])
    m = np.stack([np.load(IMAGE_ROOT / "mask" / f"{int(s)}.npy")[0].astype(bool) for s in ids])
    return v, m


def slice_positions():
    """Posisi slice dari bbox mask template MNI (bukan dari data subjek): fraksi 0,3-0,7 tiap sumbu."""
    import nibabel as nib
    ref = json.loads((IMAGE_ROOT / "reference_space.json").read_text(encoding="utf-8"))
    t = np.asanyarray(nib.load(str(IMAGE_ROOT / "templates" / "mni152_mask_2mm.nii.gz")).dataobj) > 0
    t = t[tuple(slice(a, b) for a, b in ref["crop"])]
    t = np.pad(t, ref["pad"])
    assert t.shape == (96, 112, 96), t.shape
    pos = {}
    for ax, plane in enumerate(["sag", "cor", "ax"]):
        idx = np.where(t.any(axis=tuple(i for i in range(3) if i != ax)))[0]
        lo, hi = idx.min(), idx.max()
        pos[plane] = [int(np.clip(round(lo + f * (hi - lo)), 1, t.shape[ax] - 2)) for f in CONFIG["slices"]["fractions"]]
    return pos


# --------------------------------------------------------------------------------------------------
# Metrik
# --------------------------------------------------------------------------------------------------
def metrics(y, p, thr=None):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    thr = CONFIG["threshold"] if thr is None else thr
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    yh = (p >= thr).astype(int)
    tp, tn = int(((yh == 1) & (y == 1)).sum()), int(((yh == 0) & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return {"n": len(y), "balanced_accuracy": float(balanced_accuracy_score(y, yh)) if pos and neg else np.nan,
            "f1_macro": float(f1_score(y, yh, average="macro", zero_division=0)),
            "auc": float(roc_auc_score(y, p)) if pos and neg else np.nan,
            "sensitivity": tp / pos if pos else np.nan, "specificity": tn / neg if neg else np.nan}


def youden_threshold(y, p):
    """Threshold Youden (maks sensitivitas + spesifisitas - 1) dari probabilitas INNER validation; prediksi ASD bila
    p >= threshold. Seri -> threshold terdekat 0,5. Tanpa dua kelas -> threshold default."""
    from sklearn.metrics import roc_curve
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if y.min() == y.max():
        return float(CONFIG["threshold"])
    fpr, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    ok = np.isfinite(thr)
    j, thr = (tpr - fpr)[ok], thr[ok]
    best = np.flatnonzero(np.isclose(j, j.max()))
    return float(thr[best[np.argmin(np.abs(thr[best] - 0.5))]])


def bootstrap_ci(y, p, n=None, seed=42):
    n = n or CONFIG["bootstrap_n"]
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    keys = ["balanced_accuracy", "f1_macro", "auc", "sensitivity", "specificity"]
    vals = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        m = metrics(y[i], p[i])
        for k in keys:
            vals[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


# --------------------------------------------------------------------------------------------------
# Status unit
# --------------------------------------------------------------------------------------------------
STATUS_COLS = ["unit", "model", "fold", "status", "last_epoch", "max_epochs", "best_epoch", "best_val_loss",
               "seconds", "vram_peak_mib", "params_total", "params_trainable", "error", "config_hash", "updated"]


def read_status(rdir):
    p = rdir / "status.csv"
    return pd.read_csv(p, keep_default_na=False, na_values=[""]) if p.exists() else pd.DataFrame(columns=STATUS_COLS)


def update_status(rdir, unit, **kw):
    recs = {r["unit"]: r for r in read_status(rdir).to_dict("records")}
    base = recs.get(unit, {"unit": unit})
    base.update(kw)
    base["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    recs[unit] = base
    df = pd.DataFrame(list(recs.values()))
    df = df.reindex(columns=STATUS_COLS + [c for c in df.columns if c not in STATUS_COLS])
    sp.write_csv_atomic(rdir / "status.csv", df)


def unit_dir(rdir, unit):
    model, _, fold = unit.partition(":")
    return rdir / "units" / model / (f"fold{fold}" if fold else "_")


def unit_done_valid(rdir, unit):
    """done hanya jika preds.csv + metrics.json ada dan konsisten."""
    d = unit_dir(rdir, unit)
    try:
        if unit.startswith("cache_"):
            return (d / "cache_done.json").exists()
        p = pd.read_csv(d / "preds.csv")
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        return len(p) > 0 and p["prob"].between(0, 1).all() and np.isfinite(p["prob"]).all() and "test" in m
    except Exception:
        return False


# ==================================================================================================
# PILOT v2 (perbaikan training citra, fitur ROI, ComBat, CNN dari nol). CONFIG v1 di atas TIDAK diubah.
# ==================================================================================================
_V2_TRAIN = {"min_epochs": 5, "patience": 10, "smooth": 3, "warmup": 2}          # aturan early stopping/warmup v2
# bugfix 2026-10-27 (bukan tuning): running stats BN model dari nol basi saat eval (overfit test: loss train ~0,02 tetapi
# eval ~12-17). Sebelum tiap evaluasi & prediksi akhir, statistik BN dihitung ulang (rerata kumulatif) dari inner_train
# tanpa augmentasi; val/test tidak dipakai.
_BN_RECAL = "inner_train_tanpa_augmentasi"
CONFIG_V2 = {
    "pilot_version": "06-pilot-v2",
    "seed": 42, "inner_val_frac": 0.2, "preproc_config_fp": "682404ae0f5d2624",
    "pheno_features": ["AGE_AT_SCAN", "SEX", "FIQ", "volume_mask_ml"], "missing_max_gap_pp": 10.0,
    "threshold": 0.5, "bootstrap_n": 2000,
    "slices": copy.deepcopy(CONFIG["slices"]), "augment": copy.deepcopy(CONFIG["augment"]),
    "slice_jitter_vox": 2,
    "tabular": {"inner_cv_folds": 5, "es_frac": 0.2,
                "note": "model tabel: OOF inner CV 5-fold pada train luar (untuk seleksi & stacking), "
                        "model akhir di-fit ulang pada seluruh train luar; ES model iteratif memakai 20% train"},
    "roi": {"atlas": "Harvard-Oxford cortl+sub maxprob thr25 2mm (FSL) -> grid final 05c",
            "features": ["mean", "sd"], "min_voxels": 10,
            "note": "registrasi afin saja -> volume per region dikeluarkan; fitur = mean & SD intensitas z-score"},
    "combat": {"tool": "neuroHarmonize learn + apply kustom (diverifikasi identik)", "covariates": ["AGE_AT_SCAN", "SEX"],
               "no_diagnosis": True, "min_site_n": 2,
               "unseen_site": "tidak diharmonisasi & ditandai (berlaku juga untuk LOSO)"},
    "models": {
        "logreg": copy.deepcopy(CONFIG["models"]["logreg"]), "xgb": copy.deepcopy(CONFIG["models"]["xgb"]),
        "mlp_tab": copy.deepcopy(CONFIG["models"]["mlp_tab"]), "tabm": copy.deepcopy(CONFIG["models"]["tabm"]),
        "roi_logreg": {"kind": "roi", "base": "logreg", "combat": False, "group": "CNN/ML klasik (ROI)"},
        "roi_xgb": {"kind": "roi", "base": "xgb", "combat": False, "group": "CNN/ML klasik (ROI)"},
        "roi_tabm": {"kind": "roi", "base": "tabm", "combat": False, "group": "arsitektur modern (ROI)"},
        "roi_logreg_cb": {"kind": "roi", "base": "logreg", "combat": True, "group": "CNN/ML klasik (ROI)"},
        "roi_xgb_cb": {"kind": "roi", "base": "xgb", "combat": True, "group": "CNN/ML klasik (ROI)"},
        "roi_tabm_cb": {"kind": "roi", "base": "tabm", "combat": True, "group": "arsitektur modern (ROI)"},
        "effnet_b0": {"kind": "2d", "group": "pretrained: CNN klasik", "timm": "efficientnet_b0.ra_in1k",
                      "train_modules": ["blocks.6", "conv_head", "bn2"], "lp_epochs": 5, "freeze_bn": True,
                      "epochs": 40, "batch": 4, "lr_head": 1e-3, "lr_backbone": 3e-5, "wd": 1e-4, "budget_min": 60,
                      **_V2_TRAIN},
        "resnet18": {"kind": "2d", "group": "pretrained: CNN klasik", "timm": "resnet18.a1_in1k",
                     "train_modules": ["layer4"], "lp_epochs": 5, "freeze_bn": True,
                     "epochs": 40, "batch": 4, "lr_head": 1e-3, "lr_backbone": 3e-5, "wd": 1e-4, "budget_min": 60,
                     **_V2_TRAIN},
        "scratch2d": {"kind": "scratch2d", "group": "CNN dari nol", "channels": [32, 64, 128, 256], "img_size": 112,
                      "dropout": 0.5, "epochs": 60, "batch": 4, "lr": 1e-3, "wd": 1e-2, "budget_min": 60, **_V2_TRAIN,
                      "bn_recal": _BN_RECAL},
        "scratch3d": {"kind": "scratch3d", "group": "CNN dari nol", "channels": [16, 32, 64, 128, 128], "se_ratio": 8,
                      "dropout": 0.5, "epochs": 60, "batch": 4, "lr": 1e-3, "wd": 1e-2, "budget_min": 75, **_V2_TRAIN,
                      "bn_recal": _BN_RECAL},
        "sfcn_lp": {"kind": "probe", "group": "pretrained: CNN otak (frozen)", "source": "sfcn64",
                    "C_grid": [0.001, 0.01, 0.1, 1.0]},
        "sfcn_head": {"kind": "fhead", "group": "pretrained: CNN otak (frozen)", "source": "sfcn64",
                      "epochs": 200, "batch": 16, "lr": 1e-3, "wd": 1e-2, "dropout": 0.5, "budget_min": 10, **_V2_TRAIN},
        "dinov2_lp": {"kind": "probe", "group": "foundation model (frozen)", "source": "dino768",
                      "C_grid": [0.001, 0.01, 0.1, 1.0]},
        "dinov2_head": {"kind": "fhead", "group": "foundation model (frozen)", "source": "dino768",
                        "epochs": 200, "batch": 16, "lr": 1e-3, "wd": 1e-2, "dropout": 0.5, "budget_min": 10, **_V2_TRAIN},
        "convnextv2_t": {"kind": "2d", "group": "pretrained: arsitektur modern", "timm": "convnextv2_tiny.fcmae_ft_in22k_in1k",
                         "train_modules": ["stages.3.blocks.2", "head"], "lp_epochs": 5, "freeze_bn": True,
                         "epochs": 40, "batch": 2, "lr_head": 1e-3, "lr_backbone": 3e-5, "wd": 1e-4, "budget_min": 120,
                         **_V2_TRAIN},
    },
    "sfcn": copy.deepcopy(CONFIG["models"]["sfcn"]),
    "dinov2": copy.deepcopy(CONFIG["models"]["dinov2_s"]),
    "modality": {"pheno": ["logreg", "xgb", "mlp_tab", "tabm"],
                 "roi": ["roi_logreg", "roi_xgb", "roi_tabm", "roi_logreg_cb", "roi_xgb_cb", "roi_tabm_cb"],
                 "img25d": ["effnet_b0", "resnet18", "scratch2d", "dinov2_lp", "dinov2_head", "convnextv2_t"],
                 "img3d": ["scratch3d", "sfcn_lp", "sfcn_head"]},
    "fusion": {"scenarios": {"PR": ["pheno", "roi"], "P25": ["pheno", "img25d"], "P3": ["pheno", "img3d"],
                             "PRI": ["pheno", "roi", "img"]},
               "pca_dims": 16, "selection": "AUC pooled inner validation lima fold",
               "note": "stacking pheno+ROI dilatih pada OOF inner CV (160 subjek/fold); dengan citra dilatih pada "
                       "inner holdout (±32 subjek/fold) karena model citra tidak dilatih ulang per inner fold"},
    "ensemble": {"top_k": 3, "rule": "rata-rata probabilitas 3 model teratas menurut AUC inner val (semua modalitas)"},
    "sfcn_sanity_rule": {"min_pred_sd_years": 2.0, "min_abs_r_adult": 0.3,
                         "action": "model SFCN tetap dijalankan & dilaporkan, tetapi dikeluarkan dari seleksi/fusion "
                                   "bila cek kewajaran brain age gagal"},
    "learned_criterion": "train loss terakhir < 0,98 x epoch 1 DAN (val loss halus terbaik < loss prediksi konstan "
                         "ATAU AUC inner val di epoch terbaik >= 0,55)",
    "deferred": {"tabpfn_v2": "opsional (.venv-tabpfn)", "dinov3": "gated", "brainiac": "verifikasi data pretraining",
                 "ssl_pretraining": "perlu izin dosbim", "subgroup": "perlu izin dosbim"},
    "order": ["logreg", "xgb", "mlp_tab", "tabm", "cache_roi", "roi_logreg", "roi_xgb", "roi_tabm", "roi_logreg_cb",
              "roi_xgb_cb", "roi_tabm_cb", "effnet_b0", "resnet18", "scratch2d", "scratch3d", "cache_sfcn64", "sfcn_lp",
              "sfcn_head", "cache_dino768", "dinov2_lp", "dinov2_head", "convnextv2_t", "EMBCOMBAT", "FUSION", "ENSEMBLE"],
}


def load_atlas():
    a = np.load(PRETRAINED / "atlas" / "harvard_oxford_final_grid.npy")
    lab = json.loads((PRETRAINED / "atlas" / "harvard_oxford_labels.json").read_text(encoding="utf-8"))["labels"]
    return a, {int(k): v for k, v in lab.items()}


def roi_features(ids, min_voxels=10):
    """Mean & SD intensitas (z-score) per region HO di dalam mask otak. Tidak memakai label; per subjek."""
    atlas, labels = load_atlas()
    regs = sorted(labels)
    idx = {r: atlas == r for r in regs}
    rows = []
    for s in ids:
        v = np.load(IMAGE_ROOT / "vol" / f"{int(s)}.npy")[0]
        m = np.load(IMAGE_ROOT / "mask" / f"{int(s)}.npy")[0].astype(bool)
        row = {"SUB_ID": int(s)}
        for r in regs:
            x = v[idx[r] & m]
            ok = x.size >= min_voxels
            row[f"roi{r:03d}_mean"] = float(x.mean()) if ok else np.nan
            row[f"roi{r:03d}_sd"] = float(x.std()) if ok else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("SUB_ID")


def combat_fit(X, site, cov, min_site_n=2):
    """neuroHarmonize harmonizationLearn pada situs dengan >= min_site_n subjek. cov: DataFrame AGE_AT_SCAN, SEX."""
    from neuroHarmonize import harmonizationLearn
    site = np.asarray(site)
    vc = pd.Series(site).value_counts()
    ok = np.isin(site, vc[vc >= min_site_n].index)
    X = np.asarray(X, float)
    cols = np.where(X[ok].std(0) > 1e-8)[0]          # fitur konstan (varians 0) dilewatkan tanpa harmonisasi
    df = pd.DataFrame({"SITE": site[ok], **{c: cov[c].to_numpy(float)[ok] for c in cov.columns}})
    model, _ = harmonizationLearn(X[ok][:, cols], df)
    return {"model": model, "cov_cols": list(cov.columns), "cols": cols}


def combat_apply(cm, X, site, cov):
    """Apply ComBat dengan parameter hasil learn (identik dengan output learn pada data train; diverifikasi).
    Subjek dari situs yang tidak ada di model dibiarkan apa adanya dan ditandai (unseen)."""
    m = cm["model"]
    labels = list(m["SITE_labels"]); nb = len(labels); site = np.asarray(site)
    full = np.asarray(X, float)
    X = full[:, cm["cols"]]
    covpart = cov[cm["cov_cols"]].to_numpy(float) @ m["B_hat"][nb:]
    gm, vp = m["stand_mean"][:, 0], np.sqrt(m["var_pooled"][:, 0])
    s = (X - gm - covpart) / vp
    out = X.copy()
    for i, lab in enumerate(labels):
        r = site == lab
        out[r] = (s[r] - m["gamma_star"][i]) / np.sqrt(m["delta_star"][i]) * vp + gm + covpart[r]
    res = full.copy()
    res[:, cm["cols"]] = out
    return res, ~np.isin(site, labels)
