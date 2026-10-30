"""EVALUASI FINAL — SEKALI — pada 149 subjek test held-out (config: results/final/locked_config_final.json + .sha256).

    .venv-model\\Scripts\\python.exe src\\final_eval.py

Model final = model yang SUDAH dilatih per fold (5 fold, dan 5 seed untuk TabM); prediksi test = rerata 5 model fold.
Tidak ada pelatihan ulang dengan hyperparameter baru dan tidak ada seleksi apa pun dari skor test.
Sebelum test diprediksi, prediksi fold-CV direproduksi & dibandingkan dengan yang tersimpan (bukti pipeline identik).
Menolak jalan bila hasil sudah ada (evaluasi sekali).
"""
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_full_v2 as af  # noqa: E402
import analysis_v3 as a3  # noqa: E402
import pilot_common as pc  # noqa: E402
import run_pilot as rp  # noqa: E402
import run_v3_tabular as rv  # noqa: E402

LOG = logging.getLogger("pilot")
OUT = pc.RESULTS_ROOT / "final"
CFG = OUT / "locked_config_final.json"
FULL, V3 = pc.RESULTS_ROOT / "full_v2", pc.RESULTS_ROOT / "v3"


def check_lock():
    raw = CFG.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    lock = CFG.with_suffix(".sha256")
    if not lock.exists() or lock.read_text(encoding="utf-8").split()[0] != h:
        raise SystemExit(f"config final belum dikunci / berubah: {h}")
    return json.loads(raw), h


# ---------------------------------------------------------------- data (dev + test)
def cohorts(fv2):
    dev, _ = pc.load_cohort("all")
    folds = pd.read_csv(pc.PROJECT / "data" / "folds.csv")
    test = folds[folds.split == "test"][["SUB_ID", "SITE_ID", "label"]].copy()
    st = pd.read_csv(pc.IMAGE_ROOT / "status.csv")
    ok = set(st.loc[st.status.isin(["success", "flagged"]) & (st.config_hash == fv2["preproc_config_fp"]), "subject_id"].astype(int))
    assert set(test.SUB_ID) <= ok and len(test) == 149, "ada subjek test tanpa output preprocessing valid"
    assert not set(test.SUB_ID) & set(dev.SUB_ID)
    test["fold_strat"] = -1
    allc = pd.concat([dev, test], ignore_index=True).sort_values("SUB_ID").reset_index(drop=True)
    return dev, np.sort(test.SUB_ID.to_numpy()), allc


# ---------------------------------------------------------------- anggota tabel v3 (e1/e3)
def v3_member(r, key, heldout, splits, rep):
    _, name = key.split(":")
    exp, feat, model = name.split("_", 2)
    seeds = r.seeds if model in rv.NEURAL else [0]
    P = []
    for k, s_ in splits.items():
        T = np.concatenate([s_["inner_train"], s_["inner_val"]])
        if exp == "e3":
            best = json.loads((pc.unit_dir(V3, f"e3tune_{feat}_{model}:{k}") / "best.json").read_text(encoding="utf-8"))
            hp = r.hp(model, best["best_params"])
        else:
            hp = r.hp(model)
        for s in seeds:
            seed = r.cfg["seed"]["basis"] + k + 1000 * s
            res, _ = r.jobs_fit(feat, model, hp, [(T, [s_["test"], heldout])], seed)
            saved = pd.read_csv(pc.unit_dir(V3, f"{name}_s{s}:{k}") / "preds.csv").query("part == 'test'").set_index("SUB_ID").loc[s_["test"], "prob"]
            rep.append({"anggota": key, "fold": k, "seed": s, "max|Δ| prediksi fold-CV": float(np.abs(res[0][0] - saved.to_numpy()).max())})
            P.append(res[0][1])
    return np.mean(P, 0)


# ---------------------------------------------------------------- fusion full_v2 PR (pembanding)
def fusion_pr(fv2, allc, pheno, roi, heldout, splits, rep):
    import pilot_models as pm
    from sklearn.linear_model import LogisticRegression
    cov = pheno[["AGE_AT_SCAN", "SEX"]]
    mem = json.loads((pc.unit_dir(FULL, "fusion_PR:0") / "metrics.json").read_text(encoding="utf-8"))["members"]
    out = []
    for k, s_ in splits.items():
        sp = {"inner_train": s_["inner_train"], "inner_val": s_["inner_val"], "test": heldout}
        oof, te = [], []
        for n in mem:
            hp = fv2["models"][n]
            if hp["kind"] == "pheno":
                X, base, kind, feats, cb = pheno, n, "pheno", fv2["pheno_features"], False
            else:
                X, base, kind, feats, cb = roi, hp["base"], "roi", list(roi.columns), hp["combat"]
            probs, ids, _ = pm.run_tabular_v2(base, fv2["models"][base], X, allc, sp, fv2["seed"] + k, kind, feats, cb, cov,
                                              fv2["tabular"]["inner_cv_folds"])
            saved = pd.read_csv(pc.unit_dir(FULL, f"{n}:{k}") / "preds.csv").query("part == 'inner_oof'").set_index("SUB_ID").loc[ids["inner_oof"], "prob"]
            rep.append({"anggota": f"full_v2:{n} (fusion PR)", "fold": k, "seed": 0,
                        "max|Δ| prediksi fold-CV": float(np.abs(probs["inner_oof"] - saved.to_numpy()).max())})
            oof.append(rv.lg(probs["inner_oof"])); te.append(rv.lg(probs["test"]))
            T = ids["inner_oof"]
        yT = allc.set_index("SUB_ID").loc[T, "label"].to_numpy()
        out.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(np.column_stack(oof), yT)
                   .predict_proba(np.column_stack(te))[:, 1])
    return np.mean(out, 0)


# ---------------------------------------------------------------- anggota citra full_v2
def image_member(name, fv2, allc, heldout, splits, rep):
    import torch
    import pilot_models as pm
    hp = fv2["models"][name]
    kind = hp["kind"]
    pos = pc.slice_positions()
    lab = allc.set_index("SUB_ID")["label"]
    B = hp["batch"]

    def vol(s):                                     # baca dari disk per batch (hemat RAM); dtype sama dengan pc.load_volumes
        return (np.load(pc.IMAGE_ROOT / "vol" / f"{int(s)}.npy")[0].astype(np.float16),
                np.load(pc.IMAGE_ROOT / "mask" / f"{int(s)}.npy")[0].astype(bool))

    def prep(ids):
        v = torch.from_numpy(np.stack([vol(s)[0] for s in ids])).to(pm.DEV).float()
        m = torch.from_numpy(np.stack([vol(s)[1] for s in ids])).to(pm.DEV)
        if kind == "scratch3d":
            return torch.where(m, v, torch.zeros_like(v))[:, None]
        return pm.make_slices(v, m, pos, hp.get("img_size", fv2["slices"]["size"]), jitter=0)

    out = []
    for k, s_ in splits.items():
        pm.seed_all(fv2["seed"] + k)
        if kind == "2d":
            model = pm.Slice25D(hp["timm"], hp["train_modules"], freeze_bn=hp.get("freeze_bn", False))
        elif kind == "scratch2d":
            model = pm.Scratch2D(hp["channels"], hp["dropout"])
        else:
            model = pm.Scratch3DSE(hp["channels"], hp["se_ratio"], hp["dropout"])
        model.load_state_dict(torch.load(pc.unit_dir(FULL, f"{name}:{k}") / "ckpt" / "best.pt", map_location="cpu", weights_only=True))
        model = model.to(pm.DEV).eval()
        if hp.get("bn_recal"):
            tr = s_["inner_train"]
            pm.recalibrate_bn(model, lambda: (prep(tr[i:i + B]) for i in range(0, len(tr), B)))

        def predict(ids):
            P = []
            with torch.no_grad(), torch.autocast(pm.DEV.type, dtype=torch.float16):
                for i in range(0, len(ids), B):
                    lg_, _ = model(prep(ids[i:i + B]))
                    P.append(lg_.float().cpu())
            return torch.sigmoid(torch.cat(P)).numpy()
        pf, ph = predict(s_["test"]), predict(heldout)
        saved = pd.read_csv(pc.unit_dir(FULL, f"{name}:{k}") / "preds.csv").query("part == 'test'").set_index("SUB_ID").loc[s_["test"], "prob"]
        rep.append({"anggota": f"full:{name}", "fold": k, "seed": 0, "max|Δ| prediksi fold-CV": float(np.abs(pf - saved.to_numpy()).max())})
        LOG.info("%s fold %d: reproduksi fold-CV max|Δ| %.2e", name, k, rep[-1]["max|Δ| prediksi fold-CV"])
        out.append(ph)
        del model
        torch.cuda.empty_cache()
    return np.mean(out, 0)


def youden_final(members, how):
    """Threshold Youden final = rerata threshold Youden 5 fold (dari prob. inner, seperti analisis v3)."""
    if members == "PR":
        _, inner = af.load(FULL, "PR|F1_stack_logreg")
    else:
        _, inner = a3.ensemble(members, how, None)
    return float(np.mean([pc.youden_threshold(g.label, g.prob) for _, g in inner.groupby("fold")]))


def main():
    cfg, h = check_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    rp.setup_logging(OUT / "_cek" if CEK else OUT)
    if (OUT / "heldout_predictions.csv").exists():
        LOG.error("Evaluasi test SUDAH pernah dijalankan (%s). Evaluasi final hanya sekali.", OUT / "heldout_predictions.csv")
        return 2
    if not CEK:
        import psutil
        ram = psutil.virtual_memory().available / 2**30
        if ram < 4.0:
            LOG.error("RAM tersedia %.2f GiB < 4 GiB: tidak dimulai (tutup aplikasi lain lalu ulangi).", ram)
            return 2
    LOG.info("EVALUASI FINAL | config sha256 %s", h)
    fv2 = json.loads((FULL / "config.json").read_text(encoding="utf-8"))["config"]
    assert json.loads((FULL / "config.json").read_text(encoding="utf-8"))["config_hash"] == cfg["sumber"]["full_v2_config_hash"]
    sel = json.loads((V3 / "report" / "selection_final.json").read_text(encoding="utf-8"))
    assert sel["ENS_utama"] == cfg["model_final"]["anggota"], "anggota ENS_utama berbeda dengan config final"
    dev, heldout, allc = cohorts(fv2)
    splits = {int(k): {p: np.asarray(v) for p, v in s.items()} for k, s in json.loads((FULL / "splits.json").read_text(encoding="utf-8")).items()}
    pheno, ov = pc.load_pheno(allc)
    LOG.info("dev %d | test held-out %d | override %s", len(dev), len(heldout), ov)

    # ROI test: fitur deterministik (tanpa parameter yang dipelajari) dengan fungsi yang sama; cek reproduksi pada dev
    roi_dev = pd.read_csv(FULL / "cache" / "roi" / "roi_features.csv").set_index("SUB_ID")
    chk = pc.roi_features(roi_dev.index[:5].to_numpy(), fv2["roi"]["min_voxels"])
    assert np.allclose(chk.to_numpy(), roi_dev.loc[chk.index, chk.columns].to_numpy(), equal_nan=True), "fitur ROI tidak tereproduksi"
    if CEK:                                            # uji skrip: fold 0, 6 subjek DEV menggantikan test (test tidak disentuh)
        splits = {0: splits[0]}
        heldout = splits[0]["test"][:6]
        roi = roi_dev.reindex(allc.SUB_ID)
    else:
        roi_test = pc.roi_features(heldout, fv2["roi"]["min_voxels"])
        roi = pd.concat([roi_dev, roi_test[roi_dev.columns]]).loc[allc.SUB_ID]
        LOG.info("ROI test: %d subjek x %d fitur (reproduksi 5 subjek dev: identik)", *roi_test.shape)

    class A:
        test, results_dir, stop_at = False, str(V3), None
    r = rv.V3(A())
    r.data()                                           # data dev v3 (cek split & fitur)
    c = allc.set_index("SUB_ID")
    r.cohort, r.pheno, r.roi, r.cov, r.lab, r.site = allc, pheno, roi, pheno[["AGE_AT_SCAN", "SEX"]], c["label"], c["SITE_ID"]

    rep, P = [], {}
    t0 = time.time()
    for key in cfg["model_final"]["anggota"]:
        if key.startswith("v3:"):
            P[key] = v3_member(r, key, heldout, splits, rep)
        else:
            P[key] = image_member(key.split(":")[1], fv2, allc, heldout, splits, rep)
        LOG.info("anggota %s selesai (%.1f menit)", key, (time.time() - t0) / 60)
    P["full_v2:fusion_PR"] = fusion_pr(fv2, allc, pheno, roi, heldout, splits, rep)
    R = pd.DataFrame(rep)
    worst = R.groupby("anggota")["max|Δ| prediksi fold-CV"].max()
    LOG.info("Reproduksi prediksi fold-CV (max|Δ|):\n%s", worst.to_string())
    if CEK:
        pc.sp.write_csv_atomic(OUT / "_cek" / "reproduksi_fold0.csv", R)
        LOG.info("CEK selesai (%.1f menit): %s", (time.time() - t0) / 60,
                 "LULUS" if (worst < cfg["toleransi_reproduksi"]).all() else "GAGAL (reproduksi di atas toleransi)")
        return 0
    pc.sp.write_csv_atomic(OUT / "reproduksi_fold_cv.csv", R)
    assert (worst < cfg["toleransi_reproduksi"]).all(), "pipeline prediksi TIDAK mereproduksi hasil CV -> evaluasi dibatalkan"

    ens = {"ENS_utama (model final)": np.mean([P[k] for k in cfg["model_final"]["anggota"]], 0),
           "ENS_tabel (pembanding)": np.mean([P[k] for k in cfg["pembanding"]["ENS_tabel"]], 0),
           "full_v2 fusion PR (pembanding)": P["full_v2:fusion_PR"]}
    for k in cfg["model_final"]["anggota"]:
        ens[f"anggota: {k}"] = P[k]
    pred = pd.DataFrame({"SUB_ID": heldout, **{k: v for k, v in ens.items()}})
    pc.sp.write_csv_atomic(OUT / "heldout_predictions.csv", pred)     # prediksi disimpan SEBELUM label test dipakai
    LOG.info("Prediksi test disimpan; menghitung metrik dengan label test (sekali).")

    thr = {"ENS_utama (model final)": youden_final(cfg["model_final"]["anggota"], "mean"),
           "ENS_tabel (pembanding)": youden_final(cfg["pembanding"]["ENS_tabel"], "mean"),
           "full_v2 fusion PR (pembanding)": youden_final("PR", None)}
    y = c.loc[heldout, "label"].to_numpy()
    site = c.loc[heldout, "SITE_ID"].to_numpy()
    rows = []
    for name, p in ens.items():
        for rule, t in (("0,5", 0.5), ("Youden (rerata 5 fold, dari inner)", thr.get(name))):
            if t is None:
                continue
            tv = np.full(len(y), t)
            m, ci = af.mfull(y, p, tv), af.boot(y, p, tv)
            rows.append({"model": name, "threshold": rule if t == 0.5 else f"{rule} = {t:.3f}",
                         **{f"{af.SHORT[k]} [95% CI]": f"{m[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in af.KEYS},
                         **{k: m[k] for k in ("tn", "fp", "fn", "tp")}})
    M = pd.DataFrame(rows)
    pc.sp.write_csv_atomic(OUT / "metrik_test.csv", M)

    f, b = ens["ENS_utama (model final)"], ens["full_v2 fusion PR (pembanding)"]
    auc_f, auc_b, p_d = a3.delong(y, f, b)
    bb, cc, p_m = af.mcnemar(y, (b >= 0.5).astype(int), (f >= 0.5).astype(int))
    comp = {"AUC model final": auc_f, "AUC fusion PR": auc_b, "p DeLong": p_d, "Acc model final": float(((f >= 0.5) == y).mean()),
            "Acc fusion PR": float(((b >= 0.5) == y).mean()), "hanya fusion PR benar": bb, "hanya model final benar": cc, "p McNemar": p_m,
            "catatan": "uji sekunder deskriptif (n = 149); hipotesis utama = performa model final di test"}
    pc.sp.write_json_atomic(OUT / "uji_final_vs_fusion.json", comp)
    sr = []
    for s in sorted(set(site)):
        i = site == s
        m = af.mfull(y[i], f[i], 0.5)
        sr.append({"site": s, "n": int(i.sum()), "n ASD": int(y[i].sum()), "accuracy": round(m["accuracy"], 3),
                   "BA": round(m["balanced_accuracy"], 3), "AUC": round(m["auc"], 3) if not np.isnan(m["auc"]) else np.nan})
    pc.sp.write_csv_atomic(OUT / "metrik_test_per_site.csv", pd.DataFrame(sr))

    # perbandingan dengan estimasi CV (dev)
    v2 = pd.read_csv(V3 / "report" / "tables" / "V2_metrik_semua_model.csv").query("threshold == '0,5'").set_index("model")
    cvmap = {"ENS_utama (model final)": "ENS_utama (rerata prob., semua modalitas terpilih)", "ENS_tabel (pembanding)": "ENS_tabel (rerata prob. fenotipik + ROI)",
             "full_v2 fusion PR (pembanding)": "full_v2 · fusion PR (F1_stack_logreg)"}
    cmp = [{"model": k, "Accuracy CV dev (853)": v2.loc[v, "Accuracy pooled [95% CI]"], "Accuracy TEST (149)": M[(M.model == k) & (M.threshold == "0,5")].iloc[0]["Accuracy [95% CI]"],
            "AUC CV dev (853)": v2.loc[v, "AUC pooled [95% CI]"], "AUC TEST (149)": M[(M.model == k) & (M.threshold == "0,5")].iloc[0]["AUC [95% CI]"]}
           for k, v in cvmap.items()]
    pc.sp.write_csv_atomic(OUT / "cv_vs_test.csv", pd.DataFrame(cmp))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, k in enumerate(["ENS_utama (model final)", "ENS_tabel (pembanding)", "full_v2 fusion PR (pembanding)"]):
        fpr, tpr, _ = roc_curve(y, ens[k]); axes[0].plot(fpr, tpr, color=af.COL[i], lw=2, label=f"{k} (AUC {af.mfull(y, ens[k], 0.5)['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#9a9a94", ls="--", lw=1, label="acak")
    axes[0].set(xlabel="1 − spesifisitas", ylabel="sensitivitas", title="ROC — 149 subjek test held-out"); axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    m = af.mfull(y, f, 0.5); cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    axes[1].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        axes[1].text(j, i, f"{v}\n({v / cm[i].sum():.0%})", ha="center", va="center", color="white" if v / cm[i].sum() > 0.6 else "#1a1a19")
    axes[1].set_xticks([0, 1], ["TD", "ASD"]); axes[1].set_yticks([0, 1], ["TD", "ASD"])
    axes[1].set(xlabel="prediksi", ylabel="label asli", title=f"Model final (threshold 0,5) — acc {m['accuracy']:.3f}")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "test_roc_cm.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    txt = ["# EVALUASI FINAL — 149 subjek test held-out (sekali)\n", f"config final sha256 `{h}`\n",
           "## Reproduksi prediksi fold-CV sebelum test (max|Δ|)\n", af.md(worst.reset_index()),
           "\n## Metrik test\n", af.md(M.drop(columns=["tn", "fp", "fn", "tp"])), "\n## CV dev vs TEST\n", af.md(pd.DataFrame(cmp)),
           "\n## Model final vs fusion full_v2 (sekunder)\n", json.dumps(comp, indent=1), "\n## Per site (model final, threshold 0,5)\n", af.md(pd.DataFrame(sr))]
    (OUT / "ringkasan_final.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    return 0


CEK = "--cek" in sys.argv      # uji skrip tanpa menyentuh test: reproduksi fold 0 saja

if __name__ == "__main__":
    sys.exit(main())
