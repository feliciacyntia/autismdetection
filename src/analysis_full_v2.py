"""Analisis hasil DATA PENUH v2 (853 subjek dev, 5 fold; config terkunci). Tidak menyentuh 149 subjek test held-out.

    .venv-model\\Scripts\\python.exe src\\analysis_full_v2.py            # tabel & gambar (cepat)
    .venv-model\\Scripts\\python.exe src\\analysis_full_v2.py --lc       # + kurva belajar fenotipik/ROI/fusion tabel (bisa di-resume)

Output: <results-dir>\\report\\tables & figures. Threshold Youden dari probabilitas inner (aturan di locked config).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc  # noqa: E402

RD = pc.RESULTS_ROOT / "full_v2"                                   # env TA06_RESULTS_ROOT (default C:\ta_local\results)
PILOT, PILOT_BNFIX = pc.RESULTS_ROOT / "pilot200_v2", pc.RESULTS_ROOT / "pilot200_v2_bnfix"
BASE = [("fenotipik", "logreg"), ("fenotipik", "mlp_tab"), ("fenotipik", "tabm"),
        ("ROI", "roi_logreg_cb"), ("ROI", "roi_tabm_cb"),
        ("citra 2.5D pretrained", "effnet_b0"), ("citra 2.5D pretrained", "resnet18"),
        ("citra 2.5D dari nol", "scratch2d"), ("citra 3D dari nol", "scratch3d")]
FUS_METHODS = ["F1_stack_logreg", "F1_mean_prob", "F2_pca16_logreg"]          # F1_stack_logreg = utama (dikunci)
FUSION = [(s, m) for s in ("PR", "P25", "P3", "PRI") for m in FUS_METHODS] + [("ENS", "ENS_mean_top3")]
BASELINE = "logreg"                                                             # pembanding McNemar
KEYS = ["accuracy", "balanced_accuracy", "f1_macro", "auc", "sensitivity", "specificity"]
SHORT = {"accuracy": "Accuracy", "balanced_accuracy": "BA", "f1_macro": "F1 macro", "auc": "AUC", "sensitivity": "Sens", "specificity": "Spes"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LC_FRACS = [0.20, 0.35, 0.50, 0.65, 0.80, 1.00]


# ---------------------------------------------------------------- data
def folds_of(rdir):
    return sorted(int(k) for k in json.loads((rdir / "splits.json").read_text(encoding="utf-8")))


def unit_preds(rdir, model):
    fs = folds_of(rdir)
    ps = [pc.unit_dir(rdir, f"{model}:{k}") / "preds.csv" for k in fs]
    if not all(p.exists() for p in ps):
        return None
    return pd.concat([pd.read_csv(p) for p in ps], ignore_index=True)


def load(rdir, key):
    """-> (test, inner) untuk satu model atau 'SKEN|metode'. inner = sumber threshold Youden (bukan test)."""
    if "|" in key:
        s, m = key.split("|")
        p = unit_preds(rdir, f"fusion_{s}")
        if p is None:
            return None, None
        p = p[p.method == m]
        return p[p.part == "test"].reset_index(drop=True), p[p.part == "inner_cv"]
    p = unit_preds(rdir, key)
    if p is None:
        return None, None
    return p[p.part == "test"].reset_index(drop=True), p[p.part == "inner_val"]


def with_youden(t, inner):
    thr = {int(k): pc.youden_threshold(g.label, g.prob) for k, g in inner.groupby("fold")} if inner is not None and len(inner) else {}
    t = t.copy()
    t["thr_05"] = 0.5
    t["thr_youden"] = t.fold.astype(int).map(thr).astype(float) if len(thr) == t.fold.nunique() else np.nan
    return t, thr


# ---------------------------------------------------------------- metrik
def mfull(y, p, thr):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    yh = (p >= np.asarray(thr, float)).astype(int)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    tp, tn = ((yh == 1) & (y == 1)).sum(), ((yh == 0) & (y == 0)).sum()
    fp, fn = neg - tn, pos - tp
    sens, spec = tp / pos if pos else np.nan, tn / neg if neg else np.nan
    f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": (sens + spec) / 2, "f1_macro": (f1p + f1n) / 2,
            "auc": roc_auc_score(y, p) if pos and neg else np.nan, "sensitivity": sens, "specificity": spec,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": len(y)}


def boot(y, p, thr, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    y, p, thr = np.asarray(y).astype(int), np.asarray(p, float), np.asarray(thr, float)
    v = {k: [] for k in KEYS}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        m = mfull(y[i], p[i], thr[i])
        for k in KEYS:
            v[k].append(m[k])
    return {k: (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)) for k, a in v.items()}


def summary_row(t, thr_col):
    pf = pd.DataFrame([mfull(g.label, g.prob, g[thr_col]) for _, g in t.groupby("fold")])
    po = mfull(t.label, t.prob, t[thr_col])
    ci = boot(t.label, t.prob, t[thr_col])
    r = {}
    for k in KEYS:
        r[f"{SHORT[k]} fold (mean ± SD)"] = f"{pf[k].mean():.3f} ± {pf[k].std(ddof=1):.3f}"
        r[f"{SHORT[k]} pooled [95% CI]"] = f"{po[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
    return r, po


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def mcnemar(y, ya, yb):
    from scipy.stats import binomtest
    ca, cb = ya == y, yb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- analisis utama
def main_analysis(rdir, out_t, out_f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    keys = [(mod, n, n) for mod, n in BASE] + [("fusion" if s != "ENS" else "ensemble", f"{s} · {m}", f"{s}|{m}") for s, m in FUSION]
    members = {s: json.loads((pc.unit_dir(rdir, f"fusion_{s}:0") / "metrics.json").read_text(encoding="utf-8"))["members"]
               for s in ("PR", "P25", "P3", "PRI", "ENS")}
    T, rows, cms, thr_rows = {}, [], [], []
    for mod, label, key in keys:
        t, inn = load(rdir, key)
        if t is None:
            print("belum ada:", key); continue
        t, thr = with_youden(t, inn)
        T[key] = t
        for k, v in thr.items():
            thr_rows.append({"model": label, "fold": k, "threshold_youden": round(v, 4)})
        for rule, col in (("0,5", "thr_05"), ("Youden (inner)", "thr_youden")):
            if t[col].isna().any():
                continue
            r, po = summary_row(t, col)
            extra = {"anggota": "+".join(members[key.split("|")[0]])} if "|" in key else {}
            rows.append({"modalitas": mod, "model": label, "threshold": rule, **extra, **r})
            cms.append({"model": label, "threshold": rule, **{k: po[k] for k in ("tn", "fp", "fn", "tp", "n")}})
    M = pd.DataFrame(rows)
    M.to_csv(out_t / "F1_metrik_utama.csv", index=False)
    pd.DataFrame(cms).to_csv(out_t / "F2_confusion_matrix.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(out_t / "F3_threshold_youden_per_fold.csv", index=False)

    # McNemar vs baseline LogReg fenotipik (prediksi test pooled, subjek sama), Holm per aturan threshold
    mc = []
    for rule, col in (("0,5", "thr_05"), ("Youden (inner)", "thr_youden")):
        base = T[BASELINE].set_index("SUB_ID")
        yb_ = (base.prob >= base[col]).astype(int)
        part = []
        for key, t in T.items():
            if key == BASELINE or t[col].isna().any():
                continue
            a = t.set_index("SUB_ID").loc[base.index]
            ya = (a.prob >= a[col]).astype(int)
            b, c, p = mcnemar(base.label.to_numpy(), yb_.to_numpy(), ya.to_numpy())
            ba_b = mfull(base.label, base.prob, base[col])["balanced_accuracy"]
            ba_a = mfull(a.label, a.prob, a[col])["balanced_accuracy"]
            part.append({"threshold": rule, "model": key.replace("|", " · "), "BA model": round(ba_a, 3), "BA LogReg": round(ba_b, 3),
                         "ΔBA": round(ba_a - ba_b, 3), "hanya LogReg benar": b, "hanya model benar": c, "p McNemar (eksak)": p})
        if part:
            adj = holm([r["p McNemar (eksak)"] for r in part])
            for r, a_ in zip(part, adj):
                r["p Holm"] = a_
                r["signifikan (Holm < 0,05)"] = "ya" if a_ < 0.05 else "tidak"
                r["p McNemar (eksak)"] = round(r["p McNemar (eksak)"], 4); r["p Holm"] = round(a_, 4)
            mc += part
    MC = pd.DataFrame(mc)
    MC.to_csv(out_t / "F4_mcnemar_vs_logreg_holm.csv", index=False)

    # metrik per site (threshold 0,5): model tunggal + fusion utama
    site_keys = [n for _, n in BASE] + [f"{s}|F1_stack_logreg" for s in ("PR", "P25", "P3", "PRI")]
    srows = []
    for key in site_keys:
        if key not in T:
            continue
        for site, g in T[key].groupby("SITE_ID"):
            m = mfull(g.label, g.prob, g.thr_05)
            srows.append({"model": key.replace("|", " · "), "site": site, "n": len(g), "n ASD": int(g.label.sum()),
                          "BA": round(m["balanced_accuracy"], 3), "AUC": round(m["auc"], 3) if not np.isnan(m["auc"]) else np.nan,
                          "Sens": round(m["sensitivity"], 3) if not np.isnan(m["sensitivity"]) else np.nan,
                          "Spes": round(m["specificity"], 3) if not np.isnan(m["specificity"]) else np.nan})
    S = pd.DataFrame(srows)
    S.to_csv(out_t / "F5_metrik_per_site.csv", index=False)
    S.pivot(index="site", columns="model", values="BA").to_csv(out_t / "F5b_BA_per_site_pivot.csv")

    # site learning sebelum/sesudah ComBat (dari runner)
    sl = rdir / "tables" / "tabel_site_learning_combat.csv"
    if sl.exists():
        pd.read_csv(sl).to_csv(out_t / "F6_site_learning_combat.csv", index=False)

    # pilot (uji coba, n=200) vs data penuh (hasil utama dev, n=853), threshold 0,5
    pv = []
    for mod, label, key in keys:
        if key not in T:
            continue
        src = PILOT_BNFIX if key in ("scratch2d", "scratch3d") else PILOT
        tp_, _ = load(src, key)
        r = {"modalitas": mod, "model": label}
        for nm, t in (("pilot = uji coba (n=200)", tp_), ("data penuh = hasil utama dev (n=853)", T[key])):
            if t is None:
                for k in ("BA", "F1 macro", "AUC"):
                    r[f"{k} · {nm}"] = "belum ada"
                continue
            t = t.assign(thr_05=0.5)
            po, ci = mfull(t.label, t.prob, t.thr_05), boot(t.label, t.prob, t.thr_05)
            for k in ("accuracy", "balanced_accuracy", "f1_macro", "auc"):
                r[f"{SHORT[k]} · {nm}"] = f"{po[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
        if "|" in key:
            pm_ = PILOT / "units" / f"fusion_{key.split('|')[0]}" / "fold0" / "metrics.json"
            r["anggota pilot"] = "+".join(json.loads(pm_.read_text(encoding="utf-8"))["members"]) if pm_.exists() else ""
            r["anggota penuh"] = "+".join(members[key.split("|")[0]])
        pv.append(r)
    PV = pd.DataFrame(pv)
    PV.to_csv(out_t / "F7_pilot_vs_data_penuh.csv", index=False)

    # ---- gambar: ROC pooled (3 panel) & confusion matrix
    panels = [("Fenotipik & ROI", [n for m, n in BASE if m in ("fenotipik", "ROI")]),
              ("Citra (2.5D & 3D)", [n for m, n in BASE if m.startswith("citra")]),
              ("Fusion (F1_stack_logreg) & ensemble", [f"{s}|F1_stack_logreg" for s in ("PR", "P25", "P3", "PRI")] + ["ENS|ENS_mean_top3"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (title, ks) in zip(axes, panels):
        for i, key in enumerate([k for k in ks if k in T]):
            t = T[key]
            fpr, tpr, _ = roc_curve(t.label, t.prob)
            auc = mfull(t.label, t.prob, 0.5)["auc"]
            ax.plot(fpr, tpr, color=COL[i], lw=2, label=f"{key.replace('|F1_stack_logreg', '').replace('|', ' · ')} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#9a9a94", lw=1, ls="--", label="acak")
        ax.set(title=title, xlabel="1 − spesifisitas", ylabel="sensitivitas", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right", frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("ROC pooled test lima fold — data penuh (853 subjek dev)")
    fig.tight_layout(); fig.savefig(out_f / "F_roc_pooled.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    cm_keys = [k for k in ("logreg", "tabm", "roi_tabm_cb", "effnet_b0", "scratch3d", "PR|F1_stack_logreg", "PRI|F1_stack_logreg") if k in T]
    for rule, col, fn in (("threshold 0,5", "thr_05", "F_confusion_05.png"), ("threshold Youden (inner)", "thr_youden", "F_confusion_youden.png")):
        fig, axes = plt.subplots(1, len(cm_keys), figsize=(2.6 * len(cm_keys), 3.0))
        for ax, key in zip(axes, cm_keys):
            t = T[key]
            m = mfull(t.label, t.prob, t[col])
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v}\n({v / cm[i].sum():.0%})", ha="center", va="center", fontsize=9,
                        color="white" if v / cm[i].sum() > 0.6 else "#1a1a19")
            ax.set_xticks([0, 1], ["TD", "ASD"]); ax.set_yticks([0, 1], ["TD", "ASD"])
            ax.set_title(f"{key.replace('|F1_stack_logreg', ' (fusion)')}\nBA {m['balanced_accuracy']:.3f}", fontsize=9)
            ax.set_xlabel("prediksi", fontsize=8)
        axes[0].set_ylabel("label asli")
        fig.suptitle(f"Confusion matrix pooled test ({rule})", y=1.04)
        fig.tight_layout(); fig.savefig(out_f / fn, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ringkasan markdown
    show = M[M.threshold == "0,5"][["modalitas", "model", "Accuracy pooled [95% CI]", "BA pooled [95% CI]", "F1 macro pooled [95% CI]", "AUC pooled [95% CI]",
                                     "Sens pooled [95% CI]", "Spes pooled [95% CI]"]]
    showy = M[M.threshold != "0,5"][["model", "BA pooled [95% CI]", "F1 macro pooled [95% CI]", "Sens pooled [95% CI]", "Spes pooled [95% CI]"]]
    txt = ["# Analisis data penuh v2 (853 subjek dev, 5 fold)\n", "## Metrik threshold 0,5\n", md(show),
           "\n## Metrik threshold Youden (dari inner)\n", md(showy), "\n## McNemar vs LogReg fenotipik (Holm)\n", md(MC),
           "\n## Pilot vs data penuh (threshold 0,5)\n", md(PV)]
    (out_t.parent / "ringkasan_analisis.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


# ---------------------------------------------------------------- kurva belajar (tabel saja)
def learning_curve(rdir, out_t, out_f, repeats):
    import pilot_models as pm
    from sklearn.linear_model import LogisticRegression
    from run_pilot_v2 import lg
    cfg = json.loads((rdir / "config.json").read_text(encoding="utf-8"))["config"]
    cohort, _ = pc.load_cohort("all")
    pheno, _ = pc.load_pheno(cohort)
    feats = cfg["pheno_features"]
    roi = pd.read_csv(rdir / "cache" / "roi" / "roi_features.csv").set_index("SUB_ID")
    cov = pheno[["AGE_AT_SCAN", "SEX"]]
    lab = cohort.set_index("SUB_ID")["label"]
    splits = {int(k): {p: np.asarray(v) for p, v in s.items()} for k, s in json.loads((rdir / "splits.json").read_text(encoding="utf-8")).items()}
    pr_mem = json.loads((pc.unit_dir(rdir, "fusion_PR:0") / "metrics.json").read_text(encoding="utf-8"))["members"]
    models = ["logreg", "mlp_tab", "tabm", "roi_logreg_cb", "roi_tabm_cb"]
    raw = out_t / "F8_learning_curve_raw.csv"
    done = pd.read_csv(raw) if raw.exists() else pd.DataFrame(columns=["model", "frac", "rep", "fold"])
    have = set(zip(done.model, done.frac.astype(float).round(2), done.rep.astype(int), done.fold.astype(int)))
    t0 = time.time()

    def sub(ids, f, rng):
        y = lab.loc[ids].to_numpy()
        return np.sort(np.concatenate([rng.choice(ids[y == c], max(2, int(round(f * (y == c).sum()))), replace=False) for c in (0, 1)]))

    for f in LC_FRACS:
        for r in range(1 if f == 1.0 else repeats):
            for k, s in splits.items():
                if all((m, round(f, 2), r, k) in have for m in models + ["PR_F1_stack"]):
                    continue
                rng = np.random.default_rng(100000 + 1000 * k + 10 * r + int(round(f * 100)))
                sp_ = s if f == 1.0 else {"inner_train": sub(s["inner_train"], f, rng), "inner_val": sub(s["inner_val"], f, rng), "test": s["test"]}
                n_tr = len(sp_["inner_train"]) + len(sp_["inner_val"])
                P, rows = {}, []
                for m in models:
                    hp = cfg["models"][m]
                    if hp["kind"] == "pheno":
                        X, base, kind, fe, cb = pheno, m, "pheno", feats, False
                    else:
                        X, base, kind, fe, cb = roi, hp["base"], "roi", list(roi.columns), hp["combat"]
                    probs, ids, _ = pm.run_tabular_v2(base, cfg["models"][base], X, cohort, sp_, cfg["seed"] + k, kind, fe, cb, cov,
                                                      cfg["tabular"]["inner_cv_folds"])
                    P[m] = (probs, ids)
                    rows.append((m, probs["test"]))
                T_ = P[pr_mem[0]][1]["inner_oof"]
                Xs = np.column_stack([lg(pd.Series(P[n][0]["inner_oof"], index=P[n][1]["inner_oof"]).loc[T_]) for n in pr_mem])
                Xt = np.column_stack([lg(P[n][0]["test"]) for n in pr_mem])
                rows.append(("PR_F1_stack", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
                             .fit(Xs, lab.loc[T_].to_numpy()).predict_proba(Xt)[:, 1]))
                yt = lab.loc[s["test"]].to_numpy()
                new = []
                for m, p in rows:
                    mm = mfull(yt, p, 0.5)
                    new.append({"model": m, "frac": f, "rep": r, "fold": k, "n_latih": n_tr, "auc": mm["auc"],
                                "balanced_accuracy": mm["balanced_accuracy"], "f1_macro": mm["f1_macro"]})
                done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
                pc.sp.write_csv_atomic(raw, done)
                print(f"LC frac {f:.2f} rep {r} fold {k} | n latih {n_tr} | AUC " +
                      " ".join(f"{d['model']}={d['auc']:.3f}" for d in new) + f" | {time.time() - t0:.0f} s", flush=True)
    done = done.astype({"frac": float, "fold": int, "rep": int})
    per_fold = done.groupby(["model", "frac", "fold"], as_index=False)[["auc", "balanced_accuracy", "f1_macro", "n_latih"]].mean()
    S = per_fold.groupby(["model", "frac"]).agg(n_latih=("n_latih", "mean"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                                                ba_mean=("balanced_accuracy", "mean"), ba_sd=("balanced_accuracy", "std"),
                                                f1_mean=("f1_macro", "mean")).reset_index()
    S.round(4).to_csv(out_t / "F8_learning_curve.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (mcol, sdcol, nm) in zip(axes, (("auc_mean", "auc_sd", "AUC"), ("ba_mean", "ba_sd", "BA (threshold 0,5)"))):
        for i, (m, g) in enumerate(S.groupby("model", sort=False)):
            ax.errorbar(g.n_latih, g[mcol], yerr=g[sdcol], color=COL[i], lw=2, marker="o", ms=5, capsize=3, label=m)
        ax.axhline(0.5, color="#9a9a94", lw=1, ls="--")
        ax.set(xlabel="subjek latih per fold", ylabel=nm, title=f"Kurva belajar — {nm} (mean ± SD antar fold)")
        ax.grid(alpha=0.25)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out_f / "F_learning_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(S.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RD))
    ap.add_argument("--lc", action="store_true", help="jalankan juga kurva belajar (fenotipik/ROI/fusion tabel)")
    ap.add_argument("--lc-only", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    rdir = Path(a.results_dir)
    out_t, out_f = rdir / "report" / "tables", rdir / "report" / "figures"
    out_t.mkdir(parents=True, exist_ok=True); out_f.mkdir(parents=True, exist_ok=True)
    if not a.lc_only:
        main_analysis(rdir, out_t, out_f)
    if a.lc or a.lc_only:
        learning_curve(rdir, out_t, out_f, a.repeats)


if __name__ == "__main__":
    main()
