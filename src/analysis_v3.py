"""Analisis v3 (model tabel + ensemble) sesuai locked_config_v3.json. Hanya membaca prediksi; tidak melatih model.

    .venv-model\\Scripts\\python.exe src\\analysis_v3.py

Output: <RESULTS_ROOT>\\v3\\report\\tables & figures + ringkasan_v3.md
Seleksi model & anggota ensemble memakai skor inner (bukan test); skor test fold hanya dilaporkan.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_full_v2 as af  # noqa: E402
import pilot_common as pc  # noqa: E402
import run_v3_tabular as rv  # noqa: E402

V3, FULL = pc.RESULTS_ROOT / "v3", pc.RESULTS_ROOT / "full_v2"
REF = "full_v2 · fusion PR (F1_stack_logreg)"
NAMA = {"logreg_cv": "LogReg (C dari inner CV)", "logreg": "LogReg", "tabm": "TabM (5 seed)", "catboost": "CatBoost", "tabpfn": "TabPFN v2"}


# ---------------------------------------------------------------- data
def preds(key):
    """-> (test, inner) ; inner = sumber threshold Youden (tabel: inner_val dari OOF; citra: inner holdout)."""
    if key == REF:
        return af.load(FULL, "PR|F1_stack_logreg")
    p = rv.model_preds(key, V3, FULL)
    if p is None:
        return None, None
    return p[p.part == "test"].reset_index(drop=True), p[p.part == "inner_val"]


def label(key):
    if key == REF:
        return key
    src, name = key.split(":")
    if src == "full":
        return f"full_v2 · {name}"
    exp, feat, m = name.split("_", 2)
    return f"{exp.upper()} · {'fenotipik' if feat == 'pheno' else 'ROI'} · {NAMA[m]}"


def ensemble(members, how, cfg):
    """Rerata prob. (utama/tabel) atau stacking LogReg pada logit (dilatih di inner_val karena ada cabang citra)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    T, I = {}, {}
    for k in members:
        t, i = preds(k)
        T[k], I[k] = t.set_index("SUB_ID"), i.set_index("SUB_ID")
    base = T[members[0]]
    test_rows, inner_rows = [], []
    for f in sorted(base.fold.unique()):
        te = base.index[base.fold == f]
        iv = I[members[0]].index[I[members[0]].fold == f]
        Pt = np.column_stack([T[k].loc[te, "prob"] for k in members])
        Pi = np.column_stack([I[k].loc[iv, "prob"] for k in members])
        yi = I[members[0]].loc[iv, "label"].to_numpy()
        if how == "mean":
            pt, pi = Pt.mean(1), Pi.mean(1)
        else:
            lr = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
            pt = lr().fit(rv.lg(Pi), yi).predict_proba(rv.lg(Pt))[:, 1]
            pi = cross_val_predict(lr(), rv.lg(Pi), yi, cv=StratifiedKFold(5, shuffle=True, random_state=f), method="predict_proba")[:, 1]
        test_rows.append(base.loc[te, ["SITE_ID", "fold", "label"]].assign(prob=pt))
        inner_rows.append(I[members[0]].loc[iv, ["SITE_ID", "fold", "label"]].assign(prob=pi))
    return pd.concat(test_rows).reset_index(), pd.concat(inner_rows).reset_index()


# ---------------------------------------------------------------- uji statistik
def delong(y, a, b):
    """DeLong berpasangan: (AUC a, AUC b, p) untuk H0 AUC a = AUC b."""
    from scipy.stats import norm
    y = np.asarray(y).astype(int)
    V10, V01, A = [], [], []
    for s in (np.asarray(a, float), np.asarray(b, float)):
        x, z = s[y == 1], s[y == 0]
        c = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
        V10.append(c.mean(1)); V01.append(c.mean(0)); A.append(c.mean())
    S = np.cov(np.vstack(V10)) / (y == 1).sum() + np.cov(np.vstack(V01)) / (y == 0).sum()
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    zv = (A[0] - A[1]) / np.sqrt(var) if var > 0 else 0.0
    return A[0], A[1], float(2 * norm.sf(abs(zv)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import kruskal, spearmanr
    from sklearn.metrics import roc_curve

    cfg = json.loads((V3 / "locked_config_v3.json").read_text(encoding="utf-8"))
    out_t, out_f = V3 / "report" / "tables", V3 / "report" / "figures"
    out_t.mkdir(parents=True, exist_ok=True); out_f.mkdir(parents=True, exist_ok=True)

    # ---- 1. seleksi (skor inner, dikunci) -> anggota ensemble
    sel, scores = rv.selection(V3, cfg)
    scores["model"] = scores.model.map(label)
    scores.sort_values(["modalitas", "skor_seleksi"], ascending=[True, False]).round(4).to_csv(out_t / "V1_seleksi_inner.csv", index=False)
    masuk = [v["model"] for v in sel.values() if v["masuk"]]
    tabel = [sel[m]["model"] for m in ("fenotipik", "roi") if m in sel]

    # ---- 2. semua model + ensemble
    keys = [f"v3:e1_{f}_{m}" for f in rv.FEATS for m in rv.E1_MODELS] + [f"v3:e3_{f}_{m}" for f in rv.FEATS for m in rv.E3_MODELS]
    T = {}
    for k in keys + [REF, "full:effnet_b0", "full:scratch3d"]:
        t, i = preds(k)
        if t is not None:
            T[label(k)] = af.with_youden(t, i)[0]
    ens = {"ENS_utama (rerata prob., semua modalitas terpilih)": ("mean", masuk),
           "ENS_stack (stacking LogReg, pembanding)": ("stack", masuk),
           "ENS_tabel (rerata prob. fenotipik + ROI)": ("mean", tabel)}
    for name, (how, mem) in ens.items():
        t, i = ensemble(mem, how, cfg)
        T[name] = af.with_youden(t, i)[0]
    ref = T[REF].set_index("SUB_ID")
    assert all(len(t) == 853 for t in T.values()), "jumlah subjek test pooled bukan 853"

    rows, tests = [], []
    for name, t in T.items():
        for rule, col in (("0,5", "thr_05"), ("Youden (inner)", "thr_youden")):
            if t[col].isna().any():
                continue
            r, po = af.summary_row(t, col)
            rows.append({"model": name, "threshold": rule, **r, **{k: po[k] for k in ("tn", "fp", "fn", "tp")}})
        if name == REF:
            continue
        a = t.set_index("SUB_ID").loc[ref.index]
        auc_a, auc_r, p_d = delong(a.label, a.prob, ref.prob)
        b, c, p_m = af.mcnemar(a.label.to_numpy(), (ref.prob >= 0.5).astype(int).to_numpy(), (a.prob >= 0.5).astype(int).to_numpy())
        acc_a = ((a.prob >= 0.5).astype(int) == a.label).mean(); acc_r = ((ref.prob >= 0.5).astype(int) == ref.label).mean()
        tests.append({"model": name, "AUC model": round(auc_a, 3), "AUC full_v2": round(auc_r, 3), "ΔAUC": round(auc_a - auc_r, 3), "p DeLong": p_d,
                      "Acc model": round(acc_a, 3), "Acc full_v2": round(acc_r, 3), "ΔAcc": round(acc_a - acc_r, 3),
                      "hanya full_v2 benar": b, "hanya model benar": c, "p McNemar": p_m})
    M = pd.DataFrame(rows); M.to_csv(out_t / "V2_metrik_semua_model.csv", index=False)
    TS = pd.DataFrame(tests)
    for col in ("p DeLong", "p McNemar"):
        TS[col + " (Holm)"] = af.holm(TS[col].to_numpy())
    TS["signifikan (Holm < 0,05)"] = np.where((TS["p DeLong (Holm)"] < 0.05) | (TS["p McNemar (Holm)"] < 0.05), "ya", "tidak")
    TS = TS.round({c: 4 for c in TS.columns if c.startswith("p ")})
    TS.to_csv(out_t / "V3_uji_vs_full_v2.csv", index=False)

    # ---- 3. ablasi
    abl_keys = [(f"fenotipik ({label(sel['fenotipik']['model'])})", label(sel["fenotipik"]["model"])),
                (f"ROI ({label(sel['roi']['model'])})", label(sel["roi"]["model"])),
                ("citra 2.5D (full_v2 · effnet_b0)", "full_v2 · effnet_b0"), ("citra 3D (full_v2 · scratch3d)", "full_v2 · scratch3d"),
                ("fenotipik + ROI (ENS_tabel)", "ENS_tabel (rerata prob. fenotipik + ROI)"),
                ("semua modalitas (ENS_utama)", "ENS_utama (rerata prob., semua modalitas terpilih)"),
                ("semua modalitas (ENS_stack)", "ENS_stack (stacking LogReg, pembanding)"), ("pembanding: " + REF, REF)]
    m05 = M[M.threshold == "0,5"].set_index("model"); tsi = TS.set_index("model")
    ab = []
    for nm, k in abl_keys:
        r = m05.loc[k]
        ab.append({"komponen": nm, "Accuracy": r["Accuracy pooled [95% CI]"], "BA": r["BA pooled [95% CI]"], "F1 macro": r["F1 macro pooled [95% CI]"],
                   "AUC": r["AUC pooled [95% CI]"], "Sens": r["Sens pooled [95% CI]"].split(" ")[0], "Spes": r["Spes pooled [95% CI]"].split(" ")[0],
                   "p DeLong (Holm)": tsi.loc[k, "p DeLong (Holm)"] if k in tsi.index else "", "p McNemar (Holm)": tsi.loc[k, "p McNemar (Holm)"] if k in tsi.index else ""})
    AB = pd.DataFrame(ab); AB.to_csv(out_t / "V4_ablasi.csv", index=False)

    # ---- 4. konfounder
    cohort, _ = pc.load_cohort("all"); ph, _ = pc.load_pheno(cohort)
    kf = []
    for k in ["ENS_utama (rerata prob., semua modalitas terpilih)", "ENS_tabel (rerata prob. fenotipik + ROI)",
              label(sel["fenotipik"]["model"]), label(sel["roi"]["model"]), REF]:
        t = T[k].set_index("SUB_ID").join(ph[["AGE_AT_SCAN", "FIQ", "volume_mask_ml"]])
        r = {"model": k}
        for v in ("AGE_AT_SCAN", "FIQ", "volume_mask_ml"):
            for grp, q in (("semua", t), ("TD saja", t[t.label == 0]), ("ASD saja", t[t.label == 1])):
                q = q.dropna(subset=[v]); rho, p = spearmanr(q.prob, q[v])
                r[f"ρ {v} ({grp})"] = f"{rho:+.2f} (p {p:.3f})"
        h, p = kruskal(*[g.prob.to_numpy() for _, g in t.groupby("SITE_ID")])
        r["Kruskal prob ~ site"] = f"H {h:.1f} (p {p:.2g})"
        kf.append(r)
    pd.DataFrame(kf).to_csv(out_t / "V5_konfounder.csv", index=False)

    # ---- 5. LOSO
    loso_keys = sorted({d.name.rsplit("_s", 1)[0] for d in (V3 / "units").glob("loso_*")})
    LO = {}
    for nm in loso_keys:
        p = rv.model_preds(f"v3:{nm}", V3)
        LO[nm] = p[p.part == "test"].set_index("SUB_ID")
    if len(LO) == 2:
        a, b = LO.values()
        LO["loso_ENS_tabel"] = a.assign(prob=(a.prob + b.loc[a.index, "prob"]) / 2)
    lrows, srows = [], []
    for nm, t in LO.items():
        t = t.assign(thr_05=0.5).reset_index()
        po, ci = af.mfull(t.label, t.prob, 0.5), af.boot(t.label, t.prob, t.thr_05)
        k5 = {"loso_ENS_tabel": "ENS_tabel (rerata prob. fenotipik + ROI)"}.get(nm, label(sel["fenotipik" if "_pheno_" in nm else "roi"]["model"]))
        cv = T[k5].set_index("SUB_ID").loc[t.SUB_ID]
        cvm = af.mfull(cv.label, cv.prob, 0.5)
        lrows.append({"model LOSO": nm, "padanan 5-fold": k5, "n subjek": len(t), "n site": t.SITE_ID.nunique(),
                      **{f"{af.SHORT[k]} LOSO [95% CI]": f"{po[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in ("accuracy", "balanced_accuracy", "auc")},
                      **{f"{af.SHORT[k]} 5-fold (subjek sama)": round(cvm[k], 3) for k in ("accuracy", "balanced_accuracy", "auc")}})
        for site, g in t.groupby("SITE_ID"):
            m = af.mfull(g.label, g.prob, 0.5)
            srows.append({"model": nm, "site": site, "n": len(g), "BA": round(m["balanced_accuracy"], 3), "AUC": round(m["auc"], 3) if not np.isnan(m["auc"]) else np.nan})
    LT = pd.DataFrame(lrows); LT.to_csv(out_t / "V6_loso_ringkasan.csv", index=False)
    LS = pd.DataFrame(srows); LS.to_csv(out_t / "V6b_loso_per_site.csv", index=False)

    # ---- 6. gambar
    show = [label(sel["fenotipik"]["model"]), label(sel["roi"]["model"]), "ENS_tabel (rerata prob. fenotipik + ROI)",
            "ENS_utama (rerata prob., semua modalitas terpilih)", "ENS_stack (stacking LogReg, pembanding)", REF]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, k in enumerate(show):
        t = T[k]; fpr, tpr, _ = roc_curve(t.label, t.prob)
        ax.plot(fpr, tpr, color=af.COL[i], lw=2, label=f"{k} (AUC {af.mfull(t.label, t.prob, 0.5)['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#9a9a94", lw=1, ls="--", label="acak")
    ax.set(xlabel="1 − spesifisitas", ylabel="sensitivitas", title="ROC pooled test 5 fold — v3 vs full_v2 (853 subjek dev)")
    ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out_f / "V_roc.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, k in zip(axes, [label(sel["fenotipik"]["model"]), "ENS_tabel (rerata prob. fenotipik + ROI)",
                            "ENS_utama (rerata prob., semua modalitas terpilih)", REF]):
        t = T[k]; m = af.mfull(t.label, t.prob, 0.5)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}\n({v / cm[i].sum():.0%})", ha="center", va="center", fontsize=9, color="white" if v / cm[i].sum() > 0.6 else "#1a1a19")
        ax.set_xticks([0, 1], ["TD", "ASD"]); ax.set_yticks([0, 1], ["TD", "ASD"]); ax.set_xlabel("prediksi", fontsize=8)
        ax.set_title(f"{k.split(' (')[0]}\nacc {m['accuracy']:.3f}", fontsize=8.5)
    axes[0].set_ylabel("label asli")
    fig.suptitle("Confusion matrix pooled test (threshold 0,5)", y=1.05)
    fig.tight_layout(); fig.savefig(out_f / "V_confusion.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    if len(LS):
        piv = LS[LS.model == "loso_ENS_tabel"].sort_values("BA") if "loso_ENS_tabel" in LO else LS
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(piv.site, piv.BA, color=af.COL[0], height=0.6)
        ax.axvline(0.5, color="#9a9a94", ls="--", lw=1)
        for y, (b, n) in enumerate(zip(piv.BA, piv.n)):
            ax.text(b + 0.01, y, f"{b:.2f} (n={n})", va="center", fontsize=8)
        ax.set(xlabel="balanced accuracy (site ditinggalkan)", title="LOSO — ENS_tabel per site (dilatih tanpa site tsb)", xlim=(0, 1))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(out_f / "V_loso_per_site.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    pc.sp.write_json_atomic(V3 / "report" / "selection_final.json", {"seleksi": sel, "ENS_utama": masuk, "ENS_tabel": tabel})
    top = M[M.threshold == "0,5"][["model", "Accuracy pooled [95% CI]", "BA pooled [95% CI]", "F1 macro pooled [95% CI]", "AUC pooled [95% CI]"]]
    txt = ["# Analisis v3 (853 subjek dev, 5 fold)\n", "## Seleksi (skor inner)\n", json.dumps(sel, indent=1),
           "\n## Metrik threshold 0,5\n", af.md(top), "\n## Uji vs full_v2 (DeLong AUC, McNemar accuracy, Holm)\n", af.md(TS),
           "\n## Ablasi\n", af.md(AB), "\n## LOSO\n", af.md(LT), "\n## Konfounder\n", af.md(pd.DataFrame(kf))]
    (V3 / "report" / "ringkasan_v3.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
