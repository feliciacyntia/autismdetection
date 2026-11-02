"""Analisis v4 (morfometri FreeSurfer PCP + ensemble) sesuai locked_config_v4.json. Hanya membaca prediksi; tidak melatih model.
Hanya 853 dev (CV); 149 test tidak dibaca. Seleksi model memakai skor inner; skor test fold hanya dilaporkan.

    .venv-model\\Scripts\\python.exe src\\analysis_v4.py

Output: <RESULTS_ROOT>\\v4\\report\\tables & figures + ringkasan_v4.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_full_v2 as af  # noqa: E402
import analysis_v3 as av  # noqa: E402
import pilot_common as pc  # noqa: E402
import run_v3_tabular as rv  # noqa: E402
import run_v4 as r4  # noqa: E402

V4, V3, FULL = pc.RESULTS_ROOT / "v4", pc.RESULTS_ROOT / "v3", pc.RESULTS_ROOT / "full_v2"
REF_F = av.REF
REF_E = "ENS_utama v3 (model final)"
ENS_T = "ENS_tabel v3 (fenotipik + ROI)"
ENS_4 = "ENS_v4 (ENS_utama v3 + morfometri)"
VARIAN = {"morph_raw": "mentah", "morph_icv": "relatif ICV", "morph_cb": "relatif ICV + ComBat"}


def preds(key):
    """key 'v4:<prefix>' / 'v3:<prefix>' / 'full:<model>' / REF_F -> (test, inner_val)."""
    if key == REF_F:
        return af.load(FULL, "PR|F1_stack_logreg")
    p = rv.model_preds("v3:" + key[3:], V4) if key.startswith("v4:") else rv.model_preds(key, V3, FULL)
    if p is None:
        return None, None
    return p[p.part == "test"].reset_index(drop=True), p[p.part == "inner_val"]


def label(key):
    if key.startswith("v4:"):
        feat, m = r4.UNIT_RE.match(key[3:]).groups()
        return f"{key[3:5].upper()} · morfometri {VARIAN[feat]} · {av.NAMA[m]}"
    return av.label(key)


def ens_mean(members):
    T, I = zip(*[preds(k) for k in members])
    T, I = [t.set_index("SUB_ID") for t in T], [i.set_index(["fold", "SUB_ID"]) for i in I]     # inner_val: subjek muncul di beberapa fold
    t0, i0 = T[0].sort_values("fold", kind="stable"), I[0]                                       # urutan sama dengan analysis_v3 (CI bootstrap identik)
    pt =np.mean([t.loc[t0.index, "prob"].to_numpy() for t in T], 0)
    pi = np.mean([i.loc[i0.index, "prob"].to_numpy() for i in I], 0)
    return t0[["SITE_ID", "fold", "label"]].assign(prob=pt).reset_index(), i0[["SITE_ID", "label"]].assign(prob=pi).reset_index()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import kruskal, spearmanr
    from sklearn.metrics import roc_curve

    raw = (V4 / "locked_config_v4.json").read_bytes()
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == (V4 / "locked_config_v4.sha256").read_text(encoding="utf-8").split()[0], "config v4 berubah"
    out_t, out_f = V4 / "report" / "tables", V4 / "report" / "figures"
    out_t.mkdir(parents=True, exist_ok=True); out_f.mkdir(parents=True, exist_ok=True)

    # ---- 1. seleksi morfometri (skor inner_oof), anggota lama dari seleksi v3 (tidak berubah)
    sv3 = json.loads((V3 / "report" / "selection_final.json").read_text(encoding="utf-8"))
    keys = [f"v4:e1_{f}_{m}" for f in r4.MORPH for m in rv.E1_MODELS] + [f"v4:e3_{f}_{m}" for f in r4.MORPH for m in rv.E3_MODELS]
    sc = []
    for k in keys:
        p = rv.model_preds("v3:" + k[3:], V4)
        q = p[p.part == "inner_oof"]
        assert q.fold.nunique() == 5, k
        sc.append({"model": k, "skor_seleksi": rv.auc(q.label, q.prob)})
    SC = pd.DataFrame(sc).sort_values("skor_seleksi", ascending=False).reset_index(drop=True)
    best, best_s = SC.model[0], float(SC.skor_seleksi[0])
    saved = json.loads((V4 / "selection_morfometri.json").read_text(encoding="utf-8"))
    assert "v4:" + saved["model"] == best, "seleksi berbeda dengan yang dipakai LOSO"
    masuk = best_s >= 0.55
    SC.assign(model=SC.model.map(label)).round(4).to_csv(out_t / "M1_seleksi_inner_morfometri.csv", index=False)
    roi_key, phe_key = sv3["seleksi"]["roi"]["model"], sv3["seleksi"]["fenotipik"]["model"]
    mem_v3 = sv3["ENS_utama"]
    mem_v4 = mem_v3 + ([best] if masuk else [])

    # ---- 2. metrik semua model + ensemble (rerata prob.)
    T = {}
    for k in keys + [phe_key, roi_key, REF_F]:
        t, i = preds(k)
        T[label(k)] = af.with_youden(t, i)[0]
    L_M, L_R, L_P = label(best), label(roi_key), label(phe_key)
    ens = {REF_E: mem_v3, ENS_T: sv3["ENS_tabel"], ENS_4: mem_v4, "fenotipik + morfometri": [phe_key, best],
           "fenotipik + ROI + morfometri": [phe_key, roi_key, best]}
    for name, mem in ens.items():
        t, i = ens_mean(mem)
        T[name] = af.with_youden(t, i)[0]
    assert all(len(t) == 853 for t in T.values()), "jumlah subjek test pooled bukan 853"
    rows = []
    for name, t in T.items():
        for rule, col in (("0,5", "thr_05"), ("Youden (inner)", "thr_youden")):
            r, po = af.summary_row(t, col)
            rows.append({"model": name, "threshold": rule, **r, **{k: po[k] for k in ("tn", "fp", "fn", "tp")}})
    M = pd.DataFrame(rows); M.to_csv(out_t / "M2_metrik_semua_model.csv", index=False)

    # ---- 3. uji berpasangan; Holm mencakup semua model v4 yang dilaporkan
    tests = []

    def uji(name, ref):
        a, b = T[name].set_index("SUB_ID"), T[ref].set_index("SUB_ID")
        a = a.loc[b.index]
        ua, ub, pd_ = av.delong(a.label, a.prob, b.prob)
        pa, pb = (a.prob >= 0.5).astype(int), (b.prob >= 0.5).astype(int)
        nb, nc, pm = af.mcnemar(a.label.to_numpy(), pb.to_numpy(), pa.to_numpy())
        tests.append({"model": name, "pembanding": ref, "AUC model": round(ua, 3), "AUC pembanding": round(ub, 3), "ΔAUC": round(ua - ub, 3), "p DeLong": pd_,
                      "Acc model": round((pa == a.label).mean(), 3), "Acc pembanding": round((pb == b.label).mean(), 3),
                      "ΔAcc": round((pa == a.label).mean() - (pb == b.label).mean(), 3),
                      "hanya pembanding benar": nb, "hanya model benar": nc, "p McNemar": pm})

    v4_names = [label(k) for k in keys] + [ENS_4, "fenotipik + morfometri", "fenotipik + ROI + morfometri"]
    for n in v4_names:
        uji(n, REF_E); uji(n, REF_F)
    uji(L_M, L_R); uji(L_M, L_P)
    TS = pd.DataFrame(tests)
    for col in ("p DeLong", "p McNemar"):
        TS[col + " (Holm)"] = af.holm(TS[col].to_numpy())
    TS["signifikan (Holm < 0,05)"] = np.where((TS["p DeLong (Holm)"] < 0.05) | (TS["p McNemar (Holm)"] < 0.05), "ya", "tidak")
    TS = TS.round({c: 4 for c in TS.columns if c.startswith("p ")})
    TS.to_csv(out_t / "M3_uji_berpasangan.csv", index=False)

    # ---- 4. ablasi
    m05 = M[M.threshold == "0,5"].set_index("model")
    ab = []
    for nm, k in [(f"fenotipik ({L_P})", L_P), (f"ROI ({L_R})", L_R), (f"morfometri ({L_M})", L_M), ("citra 2.5D (full_v2 · effnet_b0)", None), ("citra 3D (full_v2 · scratch3d)", None),
                  ("fenotipik + ROI", ENS_T), ("fenotipik + morfometri", "fenotipik + morfometri"), ("fenotipik + ROI + morfometri", "fenotipik + ROI + morfometri"),
                  ("semua modalitas tanpa morfometri (ENS_utama v3)", REF_E), ("semua modalitas + morfometri (ENS_v4)", ENS_4), ("pembanding: " + REF_F, REF_F)]:
        if k is None:
            mod = "full:effnet_b0" if "2.5D" in nm else "full:scratch3d"
            t, i = preds(mod); k = label(mod); T[k] = af.with_youden(t, i)[0]
            r_, _ = af.summary_row(T[k], "thr_05"); r = pd.Series(r_)
        else:
            r = m05.loc[k]
        ab.append({"komponen": nm, "Accuracy": r["Accuracy pooled [95% CI]"], "BA": r["BA pooled [95% CI]"], "F1 macro": r["F1 macro pooled [95% CI]"],
                   "AUC": r["AUC pooled [95% CI]"], "Sens": r["Sens pooled [95% CI]"].split(" ")[0], "Spes": r["Spes pooled [95% CI]"].split(" ")[0]})
    AB = pd.DataFrame(ab); AB.to_csv(out_t / "M4_ablasi.csv", index=False)

    # ---- 5. konfounder
    cohort, _ = pc.load_cohort("all"); ph, _ = pc.load_pheno(cohort)
    icv = r4.morph_variants()["morph_raw"]["ICV"].rename("ICV")
    kf = []
    for k in (L_M, ENS_4, REF_E, L_R):
        t = T[k].set_index("SUB_ID").join(ph[["AGE_AT_SCAN", "FIQ", "volume_mask_ml"]]).join(icv)
        r = {"model": k}
        for v in ("AGE_AT_SCAN", "FIQ", "volume_mask_ml", "ICV"):
            for grp, q in (("semua", t), ("TD saja", t[t.label == 0]), ("ASD saja", t[t.label == 1])):
                q = q.dropna(subset=[v]); rho, p = spearmanr(q.prob, q[v])
                r[f"ρ {v} ({grp})"] = f"{rho:+.2f} (p {p:.3f})"
        h, p = kruskal(*[g.prob.to_numpy() for _, g in t.groupby("SITE_ID")])
        r["Kruskal prob ~ site"] = f"H {h:.1f} (p {p:.2g})"
        kf.append(r)
    KF = pd.DataFrame(kf); KF.to_csv(out_t / "M5_konfounder.csv", index=False)

    # ---- 6. LOSO model morfometri terpilih
    nm = next(d.name.rsplit("_s", 1)[0] for d in (V4 / "units").glob("loso_*"))
    p = rv.model_preds(f"v3:{nm}", V4)
    t = p[p.part == "test"].reset_index(drop=True)
    po, ci = af.mfull(t.label, t.prob, 0.5), af.boot(t.label, t.prob, t.assign(thr_05=0.5).thr_05)
    cv = T[L_M].set_index("SUB_ID").loc[t.SUB_ID]; cvm = af.mfull(cv.label, cv.prob, 0.5)
    LT = pd.DataFrame([{"model LOSO": nm, "padanan 5-fold": L_M, "n subjek": len(t), "n site": t.SITE_ID.nunique(),
                        **{f"{af.SHORT[k]} LOSO [95% CI]": f"{po[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in ("accuracy", "balanced_accuracy", "auc")},
                        **{f"{af.SHORT[k]} 5-fold (subjek sama)": round(cvm[k], 3) for k in ("accuracy", "balanced_accuracy", "auc")}}])
    LT.to_csv(out_t / "M6_loso_ringkasan.csv", index=False)
    LS = []
    for site, g in t.groupby("SITE_ID"):
        m = af.mfull(g.label, g.prob, 0.5)
        LS.append({"site": site, "n": len(g), "BA": round(m["balanced_accuracy"], 3), "AUC": round(m["auc"], 3) if not np.isnan(m["auc"]) else np.nan})
    LS = pd.DataFrame(LS); LS.to_csv(out_t / "M6b_loso_per_site.csv", index=False)

    # ---- 7. gambar
    show = [L_P, L_R, L_M, ENS_T, REF_E, ENS_4]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, k in enumerate(show):
        q = T[k]; fpr, tpr, _ = roc_curve(q.label, q.prob)
        ax.plot(fpr, tpr, color=af.COL[i], lw=2, label=f"{k} (AUC {af.mfull(q.label, q.prob, 0.5)['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#9a9a94", lw=1, ls="--", label="acak")
    ax.set(xlabel="1 − spesifisitas", ylabel="sensitivitas", title="ROC pooled test 5 fold — morfometri (853 subjek dev)")
    ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out_f / "M_roc.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    pc.sp.write_json_atomic(V4 / "report" / "selection_v4.json", {"morfometri": {"model": best, "skor_inner_oof": best_s, "masuk": bool(masuk)},
                                                                  "ENS_v4": mem_v4, "ENS_utama_v3": mem_v3})
    top = M[M.threshold == "0,5"][["model", "Accuracy pooled [95% CI]", "BA pooled [95% CI]", "F1 macro pooled [95% CI]", "AUC pooled [95% CI]"]]
    utama = TS[TS.model.isin([L_M, ENS_4, "fenotipik + morfometri", "fenotipik + ROI + morfometri"])]
    txt = ["# Analisis v4 — morfometri FreeSurfer PCP (853 subjek dev, 5 fold)\n",
           f"Model morfometri terpilih (skor inner_oof): **{L_M}**, AUC inner {best_s:.3f}; masuk ensemble: {masuk}.\n",
           "## Ablasi (threshold 0,5)\n", af.md(AB), "\n## Uji utama (DeLong AUC, McNemar accuracy; Holm mencakup semua uji v4)\n", af.md(utama),
           "\n## LOSO\n", af.md(LT), "\n## Konfounder\n", af.md(KF), "\n## Semua model\n", af.md(top),
           f"\n## Seleksi inner morfometri\n", af.md(SC.assign(model=SC.model.map(label)).round(4))]
    (V4 / "report" / "ringkasan_v4.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt[:9]))


if __name__ == "__main__":
    main()
