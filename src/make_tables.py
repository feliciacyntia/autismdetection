"""06 — Tabel A–E, ROC, boxplot, ringkasan dari unit yang sudah done. Bisa dijalankan KAPAN SAJA (juga saat runner
berjalan); unit yang belum selesai ditandai "belum ada".

    .venv-model\\Scripts\\python.exe src\\make_tables.py --subset pilot200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc  # noqa: E402

METRICS = ["balanced_accuracy", "f1_macro", "auc", "sensitivity", "specificity"]
SHORT = {"balanced_accuracy": "BA", "f1_macro": "F1m", "auc": "AUC", "sensitivity": "Sens", "specificity": "Spes"}
F1_MAIN, F2_MAIN = "F1_stack_logreg", "F2_pca16_logreg"          # ditetapkan sebelum melihat hasil


def load_preds(rdir, model, folds, method=None):
    us = [f"{model}:{k}" for k in folds]
    if not all(pc.unit_done_valid(rdir, u) for u in us):
        return None
    p = pd.concat([pd.read_csv(pc.unit_dir(rdir, u) / "preds.csv") for u in us], ignore_index=True)
    if method is not None:
        p = p[p["method"] == method]
    return p


def summarize(p, part="test"):
    t = p[p["part"] == part]
    per_fold = pd.DataFrame([pc.metrics(g.label, g.prob) for _, g in t.groupby("fold")])
    pooled = pc.metrics(t.label, t.prob)
    ci = pc.bootstrap_ci(t.label.to_numpy(), t.prob.to_numpy())
    row = {}
    for m in METRICS:
        row[f"{SHORT[m]} fold"] = f"{per_fold[m].mean():.3f} ± {per_fold[m].std(ddof=1):.3f}"
        row[f"{SHORT[m]} pooled [95% CI]"] = f"{pooled[m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]"
    return row, pooled, per_fold


def unit_info(rdir, model, folds):
    infos = []
    for k in folds:
        try:
            infos.append(json.loads((pc.unit_dir(rdir, f"{model}:{k}") / "metrics.json").read_text(encoding="utf-8"))["info"])
        except Exception:
            pass
    return infos


def md_table(df):
    if df is None or not len(df):
        return "_belum ada_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


LEAK_BA = 0.75
MIRROR_ITEMS = ["tables", "figures", "ringkasan_tabel.md", "ringkasan_pilot.md", "status.csv", "missing_report.csv",
                "locked_config_kandidat.json", "config.json", "splits.json"]


def majority_preds(rdir, splits, folds):
    """S0: probabilitas = prevalensi ASD di train fold (konstan) -> selalu memprediksi kelas mayoritas."""
    lab = pd.read_csv(pc.PROJECT / "data" / "folds.csv").set_index("SUB_ID")["label"]
    # satu konstanta (rerata prevalensi train antar fold) agar AUC pooled tepat 0,5
    prev = float(np.mean([lab.loc[splits[str(k)]["inner_train"] + splits[str(k)]["inner_val"]].mean() for k in folds]))
    rows = []
    for k in folds:
        s = splits[str(k)]
        for part in ("inner_val", "test"):
            rows += [{"SUB_ID": i, "fold": k, "part": part, "label": int(lab.at[i]), "prob": prev} for i in s[part]]
    return pd.DataFrame(rows)


def table_A(rdir, cfg, folds, splits, best_pheno):
    """Satu baris per model: BA, F1 macro, AUC, sens, spes (mean ± SD fold) + BA & AUC pooled [95% CI]."""
    rows = []
    for n in ["majority"] + cfg["modality"]["pheno"]:
        p = majority_preds(rdir, splits, folds) if n == "majority" else load_preds(rdir, n, folds)
        grp = "baseline (S0)" if n == "majority" else cfg["models"][n]["group"]
        if p is None:
            rows.append({"model": n, "kelompok": grp, "status": "belum ada"})
            continue
        t = p[p.part == "test"]
        pf = pd.DataFrame([pc.metrics(g.label, g.prob) for _, g in t.groupby("fold")])
        pooled, ci = pc.metrics(t.label, t.prob), pc.bootstrap_ci(t.label.to_numpy(), t.prob.to_numpy())
        v = p[p.part == "inner_val"]
        row = {"model": n, "kelompok": grp, "status": "done",
               "terbaik": "★" if best_pheno and n == best_pheno[0] else "",
               "AUC inner val": round(pc.metrics(v.label, v.prob)["auc"], 3)}
        for m in METRICS:
            row[f"{SHORT[m]} (mean±SD)"] = f"{pf[m].mean():.3f}±{pf[m].std(ddof=1):.3f}"
        for m in ("balanced_accuracy", "auc"):
            row[f"{SHORT[m]} pooled [95% CI]"] = f"{pooled[m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]"
        row["_ba_pooled"] = pooled["balanced_accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)


def leak_warnings(tA):
    if "_ba_pooled" not in tA:
        return []
    hi = tA[tA["_ba_pooled"] > LEAK_BA]
    return [f"PERINGATAN: balanced accuracy pooled {r.model} = {r._ba_pooled:.3f} > {LEAK_BA} -> cek kemungkinan leakage"
            for r in hi.itertuples()]


def mirror(rdir, subset):
    """Salin hasil ringan (tanpa checkpoint/cache) ke folder proyek: <proyek>/results/<subset>/."""
    import shutil
    dst = pc.PROJECT / "results" / subset
    dst.mkdir(parents=True, exist_ok=True)
    for it in MIRROR_ITEMS:
        src = rdir / it
        if src.is_dir():
            for f in src.glob("*"):
                if f.is_file():
                    (dst / it).mkdir(exist_ok=True)
                    pc.sp.copy_atomic(f, dst / it / f.name)
        elif src.exists():
            pc.sp.copy_atomic(src, dst / it)
    return dst


def console_A(rdir):
    """Teks Tabel A + missing + peringatan untuk layar PowerShell."""
    tA = pd.read_csv(rdir / "tables" / "tabel_A_fenotipik.csv")
    mr = pd.read_csv(rdir / "missing_report.csv")
    cols = [c for c in tA.columns if not c.startswith("_") and c not in ("kelompok",)]
    show = tA[cols].fillna("")
    lines = ["", "=" * 100, "TABEL A — FENOTIPIK (test per fold mean±SD; pooled OOF + 95% CI bootstrap 2000x)",
             "★ = terbaik menurut AUC pooled inner validation lima fold (aturan seleksi, bukan skor test)", "=" * 100,
             show.to_string(index=False), "", "Missing per fitur per grup (%):",
             mr.to_string(index=False)]
    w = leak_warnings(tA)
    lines += [""] + (w if w else [f"Tidak ada model dengan BA pooled > {LEAK_BA} (tidak ada tanda leakage dari angka ini)."])
    lines += ["Catatan: pilot n=200 (±40 subjek test per fold) -> CI lebar; peringkat hanya indikasi.", "=" * 100, ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["pilot200", "all"], default="pilot200")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--no-mirror", action="store_true", help="jangan salin ke folder proyek")
    args = ap.parse_args()
    rdir = Path(args.results_dir) if args.results_dir else pc.results_dir(args.subset)
    build(rdir, args.subset)
    if not args.no_mirror and not args.results_dir:
        print("Disalin ke", mirror(rdir, args.subset))


def build(rdir, subset):
    class A:
        pass
    args = A(); args.subset = subset
    cfgj = json.loads((rdir / "config.json").read_text(encoding="utf-8"))
    cfg = cfgj["config"]
    splits = json.loads((rdir / "splits.json").read_text(encoding="utf-8"))
    folds = sorted(int(k) for k in splits)
    tdir, fdir = rdir / "tables", rdir / "figures"
    tdir.mkdir(exist_ok=True); fdir.mkdir(exist_ok=True)
    status = pc.read_status(rdir)
    md = [f"# Hasil eksperimen {args.subset} (config_hash {cfgj['config_hash']})", "",
          f"Status unit: {status['status'].value_counts().to_dict()}", ""]

    # ---------------- seleksi (inner val pooled) ----------------
    sel_rows, pooled_test, oof = [], {}, {}
    for mod, names in cfg["modality"].items():
        for n in names:
            p = load_preds(rdir, n, folds)
            if p is None:
                sel_rows.append({"modalitas": mod, "model": n, "kelompok": cfg["models"][n]["group"],
                                 "AUC inner val (pooled)": "belum ada"})
                continue
            v = p[p.part == "inner_val"]
            sel_rows.append({"modalitas": mod, "model": n, "kelompok": cfg["models"][n]["group"],
                             "AUC inner val (pooled)": round(pc.metrics(v.label, v.prob)["auc"], 3)})
            oof[n] = p[p.part == "test"]
    sel = pd.DataFrame(sel_rows)
    best = {}
    for mod in cfg["modality"]:
        s = sel[(sel.modalitas == mod) & (sel["AUC inner val (pooled)"] != "belum ada")]
        best[mod] = s.sort_values("AUC inner val (pooled)", ascending=False)["model"].tolist()

    # ---------------- Tabel A & B ----------------
    def model_table(names):
        rows = []
        for n in names:
            p = load_preds(rdir, n, folds)
            if p is None:
                rows.append({"model": n, "kelompok": cfg["models"][n]["group"], "status": "belum ada"})
                continue
            row, pooled, _ = summarize(p)
            infos = unit_info(rdir, n, folds)
            extra = {}
            if infos and "params_total" in infos[0]:
                extra["params (latih/total)"] = f"{infos[0].get('params_trainable', '-')}/{infos[0]['params_total']}"
            if infos and "vram_peak_mib" in infos[0]:
                extra["VRAM puncak (MiB)"] = round(max(i["vram_peak_mib"] for i in infos))
                extra["s/epoch"] = round(np.mean([i["s_per_epoch"] for i in infos]), 1)
                extra["waktu total (menit)"] = round(sum(i["seconds"] for i in infos) / 60, 1)
                extra["epoch terbaik (median)"] = int(np.median([i["best_epoch"] for i in infos]))
            v = p[p.part == "inner_val"]
            rows.append({"model": n, "kelompok": cfg["models"][n]["group"], "status": "done",
                         "AUC inner val": round(pc.metrics(v.label, v.prob)["auc"], 3), **row, **extra})
        return pd.DataFrame(rows)

    mr = pd.read_csv(rdir / "missing_report.csv") if (rdir / "missing_report.csv").exists() else None
    tA = table_A(rdir, cfg, folds, splits, best["pheno"])
    tB = model_table(cfg["modality"]["img25d"] + cfg["modality"]["img3d"])
    tA.to_csv(tdir / "tabel_A_fenotipik.csv", index=False); tB.to_csv(tdir / "tabel_B_citra.csv", index=False)

    # ---------------- Tabel C fusion ----------------
    rowsC, fus = [], {}
    for scen in ("S4", "S5"):
        try:
            p = pd.concat([pd.read_csv(pc.unit_dir(rdir, f"fusion_{scen}:{k}") / "preds.csv") for k in folds])
            selj = json.loads((pc.unit_dir(rdir, f"fusion_{scen}:{folds[0]}") / "metrics.json").read_text(encoding="utf-8"))["selection"]
        except Exception:
            rowsC.append({"skenario": scen, "metode": "belum ada"})
            continue
        for mth, g in p.groupby("method"):
            row, pooled, _ = summarize(g.assign(part="test"))
            rowsC.append({"skenario": scen, "citra": selj["img"], "fenotipik": selj["pheno"], "metode": mth,
                          "jenis": "F1 late" if mth.startswith("F1") else "F2 intermediate",
                          "utama": "ya" if mth in (F1_MAIN, F2_MAIN) else "", **row})
            fus[(scen, mth)] = g
    tC = pd.DataFrame(rowsC); tC.to_csv(tdir / "tabel_C_fusion.csv", index=False)

    # ---------------- Tabel D ablasi ----------------
    rowsD, roc = [], {}
    any_oof = next(iter(oof.values()), None)
    if any_oof is not None:
        s0 = majority_preds(rdir, splits, folds)      # sama dengan baris majority di Tabel A
        s0 = s0[s0.part == "test"].merge(any_oof[["SUB_ID", "SITE_ID"]], on="SUB_ID")
        scen = {"S0 majority": s0}
        names = {"S1 fenotipik": best["pheno"][:1], "S2 2.5D": best["img25d"][:1], "S3 3D": best["img3d"][:1]}
        for sname, b in names.items():
            if b:
                scen[f"{sname} ({b[0]})"] = oof[b[0]]
        for sc_, lbl in (("S4", "2.5D+fenotipik"), ("S5", "3D+fenotipik")):
            for mth in (F1_MAIN, F2_MAIN):
                if (sc_, mth) in fus:
                    scen[f"{sc_} {lbl} {mth.split('_')[0]}"] = fus[(sc_, mth)].assign(part="test")
        for sname, p in scen.items():
            row, pooled, pf = summarize(p.assign(part="test"))
            rowsD.append({"skenario": sname, **row})
            roc[sname] = p
            pf.assign(skenario=sname).to_csv(tdir / f"_perfold_{sname.split()[0]}.csv", index=False)
    tD = pd.DataFrame(rowsD); tD.to_csv(tdir / "tabel_D_ablasi.csv", index=False)

    # ---------------- Tabel E: site learning + per situs ----------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    rowsE = []
    coh = pd.read_csv(pc.PROJECT / "data" / "folds.csv").set_index("SUB_ID")["SITE_ID"]
    for n in cfg["modality"]["img25d"] + cfg["modality"]["img3d"]:
        if load_preds(rdir, n, folds) is None:
            continue
        yt, yp = [], []
        for k in folds:
            e = np.load(pc.unit_dir(rdir, f"{n}:{k}") / "emb.npz")
            sc = StandardScaler().fit(e["emb_inner_train"])
            clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(e["emb_inner_train"]),
                                                               coh.loc[e["ids_inner_train"]].to_numpy())
            yt += coh.loc[e["ids_test"]].tolist(); yp += clf.predict(sc.transform(e["emb_test"])).tolist()
        nsite = len(set(yt))
        rowsE.append({"embedding": n, "BA prediksi SITE_ID (test)": round(balanced_accuracy_score(yt, yp), 3),
                      "peluang acak": round(1 / nsite, 3), "n situs": nsite})
    tE = pd.DataFrame(rowsE)
    rowsS = []
    for sname, p in roc.items():
        if sname.startswith("S0"):
            continue
        for site, g in p.groupby("SITE_ID"):
            m = pc.metrics(g.label, g.prob)
            rowsS.append({"skenario": sname, "SITE_ID": site, "n": m["n"], "BA": round(m["balanced_accuracy"], 3),
                          "AUC": round(m["auc"], 3) if not np.isnan(m["auc"]) else np.nan})
    tS = pd.DataFrame(rowsS)
    tE.to_csv(tdir / "tabel_E_site_learning.csv", index=False); tS.to_csv(tdir / "tabel_E_per_situs.csv", index=False)

    # ---------------- gambar ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    if roc:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for sname, p in roc.items():
            if sname.startswith("S0"):
                continue
            fpr, tpr, _ = roc_curve(p.label, p.prob)
            ax.plot(fpr, tpr, label=f"{sname} (AUC {pc.metrics(p.label, p.prob)['auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 - spesifisitas"); ax.set_ylabel("sensitivitas"); ax.set_title(f"ROC pooled OOF ({args.subset})")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(fdir / "roc_skenario.png", dpi=130); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        data, labels = [], []
        for sname, p in roc.items():
            data.append([pc.metrics(g.label, g.prob)["auc"] for _, g in p.groupby("fold")]); labels.append(sname)
        ax.boxplot(data, tick_labels=labels); ax.set_ylabel("AUC per fold"); ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7); fig.tight_layout()
        fig.savefig(fdir / "boxplot_auc_per_fold.png", dpi=130); plt.close(fig)

    # ---------------- ringkasan & konfigurasi terkunci ----------------
    groups = {}
    for n, hp in cfg["models"].items():
        groups.setdefault(hp["group"], []).append(n)
    cand = {mod: best[mod][:3] for mod in best}
    # LogReg selalu ikut sebagai baseline standar (keputusan peneliti), bukan karena skor test
    if "logreg" in best.get("pheno", []):
        cand["pheno"] = ["logreg"] + [m for m in best["pheno"] if m != "logreg"][:2]
    locked = {"config_hash_pilot": cfgj["config_hash"], "kandidat_data_penuh": cand,
              "aturan_kandidat": "2-3 teratas menurut AUC pooled inner val; LogReg selalu disertakan sebagai baseline "
                                 "standar. Seleksi via inner val tidak stabil pada n=200 (±32 subjek inner val/fold).",
              "fusion": {"F1": F1_MAIN, "F2": F2_MAIN, "pca_dims": cfg["fusion"]["pca_dims"]},
              "hyperparameter": {n: cfg["models"][n] for mod in cand for n in cand[mod]},
              "catatan": "pilot n=200 (inner holdout tunggal); data penuh memakai inner CV penuh"}
    pc.sp.write_json_atomic(rdir / "locked_config_kandidat.json", locked)

    md += ["## Fitur fenotipik — missing per grup", md_table(mr),
           "## Seleksi (AUC pooled inner validation lima fold; BUKAN skor test)", md_table(sel),
           "## Tabel A — fenotipik (test fold, mean ± SD; pooled OOF + 95% CI bootstrap)",
           "★ = terbaik menurut AUC pooled inner validation lima fold (aturan seleksi, bukan skor test).", "",
           md_table(tA[[c for c in tA.columns if not c.startswith("_")]]),
           "\n".join(leak_warnings(tA)) or f"Tidak ada model dengan BA pooled > {LEAK_BA}.", "",
           "## Tabel B — citra 2.5D & 3D", md_table(tB),
           "## Tabel C — fusion F1 vs F2 (utama: F1 = stacking LogReg, F2 = PCA16 + LogReg; ditetapkan sebelum hasil)",
           md_table(tC), "## Tabel D — ablasi S0–S5", md_table(tD),
           "## Tabel E — site learning (prediksi SITE_ID dari embedding citra)", md_table(tE),
           "## Tabel E — performa per situs (pooled OOF)", md_table(tS),
           "## Kelompok model", "\n".join(f"- **{g}**: {', '.join(v)}" for g, v in groups.items()), "",
           "## Kandidat untuk data penuh (2–3 per modalitas, urut AUC inner val)",
           "\n".join(f"- {mod}: {', '.join(v) if v else 'belum ada'}" for mod, v in cand.items()), "",
           "## Ditunda", "\n".join(f"- {k}: {v}" for k, v in cfg["deferred"].items()), "",
           "## Catatan jujur",
           "- Pilot 20% (n=200, ±40 subjek test per fold): CI lebar; selisih antar model yang kecil belum tentu bermakna.",
           "- Peringkat hanya indikasi; data penuh memakai inner CV penuh, bukan inner holdout tunggal.",
           "- Head fusion dilatih pada inner holdout (±32 subjek per fold) — penyederhanaan pilot.",
           "- SFCN & DINOv2: bagian beku di-cache tanpa augmentasi."]
    (rdir / "ringkasan_tabel.md").write_text("\n".join(md), encoding="utf-8")
    print("Tabel ditulis ke", tdir, "| ringkasan:", rdir / "ringkasan_tabel.md")
    print("Status:", status["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
