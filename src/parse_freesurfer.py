"""Parsing + QC statistik FreeSurfer PCP -> data/freesurfer_pcp/morfometri.csv, qc_morfometri.csv, sanity_morfometri.json.

Fitur: ICV & volume global, volume subkortikal (aseg); ketebalan/luas/volume GM 34 region Desikan per belahan (aparc).
QC (ditetapkan sebelum modeling): ICV |z robust| > 4, ICV < 1,2 x GM total, region hilang, ketebalan rata-rata < 1,5 / > 4 mm,
identitas file (# subjectname) tidak cocok. Tiga subjek OHSU/USM memakai format aseg versi lain (tanpa ICV, definisi SubCortGray beda) -> gagal QC. Gagal QC / tidak tersedia = semua fitur subjek NaN, subjek TETAP di kohort.
Median & MAD ICV dihitung dari subjek dev saja. Tidak memakai label diagnosis.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
FS = PROJECT / "data" / "freesurfer_pcp"
GLOBAL = {"ICV": "ICV", "SupraTentorialVol": "SupraTentorialVol", "CortexVol": "CortexVol", "CorticalWhiteMatterVol": "CorticalWhiteMatterVol",
          "SubCortGrayVol": "SubCortGrayVol", "TotalGrayVol": "TotalGrayVol"}
SUBCORT = ["Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum", "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area",
           "Left-VentralDC", "Left-Lateral-Ventricle", "Left-Cerebellum-Cortex", "Left-Cerebellum-White-Matter",
           "Right-Thalamus-Proper", "Right-Caudate", "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
           "Right-VentralDC", "Right-Lateral-Ventricle", "Right-Cerebellum-Cortex", "Right-Cerebellum-White-Matter",
           "Brain-Stem", "CSF", "3rd-Ventricle", "4th-Ventricle", "CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior"]
DESIKAN = ["bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
           "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
           "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral", "posteriorcingulate", "precentral",
           "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal",
           "frontalpole", "temporalpole", "transversetemporal", "insula"]


def subject_ok(text, sid):
    m = re.search(r"#\s*subjectname\s+(\S+)", text)
    return bool(m) and str(sid) in m.group(1)


def measures(text):
    """# Measure a, b, deskripsi, nilai, unit -> {b: nilai}."""
    return {m.group(1).strip(): float(m.group(2)) for m in re.finditer(r"#\s*Measure\s+[^,]+,\s*([^,]+),[^,]*,\s*([-\d.eE+]+)\s*,", text)}


def parse_subject(sid):
    """-> (fitur dict, alasan gagal list)."""
    d, f, why = FS / str(sid), {}, []
    paths = {k: d / k for k in ("aseg.stats", "lh.aparc.stats", "rh.aparc.stats")}
    if not all(p.exists() for p in paths.values()):
        return {}, ["file_tidak_tersedia"]
    a = paths["aseg.stats"].read_text(errors="replace")
    if not subject_ok(a, sid):
        why.append("identitas_aseg")
    me = measures(a)
    for k, n in GLOBAL.items():
        f[n] = me.get(k, np.nan)
    rows = {}
    for ln in a.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        rows[p[4]] = float(p[3])
    for n in SUBCORT:
        f[f"vol_{n}"] = rows.get(n, np.nan)
    miss = [n for n in SUBCORT if n not in rows]
    if miss:
        why.append("subkortikal_hilang:" + "|".join(miss))
    for h in ("lh", "rh"):
        t = paths[f"{h}.aparc.stats"].read_text(errors="replace")
        if not subject_ok(t, sid):
            why.append(f"identitas_{h}")
        m2 = measures(t)
        f[f"{h}_MeanThickness"], f[f"{h}_WhiteSurfArea"] = m2.get("MeanThickness", np.nan), m2.get("WhiteSurfArea", np.nan)
        reg = {}
        for ln in t.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split()     # StructName NumVert SurfArea GrayVol ThickAvg ...
            reg[p[0]] = (float(p[2]), float(p[3]), float(p[4]))
        miss = [r for r in DESIKAN if r not in reg]
        if miss:
            why.append(f"region_hilang_{h}:" + "|".join(miss))
        for r in DESIKAN:
            ar, gv, th = reg.get(r, (np.nan, np.nan, np.nan))
            f[f"{h}_{r}_area"], f[f"{h}_{r}_gmvol"], f[f"{h}_{r}_thick"] = ar, gv, th
    return f, why


def main():
    folds = pd.read_csv(PROJECT / "data" / "folds.csv")
    rows, qc = [], []
    for sid in folds.SUB_ID.astype(int):
        f, why = parse_subject(sid)
        rows.append({"SUB_ID": sid, **f}); qc.append({"SUB_ID": sid, "alasan": why})
    X = pd.DataFrame(rows).set_index("SUB_ID")
    Q = pd.DataFrame(qc).set_index("SUB_ID")
    dev = folds[folds.split == "dev"].SUB_ID.astype(int)
    med = X.loc[dev, "ICV"].median()
    mad = 1.4826 * (X.loc[dev, "ICV"] - med).abs().median()
    z = (X["ICV"] - med) / mad
    thick = X[["lh_MeanThickness", "rh_MeanThickness"]].mean(axis=1)
    for sid in X.index:
        w = Q.at[sid, "alasan"]
        if "file_tidak_tersedia" not in w and np.isnan(X.at[sid, "ICV"]):
            w.append("versi_aseg_beda_tanpa_ICV")
        if not np.isnan(z[sid]) and abs(z[sid]) > 4:
            w.append("icv_z_robust>4")
        if X.at[sid, "ICV"] < 1.2 * X.at[sid, "TotalGrayVol"]:
            w.append("icv<1.2xgm")
        if thick[sid] < 1.5 or thick[sid] > 4:
            w.append("ketebalan_di_luar_1.5-4mm")
    Q["lolos"] = Q.alasan.map(len) == 0
    Q["ICV_z_robust"], Q["mean_thickness"] = z, thick
    Q["alasan"] = Q.alasan.map("; ".join)
    Xqc = X.copy().astype(float)
    Xqc.loc[~Q.lolos.reindex(Xqc.index), :] = np.nan
    bad = [c for c in Xqc.columns if c.upper().startswith(("SITE", "DX_", "ADOS", "ADI", "SRS", "SCQ", "AQ", "DSM"))]
    assert not bad, bad
    Xqc.reset_index().to_csv(FS / "morfometri.csv", index=False)
    Q.reset_index().to_csv(FS / "qc_morfometri.csv", index=False)

    ph = pd.read_csv(PROJECT / "data" / "pheno_clean.csv").set_index("SUB_ID")
    d = Xqc.loc[dev].join(ph[["AGE_AT_SCAN", "SEX"]])
    th = d[["lh_MeanThickness", "rh_MeanThickness"]].mean(axis=1)
    ok = th.notna() & d.AGE_AT_SCAN.notna()
    sanity = {"n_total": int(len(Q)), "n_lolos": int(Q.lolos.sum()), "n_dev": int(len(dev)), "n_dev_lolos": int(Q.loc[dev, "lolos"].sum()),
              "icv_median_dev": float(med), "icv_mad_dev": float(mad),
              "r_spearman_usia_vs_ketebalan_dev": float(pd.concat([d.AGE_AT_SCAN[ok], th[ok]], axis=1).corr(method="spearman").iloc[0, 1]),
              "icv_median_laki_dev": float(d[d.SEX == 1].ICV.median()), "icv_median_perempuan_dev": float(d[d.SEX == 2].ICV.median()),
              "n_fitur": int(Xqc.shape[1]), "gagal_per_alasan": Q[~Q.lolos].alasan.str.split(":").str[0].value_counts().to_dict()}
    sanity["uji_usia_ketebalan_lulus"] = sanity["r_spearman_usia_vs_ketebalan_dev"] < 0
    sanity["uji_icv_lk_gt_pr_lulus"] = sanity["icv_median_laki_dev"] > sanity["icv_median_perempuan_dev"]
    (FS / "sanity_morfometri.json").write_text(json.dumps(sanity, indent=1), encoding="utf-8")
    print(json.dumps(sanity, indent=1))
    print(Q[~Q.lolos][["alasan", "ICV_z_robust", "mean_thickness"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
