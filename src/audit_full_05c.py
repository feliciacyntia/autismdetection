"""Tahap 0 — audit konsistensi output 05c untuk SEMUA subjek (read-only; hasil ke <out>/audit_full/).

    .venv-model\\Scripts\\python.exe src\\audit_full_05c.py

1 fingerprint · 2 versi tool/bobot/template per subjek & per run · 3 kelengkapan & validitas output · 4 metrik QC pilot vs
subjek baru (KS + Holm) · 6 header voxel/FOV citra mentah · 7 subjek test held-out tidak ada di hasil modeling.
(Item 5 = repro_check_05c.py.) Kode keluar 0 = semua lulus, 1 = ada yang gagal/perlu keputusan.
"""
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT = Path(__file__).resolve().parent.parent
PRE = Path(r"C:\ta_local\data\smri_preproc")
RESULTS = [Path(r"C:\ta_local\results"), PROJECT / "results"]
EXPECTED_FP = "682404ae0f5d2624"
OUT = PRE / "audit_full"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); m = len(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    folds = pd.read_csv(PROJECT / "data" / "folds.csv")
    pilot = set(pd.read_csv(PROJECT / "data" / "subset_20pct.csv").SUB_ID)
    ids = folds.SUB_ID.astype(int).tolist()
    rep, fails, flags = {}, [], []
    st = pd.read_csv(PRE / "status.csv")
    payload = json.loads((PRE / "preprocess_config.json").read_text(encoding="utf-8"))["fingerprint_payload"]

    # ---- 1. fingerprint ----
    fp_status = st.config_hash.value_counts().to_dict()
    qc = pd.read_csv(PRE / "qc_report.csv")
    metas = {s: json.loads((PRE / "meta" / f"{s}.json").read_text(encoding="utf-8")) for s in ids}
    fp_meta = Counter(m.get("config_fp") for m in metas.values())
    rep["1_fingerprint"] = {"status.csv": fp_status, "qc_report.csv": qc.config_fp.value_counts().to_dict(), "meta/*.json": dict(fp_meta),
                            "subjek_di_status": int(st.subject_id.isin(ids).sum()), "subjek_folds": len(ids)}
    if set(fp_status) != {EXPECTED_FP} or set(fp_meta) != {EXPECTED_FP} or rep["1_fingerprint"]["subjek_di_status"] != len(ids):
        fails.append("1 fingerprint tidak tunggal / subjek tidak lengkap")

    # ---- 2. versi tool per subjek + per run + hash template ----
    tool = Counter(json.dumps({k: m["tool"].get(k) for k in ("tool", "version", "nnunetv2", "torch", "weights_sha256", "device")},
                              sort_keys=True) for m in metas.values())
    pipe = Counter(m.get("pipeline_version") for m in metas.values())
    tmpl = {}                                       # sama dengan smri_preprocessing: sha256(array T1 + mask + affine)
    for res in (1, 2):
        ti = nib.load(str(PRE / "templates" / f"mni152_t1_{res}mm.nii.gz"))
        t = np.asanyarray(ti.dataobj).astype(np.float32)
        m = (np.asanyarray(nib.load(str(PRE / "templates" / f"mni152_mask_{res}mm.nii.gz")).dataobj) > 0).astype(np.uint8)
        tmpl[f"{res}mm"] = hashlib.sha256(np.ascontiguousarray(t).tobytes() + np.ascontiguousarray(m).tobytes()
                                          + np.asarray(ti.affine, dtype=np.float64).tobytes()).hexdigest()
    runs_env = Counter()
    for f in sorted((PRE / "runs").glob("*.json")):
        if f.name.endswith("_subjects.json"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        env = r.get("env") or r.get("environment") or {}
        if env:
            runs_env[json.dumps({k: env.get(k) for k in ("antspyx", "nibabel", "numpy", "nilearn", "scipy", "python")}, sort_keys=True)] += 1
    rep["2_versi_tool"] = {"kombinasi_tool_per_subjek": {k: v for k, v in tool.items()}, "pipeline_version": dict(pipe),
                           "versi_di_fingerprint": payload["versions"], "hdbet_weights_fingerprint": payload["hdbet_weights_sha256"],
                           "template_sha_sekarang": tmpl, "template_sha_fingerprint": payload["templates"],
                           "env_per_run (antspyx/nibabel/numpy/...)": dict(runs_env)}
    if len(tool) != 1 or len(pipe) != 1 or tmpl != payload["templates"] or len(runs_env) > 1:
        fails.append("2 campuran versi tool/template")

    # ---- 3. kelengkapan & validitas output ----
    rows = []
    for s in ids:
        r = {"SUB_ID": s, "pilot": s in pilot}
        try:
            v = np.load(PRE / "vol" / f"{s}.npy"); m = np.load(PRE / "mask" / f"{s}.npy")
            vn = np.asanyarray(nib.load(str(PRE / "nii" / f"{s}.nii.gz")).dataobj)
            mn = np.asanyarray(nib.load(str(PRE / "nii" / f"{s}_mask.nii.gz")).dataobj)
            r.update(npy_shape_ok=v.shape == (1, 96, 112, 96) and m.shape == (1, 96, 112, 96), nii_shape_ok=vn.shape == (96, 112, 96) == mn.shape,
                     mask_biner=bool(np.isin(np.unique(m), [0, 1]).all() and np.isin(np.unique(mn), [0, 1]).all()),
                     mask_tidak_kosong=bool(m.sum() > 0), luar_mask_nol=bool((v[m == 0] == 0).all() and (vn[mn == 0] == 0).all()),
                     finite=bool(np.isfinite(v).all() and np.isfinite(vn).all()), npy_sama_nii=bool(np.array_equal(v[0], vn) and np.array_equal(m[0].astype(bool), mn.astype(bool))),
                     montase_ada=(PRE / "qc_figures" / f"{s}.png").exists(), mask_voxels=int(m.sum()))
        except Exception as e:  # noqa: BLE001
            r["error"] = repr(e)[:200]
        rows.append(r)
    V = pd.DataFrame(rows); V.to_csv(OUT / "3_validitas_output.csv", index=False)
    chk = [c for c in V.columns if c not in ("SUB_ID", "pilot", "mask_voxels", "error")]
    bad = V[(~V[chk].fillna(False).astype(bool)).any(axis=1) | (V["error"].notna() if "error" in V else False)]
    rep["3_validitas_output"] = {"subjek_dicek": len(V), "gagal": bad.SUB_ID.tolist(), **{c: int(V[c].fillna(False).astype(bool).sum()) for c in chk}}
    if len(bad):
        fails.append(f"3 output tidak valid: {bad.SUB_ID.tolist()[:20]}")

    # ---- 4. metrik QC pilot vs baru ----
    q = qc.set_index("SUB_ID").loc[ids].copy()
    for k in ("zscore_mean_raw", "zscore_std_raw", "clip_low", "clip_high", "brain_volume_native_ml", "vol_min", "vol_max"):
        q[k] = [metas[s].get(k) for s in q.index]
    q["pilot"] = q.index.isin(pilot)
    M = ["volume_mask_ml", "mask_ratio_template", "dice_mask_template", "reg_det", "reg_scale_min", "reg_scale_max", "ncc_template",
         "frac_zero", "zscore_mean_raw", "zscore_std_raw", "clip_low", "clip_high", "brain_volume_native_ml", "vol_min", "vol_max"]
    krows = []
    for k in M:
        a, b = q.loc[q.pilot, k].dropna().astype(float), q.loc[~q.pilot, k].dropna().astype(float)
        ks = ks_2samp(a, b)
        krows.append({"metrik": k, "pilot median [IQR]": f"{a.median():.4g} [{a.quantile(.25):.4g}–{a.quantile(.75):.4g}]",
                      "baru median [IQR]": f"{b.median():.4g} [{b.quantile(.25):.4g}–{b.quantile(.75):.4g}]",
                      "KS D": round(ks.statistic, 4), "p": ks.pvalue})
    K = pd.DataFrame(krows); K["p Holm"] = holm(K.p); K["beda sistematis (Holm < 0,05)"] = K["p Holm"] < 0.05
    K.to_csv(OUT / "4_qc_pilot_vs_baru.csv", index=False)
    rep["4_qc_pilot_vs_baru"] = {"n_pilot": int(q.pilot.sum()), "n_baru": int((~q.pilot).sum()),
                                 "metrik_beda_holm": K.loc[K["beda sistematis (Holm < 0,05)"], "metrik"].tolist()}
    if K["beda sistematis (Holm < 0,05)"].any():
        fails.append("4 perbedaan sistematis metrik QC pilot vs baru")
    fl = qc[qc.n_flags > 0][["SUB_ID", "SITE_ID", "flag_reasons"]]
    known = set(pd.read_csv(PRE / "qc_decisions.csv").subject_id) | {50009, 50118, 50345, 50482, 50708, 50955, 51220, 51324, 51461, 51581}
    rep["4b_flag_qc"] = {"semua": fl.to_dict("records"), "flag_baru_belum_diputuskan": fl[~fl.SUB_ID.isin(known)].SUB_ID.tolist()}

    # ---- 6. header voxel/FOV citra mentah ----
    cm = pd.read_csv(PROJECT / "dataset" / "cleaned_dataset" / "cleaned_master_dataset.csv").set_index("SUB_ID")
    hrows = []
    for s in ids:
        h = nib.load(str(PROJECT / cm.at[s, "smri_path"])).header
        z = [round(float(x), 2) for x in h.get_zooms()[:3]]; sh = [int(x) for x in h.get_data_shape()[:3]]
        hrows.append({"SUB_ID": s, "SITE_ID": folds.set_index("SUB_ID").at[s, "SITE_ID"], "pilot": s in pilot,
                      "voxel_mm": tuple(sorted(z)), "matrix": tuple(sorted(sh)),
                      "fov_mm": tuple(sorted(int(round(a * b)) for a, b in zip(z, sh)))})
    H = pd.DataFrame(hrows)
    mode = lambda x: Counter(x).most_common(1)[0][0]
    H["site_mode_voxel"] = H.SITE_ID.map(H.groupby("SITE_ID").voxel_mm.agg(mode))
    H["flag_voxel_vs_site_mode"] = H.voxel_mm != H.site_mode_voxel
    # FOV dibandingkan dengan subjek SITE yang sama dan matriks yang sama (kasus 50404: matriks sama, voxel lebih kecil)
    key = list(zip(H.SITE_ID, H.matrix))
    H["fov_mode_same_matrix"] = pd.Series(key).map(H.assign(k=key).groupby("k").fov_mm.agg(mode)).to_numpy()
    H["n_same_matrix"] = pd.Series(key).map(Counter(key)).to_numpy()
    H["fov_dev_same_matrix"] = [max(abs(a - b) / b for a, b in zip(f, m)) for f, m in zip(H.fov_mm, H.fov_mode_same_matrix)]
    H["flag_fov_vs_same_matrix"] = (H.fov_dev_same_matrix > 0.05) & (H.n_same_matrix >= 5)
    H.to_csv(OUT / "6_header_voxel_fov.csv", index=False)
    prev = pd.read_csv(PRE / "voxel_spacing_check.csv")
    rep["6_header"] = {"flag_voxel_vs_site_mode": int(H.flag_voxel_vs_site_mode.sum()),
                       "flag_fov_janggal (seperti 50404)": H.loc[H.flag_fov_vs_same_matrix, ["SUB_ID", "SITE_ID", "voxel_mm", "matrix", "fov_mm",
                                                                                          "fov_mode_same_matrix"]].astype(str).to_dict("records"),
                       "sama_dengan_cek_sebelumnya": sorted(H.loc[H.flag_fov_vs_same_matrix, "SUB_ID"]) == sorted(prev.loc[prev.flag_fov_vs_same_matrix, "SUB_ID"])}
    new_fov = sorted(set(H.loc[H.flag_fov_vs_same_matrix, "SUB_ID"]) - {50404})
    if new_fov:
        fails.append(f"6 header FOV janggal baru (perlu keputusan): {new_fov}")

    # ---- 7. test held-out tidak ada di hasil modeling ----
    test = set(folds.loc[folds.split == "test", "SUB_ID"].astype(int))
    hit, nfile = Counter(), 0
    for root in RESULTS:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            found = set()
            if f.suffix == ".csv":
                try:
                    d = pd.read_csv(f, nrows=None)
                except Exception:  # noqa: BLE001
                    continue
                for c in [c for c in d.columns if c.upper() in ("SUB_ID", "SUBJECT_ID", "EKSPERIMEN_SUB_ID")]:
                    found |= set(pd.to_numeric(d[c], errors="coerce").dropna().astype(int))
            elif f.suffix == ".npz":
                z = np.load(f)
                for k in z.files:
                    if k.startswith("ids_"):
                        found |= set(z[k].astype(int))
            elif f.suffix == ".npy" and f.stem.isdigit():
                found.add(int(f.stem))
            elif f.name == "splits.json":
                for s_ in json.loads(f.read_text(encoding="utf-8")).values():
                    for v in s_.values():
                        found |= set(int(x) for x in v)
            else:
                continue
            nfile += 1
            for s in found & test:
                hit[str(f)] += 1
    rep["7_test_heldout"] = {"n_test": len(test), "file_dicek": nfile, "file_berisi_subjek_test": dict(hit)}
    if hit:
        fails.append(f"7 subjek test ditemukan di {len(hit)} file hasil")

    rep["ringkasan"] = {"lulus": not fails, "masalah": fails, "detik": round(time.time() - t0)}
    (OUT / "audit_full.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str)[:6000])
    print("\nTabel KS:\n" + K.drop(columns="p").round(4).to_string(index=False))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
