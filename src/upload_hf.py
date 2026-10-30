"""Siapkan & unggah hasil preprocessing sMRI 05c (HANYA 853 subjek dev; 149 test held-out TIDAK ikut) ke Hugging Face.

    .venv-model\\Scripts\\python.exe src\\upload_hf.py --stage                      # buat folder staging (hardlink) + verifikasi
    hf auth login                                                                  # sekali, token write
    .venv-model\\Scripts\\python.exe src\\upload_hf.py --upload --repo-id feliciacyntia/abide1-smri-prepro

Staging: C:\\ta_local\\hf_stage\\abide1-smri-preproc-05c (hardlink -> tidak memakan ruang tambahan; sumber tidak diubah).
Upload memakai upload_large_folder (bisa dilanjutkan bila terputus: jalankan perintah yang sama lagi).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SRC = Path(r"C:\ta_local\data\smri_preproc")
STAGE = Path(r"C:\ta_local\hf_stage\abide1-smri-preproc-05c")
FP = "682404ae0f5d2624"
PER_SUBJ = {"vol": ["{s}.npy"], "mask": ["{s}.npy"], "nii": ["{s}.nii.gz", "{s}_mask.nii.gz"],
            "mask_native": ["{s}_mask_native.nii.gz"], "xfm": ["{s}_0GenericAffine.mat"], "qc_figures": ["{s}.png"]}
# meta/ & metadata tidak diunggah: sudah ada di GitHub (data/folds.csv, data/smri_preproc/) -> tanpa duplikat


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage():
    folds = pd.read_csv(PROJECT / "data" / "folds.csv")
    dev = folds[folds.split == "dev"]
    test_ids = set(folds.loc[folds.split == "test", "SUB_ID"].astype(int))
    st = pd.read_csv(SRC / "status.csv")
    ids = sorted(dev.SUB_ID.astype(int))
    assert len(ids) == 853, len(ids)
    s_dev = st[st.subject_id.astype(int).isin(ids)]
    assert s_dev.status.isin(["success", "flagged"]).all() and len(s_dev) == 853, "ada subjek dev yang gagal preprocessing"
    assert set(s_dev.config_hash.astype(str)) == {FP}, f"fingerprint campur: {set(s_dev.config_hash)}"
    for d, pats in PER_SUBJ.items():
        for s in ids:
            for p in pats:
                f = SRC / d / p.format(s=s)
                assert f.exists(), f"tidak ada: {f}"
                link(f, STAGE / d / f.name)
    for f in (SRC / "templates").iterdir():
        link(f, STAGE / "templates" / f.name)
    # bukti uji reproducibility & output config lama (HD-BET non-deterministik): hanya subjek dev, dan hanya file yang
    # isinya belum ada (output repro identik byte-per-byte dengan output utama -> cukup laporan repro_*.json, tanpa duplikat)
    have = {hashlib.sha256(f.read_bytes()).hexdigest() for d in list(PER_SUBJ) + ["templates"] for f in (STAGE / d).iterdir()}
    for sub in ("repro", "_superseded"):
        for d in [STAGE / sub]:
            if d.exists():
                shutil.rmtree(d)                     # staging = hardlink; sumber tidak tersentuh
        for f in sorted((SRC / sub).rglob("*")):
            r = f.relative_to(SRC)
            if not f.is_file() or any(p.split("_")[0].split(".")[0].isdigit() and int(p.split("_")[0].split(".")[0]) in test_ids
                                      for p in [x.replace("repro_", "") for x in r.parts]):
                continue
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            if h in have:
                continue
            have.add(h)
            link(f, STAGE / r)
    shutil.copy2(PROJECT / "src" / "hf_dataset_card.md", STAGE / "README.md")
    # verifikasi: tidak ada subjek test, jumlah file benar
    files = [f for f in STAGE.rglob("*") if f.is_file() and ".cache" not in f.parts]

    def sid(p):
        h = p.replace("repro_", "").split("_")[0].split(".")[0]
        return int(h) if h.isdigit() and len(h) == 5 else None
    leak = [f for f in files if any(sid(p) in test_ids for p in f.relative_to(STAGE).parts)]
    assert not leak, f"subjek test ikut: {leak[:3]}"
    for d, pats in PER_SUBJ.items():
        n = len(list((STAGE / d).iterdir()))
        assert n == 853 * len(pats), (d, n)
    lines = []
    for f in sorted(files):
        if f.name == "SHA256SUMS":
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append(f"{h}  {f.relative_to(STAGE).as_posix()}")
    (STAGE / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in files) / 2**30
    print(f"STAGING OK: {STAGE} | {len(files)} file | {size:.2f} GiB | 853 subjek dev | 0 subjek test | fingerprint {FP}")


def upload(repo_id, private):
    from huggingface_hub import HfApi
    api = HfApi()
    print("login sebagai:", api.whoami()["name"])
    assert (STAGE / "SHA256SUMS").exists(), "jalankan --stage dulu"
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, folder_path=str(STAGE), repo_type="dataset")
    print(f"SELESAI: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--private", action="store_true", help="default publik (keputusan pemilik TA 2026-10-29)")
    a = ap.parse_args()
    if a.stage:
        stage()
    if a.upload:
        if not a.repo_id:
            sys.exit("--repo-id wajib, mis. USER/abide1-smri-preproc-05c")
        upload(a.repo_id, a.private)
