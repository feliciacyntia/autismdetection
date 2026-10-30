"""Unggah file biner hasil modeling ke Hugging Face (repo model): checkpoint .pt, cache embedding .npy, atlas ROI.
Struktur sama dengan repo GitHub (results/..., pretrained/...), jadi bisa diunduh langsung ke root repo:

    hf download feliciacyntia/abide1-asd-results --local-dir . --include "results/*" --include "pretrained/*"

    .venv-model\\Scripts\\python.exe src\\upload_hf_results.py --stage
    .venv-model\\Scripts\\python.exe src\\upload_hf_results.py --upload --repo-id feliciacyntia/abide1-asd-results
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RES = Path(r"C:\ta_local\results")
PRE = Path(r"C:\ta_local\pretrained")
STAGE = Path(r"C:\ta_local\hf_stage\abide1-asd-results")
RUNS = ["pilot200", "pilot200_v2", "pilot200_v2_bnfix", "full_v2"]       # run uji (_smoke dll.) tidak disertakan


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def stage():
    for run in RUNS:
        for f in (RES / run).rglob("*"):
            if f.is_file() and f.suffix in (".pt", ".npy"):
                link(f, STAGE / "results" / run / f.relative_to(RES / run))
    for f in (PRE / "atlas").rglob("*"):
        if f.is_file():
            link(f, STAGE / "pretrained" / "atlas" / f.relative_to(PRE / "atlas"))
    shutil.copy2(PROJECT / "src" / "hf_results_card.md", STAGE / "README.md")
    files = sorted(f for f in STAGE.rglob("*") if f.is_file() and ".cache" not in f.parts and f.name != "SHA256SUMS")
    (STAGE / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(STAGE).as_posix()}\n"
                                              for f in files), encoding="utf-8")
    by = {}
    for f in files:
        k = "/".join(f.relative_to(STAGE).parts[:2])
        n, s = by.get(k, (0, 0)); by[k] = (n + 1, s + f.stat().st_size)
    for k, (n, s) in sorted(by.items()):
        print(f"  {k:32s} {n:5d} file {s / 2**30:6.2f} GiB")
    print(f"STAGING OK: {STAGE} | {len(files)} file | {sum(f.stat().st_size for f in files) / 2**30:.2f} GiB")


def upload(repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    print("login sebagai:", api.whoami()["name"])
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, folder_path=str(STAGE), repo_type="model")
    print(f"SELESAI: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--repo-id", default="feliciacyntia/abide1-asd-results")
    a = ap.parse_args()
    if a.stage:
        stage()
    if a.upload:
        upload(a.repo_id)
