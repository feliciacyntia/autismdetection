"""Unduh statistik FreeSurfer PCP ABIDE I (aseg.stats, lh/rh.aparc.stats) untuk subjek di data/folds.csv.

Hanya subjek folds.csv; retry 3x; 404 = tidak tersedia (dicatat). Resume: file yang sudah ada dilewati.
Map SUB_ID -> folder memakai listing bucket (FILE_ID fenotipik berisi 'no_filename' untuk sebagian subjek).
Output: data/freesurfer_pcp/{SUB_ID}/*.stats + data/freesurfer_pcp/fetch_log.csv
"""
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "data" / "freesurfer_pcp"
BASE = "https://s3.amazonaws.com/fcp-indi"
PREFIX = "data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/"
FILES = ["aseg.stats", "lh.aparc.stats", "rh.aparc.stats"]


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
        time.sleep(2 * (i + 1))


def list_folders():
    folders, marker = {}, ""
    while True:
        q = urllib.parse.urlencode({"prefix": PREFIX, "delimiter": "/", "marker": marker})
        xml = get(f"{BASE}?{q}").decode()
        names = re.findall(rf"<Prefix>{re.escape(PREFIX)}([^<]+)/</Prefix>", xml)
        for n in names:
            m = re.search(r"(\d{5,7})$", n)
            if m:
                folders.setdefault(int(m.group(1)), []).append(n)
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        if "<IsTruncated>true" not in xml or not nxt:
            return folders
        marker = nxt.group(1)


def fetch(sid, folder):
    rows = []
    for f in FILES:
        dst = OUT / str(sid) / f
        if dst.exists() and dst.stat().st_size > 0:
            rows.append((sid, folder, f, "ada"))
            continue
        try:
            data = get(f"{BASE}/{PREFIX}{folder}/stats/{f}")
        except Exception as e:  # dicatat, bukan menghentikan seluruh unduhan
            rows.append((sid, folder, f, f"gagal: {type(e).__name__}"))
            continue
        if data is None:
            rows.append((sid, folder, f, "404"))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(dst)
        rows.append((sid, folder, f, "unduh"))
    return rows


def main():
    ids = pd.read_csv(PROJECT / "data" / "folds.csv").SUB_ID.astype(int).tolist()
    folders = list_folders()
    print(f"Folder FreeSurfer di bucket: {len(folders)}; subjek folds.csv: {len(ids)}", flush=True)
    dup = {k: v for k, v in folders.items() if len(v) > 1 and k in set(ids)}
    if dup:
        print("Folder ganda (dipakai yang pertama, dicatat):", dup, flush=True)
    rows = [(s, "", f, "tidak ada folder") for s in ids if s not in folders for f in FILES]
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(lambda s: fetch(s, folders[s][0]), [s for s in ids if s in folders]), 1):
            rows += r
            if i % 100 == 0:
                print(f"  {i} subjek", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log = pd.DataFrame(rows, columns=["SUB_ID", "folder", "file", "status"]).sort_values(["SUB_ID", "file"])
    log.to_csv(OUT / "fetch_log.csv", index=False)
    print(log.status.str.split(":").str[0].value_counts().to_string())
    return 0 if not log.status.str.startswith("gagal").any() else 1


if __name__ == "__main__":
    sys.exit(main())
