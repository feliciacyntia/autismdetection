"""06 — Tampilkan progres runner (aman dijalankan kapan saja, hanya membaca).

    .venv-model\\Scripts\\python.exe src\\pilot_progress.py --subset pilot200
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["pilot200", "all"], default="pilot200")
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args()
    rdir = Path(a.results_dir) if a.results_dir else pc.results_dir(a.subset)
    st = pc.read_status(rdir)
    if not len(st):
        print("Belum ada status di", rdir); return
    lock = rdir / "run.lock"
    alive = lock.exists() and pc.sp._lock_owner_alive(json.loads(lock.read_text(encoding="utf-8") or "{}"))
    print(f"Folder hasil : {rdir}")
    print(f"Runner       : {'SEDANG BERJALAN' if alive else 'tidak berjalan'}" + (" | file STOP ada" if (rdir / 'STOP').exists() else ""))
    cnt = st["status"].value_counts().to_dict()
    print(f"Unit         : {cnt.get('done', 0)}/{len(st)} done | running {cnt.get('running', 0)} | pending {cnt.get('pending', 0)} | "
          f"failed {cnt.get('failed', 0)} | skipped {cnt.get('skipped', 0)}")
    p = rdir / "progress.json"
    if p.exists():
        g = json.loads(p.read_text(encoding="utf-8"))
        cur = g.get("unit")
        if cur and alive:
            ep = f" | epoch {g.get('epoch')}/{g.get('max_epochs')} ({g.get('epoch_s')} s/epoch, sisa unit ±{g.get('eta_unit_min')} menit)" \
                if g.get("epoch") else f" | {g.get('last_message', '')}"
            print(f"Sekarang     : unit {g.get('unit_index')}/{g.get('units_total')} {cur}{ep}")
        print(f"Update       : {g.get('updated')}")
    print("\nPer model (fold done / total):")
    st["m"] = st["unit"].str.split(":").str[0]
    for m, g in st.groupby("m", sort=False):
        flag = "" if (g["status"] == "done").all() else "  <-- " + ", ".join(f"{k}:{v}" for k, v in g["status"].value_counts().items() if k != "done")
        print(f"  {m:18s} {int((g['status'] == 'done').sum())}/{len(g)}{flag}")
    bad = st[st["status"].isin(["failed", "skipped"])]
    if len(bad):
        print("\nGagal/dilewati:")
        for r in bad.itertuples():
            print(f"  {r.unit}: {r.status} | {str(r.error)[:150]}")
    logs = sorted((rdir / "logs").glob("run_*.log"))
    if logs:
        print(f"\nLog terakhir ({logs[-1].name}):")
        for line in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-6:]:
            print("  " + line[:180])


if __name__ == "__main__":
    main()
