"""Uji reproducibility 05c: proses ULANG satu subjek dari awal (folder kerja + output terpisah) lalu bandingkan
dengan output utama. Output utama tidak disentuh.

    python src/repro_check_05c.py 50013
    python src/repro_check_05c.py 50013 --input-root <in> --output-root <out>

Hasil: <output>/data/smri_preproc/repro/repro_<sid>_<ts>.json (+ log). Lulus jika identik bitwise, atau jika di
dalam toleransi config (Dice mask >= repro_dice_min, max|Δ| citra <= repro_max_abs_diff); mana yang berlaku dilaporkan.
Kode keluar: 0 = lulus, 1 = gagal/di luar toleransi, 2 = prasyarat tidak terpenuhi.
"""
import argparse
import dataclasses
import datetime
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import smri_preprocessing as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub_id", type=int)
    ap.add_argument("--input-root", default=None)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()
    sid = args.sub_id

    cfg = sp.make_config(args.input_root, args.output_root, start=Path(__file__).resolve().parent)
    ts = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    rdir = cfg.out_dir / "repro"
    rdir.mkdir(parents=True, exist_ok=True)
    log_path = rdir / f"repro_{sid}_{ts}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger(); root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        h.setFormatter(fmt); root.addHandler(h)
    log = logging.getLogger("repro_05c")

    space = sp.prepare_space(cfg, write=False)
    env = sp.env_versions(cfg, write_cache=False)
    fp, _ = sp.config_fingerprint(cfg, space, env)
    st = sp.read_status(cfg)
    row = st[st["subject_id"].astype(int) == sid]
    if not len(row) or row["status"].iloc[0] not in sp.DONE_STATUSES or not sp.output_valid(cfg, sid, fp):
        log.error("SUB_ID %s belum punya output utama yang valid untuk config %s.", sid, fp)
        return 2
    ok, pre = sp.preflight(cfg, 1)
    if not ok:
        log.error("Preflight gagal: %s", pre["problems"])
        return 2

    # salinan config: output dan WORK_DIR terpisah, parameter metode sama (fp harus sama)
    cfg2 = dataclasses.replace(cfg, out_subdir=f"{cfg.out_subdir}/repro/{sid}_{ts}",
                               work_dir=cfg.work_dir / "repro" / f"{sid}_{ts}")
    cfg2.ensure_dirs()
    space2 = sp.prepare_space(cfg2, write=True)
    fp2, _ = sp.config_fingerprint(cfg2, space2, env)
    assert fp2 == fp, f"fp salinan {fp2} != {fp}"
    tool = {"tool": cfg.skullstrip_tool, "version": env["hdbet_venv"].get("hd-bet")}
    subj = sp.load_subjects(cfg)
    subj = subj[subj["SUB_ID"] == sid].reset_index(drop=True)

    t0 = time.time()
    with sp.RunLock(cfg):                                    # cegah bentrok dengan run utama
        recs = sp.process_chunk(cfg2, fp, space2, subj, sp.FailedRegistry(cfg2), sp.StopController(cfg2), tool=tool)
    if not recs or recs[0].get("status") != "ok":
        log.error("Proses ulang gagal: %s", recs and recs[0].get("error"))
        return 1

    a, b = sp.out_paths(cfg, sid), sp.out_paths(cfg2, sid)
    va, vb = np.load(a["vol"]), np.load(b["vol"])
    ma, mb = np.load(a["mask"]).astype(bool), np.load(b["mask"]).astype(bool)
    res = {"SUB_ID": sid, "config_fp": fp, "seconds": round(time.time() - t0, 1), "repro_dir": str(cfg2.out_dir),
           "bitwise": {k: sp.sha256_file(a[k]) == sp.sha256_file(b[k])
                       for k in ("vol", "mask", "xfm", "mask_native") + (("vol_nii", "mask_nii") if cfg.save_nifti else ())},
           "vol_max_abs_diff": float(np.abs(va - vb).max()),
           "mask_dice": float(2 * (ma & mb).sum() / (ma.sum() + mb.sum())),
           "tolerance": {"dice_min": cfg.repro_dice_min, "max_abs_diff": cfg.repro_max_abs_diff}}
    res["identical"] = all(res["bitwise"].values())
    res["within_tolerance"] = res["mask_dice"] >= cfg.repro_dice_min and res["vol_max_abs_diff"] <= cfg.repro_max_abs_diff
    res["criterion_used"] = "bitwise" if res["identical"] else "toleransi"
    res["passed"] = bool(res["identical"] or res["within_tolerance"])
    sp.write_json_atomic(rdir / f"repro_{sid}_{ts}.json", res)
    log.info("bitwise %s | max|Δ| %.3e | Dice mask %.6f | kriteria %s | %s", res["bitwise"], res["vol_max_abs_diff"],
             res["mask_dice"], res["criterion_used"], "LULUS" if res["passed"] else "GAGAL")
    log.info("Hasil: %s", rdir / f"repro_{sid}_{ts}.json")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
