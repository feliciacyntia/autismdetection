"""Runner terminal preprocessing sMRI 05c.

Contoh (jalankan dari root proyek, sebaiknya di terminal di LUAR VS Code agar tahan VS Code ditutup):
    python src/run_05c.py --mode pilot --dry-run
    python src/run_05c.py --mode pilot --resume
    python src/run_05c.py --mode subset200 --resume        (butuh data/subset_20pct.csv dari 05a)

Berhenti rapi: Ctrl+C sekali, atau buat file STOP di WORK_DIR; subjek yang sedang berjalan diselesaikan dulu.
Lanjutkan dengan --resume. Resume ditolak jika folder output berisi hasil dengan hash config/tool lain.
Lock basi (proses sudah mati, mis. setelah crash/sleep) diganti otomatis. Auto-sleep Windows ditahan selama batch.
Kode keluar: 0 = semua sukses, 1 = ada subjek gagal, 2 = preflight/argumen gagal, 3 = dihentikan (STOP/RAM/disk).
"""
import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smri_preprocessing as sp  # noqa: E402


def setup_logging(cfg, dry_run):
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    log_path = None
    if not dry_run:
        cfg.paths["logs"].mkdir(parents=True, exist_ok=True)
        log_path = cfg.paths["logs"] / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return log_path


def scan_state(cfg, fp, subjects, failed):
    """valid = status success/flagged DAN output lolos validasi ulang. Status 'running' (crash) = belum selesai."""
    st = sp.read_status(cfg)
    done = set(st.loc[st["status"].isin(sp.DONE_STATUSES), "subject_id"].astype(int))
    rows = []
    for r in subjects.itertuples():
        sid = int(r.SUB_ID)
        wd = sp.subject_wd(cfg, sid)
        rows.append({"SUB_ID": sid, "valid": sid in done and sp.output_valid(cfg, sid, fp),
                     "ckpt": [s for s in ("reorient", "n4", "skullstrip", "register") if (wd / f"{s}.done.json").exists()],
                     "failed_attempts": failed.attempts(sid, fp) if failed is not None else 0})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["pilot", "subset200", "all"])
    ap.add_argument("--resume", action="store_true", help="lanjutkan: lewati output valid, pakai checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="hanya rencana + preflight; tidak menulis apa pun")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=None)
    ap.add_argument("--retry-failed", action="store_true", help="proses ulang subjek yang melewati batas retry")
    ap.add_argument("--limit", type=int, default=None, help="proses maksimal N subjek yang belum valid (uji)")
    ap.add_argument("--subjects", default=None, help="SUB_ID dipisah koma; harus bagian dari daftar --mode (uji pilot kecil)")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--input-root", default=None, help="root input (dataset/, data/*.csv); default = root proyek")
    ap.add_argument("--output-root", default=None, help="root output (data/smri_preproc ditulis di sini)")
    ap.add_argument("--hdbet-mode", choices=["runner", "cli"], default=None)
    ap.add_argument("--break-stale-lock", action="store_true",
                    help="(tidak diperlukan lagi) lock basi dari proses yang sudah mati diganti otomatis")
    ap.add_argument("--clean-work", action="store_true",
                    help="hapus folder kerja subjek di WORK_DIR setelah subjek sukses (default: disimpan)")
    args = ap.parse_args(argv)

    cfg = sp.make_config(args.input_root, args.output_root, start=Path(__file__).resolve().parent)
    if args.work_dir:
        cfg.work_dir = Path(args.work_dir)
    if args.chunk_size:
        cfg.chunk_size = args.chunk_size
    if args.max_retries is not None:
        cfg.max_retries = args.max_retries
    if args.hdbet_mode:
        cfg.hdbet_mode = args.hdbet_mode
    if args.clean_work:
        cfg.keep_work_files = False
    log_path = setup_logging(cfg, args.dry_run)
    log = logging.getLogger("run_05c")
    log.info("05c %s | mode=%s | resume=%s | dry_run=%s | WORK_DIR=%s | input=%s | out=%s", sp.PIPELINE_VERSION,
             args.mode, args.resume, args.dry_run, cfg.work_dir, cfg.input_root, cfg.out_dir)
    log.info("HD-BET python=%s | CLI=%s", cfg.paths["venv_py"], cfg.paths["hdbet_exe"])

    try:
        subjects = sp.load_subjects(cfg, args.mode)
    except (FileNotFoundError, AssertionError, RuntimeError) as e:
        log.error("Daftar subjek tidak valid: %s", e)
        return 2
    if args.subjects:
        want = [int(x) for x in args.subjects.split(",") if x.strip()]
        missing = sorted(set(want) - set(subjects["SUB_ID"].astype(int)))
        if missing:
            log.error("--subjects di luar daftar mode %s: %s", args.mode, missing)
            return 2
        subjects = subjects[subjects["SUB_ID"].isin(want)].reset_index(drop=True)
    space = sp.prepare_space(cfg, write=not args.dry_run)
    env = sp.env_versions(cfg, write_cache=not args.dry_run)
    fp, payload = sp.config_fingerprint(cfg, space, env)
    tool = {"tool": cfg.skullstrip_tool, "version": env["hdbet_venv"].get("hd-bet"),
            "nnunetv2": env["hdbet_venv"].get("nnunetv2"), "torch": env["hdbet_venv"].get("torch"),
            "weights_sha256": (env.get("hdbet_weights") or {}).get("sha256"), "device": cfg.hdbet_device}
    other = sp.existing_config_hashes(cfg) - {fp}
    if other:
        log.error("Folder output %s sudah berisi hasil dengan hash config/tool lain %s (sekarang %s). Resume DITOLAK "
                  "agar hasil tidak tercampur. Pakai --output-root baru, atau tinjau folder itu dulu.",
                  cfg.out_dir, sorted(other), fp)
        return 2
    failed = sp.FailedRegistry(cfg)
    state = scan_state(cfg, fp, subjects, failed)
    todo_ids = state.loc[~state["valid"], "SUB_ID"].tolist()
    if not args.retry_failed:
        todo_ids = [s for s in todo_ids if failed.attempts(s, fp) < cfg.max_retries]
    if args.limit:
        todo_ids = todo_ids[:args.limit]
    has_progress = bool(state["valid"].any() or state["ckpt"].map(len).gt(0).any())
    st0 = sp.read_status(cfg)
    crashed = st0.loc[st0["status"].eq("running"), "subject_id"].astype(int).tolist()
    if crashed:
        log.warning("Subjek berstatus 'running' dari run sebelumnya (crash/terhenti): %s -> diulang dari checkpoint "
                    "tahap terakhir yang valid.", crashed)
    ok, pre = sp.preflight(cfg, len(todo_ids))

    log.info("Subjek mode %s: %d (dev %d, test %d) | valid (fp %s): %d | ada checkpoint: %d | akan diproses: %d",
             args.mode, len(subjects), int((subjects["split"] == "dev").sum()), int((subjects["split"] == "test").sum()),
             fp, int(state["valid"].sum()), int(state["ckpt"].map(len).gt(0).sum()), len(todo_ids))
    log.info("Versi: antspyx %s | nibabel %s | hd-bet %s | nnunetv2 %s | torch %s | bobot HD-BET sha256 %s",
             env.get("antspyx"), env.get("nibabel"), env["hdbet_venv"].get("hd-bet"), env["hdbet_venv"].get("nnunetv2"),
             env["hdbet_venv"].get("torch"), (env.get("hdbet_weights") or {}).get("sha256", "?")[:12])
    log.info("Preflight: RAM %.2f GiB (min mulai %.1f) | VRAM bebas %s MiB | disk WORK %.1f/%.1f GB | disk data %.1f/%.1f GB",
             pre["ram_available_gb"], cfg.min_start_ram_gb, (pre["gpu"] or {}).get("free"),
             pre["disk_work_free_gb"], pre["disk_work_need_gb"], pre["disk_out_free_gb"], pre["disk_out_need_gb"])
    for prob in pre["problems"]:
        log.warning("PREFLIGHT: %s", prob)
    if args.dry_run:
        log.info("Dry-run: tidak ada yang ditulis. Daftar yang akan diproses: %s", todo_ids)
        return 0
    if not ok:
        log.error("Preflight gagal; tidak dimulai.")
        return 2
    if has_progress and not args.resume:
        log.error("Sudah ada output/checkpoint untuk subjek mode ini. Gunakan --resume (tidak ada yang dihapus/ditimpa).")
        return 2
    if cfg.paths["stop"].exists():
        log.error("File STOP ada di %s. Hapus file itu dulu untuk mulai.", cfg.paths["stop"])
        return 2
    if not todo_ids:
        log.info("Tidak ada subjek yang perlu diproses.")

    run_id = f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{args.mode}"
    summary = {"run_id": run_id, "mode": args.mode, "args": vars(args), "config_fp": fp, "log": str(log_path),
               "n_subjects": len(subjects), "n_todo": len(todo_ids), "preflight": pre}
    exit_code = 0
    lock = sp.RunLock(cfg, break_stale=args.break_stale_lock)
    try:
        lock.__enter__()
    except RuntimeError as e:
        log.error("%s", e)
        return 2
    lock.__exit__(None, None, None)        # cek saja; lock diambil ulang di bawah
    with sp.RunLock(cfg, break_stale=args.break_stale_lock), sp.KeepAwake():
        cfg.ensure_dirs()
        sp.prepare_space(cfg, write=True)
        sp.write_json_atomic(cfg.paths["reference_space"], sp.reference_space_info(cfg, space))
        sp.write_json_atomic(cfg.paths["config_json"], {"pipeline_version": sp.PIPELINE_VERSION, "config_fp": fp,
                                                        "fingerprint_payload": payload, "config": cfg.to_dict(),
                                                        "env": env, "seed": cfg.seed,
                                                        "updated": datetime.datetime.now().isoformat(timespec="seconds")})
        sp.write_json_atomic(cfg.paths["runs"] / f"{run_id}_subjects.json",
                             {"mode": args.mode, "SUB_ID": subjects["SUB_ID"].astype(int).tolist(), "todo": todo_ids})
        stop = sp.StopController(cfg).install()
        todo = subjects[subjects["SUB_ID"].isin(todo_ids)].set_index("SUB_ID").loc[todo_ids].reset_index()
        site = dict(zip(subjects["SUB_ID"].astype(int), subjects["SITE_ID"]))
        known = set(st0["subject_id"].astype(int))
        pend = [{"subject_id": s, "site": site[s], "status": "pending", "config_hash": fp, "tool": tool["tool"],
                 "tool_version": tool["version"], "run_id": run_id} for s in todo_ids if s not in known]
        if pend:
            sp.update_status(cfg, pend)
        done_valid = set(state.loc[state["valid"], "SUB_ID"].astype(int))
        t_run, done_durations, counts = time.time(), [], {"ok": 0, "failed": 0, "skipped": 0}
        vram_peak, ram_min = None, None
        try:
            for i in range(0, len(todo), cfg.chunk_size):
                if stop.should_stop():
                    break
                free_w, free_o = sp.disk_free_gb(cfg.work_dir), sp.disk_free_gb(cfg.out_dir)
                if min(free_w, free_o) < cfg.min_free_disk_gb:
                    log.error("Disk bebas WORK %.1f GB / output %.1f GB < %.1f GB: berhenti rapi.", free_w, free_o,
                              cfg.min_free_disk_gb)
                    stop.requested, stop.reason = True, "disk penuh"
                    break
                chunk = todo.iloc[i:i + cfg.chunk_size]
                sp.update_status(cfg, [{"subject_id": int(s), "site": site[int(s)], "status": "running", "error": None,
                                        "config_hash": fp, "tool": tool["tool"], "tool_version": tool["version"],
                                        "run_id": run_id, "updated": datetime.datetime.now().isoformat(timespec="seconds")}
                                       for s in chunk["SUB_ID"]])
                t0 = time.time()
                ram_before = sp.ram_available_gb()
                recs = sp.process_chunk(cfg, fp, space, chunk, failed, stop, retry_failed=args.retry_failed,
                                        skip_if=lambda s: s in done_valid and sp.output_valid(cfg, s, fp), tool=tool)
                for r in recs:
                    st = r.get("status", "?")
                    key = "ok" if st == "ok" else "failed" if st == "failed" else "skipped"
                    counts[key] += 1
                    if st == "ok":
                        done_durations.append(r.get("duration_s") or 0)
                        hd = r.get("hdbet") or {}
                        if hd.get("vram_peak_mib") is not None:
                            vram_peak = max(vram_peak or 0, hd["vram_peak_mib"])
                        if hd.get("ram_min_available_gb") is not None:
                            ram_min = min(ram_min or 1e9, hd["ram_min_available_gb"])
                    log.info("SUB_ID %s: %s%s%s", r["SUB_ID"], st,
                             f" ({r.get('duration_s', 0):.0f} s)" if st == "ok" else "",
                             f" | {r.get('failed_step')}: {str(r.get('error'))[:200]}" if st == "failed" else "")
                upd = [sp.status_row(r, site[int(r["SUB_ID"])], fp, tool, run_id) for r in recs
                       if r.get("status") in ("ok", "failed")]
                if upd:
                    sp.update_status(cfg, upd)
                n_done = i + len(chunk)
                per_subj = (time.time() - t_run) / max(n_done, 1)
                eta_min = per_subj * (len(todo) - n_done) / 60
                gpu = sp.gpu_query()
                log.info("Progres %d/%d | ok %d gagal %d | chunk %.0f s | rata-rata %.0f s/subjek | sisa ±%.0f menit | "
                         "RAM %.2f→%.2f GiB | VRAM dipakai %s MiB", n_done, len(todo), counts["ok"], counts["failed"],
                         time.time() - t0, per_subj, eta_min, ram_before, sp.ram_available_gb(),
                         gpu and round(gpu["used"]))
        except sp.MemoryGuardError as e:
            log.error("RAM guard: %s", e)
            stop.requested, stop.reason = True, f"RAM guard: {e}"
            exit_code = 3
        finally:
            stop.uninstall()
            all_subj = sp.load_subjects(cfg)
            rep = sp.build_qc_report(cfg, space, all_subj, fp)
            n_flagged = 0
            if len(rep):
                # status flagged/success mengikuti flag QC terbaru (penanda, bukan pembuang)
                ok_ids = rep.loc[rep["status"].isin(sp.DONE_STATUSES), ["SUB_ID", "n_flags"]]
                sp.update_status(cfg, [{"subject_id": int(a), "status": "flagged" if b > 0 else "success"}
                                       for a, b in ok_ids.itertuples(index=False)])
                rep["status"] = np.where(rep["SUB_ID"].isin(ok_ids["SUB_ID"]),
                                         np.where(rep["n_flags"] > 0, "flagged", "success"), rep["status"])
                n_flagged = int((rep["status"] == "flagged").sum())
                sp.write_csv_atomic(cfg.paths["qc_report"], rep)
                review = sp.build_qc_review(cfg, rep)
                sp.write_csv_atomic(cfg.paths["qc_review"], review)
                sp.write_qc_decisions_template(cfg, review)
                rng = np.random.default_rng(cfg.seed)
                rand_ids = sorted(rng.choice(rep["SUB_ID"], size=min(12, len(rep)), replace=False).tolist())
                flagged = rep.loc[rep["n_flags"] > 0, "SUB_ID"].tolist()
                sp.gallery(cfg, space, rand_ids, rep, cfg.paths["fig"] / "gallery_random.png", "12 subjek acak")
                for k in range(0, len(flagged), 24):
                    sp.gallery(cfg, space, flagged[k:k + 24], rep, cfg.paths["fig"] / f"gallery_flagged_{k // 24:02d}.png",
                               f"Subjek ber-flag QC ({k + 1}–{min(k + 24, len(flagged))} dari {len(flagged)})")
            summary.update(counts=counts, stopped=stop.requested, stop_reason=stop.reason,
                           wall_s=time.time() - t_run, mean_s_per_ok=float(np.mean(done_durations)) if done_durations else None,
                           vram_peak_mib=vram_peak, ram_min_available_gb=ram_min,
                           n_valid_after=int(sum(sp.output_valid(cfg, s, fp, deep=False) for s in subjects["SUB_ID"])),
                           qc_flagged=n_flagged, finished=datetime.datetime.now().isoformat(timespec="seconds"))
            sp.write_json_atomic(cfg.paths["runs"] / f"{run_id}.json", summary)
            log.info("RINGKASAN run ini: success %d | failed %d | skipped %d | flagged (semua output valid) %d | "
                     "total %.1f menit | valid %d/%d | %s", counts["ok"], counts["failed"], counts["skipped"], n_flagged,
                     summary["wall_s"] / 60, summary["n_valid_after"], len(subjects),
                     f"dihentikan ({stop.reason})" if stop.requested else "tuntas")
    if exit_code == 0:
        exit_code = 3 if summary.get("stopped") else (1 if counts["failed"] else 0)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
