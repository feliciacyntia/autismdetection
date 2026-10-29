"""Tahap 2: validasi HD-BET runner vs CLI pada satu subjek (default 50002). Tidak menjalankan pilot/subset/all.
Log -> <output>/data/smri_preproc/logs/validate_<sid>_<ts>.log ; status -> validation_status_<sid>_<ts>.json
(ditulis di awal, diperbarui di akhir/saat error). Hasil -> hdbet_runner_validation_<ts>.json.
Lokal : python src/validate_05c.py 50002
Kaggle: python src/validate_05c.py 50002 --input-root /kaggle/input/<data> --output-root /kaggle/working/ta"""
import argparse
import datetime
import json
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import smri_preprocessing as sp

ap = argparse.ArgumentParser()
ap.add_argument("sub_id", nargs="?", type=int, default=50002)
ap.add_argument("--input-root", default=None, help="root input (dataset/, data/*.csv); default = root proyek")
ap.add_argument("--output-root", default=None, help="root output (data/smri_preproc)")
args = ap.parse_args()
SID = args.sub_id

cfg = sp.make_config(args.input_root, args.output_root, start=Path(__file__).resolve().parent)
ts = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
cfg.paths["logs"].mkdir(parents=True, exist_ok=True)
log_path = cfg.paths["logs"] / f"validate_{SID}_{ts}.log"
status_path = cfg.out_dir / f"validation_status_{SID}_{ts}.json"   # file baru per run
fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
root = logging.getLogger(); root.setLevel(logging.INFO)
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
    h.setFormatter(fmt); root.addHandler(h)
log = logging.getLogger("validate_05c")

status = {"SUB_ID": SID, "started": datetime.datetime.now().isoformat(timespec="seconds"), "status": "running",
          "log": str(log_path), "work_dir": str(cfg.work_dir), "input_root": str(cfg.input_root),
          "output_root": str(cfg.project_root), "hdbet_python": str(cfg.paths["venv_py"]),
          "hdbet_cli": str(cfg.paths["hdbet_exe"])}


def save_status():
    sp.write_json_atomic(status_path, status)


def cuda_errors(path):
    if not Path(path).exists():
        return []
    keys = ("cuda error", "out of memory", "cublas_status", "cudnn_status", "device-side", "cudaerror")
    return [l.strip() for l in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
            if not l.startswith("HDBET_RUNNER_SUMMARY") and any(k in l.lower() for k in keys)][:20]


exit_code = 1
try:
    ok, pre = sp.preflight(cfg, 1)   # sebelum cek torch: konteks CUDA di proses induk memakan ~1 GiB RAM
    import subprocess
    probe = ("import json,torch;a=torch.cuda.is_available();print(json.dumps({'version':torch.__version__,"
             "'cuda_available':a,'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0) if a else None}))")
    for key, py in (("torch_main", sys.executable), ("torch_venv", str(cfg.paths["venv_py"]))):
        r = subprocess.run([py, "-c", probe], capture_output=True, text=True)
        status[key] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-500:]}
        log.info("%s: %s", key, status[key])
    status["gpu_start"] = sp.gpu_query()
    log.info("nvidia-smi awal: %s", status["gpu_start"])

    status["preflight"] = pre
    log.info("Preflight: RAM %.2f GiB (min %.1f) | VRAM bebas %s MiB | masalah: %s",
             pre["ram_available_gb"], cfg.min_start_ram_gb, (pre["gpu"] or {}).get("free"), pre["problems"] or "-")
    if not ok:
        status["status"] = "not_started_preflight"
        save_status()
        log.error("Preflight gagal; validasi tidak dijalankan.")
        sys.exit(2)
    if cfg.hdbet_device == "cuda" and not status["torch_venv"].get("cuda_available"):
        status["status"] = "not_started_no_cuda"; save_status()
        log.error("CUDA tidak tersedia; tidak dipaksakan, validasi tidak dijalankan.")
        sys.exit(2)

    status["devices"] = {"reorient": "CPU (nibabel)", "n4": "CPU (ANTs/ITK; tidak ada implementasi GPU)",
                         "hdbet_runner": f"{cfg.hdbet_device} (torch, venv)", "hdbet_cli": f"{cfg.hdbet_device} (torch, venv)",
                         "registration": "CPU (ANTs; tidak ada implementasi GPU)",
                         "resample_intensity_crop": "CPU (ANTs/numpy)"}
    save_status()

    with sp.RunLock(cfg):
        space = sp.prepare_space(cfg, write=True)
        env = sp.env_versions(cfg, write_cache=True)
        fp, _ = sp.config_fingerprint(cfg, space, env)
        status["config_fp"] = fp
        status["versions"] = {k: env.get(k) for k in ("antspyx", "nibabel", "numpy", "nilearn")}
        status["versions"]["hdbet_venv"] = env.get("hdbet_venv")
        status["hdbet_weights_sha256"] = (env.get("hdbet_weights") or {}).get("sha256")
        log.info("fp %s | hd-bet %s | torch(venv) %s | tta=%s | device=%s | np %d/%d", fp,
                 env["hdbet_venv"].get("hd-bet"), env["hdbet_venv"].get("torch"), cfg.hdbet_tta, cfg.hdbet_device,
                 cfg.hdbet_np_preprocessing, cfg.hdbet_np_export)
        save_status()

        t0 = time.time()
        runner_mask = cfg.paths["validation"] / str(SID) / "runner" / f"{SID}_bet.nii.gz"
        res = sp.validate_runner_vs_cli(cfg, fp, space, SID)
        assert runner_mask.stat().st_mtime > t0, "mask runner bukan hasil run ini"
        status["seconds_total"] = round(time.time() - t0, 1)

        vdir = cfg.paths["validation"] / str(SID)
        # tambahan: kecocokan transform afin cli vs runner
        import ants
        A = [np.asarray(ants.read_transform(str(vdir / f"out_{n}" / "xfm.mat")).parameters) for n in ("cli", "runner")]
        res["xfm_params_max_abs_diff"] = float(np.abs(A[0] - A[1]).max())
        res["final_affine"] = space.affine_final.tolist()
        res["cuda_errors_runner"] = cuda_errors(vdir / "runner" / "hdbet_stdout.log")
        res["cuda_errors_cli"] = cuda_errors(vdir / "cli" / "hdbet_stdout.log")
        checks = {"dice>=0.999": res["mask_bitwise_identical"] or res["dice"] >= cfg.validation_dice_min,
                  "mask_shape_equal": res["shape_equal"], "mask_affine_equal(<1e-4)": res["affine_max_diff"] < 1e-4,
                  "vol_shape_equal": res["vol_shape_equal"],
                  f"max_abs_diff<={cfg.validation_max_abs_diff}": res["vol_max_abs_diff"] <= cfg.validation_max_abs_diff,
                  f"corr>={cfg.validation_min_corr}": res["vol_corr"] >= cfg.validation_min_corr}
        status.update(metrics=res, checks=checks, passed=bool(res["passed"] and all(checks.values())),
                      gpu_end=sp.gpu_query())
        status["status"] = "passed" if status["passed"] else "failed_threshold"
        save_status()
        sp.write_json_atomic(cfg.out_dir / "hdbet_runner_validation.json", res)
        sp.write_json_atomic(cfg.out_dir / f"hdbet_runner_validation_{ts}.json", res)
        for k, v in checks.items():
            log.info("CEK %-26s %s", k, "LULUS" if v else "GAGAL")
        log.info("Dice %.6f | identik bitwise %s | voxel beda %s | affine mask max diff %.2e | max|Δ| vol %.3e | "
                 "korelasi %.8f | xfm max diff %.3e | sumber CLI: %s", res["dice"], res["mask_bitwise_identical"],
                 res["n_voxel_diff"], res["affine_max_diff"], res["vol_max_abs_diff"], res["vol_corr"],
                 res["xfm_params_max_abs_diff"], res["cli_source"])
        log.info("Runner: %s", res.get("runner"))
        log.info("Error CUDA runner: %s | CLI: %s", res["cuda_errors_runner"] or "-", res["cuda_errors_cli"] or "-")
        log.info("STATUS: %s", status["status"].upper())
        exit_code = 0 if status["passed"] else 1
except SystemExit:
    raise
except BaseException as e:
    status["status"] = "error"
    status["error"] = f"{type(e).__name__}: {e}"
    status["traceback"] = traceback.format_exc()[-4000:]
    vdir = cfg.paths["validation"] / str(SID)
    status["cuda_errors"] = cuda_errors(vdir / "runner" / "hdbet_stdout.log") + cuda_errors(vdir / "cli" / "hdbet_stdout.log")
    save_status()
    log.error("Validasi berhenti karena error: %s", status["error"])
    log.error(status["traceback"])
    exit_code = 3 if isinstance(e, sp.MemoryGuardError) else 1
finally:
    status["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_status()
    log.info("Status: %s | Log: %s", status_path, log_path)
sys.exit(exit_code)
