"""Preprocessing sMRI 05c: satu volume kanonik per subjek (MNI152 2 mm, (1, 96, 112, 96), float32).

Dipakai bersama oleh notebook/05c_preprocessing_smri.ipynb dan src/run_05c.py.

Urutan per subjek (metode tidak berubah dari versi pilot pertama):
    reorient (RAS, clip intensitas negatif) -> N4 (antspyx, tanpa mask) -> input HD-BET (LAS)
    -> skull stripping HD-BET (GPU, venv terpisah) -> mask kembali ke RAS (tanpa interpolasi)
    -> registrasi afin 12 DOF otak subjek -> otak template MNI152 1 mm
    -> resample ke grid template 2 mm -> clip persentil + z-score di dalam mask (latar 0)
    -> crop box tetap + pad -> vol/{SUB_ID}.npy dan mask/{SUB_ID}.npy

Checkpoint: setiap langkah berat menulis file + penanda `<langkah>.done.json` di WORK_DIR/subjects/<SUB_ID>/,
berisi sidik jari konfigurasi dan tanda tangan file input. Output akhir hanya dilewati jika valid DAN sidik
jari konfigurasinya sama dengan run sekarang. Semua tulisan atomik (file sementara lalu os.replace).
Output lama di data/vol, data/mask, data/vol_meta, data/work tidak pernah ditulis oleh modul ini.
"""
import datetime
import hashlib
import importlib.metadata as md
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(os.cpu_count()))

import numpy as np
import pandas as pd

PIPELINE_VERSION = "05c-v2.1"
LOG = logging.getLogger("smri05c")


# ======================================================================================
# Konfigurasi
# ======================================================================================
@dataclass
class Config:
    project_root: Path                                    # root OUTPUT: data/smri_preproc ditulis di sini
    input_root: Path = None                               # root INPUT (dataset/, data/*.csv, citra); None -> env
                                                          # TA05C_INPUT_ROOT atau project_root (lokal: sama)
    hdbet_python: str = None                              # None -> env TA05C_HDBET_PYTHON / .venv-hdbet / python ini
    hdbet_cli: str = None                                 # None -> env TA05C_HDBET_CLI / .venv-hdbet / hd-bet di PATH
    work_dir: Path = Path(os.environ.get("TA05C_WORK_DIR", r"C:\ta_work\05c"))
    out_subdir: str = "data/smri_preproc"
    seed: int = 42
    # metode (bagian dari sidik jari)
    target_axcodes: tuple = ("R", "A", "S")
    hdbet_input_axcodes: tuple = ("L", "A", "S")           # padanan fslreorient2std (README HD-BET)
    hdbet_device: str = "cuda"
    hdbet_tta: bool = True
    hdbet_deterministic: bool = True                      # cuDNN deterministik (repro 50214: benchmark -> beda 3 voxel)
    clip_negative_before_n4: bool = True
    n4_params: dict = field(default_factory=lambda: {"shrink_factor": 4,
                                                     "convergence": {"iters": [50, 50, 50, 50], "tol": 1e-07}})
    reg_transform: str = "Affine"                         # 12 DOF
    ants_random_seed: int = 42
    reg_threads: int = 1                                  # ITK thread registrasi; >1 terbukti non-deterministik (diag 50002)
    template_res_reg_mm: int = 1
    target_res_mm: int = 2
    mask_interp: str = "nearestNeighbor"                  # transform mask final (ANTs interpolator)
    mask_threshold: float = 0.5
    clip_percentiles: tuple = (0.5, 99.5)
    target_shape: tuple = (96, 112, 96)
    crop_margin_vox: int = 2
    # eksekusi (bukan bagian dari sidik jari; ekuivalensi runner vs CLI divalidasi di Tahap 2)
    hdbet_mode: str = "runner"                            # "runner" | "cli"
    hdbet_np_preprocessing: int = 1
    hdbet_np_export: int = 1
    hdbet_sequential: bool = True                         # runner: predict_from_files_sequential (tanpa worker)
    chunk_size: int = 1                                   # 1 = STOP berlaku setelah subjek yang berjalan selesai
    save_nifti: bool = True                               # citra/mask final .nii.gz 3D (96,112,96) + affine 2 mm
    max_retries: int = 2
    keep_work_files: bool = True
    save_qc_figure: bool = True
    # resource guard
    min_free_ram_gb: float = 1.5                          # dicek sebelum setiap subjek dan selama HD-BET
    min_start_ram_gb: float = 4.0                         # tidak mulai jika kurang
    min_free_vram_mib: float = 3300.0                     # puncak HD-BET terukur 3.030 MiB
    work_mb_per_subject: float = 120.0                    # estimasi disk file antara (konservatif)
    out_mb_per_subject: float = 6.0
    min_free_disk_gb: float = 2.0                         # dicek sebelum tiap subjek (WORK_DIR dan output)
    # QC (hanya penanda, tidak membuang subjek)
    mask_ratio_range: tuple = (0.8, 1.25)
    brain_ml_range: tuple = (800.0, 2100.0)
    reg_det_range: tuple = (0.5, 2.0)
    min_ncc_template: float = 0.3                         # gagal jelas saja; 3 pilot lolos visual di 0,47–0,55.
                                                          # NCC rendah relatif ditangkap z per site; dikalibrasi di pilot 20
    outlier_z: float = 3.0
    min_n_global_z: int = 20
    min_n_site_z: int = 15
    # validasi runner vs CLI (Tahap 2)
    validation_dice_min: float = 0.999
    validation_max_abs_diff: float = 1e-3
    validation_min_corr: float = 0.9999
    # uji reproducibility (proses ulang 1 subjek)
    repro_dice_min: float = 0.999
    repro_max_abs_diff: float = 1e-4
    skullstrip_tool: str = "HD-BET"                       # satu tool untuk SEMUA subjek

    def __post_init__(self):
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.input_root = Path(self.input_root or os.environ.get("TA05C_INPUT_ROOT") or self.project_root).expanduser().resolve()
        self.work_dir = Path(self.work_dir)

    # ---------- path turunan ----------
    @property
    def data_dir(self):
        """data/ INPUT (folds, pilot, exclusions, audit_flags, cache lama)."""
        return self.input_root / "data"

    @property
    def out_dir(self):
        return self.project_root / self.out_subdir

    @property
    def paths(self):
        d, o, w = self.data_dir, self.out_dir, self.work_dir
        venv = Path(__file__).resolve().parent.parent / ".venv-hdbet"     # di samping src/ (bukan root output)
        venv_py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        venv_cli = venv / ("Scripts/hd-bet.exe" if os.name == "nt" else "bin/hd-bet")
        hd_py = self.hdbet_python or os.environ.get("TA05C_HDBET_PYTHON") or (venv_py if venv_py.exists() else sys.executable)
        hd_cli = (self.hdbet_cli or os.environ.get("TA05C_HDBET_CLI")
                  or (venv_cli if venv_cli.exists() else shutil.which("hd-bet") or "hd-bet"))
        return {
            "master": self.input_root / "dataset" / "cleaned_dataset" / "cleaned_master_dataset.csv",
            "smri_qc": self.input_root / "dataset" / "cleaned_dataset" / "smri_qc.csv",
            "folds": d / "folds.csv", "pilot": d / "pilot_ids.csv", "subset200": d / "subset_20pct.csv",
            "exclusions": d / "exclusions.csv", "audit_flags": d / "audit_flags.csv",
            "legacy_failed": d / "failed_subjects.csv", "legacy_work": d / "work",
            "vol": o / "vol", "mask": o / "mask", "xfm": o / "xfm", "meta": o / "meta", "fig": o / "qc_figures",
            "templates": o / "templates", "logs": o / "logs", "runs": o / "runs",
            "nii": o / "nii", "mask_native": o / "mask_native", "subj_logs": o / "logs" / "subjects",
            "failed": o / "failed_subjects.csv", "status": o / "status.csv", "qc_report": o / "qc_report.csv",
            "qc_review": o / "qc_review.csv", "qc_decisions": o / "qc_decisions.csv",
            "qc_decisions_template": o / "qc_decisions_template.csv",
            "header_check": o / "header_check.csv", "config_json": o / "preprocess_config.json",
            "reference_space": o / "reference_space.json",
            "subjects_work": w / "subjects", "hdbet_work": w / "hdbet_chunks", "validation": w / "validation",
            "cache": w / "cache", "lock": w / "run.lock", "stop": w / "STOP",
            "venv_py": Path(hd_py), "hdbet_exe": Path(hd_cli),
            "hdbet_runner": Path(__file__).resolve().parent / "hdbet_runner.py",
        }

    def method_params(self):
        keys = ["target_axcodes", "hdbet_input_axcodes", "hdbet_device", "hdbet_tta", "hdbet_deterministic", "clip_negative_before_n4",
                "n4_params", "reg_transform", "ants_random_seed", "reg_threads", "template_res_reg_mm", "target_res_mm",
                "mask_interp", "mask_threshold", "clip_percentiles", "target_shape", "crop_margin_vox"]
        return {k: getattr(self, k) for k in keys}

    def to_dict(self):
        d = asdict(self)
        d["project_root"], d["input_root"], d["work_dir"] = str(self.project_root), str(self.input_root), str(self.work_dir)
        return d

    def ensure_dirs(self):
        p = self.paths
        for k in ("vol", "mask", "xfm", "meta", "fig", "templates", "logs", "runs", "nii", "mask_native", "subj_logs"):
            p[k].mkdir(parents=True, exist_ok=True)
        for k in ("subjects_work", "hdbet_work", "validation", "cache"):
            p[k].mkdir(parents=True, exist_ok=True)


def find_project_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "dataset").exists() and (cand / "data").exists():
            return cand
    raise FileNotFoundError(f"Root proyek (berisi dataset/ dan data/) tidak ditemukan dari {p}")


def make_config(input_root=None, output_root=None, start=None):
    """Lokal (Windows): input = output = root proyek yang dicari dari `start` (perilaku lama).
    Kaggle: input_root = dataset read-only (/kaggle/input/...), output_root = /kaggle/working/...
    (argumen atau env TA05C_INPUT_ROOT / TA05C_OUTPUT_ROOT)."""
    inp = input_root or os.environ.get("TA05C_INPUT_ROOT")
    out = output_root or os.environ.get("TA05C_OUTPUT_ROOT")
    if out is None:
        if inp is not None:
            raise ValueError("input_root diberikan tanpa output_root: set --output-root / TA05C_OUTPUT_ROOT "
                             "(root input bisa read-only, mis. /kaggle/input).")
        out = find_project_root(start)
    elif inp is None:
        inp = find_project_root(start)          # output dipindah (mis. di luar OneDrive), input tetap proyek
    return Config(project_root=Path(out), input_root=inp)


# ======================================================================================
# Utilitas file: hash dan tulisan atomik
# ======================================================================================
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_sig(path):
    st = Path(path).stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _replace_retry(tmp, dst, tries=8):
    """os.replace dengan retry: OneDrive/antivirus kadang mengunci file sesaat (PermissionError)."""
    for i in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.25 * (2 ** i))


def _tmp_path(dst, ext=""):
    """File sementara di folder yang sama (agar os.replace atomik). ext dipakai untuk format (.nii.gz/.png)."""
    dst = Path(dst)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp" + ext, dir=dst.parent)
    os.close(fd)
    return Path(tmp)


def save_npy_atomic(dst, arr):
    tmp = _tmp_path(dst)
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)
        _replace_retry(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_atomic(dst, text):
    tmp = _tmp_path(dst)
    try:
        tmp.write_text(text, encoding="utf-8")
        _replace_retry(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_atomic(dst, obj):
    write_text_atomic(dst, json.dumps(obj, indent=1, default=str))


def write_csv_atomic(dst, df):
    write_text_atomic(dst, df.to_csv(index=False))


def copy_atomic(src, dst):
    tmp = _tmp_path(dst)
    try:
        shutil.copyfile(src, tmp)
        _replace_retry(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_nifti_atomic(dst, img):
    import nibabel as nib
    tmp = _tmp_path(dst, ".nii.gz")
    try:
        nib.save(img, str(tmp))
        _replace_retry(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


# ======================================================================================
# Resource
# ======================================================================================
def ram_available_gb():
    import psutil
    return psutil.virtual_memory().available / 2**30


def disk_free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


def gpu_query():
    """{'total','used','free'} MiB dari nvidia-smi; None jika tidak tersedia."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,name",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        t, u, f, name = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"total": float(t), "used": float(u), "free": float(f), "name": name}
    except Exception:
        return None


class MemoryGuardError(RuntimeError):
    """RAM tersedia di bawah ambang: proses dihentikan rapi."""


def check_ram(cfg, where):
    avail = ram_available_gb()
    if avail < cfg.min_free_ram_gb:
        raise MemoryGuardError(f"RAM tersedia {avail:.2f} GiB < {cfg.min_free_ram_gb} GiB ({where}). "
                               "Bebaskan RAM lalu jalankan ulang dengan --resume.")
    return avail


def preflight(cfg, n_subjects, require_gpu=True):
    """Cek sebelum mulai: RAM >= min_start_ram_gb, VRAM bebas, disk WORK_DIR dan data/. Return (ok, report)."""
    rep = {"ram_available_gb": round(ram_available_gb(), 2), "min_start_ram_gb": cfg.min_start_ram_gb}
    gpu = gpu_query()
    rep["gpu"] = gpu
    need_work = n_subjects * cfg.work_mb_per_subject / 1024 + 2
    need_out = n_subjects * cfg.out_mb_per_subject / 1024 + 1
    rep["disk_work_free_gb"] = round(disk_free_gb(cfg.work_dir), 1)
    rep["disk_out_free_gb"] = round(disk_free_gb(cfg.out_dir), 1)
    rep["disk_work_need_gb"], rep["disk_out_need_gb"] = round(need_work, 1), round(need_out, 1)
    problems = []
    if rep["ram_available_gb"] < cfg.min_start_ram_gb:
        problems.append(f"RAM tersedia {rep['ram_available_gb']} GiB < {cfg.min_start_ram_gb} GiB "
                        "(tutup aplikasi lain, jeda OneDrive)")
    if require_gpu:
        if gpu is None:
            problems.append("nvidia-smi tidak tersedia / GPU tidak terdeteksi")
        elif gpu["free"] < cfg.min_free_vram_mib:
            problems.append(f"VRAM bebas {gpu['free']:.0f} MiB < {cfg.min_free_vram_mib:.0f} MiB")
    if rep["disk_work_free_gb"] < need_work:
        problems.append(f"Disk WORK_DIR bebas {rep['disk_work_free_gb']} GB < {need_work:.1f} GB")
    if rep["disk_out_free_gb"] < need_out:
        problems.append(f"Disk data/ bebas {rep['disk_out_free_gb']} GB < {need_out:.1f} GB")
    rep["problems"] = problems
    return not problems, rep


# ======================================================================================
# Lock file dan penghentian rapi (Ctrl+C / stop-file)
# ======================================================================================
def _lock_owner_alive(info):
    """True jika proses pemilik lock masih hidup. PID yang dipakai ulang proses lain (create_time beda) = mati."""
    import psutil
    pid = info.get("pid")
    if pid is None or not psutil.pid_exists(pid):
        return False
    ct = info.get("create_time")
    if ct is None:                      # lock format lama: anggap hidup bila PID ada (aman)
        return True
    try:
        return abs(psutil.Process(pid).create_time() - float(ct)) < 1.0
    except psutil.Error:
        return False


class RunLock:
    """Lock eksklusif di WORK_DIR/run.lock. Lock basi (proses pemilik sudah mati, mis. setelah crash/sleep)
    diganti otomatis; lock milik proses yang masih hidup tidak pernah dibongkar. break_stale dipertahankan untuk
    kompatibilitas (tidak lagi diperlukan)."""

    def __init__(self, cfg, break_stale=False):
        self.path, self.break_stale = cfg.paths["lock"], break_stale

    def __enter__(self):
        import psutil
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                info = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            except (json.JSONDecodeError, OSError):
                info = {}
            if _lock_owner_alive(info):
                raise RuntimeError(f"Lock ada: {self.path} ({info}). Run lain masih berjalan.")
            LOG.warning("Lock basi (PID %s sudah mati) diganti otomatis.", info.get("pid"))
            self.path.unlink()
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                       "started": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node()}, f)
        return self

    def __exit__(self, *exc):
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            if info.get("pid") == os.getpid():
                self.path.unlink()
        except FileNotFoundError:
            pass


class KeepAwake:
    """Mencegah Windows auto-sleep (idle) selama batch via SetThreadExecutionState; dikembalikan normal saat keluar.
    Tidak mengubah powercfg. Bila proses crash, Windows otomatis menghapus permintaan ini. Sleep manual
    (tutup layar / menu Sleep) tetap terjadi: gunakan pause_05c.ps1 dulu. Non-Windows: tidak melakukan apa-apa."""
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def __init__(self):
        self.active = False

    def __enter__(self):
        if os.name == "nt":
            import ctypes
            self.active = bool(ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED))
            LOG.info("Auto-sleep Windows ditahan selama batch (aktif=%s).", self.active)
        return self

    def __exit__(self, *exc):
        if self.active:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            LOG.info("Auto-sleep Windows dikembalikan normal.")
            self.active = False


class StopController:
    """Ctrl+C pertama atau file STOP -> berhenti setelah langkah/subjek yang sedang berjalan selesai.
    Ctrl+C kedua -> KeyboardInterrupt (paksa)."""

    def __init__(self, cfg):
        self.stop_file, self.requested, self.reason = cfg.paths["stop"], False, None
        self._prev = None

    def install(self):
        self._prev = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            if self.requested:
                raise KeyboardInterrupt
            self.requested, self.reason = True, "Ctrl+C"
            LOG.warning("Ctrl+C diterima: berhenti setelah langkah yang sedang berjalan selesai (Ctrl+C lagi = paksa).")
        signal.signal(signal.SIGINT, handler)
        return self

    def uninstall(self):
        if self._prev is not None:
            signal.signal(signal.SIGINT, self._prev)

    def should_stop(self):
        if not self.requested and self.stop_file.exists():
            self.requested, self.reason = True, f"stop-file {self.stop_file}"
            LOG.warning("Stop-file ditemukan: berhenti setelah langkah yang sedang berjalan selesai.")
        return self.requested


# ======================================================================================
# Daftar subjek (05c hanya membaca; tidak membuat split)
# ======================================================================================
def load_subjects(cfg, mode=None):
    """Gabungan folds.csv (+ smri_qc, master, audit_flags) untuk semua subjek non-eksklusi; filter per mode."""
    p = cfg.paths
    for k in ("master", "smri_qc", "folds", "exclusions", "audit_flags"):
        if not p[k].exists():
            raise FileNotFoundError(f"{p[k]} tidak ada. Jalankan 05a (split v2) terlebih dahulu.")
    folds = pd.read_csv(p["folds"])
    if "split" not in folds.columns or set(folds["split"]) != {"dev", "test"}:
        raise RuntimeError("folds.csv bukan skema v2 (kolom split dev/test). Jalankan 05a terbaru.")
    excl = set(pd.read_csv(p["exclusions"])["SUB_ID"].astype(int))
    assert not excl & set(folds["SUB_ID"]), "Subjek eksklusi ada di folds.csv"
    master = pd.read_csv(p["master"])[["SUB_ID", "smri_path"]]
    qc = pd.read_csv(p["smri_qc"])
    qc = qc[qc["included_in_final"].astype(bool)][["SUB_ID", "qc_decision", "qc_reason"]]
    af = pd.read_csv(p["audit_flags"], keep_default_na=False, na_values=[""])[["SUB_ID", "smri_voxel_sha256"]]
    subj = (folds[["SUB_ID", "SITE_ID", "group_id", "split", "fold_strat"]]      # tanpa label: QC blind
            .merge(master, on="SUB_ID", how="left", validate="one_to_one")
            .merge(qc, on="SUB_ID", how="left", validate="one_to_one")
            .merge(af, on="SUB_ID", how="left", validate="one_to_one")
            .sort_values("SUB_ID").reset_index(drop=True))
    assert subj["smri_path"].notna().all() and subj["qc_decision"].notna().all(), "SUB_ID folds tidak cocok dengan master/QC"
    subj["qc_reason"] = subj["qc_reason"].fillna("")
    subj["smri_abspath"] = subj["smri_path"].map(lambda s: cfg.input_root / Path(str(s)))
    if mode in (None, "all"):
        return subj
    list_path = p["pilot"] if mode == "pilot" else p["subset200"] if mode == "subset200" else None
    if list_path is None:
        raise ValueError(f"mode tidak dikenal: {mode}")
    if not list_path.exists():
        raise FileNotFoundError(f"Daftar subjek {list_path.name} tidak ada. 05c hanya membaca daftar; "
                                "buat daftarnya di 05a (perlu persetujuan).")
    ids = pd.read_csv(list_path)["SUB_ID"].astype(int)
    assert ids.is_unique, f"{list_path.name}: SUB_ID ganda"
    unknown = set(ids) - set(subj["SUB_ID"])
    assert not unknown, f"{list_path.name}: SUB_ID tidak ada di folds.csv: {sorted(unknown)[:10]}"
    assert not set(ids) & excl, f"{list_path.name} memuat subjek eksklusi"
    sel = subj[subj["SUB_ID"].isin(ids)].reset_index(drop=True)
    if mode in ("pilot", "subset200"):
        bad = sel.loc[sel["split"] != "dev", "SUB_ID"].tolist()
        assert not bad, f"{list_path.name} memuat subjek non-dev: {bad}"
    return sel


# ======================================================================================
# Template, ruang referensi, crop/pad
# ======================================================================================
@dataclass
class Space:
    files: dict
    crop: tuple
    pad: tuple
    target_shape: tuple
    affine_2mm: np.ndarray
    affine_final: np.ndarray
    template_crop: np.ndarray
    template_mask_crop: np.ndarray

    def crop_pad(self, arr):
        out = np.pad(arr[self.crop], self.pad, mode="constant", constant_values=0)
        assert out.shape == self.target_shape, out.shape
        return out


def prepare_space(cfg, write=True):
    """Template nilearn -> out_dir/templates (ditulis hanya jika belum ada dan write=True; jika sudah ada harus
    identik, tidak ditimpa). Menurunkan crop/pad tetap dan affine grid akhir."""
    import nibabel as nib
    from nilearn import datasets as nl
    tdir = cfg.paths["templates"]
    if write:
        tdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for res in sorted({cfg.template_res_reg_mm, cfg.target_res_mm}):
        t_img, m_img = nl.load_mni152_template(resolution=res), nl.load_mni152_brain_mask(resolution=res)
        assert t_img.shape == m_img.shape and np.allclose(t_img.affine, m_img.affine)
        assert nib.aff2axcodes(t_img.affine) == tuple(cfg.target_axcodes)
        t = t_img.get_fdata(dtype=np.float32)
        m = (m_img.get_fdata() > 0).astype(np.uint8)
        arrays = {"t1": t, "mask": m, "brain": t * m}
        files[res] = {"affine": t_img.affine, "shape": t_img.shape, "data": t, "mask": m}
        for name, arr in arrays.items():
            dst = tdir / f"mni152_{name}_{res}mm.nii.gz"
            if dst.exists():
                old = nib.load(str(dst))
                if not (np.array_equal(np.asanyarray(old.dataobj), arr) and np.allclose(old.affine, t_img.affine)):
                    raise RuntimeError(f"Template {dst} sudah ada tetapi isinya berbeda; tidak ditimpa.")
            elif write:
                save_nifti_atomic(dst, nib.Nifti1Image(arr, t_img.affine))
            files[res][f"{name}_path"] = dst
        h = hashlib.sha256(np.ascontiguousarray(t).tobytes() + np.ascontiguousarray(m).tobytes()
                           + np.asarray(t_img.affine, dtype=np.float64).tobytes())
        files[res]["array_sha256"] = h.hexdigest()
    t2 = files[cfg.target_res_mm]
    idx = np.argwhere(t2["mask"] > 0)
    lo, hi = idx.min(0), idx.max(0)
    crop, pad = [], []
    for ax in range(3):
        dim, tgt = t2["shape"][ax], cfg.target_shape[ax]
        b0, b1 = max(lo[ax] - cfg.crop_margin_vox, 0), min(hi[ax] + cfg.crop_margin_vox, dim - 1)
        assert b1 - b0 + 1 <= tgt, f"sumbu {ax}: bbox+margin > target"
        if dim >= tgt:
            start = int(np.clip((b0 + b1 + 1) // 2 - tgt // 2, 0, dim - tgt))
            crop.append(slice(start, start + tgt)); pad.append((0, 0))
        else:
            crop.append(slice(0, dim)); extra = tgt - dim
            pad.append((extra // 2, extra - extra // 2))
        assert crop[ax].start <= b0 and b1 < crop[ax].stop, f"sumbu {ax}: crop memotong mask template"
    shift = np.array([crop[a].start - pad[a][0] for a in range(3)], dtype=float)
    aff_final = t2["affine"].copy()
    aff_final[:3, 3] = t2["affine"][:3, :3] @ shift + t2["affine"][:3, 3]
    sp = Space(files=files, crop=tuple(crop), pad=tuple(pad), target_shape=tuple(cfg.target_shape),
               affine_2mm=t2["affine"], affine_final=aff_final,
               template_crop=None, template_mask_crop=None)
    sp.template_crop = sp.crop_pad(t2["data"] * t2["mask"])
    sp.template_mask_crop = sp.crop_pad(t2["mask"]).astype(bool)
    assert sp.template_mask_crop.sum() == t2["mask"].sum(), "crop memotong mask template"
    return sp


def reference_space_info(cfg, space):
    import nibabel as nib
    return {"shape": [1, *space.target_shape], "dtype": "float32", "axcodes": "".join(nib.aff2axcodes(space.affine_final)),
            "spacing_mm": [float(np.linalg.norm(space.affine_final[:3, i])) for i in range(3)],
            "affine": space.affine_final.round(6).tolist(),
            "template": "nilearn load_mni152_template (MNI152 2009)", "template_grid_2mm_affine": space.affine_2mm.tolist(),
            "crop": [[s.start, s.stop] for s in space.crop], "pad": [list(p) for p in space.pad],
            "note": "Array .npy berindeks (x, y, z) pada grid ini; channel pertama = 1. Irisan 2.5D diambil dari volume ini."}


# ======================================================================================
# Versi tool dan sidik jari konfigurasi
# ======================================================================================
def _pkg(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def env_versions(cfg, with_weights_hash=True, write_cache=True):
    p = cfg.paths
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for n in ["antspyx", "nibabel", "numpy", "nilearn", "scipy", "pandas", "psutil", "matplotlib"]:
        env[n] = _pkg(n)
    probe = ("import json, importlib.metadata as m, torch\n"
             "def v(n):\n    try: return m.version(n)\n    except Exception: return None\n"
             "d = {k: v(k) for k in ['hd-bet', 'nnunetv2', 'SimpleITK', 'numpy']}\n"
             "d['torch'] = torch.__version__; d['torch_cuda_build'] = torch.version.cuda\n"
             "d['cuda_available'] = torch.cuda.is_available()\n"
             "d['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
             "from HD_BET import paths\n"
             "d['params_dir'] = paths.folder_with_parameter_files; d['params_url'] = paths.ZENODO_DOWNLOAD_URL\n"
             "print(json.dumps(d))")
    venv = {}
    if p["venv_py"].exists():
        r = subprocess.run([str(p["venv_py"]), "-c", probe], capture_output=True, text=True)
        if r.returncode == 0:
            venv = json.loads(r.stdout.strip().splitlines()[-1])
        else:
            venv = {"error": r.stderr[-1000:]}
    env["hdbet_venv"] = venv
    if with_weights_hash and venv.get("params_dir"):
        w = Path(venv["params_dir"]) / "fold_all" / "checkpoint_final.pth"
        env["hdbet_weights"] = _cached_sha(cfg, w, write_cache) if w.exists() else None
    return env


def _cached_sha(cfg, path, write_cache=True):
    cache = cfg.paths["cache"] / "file_sha256.json"
    data = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    sig = file_sig(path)
    key = f"{sig['path']}|{sig['size']}|{sig['mtime_ns']}"
    if key not in data:
        data[key] = sha256_file(path)
        if write_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(cache, data)
    return {"path": str(path), "size": sig["size"], "sha256": data[key]}


def config_fingerprint(cfg, space, env):
    """Hash parameter metode + versi tool + hash template + hash bobot HD-BET."""
    tmpl = {f"{res}mm": space.files[res]["array_sha256"] for res in space.files}
    venv = env.get("hdbet_venv", {})
    payload = {"pipeline_version": PIPELINE_VERSION, "method": cfg.method_params(),
               "crop": [[s.start, s.stop] for s in space.crop], "pad": [list(p) for p in space.pad],
               "versions": {k: env.get(k) for k in ("antspyx", "nibabel", "numpy", "nilearn")} |
                           {f"venv_{k}": venv.get(k) for k in ("hd-bet", "nnunetv2", "torch", "SimpleITK")},
               "hdbet_weights_sha256": (env.get("hdbet_weights") or {}).get("sha256"),
               "templates": tmpl}
    fp = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return fp, payload


# ======================================================================================
# Checkpoint per langkah
# ======================================================================================
def subject_wd(cfg, sid):
    return cfg.paths["subjects_work"] / str(int(sid))


def _marker(wd, step):
    return wd / f"{step}.done.json"


def step_valid(wd, step, fp, input_sig, outputs):
    m = _marker(wd, step)
    if not m.exists():
        return False
    try:
        info = json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        return False
    if info.get("fp") != fp or info.get("input_sig") != input_sig:
        return False
    for name, size in info.get("outputs", {}).items():
        f = wd / name
        if not f.exists() or f.stat().st_size != size or name not in outputs:
            return False
    return set(info.get("outputs", {})) == set(outputs)


def mark_step(wd, step, fp, input_sig, outputs, extra=None):
    write_json_atomic(_marker(wd, step), {"step": step, "fp": fp, "input_sig": input_sig,
                                          "outputs": {o: (wd / o).stat().st_size for o in outputs},
                                          "done": datetime.datetime.now().isoformat(timespec="seconds"),
                                          **(extra or {})})


def _sig_rel(path):
    s = file_sig(path)
    return {"name": Path(path).name, "size": s["size"], "mtime_ns": s["mtime_ns"]}


def _reorient_to(img, axcodes):
    import nibabel as nib
    t = nib.orientations.ornt_transform(nib.io_orientation(img.affine), nib.orientations.axcodes2ornt(axcodes))
    return img.as_reoriented(t)


def step_reorient(cfg, fp, sid, src, wd, rec):
    import nibabel as nib
    sig = {"source": file_sig(src)}
    if step_valid(wd, "reorient", fp, sig, ["ras.nii.gz"]):
        rec["cached_reorient"] = True
        return
    t = time.time()
    img = nib.load(str(src))
    if img.ndim == 4 and img.shape[3] == 1:
        img = img.slicer[..., 0]
    assert img.ndim == 3, f"bukan 3D: {img.shape}"
    can = nib.as_closest_canonical(img)
    data = can.get_fdata(dtype=np.float32)
    assert np.isfinite(data).all(), "voxel non-finite"
    n_neg = int((data < 0).sum())
    if cfg.clip_negative_before_n4:
        data = np.clip(data, 0, None)
    ras = nib.Nifti1Image(data, can.affine)
    ras.set_qform(can.affine, code=1); ras.set_sform(can.affine, code=1)
    assert nib.aff2axcodes(ras.affine) == tuple(cfg.target_axcodes)
    save_nifti_atomic(wd / "ras.nii.gz", ras)
    extra = {"orig_axcodes": "".join(nib.aff2axcodes(img.affine)), "orig_shape": list(img.shape),
             "orig_voxel_mm": [float(z) for z in img.header.get_zooms()[:3]], "n_negative_clipped": n_neg,
             "ras_shape": list(ras.shape)}
    mark_step(wd, "reorient", fp, sig, ["ras.nii.gz"], extra)
    rec["time_reorient"] = time.time() - t


def step_n4(cfg, fp, sid, wd, rec):
    import ants
    import nibabel as nib
    sig = {"ras": _sig_rel(wd / "ras.nii.gz")}
    if step_valid(wd, "n4", fp, sig, ["n4.nii.gz", "hdbet_in.nii.gz"]):
        rec["cached_n4"] = True
        return
    t = time.time()
    n4 = ants.n4_bias_field_correction(ants.image_read(str(wd / "ras.nii.gz")), **cfg.n4_params)
    tmp = _tmp_path(wd / "n4.nii.gz", ".nii.gz")
    try:
        ants.image_write(n4, str(tmp)); _replace_retry(tmp, wd / "n4.nii.gz")
    finally:
        if tmp.exists():
            tmp.unlink()
    n4_nib, ras = nib.load(str(wd / "n4.nii.gz")), nib.load(str(wd / "ras.nii.gz"))
    aff_diff = float(np.abs(n4_nib.affine - ras.affine).max())
    assert n4_nib.shape == ras.shape and aff_diff < 1e-3, f"N4 mengubah grid (affine max diff {aff_diff})"
    hd_in = _reorient_to(n4_nib, cfg.hdbet_input_axcodes)
    assert nib.aff2axcodes(hd_in.affine) == tuple(cfg.hdbet_input_axcodes)
    save_nifti_atomic(wd / "hdbet_in.nii.gz", hd_in)
    mark_step(wd, "n4", fp, sig, ["n4.nii.gz", "hdbet_in.nii.gz"], {"affine_max_diff_vs_ras": aff_diff})
    rec["time_n4"] = time.time() - t


def skullstrip_valid(fp, wd):
    if not (wd / "hdbet_in.nii.gz").exists():
        return False
    return step_valid(wd, "skullstrip", fp, {"hdbet_in": _sig_rel(wd / "hdbet_in.nii.gz")}, ["mask_hdbet.nii.gz"])


_REG_SCRIPT = r"""
import json, sys
import ants
fixed, moving, prefix, ttype, seed = sys.argv[1:6]
ants.config._random_seed = int(seed)                    # menambah --random-seed ke antsRegistration
reg = ants.registration(fixed=ants.image_read(fixed), moving=ants.image_read(moving), type_of_transform=ttype,
                        outprefix=prefix)
print("REG_FWD " + json.dumps(reg["fwdtransforms"]))
"""


def register_subprocess(cfg, fixed_path, moving_path, outprefix):
    """ants.registration di proses terpisah dengan ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=cfg.reg_threads
    (env harus diset sebelum ITK dimuat; proses induk memakai thread default). Return fwdtransforms."""
    env = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(cfg.reg_threads), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", _REG_SCRIPT, str(fixed_path), str(moving_path), str(outprefix),
                        cfg.reg_transform, str(cfg.ants_random_seed)], capture_output=True, text=True, env=env)
    lines = [l for l in r.stdout.splitlines() if l.startswith("REG_FWD ")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"antsRegistration gagal (kode {r.returncode}): {(r.stdout + r.stderr)[-2000:]}")
    return json.loads(lines[-1][len("REG_FWD "):])


def step_mask_and_register(cfg, fp, space, sid, wd, rec):
    """Mask HD-BET -> RAS, otak subjek, registrasi afin. Checkpoint: 'register'."""
    import ants
    import nibabel as nib
    sig = {"mask_hdbet": _sig_rel(wd / "mask_hdbet.nii.gz"), "n4": _sig_rel(wd / "n4.nii.gz")}
    outs = ["mask_ras.nii.gz", "brain_ras.nii.gz", "xfm.mat"]
    if step_valid(wd, "register", fp, sig, outs):
        rec["cached_register"] = True
        return
    t = time.time()
    n4 = nib.load(str(wd / "n4.nii.gz"))
    mask_ras = nib.as_closest_canonical(nib.load(str(wd / "mask_hdbet.nii.gz")))
    aff_diff = float(np.abs(mask_ras.affine - n4.affine).max())
    assert mask_ras.shape == n4.shape and aff_diff < 1e-3, f"mask HD-BET tidak sejajar (affine diff {aff_diff})"
    m = (np.asanyarray(mask_ras.dataobj) > 0).astype(np.uint8)
    assert m.sum() > 0, "mask HD-BET kosong"
    save_nifti_atomic(wd / "mask_ras.nii.gz", nib.Nifti1Image(m, n4.affine))
    save_nifti_atomic(wd / "brain_ras.nii.gz", nib.Nifti1Image(n4.get_fdata(dtype=np.float32) * m, n4.affine))
    brain_ml = float(m.sum() * np.prod(n4.header.get_zooms()[:3]) / 1000)
    rec["time_mask_ras"] = time.time() - t

    t = time.time()
    with tempfile.TemporaryDirectory(dir=wd) as td:
        xfm = register_subprocess(cfg, space.files[cfg.template_res_reg_mm]["brain_path"], wd / "brain_ras.nii.gz",
                                  Path(td) / "reg_")
        assert len(xfm) == 1 and xfm[0].endswith(".mat"), f"transform tak terduga: {xfm}"
        copy_atomic(xfm[0], wd / "xfm.mat")
    tx = ants.read_transform(str(wd / "xfm.mat"))
    A = np.asarray(tx.parameters[:9]).reshape(3, 3)
    sv = np.linalg.svd(A, compute_uv=False)
    reg_info = {"reg_det": float(np.linalg.det(A)), "reg_scale_min": float(sv.min()), "reg_scale_max": float(sv.max()),
                "brain_volume_native_ml": brain_ml, "mask_affine_max_diff": aff_diff}
    mark_step(wd, "register", fp, sig, outs, reg_info)
    rec["time_registration"] = time.time() - t


def finalize(cfg, fp, space, sid, wd, out, rec, figure=True):
    """Resample 2 mm, intensitas, crop/pad, simpan atomik ke out (dict path). Selalu dihitung (cepat)."""
    import ants
    t = time.time()
    xfm = [str(wd / "xfm.mat")]
    grid = ants.image_read(str(space.files[cfg.target_res_mm]["t1_path"]))
    vol = ants.apply_transforms(fixed=grid, moving=ants.image_read(str(wd / "brain_ras.nii.gz")),
                                transformlist=xfm, interpolator="linear").numpy()
    mk = ants.apply_transforms(fixed=grid, moving=ants.image_read(str(wd / "mask_ras.nii.gz")),
                               transformlist=xfm, interpolator=cfg.mask_interp).numpy() > cfg.mask_threshold
    assert vol.shape == space.files[cfg.target_res_mm]["shape"] and mk.any()
    rec["time_resample"] = time.time() - t

    t = time.time()
    lo_p, hi_p = np.percentile(vol[mk], cfg.clip_percentiles)
    v = np.clip(vol, lo_p, hi_p)
    mu, sd = float(v[mk].mean()), float(v[mk].std())
    assert sd > 0, "std di dalam mask = 0"
    v = (v - mu) / sd
    v[~mk] = 0
    rec["time_intensity"] = time.time() - t

    t = time.time()
    v_c = space.crop_pad(v).astype(np.float32)[None]
    m_c = space.crop_pad(mk.astype(np.uint8))[None]
    check_volume(cfg, v_c, m_c)
    save_npy_atomic(out["vol"], v_c)
    save_npy_atomic(out["mask"], m_c)
    copy_atomic(wd / "xfm.mat", out["xfm"])
    if cfg.save_nifti and out.get("vol_nii") is not None:
        import nibabel as nib
        for arr, dst in ((v_c[0], out["vol_nii"]), (m_c[0], out["mask_nii"])):
            img = nib.Nifti1Image(arr, space.affine_final)
            img.set_qform(space.affine_final, code=4); img.set_sform(space.affine_final, code=4)   # 4 = MNI152
            save_nifti_atomic(dst, img)
        rec.update(vol_nii_sha256=sha256_file(out["vol_nii"]), mask_nii_sha256=sha256_file(out["mask_nii"]))
    if out.get("mask_native") is not None:
        copy_atomic(wd / "mask_ras.nii.gz", out["mask_native"])
    rec["time_crop_save"] = time.time() - t
    mb = m_c.astype(bool)
    reg = json.loads(_marker(wd, "register").read_text(encoding="utf-8"))
    rec.update(clip_low=float(lo_p), clip_high=float(hi_p), zscore_mean_raw=mu, zscore_std_raw=sd,
               final_shape=list(v_c.shape), vol_min=float(v_c.min()), vol_max=float(v_c.max()),
               vol_mean_in_mask=float(v_c[mb].mean()), vol_std_in_mask=float(v_c[mb].std()),
               mask_voxels=int(m_c.sum()), mask_ratio_template=float(m_c.sum() / space.template_mask_crop.sum()),
               vol_sha256=sha256_file(out["vol"]), mask_sha256=sha256_file(out["mask"]),
               **{k: reg.get(k) for k in ("reg_det", "reg_scale_min", "reg_scale_max", "brain_volume_native_ml")})
    if figure and cfg.save_qc_figure and out.get("fig") is not None:
        save_qc_figure(cfg, space, sid, wd, v_c[0], m_c[0], out["fig"])


def check_volume(cfg, v, m):
    """Assert format output: shape, dtype, finite, mask biner, latar 0, z-score di mask."""
    assert v.shape == (1, *cfg.target_shape) and v.dtype == np.float32, (v.shape, v.dtype)
    assert m.shape == (1, *cfg.target_shape) and m.dtype == np.uint8 and set(np.unique(m)) <= {0, 1}
    assert np.isfinite(v).all(), "NaN/Inf di volume"
    mb = m.astype(bool)
    assert mb.any() and (v[~mb] == 0).all(), "latar bukan 0"
    assert abs(float(v[mb].mean())) < 0.05 and abs(float(v[mb].std()) - 1) < 0.05, "z-score di mask tidak ≈ (0, 1)"


# ======================================================================================
# HD-BET
# ======================================================================================
def _tree_rss_gb(pid):
    """Total RSS (GiB) proses + turunannya; 0 jika sudah selesai."""
    import psutil
    try:
        pr = psutil.Process(pid)
        return sum(x.memory_info().rss for x in [pr] + pr.children(recursive=True)) / 2**30
    except psutil.Error:
        return 0.0


def process_peak_rss_gb():
    """Puncak memori proses ini (Windows: peak working set; Linux: ru_maxrss)."""
    import psutil
    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):
        return mi.peak_wset / 2**30
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def kill_process_tree(pid, timeout=10):
    """Hentikan proses beserta semua turunannya (worker nnU-Net). Anak dikumpulkan sebelum induk dihentikan,
    karena di Windows anak kehilangan tautan ke induk setelah induk mati. Return ringkasan."""
    import psutil
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return {"pid": pid, "children": [], "alive_after": []}
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=timeout)
    return {"pid": pid, "children": [p.pid for p in procs[:-1]], "alive_after": [p.pid for p in alive]}


def run_hdbet(cfg, in_dir, out_dir, mode=None):
    """Satu panggilan HD-BET untuk semua .nii.gz di in_dir; memantau RAM/VRAM; RAM < ambang -> bunuh proses."""
    p = cfg.paths
    mode = mode or cfg.hdbet_mode
    if mode == "runner":
        cmd = [str(p["venv_py"]), str(p["hdbet_runner"]), "-i", str(in_dir), "-o", str(out_dir),
               "--device", cfg.hdbet_device, "--tta", str(int(cfg.hdbet_tta)),
               "--np_preprocessing", str(cfg.hdbet_np_preprocessing), "--np_export", str(cfg.hdbet_np_export),
               "--sequential", str(int(cfg.hdbet_sequential)), "--deterministic", str(int(cfg.hdbet_deterministic))]
    elif mode == "cli":
        cmd = [str(p["hdbet_exe"]), "-i", str(in_dir), "-o", str(out_dir), "-device", cfg.hdbet_device,
               "--save_bet_mask", "--no_bet_image"] + ([] if cfg.hdbet_tta else ["--disable_tta"])
    else:
        raise ValueError(mode)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    check_ram(cfg, "sebelum HD-BET")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0   # Ctrl+C tidak membunuh HD-BET
    log_path = Path(out_dir) / "hdbet_stdout.log"
    g0 = gpu_query()
    vram, ram, killed, rss_peak = [], [], None, 0.0
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags,
                                start_new_session=(os.name != "nt"))   # Linux: Ctrl+C juga tidak membunuh HD-BET
        try:
            while proc.poll() is None:
                g = gpu_query()
                vram.append(g["used"] if g else np.nan)
                ram.append(ram_available_gb())
                rss_peak = max(rss_peak, _tree_rss_gb(proc.pid))
                if ram[-1] < cfg.min_free_ram_gb:
                    killed = ram[-1]
                    break
                time.sleep(0.5)
        finally:
            if proc.poll() is None:          # guard aktif, error, atau Ctrl+C paksa: hentikan seluruh pohon proses
                info_kill = kill_process_tree(proc.pid)
                proc.wait()
                LOG.warning("HD-BET dihentikan (pohon proses): %s", info_kill)
    out_text = log_path.read_text(encoding="utf-8", errors="replace")
    info = {"mode": mode, "seconds": time.time() - t0, "returncode": proc.returncode,
            "vram_baseline_mib": g0["used"] if g0 else None,
            "vram_peak_mib": float(np.nanmax(vram)) if vram and not np.all(np.isnan(vram)) else None,
            "ram_min_available_gb": float(min(ram)) if ram else None, "tree_rss_peak_gb": round(rss_peak, 3)}
    for line in out_text.splitlines():
        if line.startswith("HDBET_RUNNER_SUMMARY "):
            info["runner"] = json.loads(line[len("HDBET_RUNNER_SUMMARY "):])
    if killed is not None:
        raise MemoryGuardError(f"HD-BET dihentikan: RAM tersedia {killed:.2f} GiB < {cfg.min_free_ram_gb} GiB.")
    if proc.returncode != 0:
        raise RuntimeError(f"HD-BET gagal (kode {proc.returncode}):\n{out_text[-3000:]}")
    return info


# ======================================================================================
# Registry: output valid, status, failed
# ======================================================================================
def out_paths(cfg, sid):
    p, sid = cfg.paths, int(sid)
    o = {"vol": p["vol"] / f"{sid}.npy", "mask": p["mask"] / f"{sid}.npy", "xfm": p["xfm"] / f"{sid}_0GenericAffine.mat",
         "meta": p["meta"] / f"{sid}.json", "fig": p["fig"] / f"{sid}.png",
         "mask_native": p["mask_native"] / f"{sid}_mask_native.nii.gz", "log": p["subj_logs"] / f"{sid}.log"}
    if cfg.save_nifti:
        o.update(vol_nii=p["nii"] / f"{sid}.nii.gz", mask_nii=p["nii"] / f"{sid}_mask.nii.gz")
    return o


def subject_log(cfg, sid, msg):
    """Log per subjek (append), termasuk traceback lengkap bila gagal."""
    d = cfg.paths["subj_logs"]
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{int(sid)}.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | {msg}\n")


def output_valid(cfg, sid, fp, deep=True):
    """Valid jika: semua output ada, meta status ok, fp sama, sha256 cocok meta, shape/dtype benar, finite,
    mask biner tidak kosong, luar mask = 0, NIfTI terbaca dan isinya sama dengan .npy."""
    o = out_paths(cfg, sid)
    try:
        req = ["vol", "mask", "xfm", "meta", "mask_native"] + (["fig"] if cfg.save_qc_figure else []) + \
              (["vol_nii", "mask_nii"] if cfg.save_nifti else [])
        if not all(o[k].exists() for k in req):
            return False
        meta = json.loads(o["meta"].read_text(encoding="utf-8"))
        if meta.get("status") != "ok" or meta.get("config_fp") != fp:
            return False
        if not deep:
            return True
        if sha256_file(o["vol"]) != meta.get("vol_sha256") or sha256_file(o["mask"]) != meta.get("mask_sha256"):
            return False
        v, m = np.load(o["vol"]), np.load(o["mask"])
        check_volume(cfg, v, m)
        if cfg.save_nifti:
            import nibabel as nib
            if sha256_file(o["vol_nii"]) != meta.get("vol_nii_sha256") or \
               sha256_file(o["mask_nii"]) != meta.get("mask_nii_sha256"):
                return False
            vn, mn = nib.load(str(o["vol_nii"])), nib.load(str(o["mask_nii"]))
            if vn.shape != tuple(cfg.target_shape) or not np.allclose(vn.affine, mn.affine):
                return False
            if not (np.array_equal(np.asanyarray(vn.dataobj), v[0]) and np.array_equal(np.asanyarray(mn.dataobj), m[0])):
                return False
        return True
    except Exception:
        return False


class FailedRegistry:
    COLS = ["SUB_ID", "failed_step", "error", "attempts", "first_failed", "last_failed", "config_fp"]

    def __init__(self, cfg):
        self.path = cfg.paths["failed"]
        self.df = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=self.COLS)

    def attempts(self, sid, fp):
        r = self.df[(self.df["SUB_ID"] == int(sid)) & (self.df["config_fp"] == fp)]
        return int(r["attempts"].iloc[0]) if len(r) else 0

    def record(self, sid, step, err, fp):
        now = datetime.datetime.now().isoformat(timespec="seconds")
        prev = self.df[self.df["SUB_ID"] == int(sid)]
        attempts = (int(prev["attempts"].iloc[0]) if len(prev) and prev["config_fp"].iloc[0] == fp else 0) + 1
        first = prev["first_failed"].iloc[0] if len(prev) and prev["config_fp"].iloc[0] == fp else now
        self.df = pd.concat([self.df[self.df["SUB_ID"] != int(sid)],
                             pd.DataFrame([{"SUB_ID": int(sid), "failed_step": step, "error": str(err)[:1000],
                                            "attempts": attempts, "first_failed": first, "last_failed": now,
                                            "config_fp": fp}])], ignore_index=True)
        write_csv_atomic(self.path, self.df.sort_values("SUB_ID"))

    def clear(self, sid):
        if (self.df["SUB_ID"] == int(sid)).any():
            self.df = self.df[self.df["SUB_ID"] != int(sid)]
            write_csv_atomic(self.path, self.df)


STATUS_COLS = ["subject_id", "site", "status", "last_stage", "error", "t_reorient_s", "t_n4_s", "t_skullstrip_s",
               "t_register_s", "t_finalize_s", "duration_s", "ram_min_available_gb", "hdbet_tree_rss_peak_gb",
               "main_peak_rss_gb", "tool", "tool_version", "config_hash", "run_id", "updated"]
DONE_STATUSES = ("success", "flagged")


def read_status(cfg):
    p = cfg.paths["status"]
    return pd.read_csv(p, keep_default_na=False, na_values=[""]) if p.exists() else pd.DataFrame(columns=STATUS_COLS)


def update_status(cfg, rows):
    """status.csv teknis (tanpa kolom fenotipik). Baris lama per subject_id diganti; kolom lain dipertahankan."""
    recs = {int(r["subject_id"]): r for r in read_status(cfg).to_dict("records")}
    for r in rows:
        sid = int(r["subject_id"])
        base = recs.get(sid, {})
        base.update(r)
        base["subject_id"] = sid
        recs[sid] = base
    out = pd.DataFrame(list(recs.values()))
    out = out.reindex(columns=STATUS_COLS + [c for c in out.columns if c not in STATUS_COLS])
    write_csv_atomic(cfg.paths["status"], out.sort_values("subject_id"))


def status_row(rec, site, fp, tool, run_id):
    hd = rec.get("hdbet") or {}
    return {"subject_id": int(rec["SUB_ID"]), "site": site, "status": {"ok": "success"}.get(rec.get("status"), rec.get("status")),
            "last_stage": rec.get("last_stage"), "error": rec.get("error"),
            "t_reorient_s": rec.get("time_reorient"), "t_n4_s": rec.get("time_n4"),
            "t_skullstrip_s": rec.get("time_skullstrip"), "t_register_s": rec.get("time_registration"),
            "t_finalize_s": sum(rec.get(k) or 0 for k in ("time_resample", "time_intensity", "time_crop_save")) or None,
            "duration_s": rec.get("duration_s"), "ram_min_available_gb": rec.get("ram_min_available_gb"),
            "hdbet_tree_rss_peak_gb": hd.get("tree_rss_peak_gb"), "main_peak_rss_gb": rec.get("main_peak_rss_gb"),
            "tool": tool.get("tool"), "tool_version": tool.get("version"), "config_hash": fp, "run_id": run_id,
            "updated": datetime.datetime.now().isoformat(timespec="seconds")}


def existing_config_hashes(cfg):
    """Hash config yang sudah ada di folder output (status.csv + meta). Dipakai untuk menolak pencampuran."""
    hs = set(read_status(cfg)["config_hash"].dropna().astype(str))
    if cfg.paths["meta"].exists():
        for f in cfg.paths["meta"].glob("*.json"):
            try:
                hs.add(str(json.loads(f.read_text(encoding="utf-8")).get("config_fp")))
            except Exception:
                hs.add("tidak-terbaca")
    return {h for h in hs if h and h != "None"}


# ======================================================================================
# Proses satu chunk
# ======================================================================================
def process_chunk(cfg, fp, space, subjects, failed, stop, retry_failed=False, skip_if=None, tool=None):
    """Proses subjek dalam chunk dengan checkpoint per langkah. Return list record. Subjek di-skip bila
    skip_if(sid) benar (default: output valid). STOP diperiksa pemanggil di antara chunk, sehingga subjek
    dalam chunk selalu diselesaikan. MemoryGuardError diteruskan ke pemanggil (berhenti rapi)."""
    p = cfg.paths
    skip_if = skip_if or (lambda s: output_valid(cfg, s, fp))
    recs = {}
    todo = []
    for r in subjects.itertuples():
        sid = int(r.SUB_ID)
        if skip_if(sid):
            recs[sid] = {"SUB_ID": sid, "status": "skipped_valid"}
            continue
        if not retry_failed and failed.attempts(sid, fp) >= cfg.max_retries:
            recs[sid] = {"SUB_ID": sid, "status": "skipped_max_retries"}
            LOG.warning("SUB_ID %s dilewati: sudah gagal %d kali (gunakan --retry-failed).", sid, failed.attempts(sid, fp))
            continue
        todo.append(r)

    # Tahap A: reorient + N4 (+ input HD-BET), per subjek
    need_hdbet = []
    for r in todo:
        sid = int(r.SUB_ID)
        check_ram(cfg, f"sebelum subjek {sid}")
        wd = subject_wd(cfg, sid); wd.mkdir(parents=True, exist_ok=True)
        rec = recs.setdefault(sid, {"SUB_ID": sid, "status": "running", "t_start": time.time(),
                                    "started": datetime.datetime.now().isoformat(timespec="seconds"),
                                    "skullstrip_tool": (tool or {}).get("tool"), "tool": tool})
        subject_log(cfg, sid, f"mulai | config {fp}")
        step = "reorient"
        try:
            step_reorient(cfg, fp, sid, r.smri_abspath, wd, rec)
            rec["last_stage"] = "reorient"
            step = "n4"
            step_n4(cfg, fp, sid, wd, rec)
            rec["last_stage"] = "n4"
            if not skullstrip_valid(fp, wd):
                need_hdbet.append(sid)
            else:
                rec["cached_skullstrip"] = True
        except MemoryGuardError:
            raise
        except Exception as e:
            rec.update(status="failed", failed_step=step, error=f"{type(e).__name__}: {e}")
            failed.record(sid, step, rec["error"], fp)
            subject_log(cfg, sid, f"GAGAL di {step}\n{traceback.format_exc()}")
            LOG.error("SUB_ID %s gagal di %s: %s", sid, step, rec["error"])

    # Tahap B: HD-BET sekali untuk semua subjek yang butuh
    hd_info = None
    if need_hdbet:
        chunk_dir = p["hdbet_work"] / f"chunk_{min(need_hdbet)}_{len(need_hdbet)}_{int(time.time())}"
        (chunk_dir / "in").mkdir(parents=True)
        for sid in need_hdbet:
            shutil.copyfile(subject_wd(cfg, sid) / "hdbet_in.nii.gz", chunk_dir / "in" / f"{sid}.nii.gz")
        try:
            hd_info = run_hdbet(cfg, chunk_dir / "in", chunk_dir / "out")
            for sid in need_hdbet:
                wd = subject_wd(cfg, sid)
                src = chunk_dir / "out" / f"{sid}_bet.nii.gz"
                if not src.exists():
                    raise FileNotFoundError(f"mask HD-BET tidak ada untuk {sid}")
                copy_atomic(src, wd / "mask_hdbet.nii.gz")
                mark_step(wd, "skullstrip", fp, {"hdbet_in": _sig_rel(wd / "hdbet_in.nii.gz")}, ["mask_hdbet.nii.gz"],
                          {"hdbet_mode": hd_info["mode"], "chunk_n": len(need_hdbet)})
                recs[sid]["time_skullstrip"] = hd_info["seconds"] / len(need_hdbet)
                recs[sid]["last_stage"] = "skullstrip"
                recs[sid]["hdbet"] = {k: hd_info.get(k) for k in ("mode", "seconds", "vram_peak_mib",
                                                                   "ram_min_available_gb", "tree_rss_peak_gb")}
        except MemoryGuardError:
            raise
        except Exception as e:
            for sid in need_hdbet:
                if not skullstrip_valid(fp, subject_wd(cfg, sid)):
                    recs[sid].update(status="failed", failed_step="skullstrip", error=str(e)[:1000])
                    failed.record(sid, "skullstrip", recs[sid]["error"], fp)
                    subject_log(cfg, sid, f"GAGAL di skullstrip\n{traceback.format_exc()}")
            LOG.error("HD-BET gagal untuk chunk %s: %s", need_hdbet, str(e)[-500:])
        finally:
            if not cfg.keep_work_files:
                shutil.rmtree(chunk_dir, ignore_errors=True)

    # Tahap C: mask -> registrasi -> finalisasi, per subjek
    for r in todo:
        sid = int(r.SUB_ID)
        rec = recs.get(sid)
        if rec is None or rec.get("status") == "failed":
            continue
        wd = subject_wd(cfg, sid)
        if not skullstrip_valid(fp, wd):
            rec.update(status="failed", failed_step="skullstrip", error="mask skull strip tidak tersedia")
            failed.record(sid, "skullstrip", rec["error"], fp)
            subject_log(cfg, sid, "GAGAL: mask skull strip tidak tersedia")
            continue
        rec.setdefault("last_stage", "skullstrip")
        step = "register"
        try:
            check_ram(cfg, f"sebelum registrasi {sid}")
            step_mask_and_register(cfg, fp, space, sid, wd, rec)
            rec["last_stage"] = "register"
            step = "finalize"
            o = out_paths(cfg, sid)
            finalize(cfg, fp, space, sid, wd, o, rec)
            rec.update(status="ok", last_stage="finalize", config_fp=fp, pipeline_version=PIPELINE_VERSION,
                       finished=datetime.datetime.now().isoformat(timespec="seconds"),
                       duration_s=time.time() - rec.pop("t_start"), ram_min_available_gb=ram_available_gb(),
                       main_peak_rss_gb=process_peak_rss_gb())
            for k in ("reorient", "n4"):
                m = _marker(wd, k)
                if m.exists():
                    rec[f"marker_{k}"] = json.loads(m.read_text(encoding="utf-8"))
            write_json_atomic(o["meta"], rec)
            if not output_valid(cfg, sid, fp):
                raise RuntimeError("output final tidak lolos validasi ulang")
            failed.clear(sid)
            subject_log(cfg, sid, f"sukses | {rec['duration_s']:.0f} s")
            if not cfg.keep_work_files:          # hanya setelah output final lolos validasi
                shutil.rmtree(wd, ignore_errors=True)
        except MemoryGuardError:
            raise
        except Exception as e:
            rec.update(status="failed", failed_step=step, error=f"{type(e).__name__}: {e}")
            failed.record(sid, step, rec["error"], fp)
            subject_log(cfg, sid, f"GAGAL di {step}\n{traceback.format_exc()}")
            LOG.error("SUB_ID %s gagal di %s: %s", sid, step, rec["error"])
    for rec in recs.values():
        rec.pop("t_start", None)
    return [recs[int(r.SUB_ID)] for r in subjects.itertuples() if int(r.SUB_ID) in recs]


# ======================================================================================
# QC
# ======================================================================================
def _mid(a):
    x, y, z = (s // 2 for s in a.shape)
    return [np.rot90(a[x, :, :]), np.rot90(a[:, y, :]), np.rot90(a[:, :, z])]


def _new_figure(figsize, nrows, ncols):
    """Figure tanpa pyplot (tidak mengubah backend notebook)."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize)
    axes = fig.subplots(nrows, ncols)
    return fig, axes


def _save_figure(fig, dst):
    tmp = _tmp_path(dst, ".png")
    try:
        fig.savefig(tmp, dpi=80)
        _replace_retry(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_qc_figure(cfg, space, sid, wd, vol, mask, dst):
    import nibabel as nib
    ras = nib.load(str(wd / "ras.nii.gz")).get_fdata(dtype=np.float32)
    n4 = nib.load(str(wd / "n4.nii.gz")).get_fdata(dtype=np.float32)
    mras = nib.load(str(wd / "mask_ras.nii.gz")).get_fdata() > 0
    fig, axes = _new_figure((10, 10), 3, 3)
    rows = ["Asli (RAS)", "N4 + mask HD-BET", "Hasil MNI 2 mm + kontur mask template"]
    for j, (a, b, mk) in enumerate(zip(_mid(ras), _mid(n4), _mid(mras.astype(float)))):
        axes[0, j].imshow(a, cmap="gray"); axes[1, j].imshow(b, cmap="gray")
        axes[1, j].contour(mk, levels=[0.5], colors="r", linewidths=0.6)
    for j, (a, t) in enumerate(zip(_mid(vol), _mid(space.template_mask_crop.astype(float)))):
        axes[2, j].imshow(np.ma.masked_where(a == 0, a), cmap="gray", vmin=-3, vmax=3)
        axes[2, j].contour(t, levels=[0.5], colors="c", linewidths=0.6)
    for i in range(3):
        axes[i, 0].set_ylabel(rows[i], fontsize=8)
        for j in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    fig.suptitle(f"SUB_ID {sid}")
    fig.tight_layout()
    _save_figure(fig, dst)


def qc_metrics(cfg, space, sid):
    o = out_paths(cfg, sid)
    v, m = np.load(o["vol"])[0], np.load(o["mask"])[0].astype(bool)
    meta = json.loads(o["meta"].read_text(encoding="utf-8"))
    tm = space.template_mask_crop
    tz = (space.template_crop - space.template_crop[tm].mean()) / space.template_crop[tm].std()
    return {"SUB_ID": int(sid), "final_shape": str(tuple(v.shape)), "n_nonfinite": int((~np.isfinite(v)).sum()),
            "frac_zero": float((v == 0).mean()), "ncc_template": float(np.corrcoef(v[tm], tz[tm])[0, 1]),
            "mask_ratio_template": float(m.sum() / tm.sum()),
            "dice_mask_template": float(2 * (m & tm).sum() / (m.sum() + tm.sum())),
            "volume_mask_ml": meta.get("brain_volume_native_ml"), "reg_det": meta.get("reg_det"),
            "reg_scale_min": meta.get("reg_scale_min"), "reg_scale_max": meta.get("reg_scale_max"),
            "duration_s": meta.get("duration_s"), "config_fp": meta.get("config_fp")}


def build_qc_report(cfg, space, subjects, fp):
    done = [int(s) for s in subjects["SUB_ID"] if output_valid(cfg, s, fp, deep=False)]
    if not done:
        return pd.DataFrame()
    # QC blind: tidak ada label/diagnosis/kolom klinis di laporan ini
    rep = pd.DataFrame([qc_metrics(cfg, space, s) for s in done]).merge(
        subjects[["SUB_ID", "SITE_ID", "qc_decision"]].rename(columns={"qc_decision": "pcp_qc_decision"}), on="SUB_ID")
    lo, hi = cfg.mask_ratio_range
    rep["flag_mask_ratio"] = ~rep["mask_ratio_template"].between(lo, hi)
    lo, hi = cfg.brain_ml_range
    rep["flag_brain_volume"] = ~rep["volume_mask_ml"].astype(float).between(lo, hi)
    lo, hi = cfg.reg_det_range
    rep["flag_reg_det"] = ~rep["reg_det"].astype(float).between(lo, hi)
    rep["flag_low_ncc"] = rep["ncc_template"] < cfg.min_ncc_template
    rep["flag_hard"] = (rep["n_nonfinite"] > 0) | (rep["final_shape"] != str(tuple(cfg.target_shape)))   # tanpa kanal
    zc = []
    for c in ("ncc_template", "mask_ratio_template", "frac_zero"):
        rep[f"z_{c}"] = ((rep[c] - rep[c].mean()) / rep[c].std(ddof=0)) if len(rep) >= cfg.min_n_global_z else np.nan
        g = rep.groupby("SITE_ID")[c]
        zs = (rep[c] - g.transform("mean")) / g.transform(lambda s: s.std(ddof=0))
        rep[f"zsite_{c}"] = np.where(g.transform("size") >= cfg.min_n_site_z, zs, np.nan)
        zc += [f"z_{c}", f"zsite_{c}"]
    rep["max_abs_z"] = rep[zc].abs().max(axis=1)
    rep["max_abs_zsite"] = rep[[c for c in zc if c.startswith("zsite_")]].abs().max(axis=1)
    rep["flag_outlier_z"] = rep["max_abs_zsite"] > cfg.outlier_z      # relatif terhadap site (butuh n >= min_n_site_z)
    flag_cols = ["flag_mask_ratio", "flag_brain_volume", "flag_reg_det", "flag_low_ncc", "flag_hard", "flag_outlier_z"]
    rep["n_flags"] = rep[flag_cols].sum(axis=1)
    rep["flag_reasons"] = rep[flag_cols].apply(lambda r: ";".join(c[5:] for c in flag_cols if r[c]), axis=1)
    st = read_status(cfg)[["subject_id", "status"]].rename(columns={"subject_id": "SUB_ID"})
    rep = rep.merge(st, on="SUB_ID", how="left")
    return rep.sort_values("SUB_ID").reset_index(drop=True)


QC_REVIEW_COLS = ["subject_id", "site", "AGE_AT_SCAN", "SEX", "volume_mask_ml", "mask_ratio_template", "reg_det",
                  "ncc_template", "zsite_ncc_template", "zsite_mask_ratio_template", "zsite_frac_zero", "n_flags",
                  "flag_reasons", "montage_path"]
QC_FORBIDDEN = ("DX_GROUP", "DSM_IV_TR", "ADOS", "ADI_R", "SRS", "SCQ", "AQ", "CURRENT_MED_STATUS", "label")


def build_qc_review(cfg, rep):
    """qc_review.csv untuk reviewer: hanya kolom yang diizinkan (AGE_AT_SCAN, SEX dari master; tanpa diagnosis)."""
    demo = pd.read_csv(cfg.paths["master"], usecols=["SUB_ID", "AGE_AT_SCAN", "SEX"])
    r = rep.merge(demo, on="SUB_ID", how="left").rename(columns={"SUB_ID": "subject_id", "SITE_ID": "site"})
    r["montage_path"] = [str(out_paths(cfg, s)["fig"]) for s in r["subject_id"]]
    r = r[QC_REVIEW_COLS].sort_values(["n_flags", "subject_id"], ascending=[False, True])
    bad = [c for c in r.columns if any(c.upper().startswith(f) for f in QC_FORBIDDEN)]
    assert not bad, f"kolom terlarang di qc_review: {bad}"
    return r


def write_qc_decisions_template(cfg, review):
    """Template qc_decisions (untuk subjek ber-flag). qc_decisions.csv buatan reviewer TIDAK pernah ditimpa."""
    t = review.loc[review["n_flags"] > 0, ["subject_id"]].copy()
    for c in ("decision", "reason", "reviewer", "date"):
        t[c] = ""
    write_csv_atomic(cfg.paths["qc_decisions_template"], t)


def gallery(cfg, space, ids, rep, dst, title):
    ids = list(ids)
    if not ids:
        return
    cols = 6
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = _new_figure((cols * 2.2, rows * 2.6), rows, cols)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    r = rep.set_index("SUB_ID")
    for ax, sid in zip(axes, ids):
        v = np.load(out_paths(cfg, sid)["vol"])[0]
        z = v.shape[2] // 2
        ax.imshow(np.rot90(v[:, :, z]), cmap="gray", vmin=-3, vmax=3)
        ax.contour(np.rot90(space.template_mask_crop[:, :, z].astype(float)), levels=[0.5], colors="c", linewidths=0.5)
        ax.set_title(f"{sid} {r.at[sid, 'SITE_ID']}\nflags {int(r.at[sid, 'n_flags'])} ncc {r.at[sid, 'ncc_template']:.2f}",
                     fontsize=7)
    fig.suptitle(title); fig.tight_layout()
    _save_figure(fig, dst)


# ======================================================================================
# Validasi runner vs CLI (Tahap 2)
# ======================================================================================
def validate_runner_vs_cli(cfg, fp, space, sid, legacy_ref=True):
    """Bandingkan mask dan volume akhir dari HD-BET runner vs CLI pada satu subjek.
    Mask CLI acuan lama (data/work/hdbet_validation) hanya DISALIN (hash diverifikasi) jika inputnya identik."""
    import nibabel as nib
    p = cfg.paths
    subj = load_subjects(cfg).set_index("SUB_ID").loc[[sid]].reset_index()
    vdir = p["validation"] / str(sid)
    for d in ("in", "cli", "runner", "out_cli", "out_runner"):
        (vdir / d).mkdir(parents=True, exist_ok=True)
    wd = subject_wd(cfg, sid); wd.mkdir(parents=True, exist_ok=True)
    rec = {}
    step_reorient(cfg, fp, sid, subj["smri_abspath"].iloc[0], wd, rec)
    step_n4(cfg, fp, sid, wd, rec)
    copy_atomic(wd / "hdbet_in.nii.gz", vdir / "in" / f"{sid}.nii.gz")
    new_in = np.asanyarray(nib.load(str(vdir / "in" / f"{sid}.nii.gz")).dataobj)

    cli_mask = vdir / "cli" / f"{sid}_bet.nii.gz"
    cli_source = None
    old_dir = p["legacy_work"] / "hdbet_validation"
    if legacy_ref and (old_dir / "cli" / f"{sid}_bet.nii.gz").exists() and (old_dir / "in" / f"{sid}.nii.gz").exists():
        old_in = nib.load(str(old_dir / "in" / f"{sid}.nii.gz"))
        if np.array_equal(np.asanyarray(old_in.dataobj), new_in):
            src = old_dir / "cli" / f"{sid}_bet.nii.gz"
            copy_atomic(src, cli_mask)
            assert sha256_file(cli_mask) == sha256_file(src)
            cli_source = f"salinan terverifikasi dari {src}"
    if cli_source is None:
        run_hdbet(cfg, vdir / "in", vdir / "cli", mode="cli")
        cli_source = "CLI dijalankan sekarang"
    run_info = run_hdbet(cfg, vdir / "in", vdir / "runner", mode="runner")

    a_img, b_img = nib.load(str(cli_mask)), nib.load(str(vdir / "runner" / f"{sid}_bet.nii.gz"))
    a, b = np.asanyarray(a_img.dataobj) > 0, np.asanyarray(b_img.dataobj) > 0
    res = {"SUB_ID": sid, "cli_source": cli_source, "shape_equal": a.shape == b.shape,
           "affine_max_diff": float(np.abs(a_img.affine - b_img.affine).max())}
    res["dice"] = float(2 * (a & b).sum() / (a.sum() + b.sum())) if res["shape_equal"] else float("nan")
    res["mask_bitwise_identical"] = bool(res["shape_equal"] and np.array_equal(a, b))
    res["n_voxel_diff"] = int((a ^ b).sum()) if res["shape_equal"] else None

    vols = {}
    for name, mpath in (("cli", cli_mask), ("runner", vdir / "runner" / f"{sid}_bet.nii.gz")):
        wd_v = vdir / f"wd_{name}"
        wd_v.mkdir(exist_ok=True)
        for f in ("n4.nii.gz", "ras.nii.gz", "hdbet_in.nii.gz"):
            copy_atomic(wd / f, wd_v / f)
        copy_atomic(mpath, wd_v / "mask_hdbet.nii.gz")
        r = {}
        step_mask_and_register(cfg, fp, space, sid, wd_v, r)
        o = {"vol": vdir / f"out_{name}" / "vol.npy", "mask": vdir / f"out_{name}" / "mask.npy",
             "xfm": vdir / f"out_{name}" / "xfm.mat", "fig": None}
        finalize(cfg, fp, space, sid, wd_v, o, r, figure=False)
        vols[name] = np.load(o["vol"])
    va, vb = vols["cli"], vols["runner"]
    res["vol_shape_equal"] = va.shape == vb.shape
    res["vol_max_abs_diff"] = float(np.abs(va - vb).max())
    res["vol_corr"] = float(np.corrcoef(va.ravel(), vb.ravel())[0, 1])
    res["vol_bitwise_identical"] = bool(np.array_equal(va, vb))
    res["runner"] = {k: run_info.get(k) for k in ("seconds", "vram_peak_mib", "ram_min_available_gb", "runner")}
    res["passed"] = bool(res["shape_equal"] and res["affine_max_diff"] < 1e-4
                         and (res["mask_bitwise_identical"] or res["dice"] >= cfg.validation_dice_min)
                         and res["vol_shape_equal"] and res["vol_max_abs_diff"] <= cfg.validation_max_abs_diff
                         and res["vol_corr"] >= cfg.validation_min_corr)
    write_json_atomic(cfg.out_dir / "hdbet_runner_validation.json", res)
    return res


# ======================================================================================
# Cek header citra (read-only terhadap citra)
# ======================================================================================
HDR_COLS = ["SUB_ID", "smri_path", "size", "mtime_ns", "exists", "ndim", "shape", "voxel_mm", "axcodes", "dtype",
            "qform_code", "sform_code", "n_nonfinite", "n_negative", "error"]


def header_check(cfg, subjects, legacy_cache=None, read_data=True):
    """Dimensi, voxel, orientasi, dtype, NaN/Inf, nilai negatif per subjek. Baris dari cache lama (read-only)
    dipakai ulang bila path, ukuran, dan mtime file sama."""
    import nibabel as nib
    cache = {}
    if legacy_cache is not None and Path(legacy_cache).exists():
        c = pd.read_csv(legacy_cache)
        cache = {(int(r.SUB_ID), r.smri_path, int(r.size), int(r.mtime_ns)): r._asdict()
                 for r in c.itertuples(index=False) if not isinstance(r.error, str) or not r.error}
    rows = []
    for r in subjects.itertuples():
        rec = {c: None for c in HDR_COLS}
        rec.update(SUB_ID=int(r.SUB_ID), smri_path=r.smri_path, exists=Path(r.smri_abspath).exists(), error="")
        try:
            st = Path(r.smri_abspath).stat()
            rec.update(size=st.st_size, mtime_ns=st.st_mtime_ns)
            hit = cache.get((int(r.SUB_ID), r.smri_path, st.st_size, st.st_mtime_ns))
            if hit is not None and (not read_data or pd.notna(hit.get("n_nonfinite"))):
                rec.update({c: hit[c] for c in HDR_COLS})
            else:
                img = nib.load(str(r.smri_abspath))
                rec.update(ndim=img.ndim, shape=str(tuple(img.shape)),
                           voxel_mm=str(tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])),
                           axcodes="".join(nib.aff2axcodes(img.affine)), dtype=str(img.get_data_dtype()),
                           qform_code=int(img.header["qform_code"]), sform_code=int(img.header["sform_code"]))
                if read_data:
                    d = np.asanyarray(img.dataobj)
                    rec.update(n_nonfinite=int((~np.isfinite(d)).sum()), n_negative=int((d < 0).sum()))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rows.append(rec)
    hdr = pd.DataFrame(rows, columns=HDR_COLS).merge(subjects[["SUB_ID", "SITE_ID"]], on="SUB_ID")
    vox = hdr["voxel_mm"].dropna().map(lambda x: tuple(float(v) for v in str(x).strip("()").split(",")))
    hdr.loc[vox.index, "max_voxel_mm"] = vox.map(max)
    hdr.loc[vox.index, "anisotropy"] = vox.map(lambda z: max(z) / min(z))
    hdr["flag_blocking"] = (~hdr["exists"].astype(bool)) | hdr["error"].fillna("").ne("") |         (hdr["ndim"].astype(float) != 3) | (hdr["n_nonfinite"].fillna(0).astype(float) > 0)
    return hdr


# ======================================================================================
# Audit warisan (read-only)
# ======================================================================================
def audit_legacy(cfg, subjects_pilot):
    """Bandingkan data/failed_subjects.csv lama dan output lama dengan daftar pilot baru (read-only)."""
    p = cfg.paths
    out = {}
    folds = pd.read_csv(p["folds"])
    excl = set(pd.read_csv(p["exclusions"])["SUB_ID"])
    pilot = set(subjects_pilot["SUB_ID"])
    if p["legacy_failed"].exists():
        f = pd.read_csv(p["legacy_failed"])
        old = set(f["SUB_ID"])
        test_ids = set(folds.loc[folds["split"] == "test", "SUB_ID"])
        out["legacy_failed_n"] = len(old)
        out["legacy_failed_in_new_pilot"] = sorted(old & pilot)
        out["legacy_failed_not_in_pilot"] = sorted(old - pilot)
        out["legacy_failed_now_test"] = sorted(old & test_ids)
        out["legacy_failed_now_excluded"] = sorted(old & excl)
        out["new_pilot_never_processed"] = sorted(pilot - old - {50002})
    legacy_vol = p["legacy_work"].parent / "vol"
    out["legacy_vol_files"] = sorted(int(x.stem) for x in legacy_vol.glob("*.npy")) if legacy_vol.exists() else []
    out["legacy_work_exists"] = p["legacy_work"].exists()
    return out
