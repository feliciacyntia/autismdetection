# Klasifikasi ASD vs TD — ABIDE I (sMRI T1w + fenotipik)

Tugas akhir: klasifikasi Autism Spectrum Disorder (ASD) vs Typically Developing (TD) pada ABIDE I memakai citra MRI
struktural T1w dan data fenotipik. Repo ini berisi kode, notebook, protokol split, dokumentasi QC, dan seluruh hasil
eksperimen. **Data citra hasil preprocessing ada di Hugging Face** (terlalu besar untuk GitHub), lihat bagian Data.

## Status (29-10-2026)
| Tahap | Status |
|---|---|
| EDA, merge, cleaning fenotipik (notebook 01–04) | selesai |
| Audit fold & split (05a), preprocessing fenotipik (05b) | selesai |
| Preprocessing sMRI 05c, 1.002 subjek + audit konsistensi | selesai, lulus audit |
| Pilot 200 subjek (uji coba) + bugfix BN + threshold Youden | selesai (notebook 06–08) |
| **Run data penuh full_v2** (853 dev, config terkunci) | selesai — hasil utama saat ini |
| v3 tabel: LogReg/TabM/CatBoost/TabPFN, tuning Optuna, LOSO | run selesai (288 unit); analisis & laporan menyusul |
| E2 fusion DL (DAFT/FiLM), morfometri FreeSurfer | ditunda |
| Evaluasi akhir 149 subjek test held-out | **belum** (menunggu config final disetujui) |

### Hasil utama full_v2 (853 subjek dev, 5-fold CV, threshold 0,5, pooled + 95 % CI bootstrap)
| Model | Accuracy | BA [95 % CI] | F1 macro | AUC |
|---|---|---|---|---|
| LogReg fenotipik (baseline) | 0,574 | 0,574 [0,541–0,608] | 0,573 | 0,598 |
| TabM fenotipik | 0,601 | 0,595 [0,562–0,628] | 0,594 | 0,624 |
| TabM ROI + ComBat | 0,599 | 0,597 [0,565–0,631] | 0,597 | 0,620 |
| EfficientNet-B0 2.5D (pretrained) | 0,571 | 0,571 [0,540–0,605] | 0,570 | 0,591 |
| CNN 3D + SE dari nol | 0,551 | 0,556 [0,523–0,590] | 0,551 | 0,567 |
| **Fusion fenotipik + ROI (stacking, utama terkunci)** | **0,628** | **0,624 [0,591–0,654]** | 0,624 | **0,664** |

Fusion utama signifikan lebih baik dari baseline LogReg (McNemar, Holm p = 0,040). Rincian: `results/full_v2/report/`,
notebook `06_modeling_report.ipynb`, ringkasan `07_ringkasan_insight.ipynb`.

## Data
- **Fenotipik**: `dataset/cleaned_dataset/cleaned_master_dataset.csv` (+ `data_dictionary.csv`, `cleaning_log.csv`), `data/pheno_clean.csv`.
- **Manifest & split**: `data/folds.csv` — `split` (dev / test), `fold_strat` (5 fold, stratified site × label),
  `repeated_fold_*`, `loso_site`, `loso_eligible`. Split inner (inner_train / inner_val, seed 42): `results/full_v2/splits.json`.
- **QC**: `data/smri_preproc/qc_report.csv`, `qc_decisions.csv` (11 keputusan: 10 KEEP outlier_z, 50404 volume mask = missing),
  `status.csv`, `meta/` (log per subjek), `audit_full/` (audit konsistensi 1.002 subjek).
- **Citra hasil preprocessing (853 dev)**: Hugging Face **https://huggingface.co/datasets/feliciacyntia/abide1-smri-prepro**
  (5,2 GB) — `vol/{SUB_ID}.npy` float32 (1, 96, 112, 96) = input model, `mask/` uint8, `nii/` (volume + mask sebagai
  NIfTI, isi sama dengan .npy), `mask_native/`, `xfm/` (affine ANTs), `qc_figures/`, `templates/` (MNI152), `SHA256SUMS`.
  Hugging Face hanya berisi file besar; daftar subjek/label/fold, QC, dan log per subjek ada di repo ini (tanpa duplikat).
  Unduh **ke dalam `data/smri_preproc/` repo ini** (digabung dengan file QC GitHub di folder yang sama) — lihat "Setup".
- **Tidak disertakan**: citra mentah ABIDE (unduh sendiri dari ABIDE/INDI setelah menyetujui ketentuannya) dan
  **citra 149 subjek test held-out** (disimpan pemilik TA untuk evaluasi akhir sekali). Label diagnosis test tetap terlihat
  di `data/folds.csv` / tabel fenotipik karena fenotipik ABIDE memang publik — jadi pengamannya adalah aturan:
  **baris `split == "test"` tidak boleh dipakai untuk melatih, memilih, atau mengevaluasi model.**

## Preprocessing sMRI (05c-v2.1, fingerprint `682404ae0f5d2624`)
Reorient → N4 → HD-BET (GPU, deterministik) → registrasi afin ANTs ke MNI152 1 mm → 2 mm → crop 96 × 112 × 96 → z-score di mask.
Kode: `src/smri_preprocessing.py`, `src/run_05c.py`; notebook `05c_preprocessing_smri.ipynb`.

## Setup untuk melanjutkan modeling (tanpa preprocessing ulang)
Preprocessing **tidak perlu diulang** (±3 hari); cukup clone repo + unduh citra dari Hugging Face. Diuji 30-10-2026 pada
clone bersih: 853 subjek termuat, split identik dengan full_v2, volume (853, 96, 112, 96) termuat, hash config terkunci cocok.
```powershell
git clone https://github.com/feliciacyntia/autismdetection.git; Set-Location autismdetection
# 1) citra hasil preprocessing (5,3 GB) masuk ke data\smri_preproc\ repo ini (file QC dari GitHub sudah ada di sana)
pip install huggingface_hub
hf download feliciacyntia/abide1-smri-prepro --repo-type dataset --local-dir data\smri_preproc
# 1b) OPSIONAL (6,8 GB): checkpoint model .pt, cache embedding .npy, atlas ROI -> langsung masuk ke results\ dan pretrained\
#     perlu hanya untuk memuat model yang sudah dilatih / ekstrak embedding / hitung ulang fitur ROI
hf download feliciacyntia/abide1-asd-results --local-dir . --include "results/*" "pretrained/*"
# 2) environment modeling (Python 3.12.3, GPU CUDA 12.1) — lihat komentar di requirements-model.txt untuk TabPFN
python -m venv .venv-model
.\.venv-model\Scripts\pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv-model\Scripts\pip install -r requirements-model.txt
# 3) arahkan kode ke data & hasil di repo (default kode = C:\ta_local\...)
$env:TA06_IMAGE_ROOT   = "$PWD\data\smri_preproc"
$env:TA06_RESULTS_ROOT = "$PWD\results"
$env:TA06_PRETRAINED   = "$PWD\pretrained"
$env:PYTHONIOENCODING  = "utf-8"
```
| Isi | Di mana |
|---|---|
| kode, notebook (+ HTML), fenotipik (asli ABIDE & bersih), `folds.csv`, split inner, QC, log per subjek, audit, config terkunci, prediksi per subjek, tabel, gambar, log semua run (termasuk run uji di `results/_uji/`) | GitHub (repo ini) |
| citra hasil preprocessing 853 dev (`vol`, `mask`, `nii`, `mask_native`, `xfm`, `qc_figures`, `templates`) + `repro/` + `_superseded/` | HF dataset `feliciacyntia/abide1-smri-prepro` |
| checkpoint `.pt` (pilot, pilot_v2, bnfix, full_v2), cache embedding `.npy` pilot, atlas Harvard-Oxford grid 2 mm | HF model `feliciacyntia/abide1-asd-results` |
| MRI mentah ABIDE, citra 149 subjek test, venv, file antara `data/work/` | tidak dibagikan |
Hasil pembanding sudah ada di `results\` (prediksi per subjek, tabel, split) — tidak perlu melatih ulang untuk membandingkan.
Untuk eksperimen baru pakai `--results-dir` folder baru, kunci config-nya dulu, dan bandingkan dengan
`results/full_v2` (fusion fenotipik+ROI: acc 0,628 / AUC 0,664) memakai prediksi pada subjek yang sama.

## Menjalankan (perintah yang dipakai pemilik TA)
```powershell
# run data penuh (config terkunci) dan analisis
.\.venv-model\Scripts\python.exe src\run_pilot_v2.py --subset all --locked results\pilot200_v2\locked_config_kandidat.json --budget-scale 5 --results-dir $env:TA06_RESULTS_ROOT\full_v2_ulang
.\.venv-model\Scripts\python.exe src\analysis_full_v2.py
# v3 tabel (config: results\v3\locked_config_v3.json + .sha256)
.\.venv-model\Scripts\python.exe src\run_v3_tabular.py --results-dir $env:TA06_RESULTS_ROOT\v3_ulang
```
Semua runner: `--resume`, file `STOP` di folder hasil atau `--stop-at HH:MM` untuk berhenti rapi, cegah sleep,
guard RAM (butuh ≥ 4 GiB RAM bebas saat mulai). `requirements-kaggle.txt` hanya untuk preprocessing.

## Aturan eksperimen (wajib dijaga)
- Semua yang dipelajari (scaling, imputasi, ComBat, PCA, seleksi fitur, SMOTE) di-fit **hanya di data latih tiap fold**.
- **Bukan fitur**: SITE_ID, DX_GROUP, ADOS, ADI-R, SRS, SCQ, AQ, DSM. Harmonisasi memakai kovariat usia & jenis kelamin, tanpa diagnosis.
- Model, hyperparameter, threshold, dan anggota ensemble dipilih **dari inner validation**, bukan skor test fold.
- Config dikunci (hash) sebelum run; tidak diubah setelah melihat hasil. Semua percobaan dicatat, termasuk yang gagal.
- 149 subjek test held-out tidak disentuh sampai evaluasi final disetujui.
- Model yang dikeluarkan saat pilot (ConvNeXt, DINOv2, SFCN, XGBoost) tidak dimasukkan kembali.

## Struktur
```
src/          kode (preprocessing, runner pilot/full/v3, analisis, upload HF)
notebook/     01–08 (EDA → cleaning → audit → preprocessing → laporan modeling → ringkasan → diagnosis)
data/         folds.csv, fenotipik bersih, log cleaning/QC, metadata preprocessing (citra di Hugging Face)
results/      pilot200, pilot200_v2 (+ _bnfix), full_v2 (hasil utama), v3 — prediksi per unit, tabel, gambar (tanpa checkpoint .pt)
insight/      catatan & rekomendasi
```

## Lisensi data
ABIDE I dibagikan dengan lisensi CC BY-NC-SA; data turunan di Hugging Face memakai lisensi yang sama.
Sitasi: Di Martino A. et al. (2014), *Molecular Psychiatry* 19, 659–667.
