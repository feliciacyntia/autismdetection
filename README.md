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
- **Citra hasil preprocessing (853 dev)**: Hugging Face dataset `abide1-smri-preproc-05c` (link diisi setelah upload) —
  `vol/{SUB_ID}.npy` float32 (1, 96, 112, 96), `mask/`, `nii/`, `xfm/`, `qc_figures/`, `templates/`, `SHA256SUMS`.
  Letakkan isinya di `C:\ta_local\data\smri_preproc\` (lokasi default kode) atau set env var `TA06_IMAGE_ROOT` ke folder tsb.
- **Tidak disertakan**: citra mentah ABIDE (unduh sendiri dari ABIDE/INDI setelah menyetujui ketentuannya) dan
  **149 subjek test held-out** (disimpan pemilik TA untuk evaluasi akhir sekali).

## Preprocessing sMRI (05c-v2.1, fingerprint `682404ae0f5d2624`)
Reorient → N4 → HD-BET (GPU, deterministik) → registrasi afin ANTs ke MNI152 1 mm → 2 mm → crop 96 × 112 × 96 → z-score di mask.
Kode: `src/smri_preprocessing.py`, `src/run_05c.py`; notebook `05c_preprocessing_smri.ipynb`.

## Menjalankan
```powershell
# environment utama (Python 3.12.3, torch 2.5.1+cu121)
python -m venv .venv-model; .\.venv-model\Scripts\pip install -r requirements-kaggle.txt -c constraints-model.txt
# TabPFN (venv terpisah, tabpfn 2.2.1 = bobot v2)
python -m venv .venv-tabpfn; .\.venv-tabpfn\Scripts\pip install -r constraints-tabpfn.txt

# run data penuh (config terkunci) dan analisis
.\.venv-model\Scripts\python.exe src\run_pilot_v2.py --subset all --locked results\pilot200_v2\locked_config_kandidat.json --budget-scale 5 --results-dir C:\ta_local\results\full_v2
.\.venv-model\Scripts\python.exe src\analysis_full_v2.py
# v3 tabel (config: results\v3\locked_config_v3.json + .sha256)
.\.venv-model\Scripts\python.exe src\run_v3_tabular.py
```
Semua runner: `--resume`, file `STOP` di folder hasil atau `--stop-at HH:MM` untuk berhenti rapi, cegah sleep, guard RAM 4 GiB.

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
