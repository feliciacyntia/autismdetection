# Laporan eksperimen modeling — pilot200 (2026-10-27)

> **⚠ PILOT n=200 — bukan hasil akhir.**

Hasil: `C:\ta_local\results\pilot200_v2` · config_hash `27c4fa66930f2ece` · preprocessing `682404ae0f5d2624` · `folds.csv` sha256 `48f8e8bf9b8df9b7…` · status run **complete** (53 eksperimen, 5 fold) · threshold tetap 0.5.

Kohort: 200 subjek dev (95 ASD / 105 TD) dari 20 site. Test held-out (149 subjek) tidak dipakai dan diverifikasi tidak muncul di hasil mana pun.

## Ringkasan (pooled OOF test, 95 % CI bootstrap 2000×)

| eksperimen | modalitas | BA | AUC | F1 macro |
|---|---|---|---|---|
| majority | baseline | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.344 [0.313, 0.373] |
| mlp_tab | Fenotipik | 0.543 [0.474, 0.612] | 0.576 [0.494, 0.654] | 0.540 [0.470, 0.609] |
| tabm | Fenotipik | 0.584 [0.512, 0.652] | 0.557 [0.474, 0.637] | 0.579 [0.507, 0.646] |
| logreg | Fenotipik | 0.558 [0.490, 0.625] | 0.564 [0.480, 0.645] | 0.558 [0.489, 0.624] |
| roi_tabm_cb | ROI | 0.521 [0.450, 0.590] | 0.556 [0.476, 0.632] | 0.520 [0.449, 0.585] |
| roi_logreg_cb | ROI | 0.550 [0.481, 0.615] | 0.546 [0.470, 0.624] | 0.550 [0.480, 0.614] |
| resnet18 | Citra 2.5D | 0.515 [0.457, 0.575] | 0.489 [0.409, 0.571] | 0.476 [0.406, 0.544] |
| effnet_b0 | Citra 2.5D | 0.531 [0.466, 0.597] | 0.522 [0.442, 0.599] | 0.521 [0.449, 0.590] |
| scratch2d | Citra 2.5D | 0.516 [0.448, 0.589] | 0.516 [0.430, 0.597] | 0.515 [0.445, 0.587] |
| scratch3d | Citra 3D | 0.449 [0.387, 0.517] | 0.461 [0.384, 0.539] | 0.443 [0.378, 0.511] |
| scratch2d [lama: bug BN] | Citra 2.5D | 0.511 [0.442, 0.576] | 0.502 [0.421, 0.582] | 0.498 [0.424, 0.567] |
| scratch3d [lama: bug BN] | Citra 3D | 0.489 [0.422, 0.556] | 0.480 [0.402, 0.564] | 0.486 [0.415, 0.552] |
| PR · F1_stack_logreg | Fusion | 0.569 [0.495, 0.637] | 0.604 [0.523, 0.680] | 0.569 [0.495, 0.635] |
| P25 · F1_stack_logreg | Fusion | 0.522 [0.452, 0.590] | 0.539 [0.456, 0.616] | 0.521 [0.449, 0.589] |
| P3 · F1_stack_logreg | Fusion | 0.522 [0.452, 0.590] | 0.539 [0.455, 0.615] | 0.521 [0.449, 0.589] |
| PRI · F1_stack_logreg | Fusion | 0.573 [0.506, 0.637] | 0.587 [0.505, 0.660] | 0.569 [0.497, 0.634] |
| PRI · F1_mean_prob | Fusion | 0.598 [0.530, 0.667] | 0.613 [0.531, 0.691] | 0.599 [0.529, 0.667] |
| PRI · F2_pca16_logreg | Fusion | 0.572 [0.501, 0.641] | 0.576 [0.492, 0.654] | 0.571 [0.498, 0.640] |
| ENS · ENS_mean_top3 | Fusion | 0.549 [0.480, 0.615] | 0.556 [0.478, 0.636] | 0.549 [0.479, 0.615] |

Baris bertanda *[lama: bug BN]* adalah hasil v2 yang **tidak valid** (bug BatchNorm); baris tanpa tanda untuk model tsb adalah rerun setelah bugfix. Fusion yang memakai anggota citra dari run lama ditandai di tabel 07D/12b.

## Perbaikan setelah pilot: bugfix BN & threshold Youden

**Bugfix BN** (CNN dari nol, threshold 0,5):

| model | versi | BA | AUC | F1 macro |
|---|---|---|---|---|
| scratch2d | lama (bug BN) | 0.511 [0.442, 0.576] | 0.502 [0.421, 0.582] | 0.498 [0.424, 0.567] |
| scratch2d | bugfix BN | 0.516 [0.448, 0.589] | 0.516 [0.430, 0.597] | 0.515 [0.445, 0.587] |
| scratch2d | Δ (bugfix − lama) | +0.006 [-0.116, +0.132] | +0.015 [-0.136, +0.157] | +0.017 [-0.110, +0.147] |
| scratch3d | lama (bug BN) | 0.489 [0.422, 0.556] | 0.480 [0.402, 0.564] | 0.486 [0.415, 0.552] |
| scratch3d | bugfix BN | 0.449 [0.387, 0.517] | 0.461 [0.384, 0.539] | 0.443 [0.378, 0.511] |
| scratch3d | Δ (bugfix − lama) | -0.041 [-0.125, +0.045] | -0.019 [-0.107, +0.068] | -0.042 [-0.128, +0.044] |

**Threshold Youden** (dipilih dari inner validation tiap fold, diterapkan ke test fold):

| eksperimen | BA @0,5 | BA @Youden | ΔBA [95% CI] | F1m @0,5 | F1m @Youden | Sens/Spes @0,5 → @Youden |
|---|---|---|---|---|---|---|
| mlp_tab | 0.543 [0.474, 0.612] | 0.520 [0.451, 0.587] | -0.023 [-0.081, +0.035] | 0.540 [0.470, 0.609] | 0.512 [0.440, 0.578] | 0.60 / 0.49 → 0.62 / 0.42 |
| tabm | 0.584 [0.512, 0.652] | 0.500 [0.433, 0.566] | -0.084 [-0.144, -0.025] | 0.579 [0.507, 0.646] | 0.475 [0.404, 0.545] | 0.66 / 0.50 → 0.69 / 0.30 |
| logreg | 0.558 [0.490, 0.625] | 0.534 [0.472, 0.595] | -0.025 [-0.078, +0.028] | 0.558 [0.489, 0.624] | 0.513 [0.442, 0.582] | 0.53 / 0.59 → 0.31 / 0.76 |
| roi_tabm_cb | 0.521 [0.450, 0.590] | 0.484 [0.419, 0.552] | -0.037 [-0.095, +0.030] | 0.520 [0.449, 0.585] | 0.484 [0.415, 0.550] | 0.55 / 0.50 → 0.47 / 0.50 |
| roi_logreg_cb | 0.550 [0.481, 0.615] | 0.511 [0.441, 0.576] | -0.039 [-0.094, +0.018] | 0.550 [0.480, 0.614] | 0.510 [0.439, 0.574] | 0.55 / 0.55 → 0.54 / 0.49 |
| resnet18 | 0.515 [0.457, 0.575] | 0.464 [0.396, 0.534] | -0.051 [-0.140, +0.035] | 0.476 [0.406, 0.544] | 0.464 [0.395, 0.532] | 0.22 / 0.81 → 0.44 / 0.49 |
| effnet_b0 | 0.531 [0.466, 0.597] | 0.525 [0.461, 0.586] | -0.006 [-0.073, +0.062] | 0.521 [0.449, 0.590] | 0.500 [0.428, 0.566] | 0.36 / 0.70 → 0.73 / 0.32 |
| scratch2d | 0.516 [0.448, 0.589] | 0.502 [0.435, 0.567] | -0.015 [-0.097, +0.066] | 0.515 [0.445, 0.587] | 0.500 [0.432, 0.565] | 0.44 / 0.59 → 0.54 / 0.47 |
| scratch3d | 0.449 [0.387, 0.517] | 0.480 [0.416, 0.548] | +0.032 [-0.054, +0.114] | 0.443 [0.378, 0.511] | 0.478 [0.411, 0.545] | 0.33 / 0.57 → 0.39 / 0.57 |
| PR · F1_mean_prob | 0.583 [0.509, 0.650] | 0.573 [0.506, 0.639] | -0.010 [-0.063, +0.050] | 0.580 [0.504, 0.645] | 0.569 [0.499, 0.635] | 0.64 / 0.52 → 0.44 / 0.70 |
| PR · F1_stack_logreg | 0.569 [0.495, 0.637] | 0.514 [0.445, 0.579] | -0.055 [-0.111, +0.004] | 0.569 [0.495, 0.635] | 0.510 [0.435, 0.576] | 0.55 / 0.59 → 0.40 / 0.63 |
| PR · F2_pca16_logreg | 0.578 [0.508, 0.646] | 0.528 [0.464, 0.594] | -0.050 [-0.104, +0.006] | 0.578 [0.506, 0.645] | 0.522 [0.453, 0.590] | 0.54 / 0.62 → 0.39 / 0.67 |
| P25 · F1_mean_prob | 0.576 [0.505, 0.645] | 0.515 [0.448, 0.581] | -0.061 [-0.120, -0.002] | 0.576 [0.502, 0.643] | 0.513 [0.443, 0.578] | 0.51 / 0.65 → 0.42 / 0.61 |
| P25 · F1_stack_logreg | 0.522 [0.452, 0.590] | 0.521 [0.455, 0.589] | -0.000 [-0.047, +0.049] | 0.521 [0.449, 0.589] | 0.520 [0.451, 0.585] | 0.45 / 0.59 → 0.55 / 0.50 |
| P25 · F2_pca16_logreg | 0.532 [0.462, 0.599] | 0.540 [0.471, 0.603] | +0.009 [-0.062, +0.075] | 0.531 [0.461, 0.597] | 0.538 [0.465, 0.601] | 0.46 / 0.60 → 0.44 / 0.64 |
| P3 · F1_mean_prob | 0.544 [0.474, 0.615] | 0.521 [0.453, 0.585] | -0.024 [-0.083, +0.038] | 0.544 [0.472, 0.613] | 0.511 [0.441, 0.575] | 0.53 / 0.56 → 0.63 / 0.41 |
| P3 · F1_stack_logreg | 0.522 [0.452, 0.590] | 0.521 [0.455, 0.589] | -0.000 [-0.047, +0.049] | 0.521 [0.449, 0.589] | 0.520 [0.451, 0.585] | 0.45 / 0.59 → 0.55 / 0.50 |
| P3 · F2_pca16_logreg | 0.488 [0.421, 0.556] | 0.468 [0.399, 0.537] | -0.019 [-0.092, +0.049] | 0.487 [0.419, 0.557] | 0.462 [0.392, 0.531] | 0.44 / 0.53 → 0.34 / 0.60 |
| PRI · F1_mean_prob | 0.598 [0.530, 0.667] | 0.577 [0.506, 0.645] | -0.022 [-0.081, +0.035] | 0.599 [0.529, 0.667] | 0.575 [0.503, 0.644] | 0.57 / 0.63 → 0.61 / 0.54 |
| PRI · F1_stack_logreg | 0.573 [0.506, 0.637] | 0.575 [0.514, 0.637] | +0.001 [-0.048, +0.052] | 0.569 [0.497, 0.634] | 0.561 [0.490, 0.627] | 0.44 / 0.70 → 0.37 / 0.78 |
| PRI · F2_pca16_logreg | 0.572 [0.501, 0.641] | 0.516 [0.443, 0.583] | -0.056 [-0.110, -0.005] | 0.571 [0.498, 0.640] | 0.514 [0.439, 0.581] | 0.51 / 0.64 → 0.43 / 0.60 |
| ENS · ENS_mean_top3 | 0.549 [0.480, 0.615] | 0.548 [0.482, 0.612] | -0.001 [-0.054, +0.051] | 0.549 [0.479, 0.615] | 0.532 [0.460, 0.601] | 0.53 / 0.57 → 0.71 / 0.39 |

> Dari 50 perbandingan sebelum/sesudah (bugfix BN: BA/AUC/F1; Youden: BA/F1), 0 naik secara signifikan, 6 turun secara signifikan (Youden vs 0,5: tabm BA; Youden vs 0,5: tabm F1m; Youden vs 0,5: P25|F1_mean_prob BA; Youden vs 0,5: P25|F1_mean_prob F1m; Youden vs 0,5: PRI|F2_pca16_logreg BA; Youden vs 0,5: PRI|F2_pca16_logreg F1m), sisanya CI 95 % bootstrap berpasangan mencakup 0 (tanpa koreksi perbandingan berganda). **Tidak ada perbaikan yang signifikan secara statistik pada n = 200.** Threshold Youden menurunkan BA pada 19 dari 22 eksperimen: threshold yang dipilih dari ±32 subjek inner validation per fold tidak stabil (SD antar fold besar), sehingga pada pilot tidak memperbaiki klasifikasi. Perubahan threshold hanya menggeser trade-off sensitivitas–spesifisitas; AUC tetap.

## Kandidat terkunci (seleksi inner validation, bukan skor test)

- **pheno**: mlp_tab, tabm, logreg
- **roi**: roi_tabm_cb, roi_logreg_cb
- **img25d_pretrained**: resnet18, effnet_b0
- **img25d_dari_nol**: scratch2d
- **img3d_dari_nol**: scratch3d
- Fusion: utama `F1_stack_logreg`, pembanding F1_mean_prob, F2_pca16_logreg

## Interpretasi & batas validasi

- **Ukuran sampel.** Pilot n = 200 (±40 subjek test per fold): CI 95 % lebar dan saling tumpang tindih; selisih antar model di bawah ±0,05 belum bermakna.
- **Tidak ada model yang jelas di atas acak secara kuat.** BA tertinggi di test adalah PRI · F1_mean_prob (0.598 [0.530, 0.667]); ini dilaporkan, tetapi **tidak** dipakai untuk memilih model.
- **Seleksi via inner validation tidak stabil** pada sampel kecil (±32 subjek inner val per fold): model yang terpilih tidak selalu yang terbaik di test, dan sebaliknya. Ini konsekuensi yang diterima demi mencegah kebocoran.
- **Model citra end-to-end belum belajar sinyal diagnosis** (fold sudah belajar: effnet_b0 1/5, resnet18 1/5, scratch2d [lama: bug BN] 1/5, dinov2_lp n/a (probe LogReg pada fitur beku), dinov2_head 0/5, convnextv2_t 1/5, scratch3d [lama: bug BN] 0/5, sfcn_lp n/a (probe LogReg pada fitur beku), sfcn_head 4/5, scratch2d 1/5, scratch3d 0/5). Overfit test menemukan bug pada scratch2d, scratch3d (running statistics BatchNorm basi saat eval). Bugfix (kalibrasi ulang statistik BN dari inner-train tanpa augmentasi) diterapkan, overfit test lulus, dan model dilatih ulang; hasil lama ditandai tidak valid. Fusion P25/P3/PRI masih memakai anggota citra dari run lama. Penyebab per model: tabel `12a_penyebab_kegagalan_citra`.
- **Registrasi hanya afin, tanpa segmentasi jaringan**: fitur ROI adalah intensitas rata-rata/SD per region atlas, bukan volume/ketebalan; variasi anatomi lokal tidak dinormalisasi.
- **Efek site kuat**: embedding citra mengandung informasi site (BA tebak site jauh di atas acak); ComBat menghapusnya, tetapi akurasi diagnosis tidak ikut naik. Generalisasi ke site baru (LOSO) belum diuji.
- **Threshold**: dilaporkan dua versi — 0,5 tetap dan Youden per fold dari inner validation (aturan dikunci untuk data penuh). Youden hanya menggeser trade-off sensitivitas–spesifisitas; AUC tidak berubah. Inner validation pilot kecil (±32 subjek/fold), sehingga threshold Youden sendiri tidak stabil antar fold.
- **SFCN (brain age UK Biobank) dikeluarkan** berdasarkan cek kewajaran pada subjek ≥ 18 tahun (n = 67): r = -0.13, SD prediksi 0.61 th (aturan: |r| ≥ 0,3 dan SD ≥ 2 th). Rentang usia model 42–82 th, sedangkan median usia subjek 23 th.
- **Signifikansi**: hanya perbandingan sebelum/sesudah (bugfix, Youden) yang diuji dengan bootstrap berpasangan; perbandingan antar model belum diuji (mis. DeLong / permutation test) dan bersifat deskriptif.
- **Test held-out (149 subjek) belum dipakai**; evaluasi akhir dilakukan sekali setelah konfigurasi data penuh dikunci.

## File

- Tabel: `report/tables/*.csv`, `report/tabel_laporan.xlsx`
- Grafik: `report/figures/*.png`
- `report/environment.json`, `report/run_validation.json`, `report/classification_reports.json`, `report/classification_reports_youden.json`