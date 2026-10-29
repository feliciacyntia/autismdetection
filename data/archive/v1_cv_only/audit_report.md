# Audit Report — Notebook 05a

- Sumber: `dataset/cleaned_dataset/cleaned_master_dataset.csv`
- SHA-256 sumber: `ad0ef352bfdf18b28d53720868131a017de6d88dacb937ea840a0f514d4f24ec`
- Subjek: 1004 (ASD 469, TD 535), situs: 20
- Encoding DX_GROUP: recode Notebook 04 (1=ASD, 0=TD); label: ASD = 1, TD = 0
- Seed: 42; N_FOLDS: 5; REPEAT_SEEDS: [43, 44, 45]; LOSO_MIN_PER_CLASS: 10
- Notebook ini read-only: tidak ada baris dibuang, tidak ada nilai diubah, tidak ada transformasi di-fit.

## Temuan

- **[Data]** DX_GROUP di file berisi [0, 1], bukan {1, 2}: sudah di-recode Notebook 04 (recode Notebook 04 (1=ASD, 0=TD)). label = DX_GROUP; DX_GROUP asli tidak diubah.
- **[Data]** Verifikasi silang DX_GROUP vs Phenotypic_V1_0b.csv: LOLOS (tidak ditemukan=0, konflik=0).
- **[Integritas]** 72 nilai FIQ tidak bulat, terkonsentrasi di {'UM_1': 47, 'UM_2': 18, 'OHSU': 7}. Kemungkinan berasal dari skor sumber situs (dibiarkan apa adanya; perlu dicek ke dokumentasi situs).
- **[Integritas]** 29 baris memiliki isi fenotipik identik dengan SUB_ID berbeda. Ini wajar pada data diskret (umur bulat, FIQ NaN); citranya dicek di CELL 3c.
- **[Integritas]** 45 outlier umur (IQR global), terkonsentrasi di {'SBL': 14, 'MAX_MUN': 10, 'CALTECH': 10, 'USM': 6, 'CMU': 2}. Umumnya mencerminkan rentang rekrutmen situs, bukan kesalahan data.
- **[Citra identik]** Hash file hanya menemukan 0 subjek duplikat, hash voxel 2. group_id memakai hash voxel.
- **[Missing]** FIQ NaN = 71 (7.1%). Asal mentah: {'sentinel -9999': 36, 'kosong di sumber': 35}. Tidak ada nilai salah format.
- **[Missing]** Missing FIQ berasosiasi kuat dengan situs (p=1.64e-153). 85% NaN berasal dari situs dengan ≥50% missing ['LEUVEN_2', 'SBL'], artinya FIQ tidak diukur/dilaporkan di situs tersebut. Sisanya sporadis di ['OHSU', 'OLIN', 'UM_1', 'MAX_MUN', 'CALTECH'].
- **[Missing]** Asosiasi missing FIQ dengan label: p=1.000.
- **[Situs]** Situs dengan < 10 subjek pada salah satu kelas: ['SDSU']
- **[Konfounding]** AGE_AT_SCAN: ASD=17.70 vs TD=17.28, tidak berbeda signifikan (p=7.89e-01, effect=0.010).
- **[Konfounding]** FIQ: ASD=106.26 vs TD=111.43, berbeda signifikan (p=2.88e-07, effect=-0.194).
- **[Konfounding]** SEX (% perempuan): ASD=13.01 vs TD=17.94, berbeda signifikan (p=3.64e-02, effect=1.463).

## Keputusan

- **[Integritas]** Outlier hanya ditandai di audit_flags.csv dan tidak dibuang. Nilainya masih dalam rentang valid, jadi tidak ada bukti data salah.
- **[Citra identik]** Tidak ada subjek yang dihapus. Subjek satu group_id selalu ditempatkan di fold yang sama (StratifiedGroupKFold). Penghapusan, jika perlu, diputuskan di 05b dan dicatat di drop_log.csv.
- **[Missing]** Tidak ada baris yang dibuang karena missing. Imputasi FIQ dilakukan di 05b, di-fit hanya pada data latih tiap fold.
- **[Situs]** SITE_ID dipakai apa adanya (20 nilai) untuk stratifikasi dan LOSO.
- **[Konfounding]** Konfounder hanya dilaporkan. Tidak ada matching atau pembuangan subjek di notebook ini.
- **[Folds]** fold_strat: StratifiedGroupKFold 5-fold, groups=group_id, strata label×SITE_ID, seed 42; repeated_fold_1..3 memakai seed [43, 44, 45]. Subjek strata fallback: 0.
- **[Folds]** LOSO: 19 situs eligible; tidak eligible (< 10 per kelas): ['SDSU']. Situs tidak eligible tetap ikut sebagai data latih, tetapi tidak dilaporkan sebagai test site tunggal.

## Perlu ditinjau manual

- **[Citra identik]** group_id 50279: SUB_ID [50279, 50286] (UM_1; label ASD) memiliki citra identik walau hash file .nii.gz berbeda. Fenotipik berbeda: kemungkinan salah satu baris salah pasangan citra.
- **[Missing]** 43 dari 71 subjek FIQ NaN memiliki VIQ dan PIQ valid. FIQ tidak bisa dihitung sebagai rata-rata VIQ/PIQ tanpa tabel norma tes, jadi tidak dipakai otomatis. Putuskan di 05b apakah VIQ/PIQ boleh dipakai sebagai sumber informasi.
- **[Missing]** FIQ_missing hampir identik dengan indikator situs ['LEUVEN_2', 'SBL']. Jika dipakai sebagai fitur, ia bisa menjadi proksi SITE_ID (shortcut). Uji di 05b sebelum dipakai.
- **[Situs]** 20 SITE_ID vs 17 situs standar. Kandidat pecahan: {'LEUVEN': ['LEUVEN_1', 'LEUVEN_2'], 'UCLA': ['UCLA_1', 'UCLA_2'], 'UM': ['UM_1', 'UM_2']}. Di ABIDE I, sampel ini adalah kohort/sampel berbeda dari institusi yang sama, dan protokolnya bisa berbeda (lihat shape/voxel). Tidak digabung otomatis. Putuskan sebelum ComBat dan LOSO.
- **[Konfounding]** FIQ berbeda antara ASD dan TD. Perbedaan IQ sebagian merupakan karakteristik populasi ASD. Dokumentasikan bahwa model fenotipik dapat memanfaatkan sinyal ini, dan laporkan hasil dengan/tanpa FIQ (ablasi) di tahap pemodelan.

## Tabel

### Verifikasi silang DX_GROUP bersih × mentah

| DX_GROUP bersih | 1 | 2 |
|---|---|---|
| 0 | 0 | 535 |
| 1 | 469 | 0 |

### Tipe data

| kolom | dtype | jenis | diharapkan | ok | n_unik | n_NaN | n_desimal_non_bulat |
|---|---|---|---|---|---|---|---|
| SUB_ID | int64 | integer | integer | True | 1004 | 0 |  |
| SITE_ID | object | string | string | True | 20 | 0 |  |
| DX_GROUP | int64 | integer | integer | True | 2 | 0 |  |
| AGE_AT_SCAN | float64 | float | float | True | 655 | 0 | 813.0 |
| SEX | int64 | integer | integer | True | 2 | 0 |  |
| FIQ | float64 | float | float | True | 120 | 71 | 72.0 |
| HANDEDNESS_CATEGORY | object | string | string | True | 3 | 14 |  |
| FIQ_missing | int64 | integer | integer | True | 2 | 0 |  |
| smri_path | object | string | string | True | 1004 | 0 |  |

### Rentang AGE dan FIQ

| kolom | min | q1 | median | q3 | max | rentang_valid | n_di_luar_rentang | n_NaN |
|---|---|---|---|---|---|---|---|---|
| AGE_AT_SCAN | 6.47 | 11.942 | 15.0 | 21.0 | 64.0 | [5, 70] | 0 | 0 |
| FIQ | 41.0 | 100.0 | 109.0 | 119.0 | 148.0 | [40, 160] | 0 | 71 |

### Outlier (ditandai, tidak dibuang)

| kolom | IQR (k=1.5) | |z|>3.0 | robust |z|>3.5 per situs |
|---|---|---|---|
| AGE_AT_SCAN | 45 | 17 | 14 |
| FIQ | 7 | 2 | 2 |

### Kelompok citra identik (group_id)

| index | group_id | SUB_ID | SITE_ID | label | AGE_AT_SCAN | FIQ | file_sha256 | voxel_sha256 | qc_decision |
|---|---|---|---|---|---|---|---|---|---|
| 743 | 50279 | 50279 | UM_1 | ASD | 12.2 | 87.5 | d27c8eed4277 | 0fd0c2564956 | REVIEW |
| 748 | 50279 | 50286 | UM_1 | ASD | 11.2 | 130.5 | 256b26ce03d1 | 0fd0c2564956 | REVIEW |

### Missing per kolom

| kolom | n_NaN | pct_NaN |
|---|---|---|
| SUB_ID | 0 | 0.0 |
| SITE_ID | 0 | 0.0 |
| DX_GROUP | 0 | 0.0 |
| AGE_AT_SCAN | 0 | 0.0 |
| SEX | 0 | 0.0 |
| FIQ | 71 | 7.07 |
| HANDEDNESS_CATEGORY | 14 | 1.39 |
| FIQ_missing | 0 | 0.0 |
| smri_path | 0 | 0.0 |

### Asal-usul FIQ NaN

| fiq_raw_origin | FIQ NaN | FIQ terisi |
|---|---|---|
| kosong di sumber | 35 | 0 |
| numerik valid | 0 | 933 |
| sentinel -9999 | 36 | 0 |

### FIQ NaN: VIQ/PIQ tersedia

| SITE_ID | n_FIQ_NaN | n_VIQ_PIQ_tersedia | FIQ_TEST_TYPE |
|---|---|---|---|
| CALTECH | 1 | 0 | {'WASI': 1} |
| LEUVEN_2 | 35 | 35 | {'WISC_III_DUTCH': 35} |
| MAX_MUN | 2 | 0 | {'-9999': 2} |
| OHSU | 2 | 0 | {'WISC_IV_4_SUBTESTS': 2} |
| OLIN | 2 | 0 | {'-9999': 2} |
| SBL | 25 | 8 | {'(kosong)': 25} |
| UM_1 | 4 | 0 | {'(kosong)': 4} |

### Missing per situs (situs dengan NaN)

| SITE_ID | n | n_FIQ_NaN | pct_FIQ_NaN | n_HAND_NaN |
|---|---|---|---|---|
| LEUVEN_2 | 35 | 35 | 100.0 | 0 |
| SBL | 30 | 25 | 83.3 | 0 |
| OHSU | 28 | 2 | 7.1 | 0 |
| OLIN | 29 | 2 | 6.9 | 0 |
| UM_1 | 91 | 4 | 4.4 | 8 |
| MAX_MUN | 52 | 2 | 3.8 | 0 |
| CALTECH | 38 | 1 | 2.6 | 0 |
| PITT | 56 | 0 | 0.0 | 1 |
| NYU | 183 | 0 | 0.0 | 4 |
| UM_2 | 34 | 0 | 0.0 | 1 |

### Uji asosiasi missing

| uji | chi2 | dof | p | %sel_expected<5 | fisher_p |
|---|---|---|---|---|---|
| FIQ NaN × SITE_ID | 781.729 | 19 | 1.64e-153 | 42.5 |  |
| FIQ NaN × label | 0.0 | 1 | 1.0 | 0.0 | 1.0 |
| HANDEDNESS NaN × SITE_ID | 46.748 | 19 | 3.88e-04 | 50.0 |  |
| HANDEDNESS NaN × label | 0.0 | 1 | 1.0 | 0.0 | 1.0 |

### Profil situs

| SITE_ID | n | n_ASD | n_TD | age_mean | age_sd | age_min | age_max | FIQ_mean | pct_FIQ_missing | pct_female | pct_REVIEW | shape_modus | voxel_modus_mm | n_protokol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CALTECH | 38 | 19 | 19 | 28.16 | 10.64 | 17.0 | 56.2 | 111.27 | 2.6 | 21.1 | 94.7 | (176, 256, 256) | 1.0x1.0x1.0 | 3 |
| CMU | 27 | 14 | 13 | 26.59 | 5.69 | 19.0 | 40.0 | 114.56 | 0.0 | 22.2 | 66.7 | (176, 256, 256) | 1.0x1.0x1.0 | 13 |
| KKI | 43 | 16 | 27 | 10.26 | 1.36 | 8.2 | 12.77 | 107.58 | 0.0 | 23.3 | 16.3 | (256, 200, 256) | 1.0x1.0x1.0 | 3 |
| LEUVEN_1 | 29 | 14 | 15 | 22.59 | 3.55 | 18.0 | 32.0 | 112.21 | 0.0 | 0.0 | 6.9 | (256, 182, 256) | 0.98x1.2x0.98 | 1 |
| LEUVEN_2 | 35 | 15 | 20 | 14.16 | 1.42 | 12.1 | 16.9 |  | 100.0 | 22.9 | 0.0 | (256, 182, 256) | 0.98x1.2x0.98 | 1 |
| MAX_MUN | 52 | 20 | 32 | 27.81 | 11.33 | 7.0 | 58.0 | 110.84 | 3.8 | 13.5 | 40.4 | (160, 240, 256) | 1.0x1.0x1.0 | 4 |
| NYU | 183 | 79 | 104 | 15.28 | 6.59 | 6.47 | 39.1 | 111.07 | 0.0 | 20.2 | 4.9 | (128, 256, 256) | 1.33x1.0x1.0 | 1 |
| OHSU | 28 | 13 | 15 | 10.81 | 1.87 | 8.0 | 15.23 | 111.6 | 7.1 | 0.0 | 17.9 | (160, 239, 200) | 1.1x1.0x1.0 | 3 |
| OLIN | 29 | 14 | 15 | 17.17 | 3.44 | 10.0 | 24.0 | 114.19 | 6.9 | 17.2 | 17.2 | (208, 256, 176) | 1.0x1.0x1.0 | 1 |
| PITT | 56 | 29 | 27 | 18.84 | 6.92 | 9.33 | 35.2 | 110.46 | 0.0 | 14.3 | 19.6 | (176, 256, 256) | 1.05x1.05x1.05 | 1 |
| SBL | 30 | 15 | 15 | 34.37 | 8.6 | 20.0 | 64.0 | 109.2 | 83.3 | 0.0 | 0.0 | (256, 256, 170) | 1.0x1.0x1.0 | 1 |
| SDSU | 28 | 8 | 20 | 14.44 | 1.94 | 8.67 | 17.15 | 112.46 | 0.0 | 21.4 | 10.7 | (172, 256, 256) | 1.0x1.0x1.0 | 1 |
| STANFORD | 27 | 14 | 13 | 10.02 | 1.63 | 7.53 | 12.94 | 113.85 | 0.0 | 29.6 | 100.0 | (256, 132, 256) | 0.86x1.5x0.86 | 1 |
| TRINITY | 46 | 21 | 25 | 17.26 | 3.67 | 12.0 | 25.91 | 109.85 | 0.0 | 0.0 | 13.0 | (160, 256, 256) | 1.0x1.0x1.0 | 2 |
| UCLA_1 | 65 | 37 | 28 | 13.31 | 2.38 | 8.36 | 17.94 | 104.29 | 0.0 | 13.8 | 35.4 | (160, 240, 256) | 1.2x1.0x1.0 | 1 |
| UCLA_2 | 21 | 11 | 10 | 12.47 | 1.67 | 9.79 | 16.47 | 102.86 | 0.0 | 9.5 | 47.6 | (160, 240, 256) | 1.2x1.0x1.0 | 3 |
| UM_1 | 91 | 37 | 54 | 13.76 | 2.93 | 8.2 | 19.2 | 106.94 | 4.4 | 27.5 | 97.8 | (256, 256, 124) | 1.02x1.02x1.2 | 12 |
| UM_2 | 34 | 13 | 21 | 16.01 | 3.36 | 12.8 | 28.8 | 112.26 | 0.0 | 5.9 | 88.2 | (106, 256, 256) | 1.4x1.0x1.0 | 10 |
| USM | 89 | 54 | 35 | 22.09 | 7.77 | 8.77 | 50.22 | 105.45 | 0.0 | 0.0 | 98.9 | (160, 480, 512) | 1.2x0.5x0.5 | 4 |
| YALE | 53 | 26 | 27 | 12.88 | 2.81 | 7.0 | 17.83 | 100.3 | 0.0 | 30.2 | 39.6 | (160, 256, 256) | 1.0x1.0x1.0 | 3 |

### Kandidat situs terpecah

| SITE_ID | grup | n | n_ASD | n_TD | age_mean | FIQ_mean | shape_modus | voxel_modus_mm |
|---|---|---|---|---|---|---|---|---|
| LEUVEN_1 | LEUVEN | 29 | 14 | 15 | 22.59 | 112.21 | (256, 182, 256) | 0.98x1.2x0.98 |
| LEUVEN_2 | LEUVEN | 35 | 15 | 20 | 14.16 |  | (256, 182, 256) | 0.98x1.2x0.98 |
| UCLA_1 | UCLA | 65 | 37 | 28 | 13.31 | 104.29 | (160, 240, 256) | 1.2x1.0x1.0 |
| UCLA_2 | UCLA | 21 | 11 | 10 | 12.47 | 102.86 | (160, 240, 256) | 1.2x1.0x1.0 |
| UM_1 | UM | 91 | 37 | 54 | 13.76 | 106.94 | (256, 256, 124) | 1.02x1.02x1.2 |
| UM_2 | UM | 34 | 13 | 21 | 16.01 | 112.26 | (106, 256, 256) | 1.4x1.0x1.0 |

### Konfounding global

| variabel | n_ASD | n_TD | ASD_mean | TD_mean | ASD_median | TD_median | uji | statistik | p | effect |
|---|---|---|---|---|---|---|---|---|---|---|
| AGE_AT_SCAN | 469 | 535 | 17.702 | 17.285 | 15.0 | 14.943 | Mann-Whitney U | 126685.0 | 0.789 | 0.01 |
| FIQ | 436 | 497 | 106.258 | 111.434 | 106.0 | 111.0 | Mann-Whitney U | 87276.0 | 2.88e-07 | -0.194 |
| SEX (% perempuan) | 469 | 535 | 13.006 | 17.944 |  |  | Fisher exact | 1.463 | 0.036 | 1.463 |

### Konfounding per situs (signifikan BH)

| index | SITE_ID | variabel | n_ASD | n_TD | ASD_mean | TD_mean | effect | p_BH |
|---|---|---|---|---|---|---|---|---|
| 7 | KKI | FIQ | 16 | 27 | 97.688 | 113.444 | -0.502 | 0.025 |
| 19 | NYU | FIQ | 79 | 104 | 107.911 | 113.471 | -0.258 | 0.018 |
| 34 | SDSU | FIQ | 8 | 20 | 123.125 | 108.2 | 0.669 | 0.025 |
| 46 | UCLA_2 | FIQ | 11 | 10 | 93.455 | 113.2 | -0.773 | 0.018 |
| 55 | USM | FIQ | 54 | 35 | 99.981 | 113.886 | -0.474 | 0.003 |

### Rasio label per fold

| fold | fold_strat | repeated_fold_1 | repeated_fold_2 | repeated_fold_3 |
|---|---|---|---|---|
| 0 | n=201, ASD=0.463 | n=201, ASD=0.473 | n=201, ASD=0.463 | n=200, ASD=0.460 |
| 1 | n=202, ASD=0.455 | n=201, ASD=0.473 | n=202, ASD=0.470 | n=202, ASD=0.455 |
| 2 | n=201, ASD=0.468 | n=202, ASD=0.475 | n=201, ASD=0.458 | n=200, ASD=0.465 |
| 3 | n=200, ASD=0.490 | n=200, ASD=0.450 | n=200, ASD=0.480 | n=201, ASD=0.493 |
| 4 | n=200, ASD=0.460 | n=200, ASD=0.465 | n=200, ASD=0.465 | n=201, ASD=0.463 |
