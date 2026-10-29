# Pheno Verification Summary — Notebook 05b

- Sumber: `dataset/cleaned_dataset/cleaned_master_dataset.csv` (SHA-256 `ad0ef352bfdf18b28d53720868131a017de6d88dacb937ea840a0f514d4f24ec`, tidak berubah)
- Folds: `data/folds.csv` (SHA-256 `48f8e8bf9b8df9b7ce96808dda4f2c450c084ac7de22fad46ff89cfe0fe285b8`, tidak berubah)
- scikit-learn 1.7.0, pandas 2.3.1, seed 42
- Subjek: 1002 (ASD 467, TD 535); dev 853, test 149
- Eksklusi 05a: [50279, 50286] (exclusions.csv); dikeluarkan 05b: 0
- Demonstrasi fit hanya pada dev; subjek test tidak dipakai untuk fit maupun validation
- Verifikasi silang DX_GROUP vs data mentah (dari audit_report.md): LOLOS (tidak ditemukan=0, konflik=0)
- DX_GROUP × label: one-to-one (1→1 ASD, 0→0 TD)

## Fitur

- Terpilih: ['AGE_AT_SCAN', 'SEX', 'FIQ', 'HANDEDNESS_CATEGORY']
- Fitur akhir pipeline tree (7): ['AGE_AT_SCAN', 'FIQ', 'SEX_1', 'SEX_2', 'HANDEDNESS_CATEGORY_Ambi', 'HANDEDNESS_CATEGORY_L', 'HANDEDNESS_CATEGORY_R']
- Fitur akhir pipeline linear/NN (7): ['AGE_AT_SCAN', 'FIQ', 'SEX_1', 'SEX_2', 'HANDEDNESS_CATEGORY_Ambi', 'HANDEDNESS_CATEGORY_L', 'HANDEDNESS_CATEGORY_R']
- Imputasi kategori: most_frequent; numerik: median train per fold

## Keputusan missing value

| fitur | n_missing | pct_missing | keputusan | alasan |
|---|---|---|---|---|
| AGE_AT_SCAN | 0 | 0.0 | dipakai | lengkap (0 NaN) dan lolos cek validitas |
| SEX | 0 | 0.0 | dipakai | lengkap (0 NaN) dan lolos cek validitas |
| FIQ | 71 | 7.09 | dipakai (imputasi median train per fold) | missing 7.1%; imputasi median di-fit pada train tiap fold. Missing terkonsentrasi di ['LEUVEN_2', 'SBL'] (struktural per situs): subjek situs ini mendapat median train, sehingga FIQ mereka tidak informatif. Audit 05a menandai FIQ sebagai konfounder; laporkan ablasi dengan/tanpa FIQ. |
| FIQ_missing | 0 | 0.0 | tidak dipakai di eksperimen awal | missing FIQ terkonsentrasi di situs ['LEUVEN_2', 'SBL'] (p situs=1.64e-153, maks 100%/situs); p label=1.000. FIQ_missing ≈ indikator situs → risiko proksi SITE_ID. Boleh diuji sebagai ablasi terpisah. |
| HANDEDNESS_CATEGORY | 14 | 1.4 | dipakai (imputasi most_frequent train per fold, one-hot) | missing 1.4%, tidak berasosiasi dengan label (p=1.000); distribusi R=845, L=81, Ambi=62, tidak ada kategori < 20. Missing terkonsentrasi di beberapa situs (p situs=3.88e-04), jadi tidak dibuat kategori 'Missing' eksplisit (bisa menjadi proksi situs). |

## Pembersihan deterministik

- Nilai diubah: 0 (tidak ada perubahan)
- Baris dikeluarkan: 0 (drop_log.csv)
- image_integrity_flag=True: -

## Demonstrasi fit per fold (tanpa melatih model)

- **fold_strat=0**: train 682, val 171, NaN sesudah transform 0, cek {'imputer_eq_train_median': True, 'scaler_eq_train_stats': True, 'cat_imputer_eq_train_mode': True}
- **loso=CALTECH**: train 821, val 32, NaN sesudah transform 0, cek {'imputer_eq_train_median': True, 'scaler_eq_train_stats': True, 'cat_imputer_eq_train_mode': True}

## Audit akhir

- Kolom terlarang di X: tidak ada
- Target tidak masuk fitur: OK
- ID hilang/ganda: tidak ada
- Transformer di-fit hanya pada train (statistics_/mean_/scale_/n_samples_seen_ = statistik train): OK
- Kebocoran citra identik antar-fold (fold_strat, repeated_fold_1, repeated_fold_2, repeated_fold_3, loso_site): tidak ada

## Perlu ditinjau

- VIQ/PIQ untuk subjek dengan FIQ kosong (43 dari 71, audit 05a): tidak dipakai di eksperimen awal; opsional sebagai analisis sensitivitas.
- Imputasi median membuat FIQ konstan untuk ['LEUVEN_2', 'SBL'], sehingga bisa menjadi penanda situs pada model pohon. Rencana: ablasi dengan dan tanpa FIQ, dan perhatikan hasil LOSO pada situs tersebut.
