# Pheno Verification Summary — Notebook 05b

- Sumber: `dataset/cleaned_dataset/cleaned_master_dataset.csv` (SHA-256 `ad0ef352bfdf18b28d53720868131a017de6d88dacb937ea840a0f514d4f24ec`, tidak berubah)
- Folds: `data/folds.csv` (SHA-256 `329b36408429a46403f825b1ac88bea018249d4b9427488d376fe13e6cbca616`, tidak berubah)
- scikit-learn 1.7.0, pandas 2.3.1, seed 42
- Subjek: 1004 (ASD 469, TD 535); dikeluarkan: 0
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
| FIQ | 71 | 7.07 | dipakai (imputasi median train per fold) | missing 7.1%; imputasi median di-fit pada train tiap fold. Missing terkonsentrasi di ['LEUVEN_2', 'SBL'] (struktural per situs): subjek situs ini mendapat median train, sehingga FIQ mereka tidak informatif. Audit 05a menandai FIQ sebagai konfounder; laporkan ablasi dengan/tanpa FIQ. |
| FIQ_missing | 0 | 0.0 | tidak dipakai di eksperimen awal | missing FIQ terkonsentrasi di situs ['LEUVEN_2', 'SBL'] (p situs=1.64e-153, maks 100%/situs); p label=1.000. FIQ_missing ≈ indikator situs → risiko proksi SITE_ID. Boleh diuji sebagai ablasi terpisah. |
| HANDEDNESS_CATEGORY | 14 | 1.39 | dipakai (imputasi most_frequent train per fold, one-hot) | missing 1.4%, tidak berasosiasi dengan label (p=1.000); distribusi R=847, L=81, Ambi=62, tidak ada kategori < 20. Missing terkonsentrasi di beberapa situs (p situs=3.88e-04), jadi tidak dibuat kategori 'Missing' eksplisit (bisa menjadi proksi situs). |

## Pembersihan deterministik

- Nilai diubah: 0 (tidak ada perubahan)
- Baris dikeluarkan: 0 (drop_log.csv)
- image_integrity_flag=True: [50279, 50286]

## Demonstrasi fit per fold (tanpa melatih model)

- **fold_strat=0**: train 803, val 201, NaN sesudah transform 0, cek {'imputer_eq_train_median': True, 'scaler_eq_train_stats': True, 'cat_imputer_eq_train_mode': True}
- **loso=CALTECH**: train 966, val 38, NaN sesudah transform 0, cek {'imputer_eq_train_median': True, 'scaler_eq_train_stats': True, 'cat_imputer_eq_train_mode': True}

## Audit akhir

- Kolom terlarang di X: tidak ada
- Target tidak masuk fitur: OK
- ID hilang/ganda: tidak ada
- Transformer di-fit hanya pada train (statistics_/mean_/scale_/n_samples_seen_ = statistik train): OK
- Kebocoran citra identik antar-fold (fold_strat, repeated_fold_1, repeated_fold_2, repeated_fold_3, loso_site): tidak ada

## Perlu ditinjau

- image_integrity_flag=True untuk SUB_ID [50279, 50286]: citra identik dengan fenotipik berbeda. Fenotipik tetap dipakai di 05b; keputusan untuk model citra di 05d.
- VIQ/PIQ untuk subjek dengan FIQ kosong (43 dari 71, audit 05a): tidak dipakai di eksperimen awal; opsional sebagai analisis sensitivitas.
- Imputasi median membuat FIQ konstan untuk ['LEUVEN_2', 'SBL'], sehingga bisa menjadi penanda situs pada model pohon. Rencana: ablasi dengan dan tanpa FIQ, dan perhatikan hasil LOSO pada situs tersebut.
