# Ringkasan Pilot v2 — subset 200 ABIDE I (config_hash 27c4fa66930f2ece)

Tanggal run: 2026-10-27 · 168/168 unit selesai · 63 menit · preprocessing 05c `682404ae0f5d2624` · `folds.csv` tidak berubah · test held-out (149) tidak disentuh.
Unit = model × fold (5 fold, ±40 subjek test/fold). Seleksi hanya dari AUC pooled inner validation; angka test di bawah **tidak** dipakai untuk memilih.

## 1. Apakah model citra sudah belajar?
Kriteria: train loss turun (< 0,98 × epoch 1) DAN (val loss halus < loss prediksi konstan ATAU AUC inner val ≥ 0,55).

| model | kelompok | sudah belajar | BA pooled test [95% CI] |
|---|---|---|---|
| EfficientNet-B0 (LP-FT) | pretrained | 1/5 | 0,531 [0,466; 0,597] |
| ResNet18 (LP-FT) | pretrained | 1/5 | 0,515 [0,457; 0,575] |
| ConvNeXt V2-T (LP-FT) | pretrained modern | 1/5 | 0,531 [0,487; 0,577] |
| CNN 2.5D dari nol | dari nol | 1/5 | 0,511 [0,442; 0,576] |
| CNN 3D+SE dari nol | dari nol | 0/5 | 0,489 [0,422; 0,556] |
| DINOv2 head | foundation (frozen) | 0/5 | 0,483 [0,415; 0,550] |

**Kesimpulan jujur:** perbaikan training v2 (warmup, LR diferensial, LP-FT, BN beku, early stopping halus, jitter) tidak membuat model citra belajar sinyal diagnosis pada ±128 subjek latih. Kurva loss: val loss menempel di garis konstan; train loss hanya turun di fold yang kemudian overfit; CNN 3D dari nol tidak menurunkan train loss sama sekali (lonjakan val loss karena statistik BatchNorm pada batch kecil). Perubahan v1 → v2 untuk citra berada di dalam noise.

## 2. Fenotipik dan ROI
- Fenotipik (usia, jenis kelamin, FIQ, volume otak): BA 0,52–0,58. TabM BA 0,584 [0,512; 0,652], LogReg 0,558. Model terpilih menurut inner val = MLP (0,554), padahal BA test-nya bukan yang tertinggi → bukti seleksi di n = 200 tidak stabil.
- ROI Harvard-Oxford (111 region, mean & SD intensitas; volume per region dikeluarkan karena registrasi hanya afin): BA 0,52–0,56. **ComBat menaikkan AUC inner val** LogReg (0,579 → 0,595) dan TabM (0,535 → 0,611), tapi efek di test tidak konsisten.

## 3. ComBat pada embedding citra (ablasi)
Kemampuan menebak site turun drastis (contoh DINOv2 0,78 → 0,03; EffNet 0,62 → 0,10; acak = 0,05). Akurasi diagnosis tidak ikut naik secara konsisten. Artinya embedding memang banyak memuat site, tetapi setelah site dihapus, sinyal diagnosis yang tersisa tetap lemah. (Satu observasi: probe ConvNeXt + ComBat AUC test 0,651 — dicatat, tidak dipakai untuk seleksi.)

## 4. Fusion dan ensemble
| skenario | F1 stacking (utama) | F1 rata-rata | F2 PCA+LogReg |
|---|---|---|---|
| PR (pheno + ROI) | 0,569 | 0,583 | 0,578 |
| P25 (pheno + 2.5D) | 0,522 | 0,576 | 0,532 |
| P3 (pheno + 3D) | 0,522 | 0,544 | 0,488 |
| PRI (pheno + ROI + 2.5D) | 0,573 | **0,598** [0,530; 0,667] | 0,572 |
| Ensemble top-3 (ROI) | – | 0,549 | – |

(BA pooled test.) Fusion pheno + ROI memberi hasil paling stabil. Citra tidak menambah informasi yang berarti. Rata-rata probabilitas cenderung lebih baik daripada stacking (stacking dengan citra hanya dilatih pada ±32 subjek inner holdout). Metode utama tetap F1 stacking sesuai rencana awal; rata-rata dilaporkan sebagai pembanding.

## 5. Percobaan gagal (dicatat)
- **SFCN (UK Biobank):** cek kewajaran brain age gagal (r = −0,08 terhadap AGE_AT_SCAN; prediksi sekitar 60 tahun untuk semua subjek). Unit tetap dijalankan: `sfcn_lp` BA test 0,578, tetapi AUC inner val 0,499. Karena gagal cek kewajaran, model ini dikeluarkan dari seleksi dan fusion.
- **DINOv2:** AUC inner val terendah (0,41–0,45).
- **TabPFN v2:** tidak dijalankan karena konflik versi scikit-learn (ditunda).

## 6. Konfigurasi DIKUNCI untuk data penuh (`locked_config_kandidat.json`)
- **Fenotipik:** MLP, TabM, LogReg (baseline).
- **ROI:** TabM + ComBat, LogReg + ComBat.
- **2.5D:** ResNet18 dan EfficientNet-B0 (pretrained, LP-FT v2), CNN 2.5D dari nol.
- **3D:** CNN 3D+SE dari nol. SFCN baru masuk lagi jika preprocessing-nya lolos cek kewajaran.
- **Dikeluarkan:** DINOv2, ConvNeXt V2 (inner val rendah, biaya sekitar 5×), XGBoost.
- **ComBat:** wajib untuk ROI; ablasi untuk embedding. Site yang tidak terlihat (LOSO) tidak diharmonisasi dan ditandai.
- **Fusion:** F1 stacking (utama), F1 rata-rata, F2 PCA16 + LogReg. Skenario PR, P25, P3, PRI. Ensemble top-3 sebagai pembanding.

## 7. Implikasi
Dengan n = 200, semua CI tumpang tindih dan berada sekitar 0,5–0,65. Sejalan dengan literatur ABIDE (sMRI ASD vs TD umumnya AUC 0,55–0,65 pada data multisite), harapan realistis untuk data penuh (sekitar 850 dev) adalah kenaikan kecil, terutama untuk model citra yang saat ini gagal belajar karena kekurangan data. Pilot dihentikan di sini sesuai batas maksimal 2 sesi. Langkah berikutnya: preprocessing 802 subjek sisa.
