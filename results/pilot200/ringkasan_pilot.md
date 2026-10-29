# Ringkasan Pilot Modeling (subset 200 subjek dev) — 27 Oktober 2026

**Status:** 57/57 unit selesai, config_hash `1797fe5bf1d67157`, preprocessing `682404ae0f5d2624`.
**Sifat hasil:** PILOT (n = 200; per fold ±128 latih, ±32 inner val, ±40 test). Semua CI 95% lebar; peringkat hanya indikasi, **bukan hasil final**. Test held-out (149 subjek) belum disentuh.

## Istilah
- **Unit = model × fold.** Contoh: `effnet_b0:3` = EfficientNet-B0 dilatih pada fold 3. 9 model × 5 fold + 2 cache fitur + 10 fusion = 57 unit. Tiap unit punya status, checkpoint, dan prediksinya sendiri sehingga bisa dilanjutkan bila terputus.
- **Ekstraksi fitur otomatis (representation learning).** Fitur citra tidak dibuat manual; jaringan (CNN/ViT) belajar sendiri mengubah citra menjadi vektor ringkas (embedding 128 dimensi per subjek). Untuk SFCN dan DINOv2 bagian ekstraktornya dibekukan dan fiturnya dihitung sekali (cache). Fitur fenotipik: usia, jenis kelamin, FIQ, volume mask otak.

## Hasil utama (pooled OOF, 95% CI bootstrap)
| Skenario | Model | BA | AUC |
|---|---|---|---|
| S0 baseline | majority | 0,500 | 0,500 |
| S1 fenotipik | LogReg | 0,584 [0,515; 0,649] | 0,580 [0,497; 0,659] |
| S1 fenotipik | XGBoost (terpilih aturan inner val) | 0,523 [0,457; 0,594] | 0,537 [0,461; 0,616] |
| S2 citra 2.5D | ConvNeXt V2-T | 0,505 [0,437; 0,571] | 0,495 [0,414; 0,576] |
| S3 citra 3D | SFCN | 0,520 [0,468; 0,578] | 0,501 [0,423; 0,583] |
| S4 2.5D + fenotipik | F1 stacking | 0,511 [0,443; 0,581] | 0,538 [0,460; 0,617] |
| S5 3D + fenotipik | F1 stacking | 0,508 [0,439; 0,576] | 0,536 [0,458; 0,616] |

Tabel lengkap: `tables/` (A–E), gambar: `figures/`, semua angka: `ringkasan_tabel.md`.

## Temuan
1. **Fenotipik** memberi sinyal lemah; hanya LogReg yang batas bawah CI BA > 0,5 (kemungkinan dari FIQ — sah, tetapi dibahas sebagai potensi konfounder).
2. **Seleksi via inner val tidak stabil pada n = 200:** aturan memilih XGBoost (AUC inner val 0,553) sedangkan LogReg lebih baik di test; dengan ±32 subjek inner val per fold pemilihan hampir acak.
3. **Model citra belum menangkap sinyal ASD** (AUC ≈ 0,48–0,50). Kurva loss (lihat bagian diagnosis) menunjukkan model tidak belajar pola yang bergeneralisasi.
4. **Efek situs kuat di embedding citra:** SITE_ID tertebak dengan BA 0,30–0,63 (acak 0,05).
5. **Fusion belum menambah performa;** F2 lebih buruk karena head dilatih pada ±32 subjek.
6. **Tidak ada tanda leakage** (BA pooled tertinggi 0,584 < 0,75).

## Kandidat untuk data penuh (2–3 per modalitas)
| Modalitas | Kandidat | Aturan |
|---|---|---|
| Fenotipik | LogReg, XGBoost, TabM | LogReg = baseline standar (keputusan peneliti, bukan skor test); sisanya AUC inner val |
| 2.5D | ConvNeXt V2-T, DINOv2 ViT-S, ResNet18 | AUC inner val |
| 3D | SFCN (+ 3D CNN+SE / BrainIAC bila siap) | satu-satunya model 3D pilot |
| Fusion | F1 stacking (utama), F1 rata-rata, F2 PCA16 + LogReg | ditetapkan sebelum hasil |

Kelompok: CNN/ML klasik (LogReg, XGBoost, MLP, EfficientNet-B0, ResNet18, SFCN) · arsitektur modern (TabM, ConvNeXt V2) · foundation model (DINOv2; TabPFN v2, DINOv3, BrainIAC ditunda).

## Rencana efek situs untuk run data penuh (keputusan)
- (a) Evaluasi utama tetap `folds.csv` (5 fold, split per subjek & situs).
- (b) **LOSO** (leave-one-site-out, 18 situs eligible) sebagai uji generalisasi antar situs.
- (c) **ComBat pada embedding citra** sebagai ablasi: di-fit hanya pada train tiap fold, kovariat usia & jenis kelamin, **tanpa diagnosis**.

## Diagnosis training model citra (dari kurva loss train vs inner val; skor test tidak dipakai)
| Model | Pola | Arti |
|---|---|---|
| EfficientNet-B0, DINOv2 head | train loss turun, inner-val naik sejak epoch 1–3 | overfitting cepat (3,3 juta parameter dilatih vs 128 subjek) |
| ResNet18, ConvNeXt V2, SFCN | train & inner-val datar ±0,71–0,73 (≈ loss prediksi konstan) | tidak belajar sama sekali (underfitting / sinyal gradien lemah) |
| ConvNeXt V2 | lonjakan loss epoch 1 | update awal terlalu besar (tanpa warmup) |

Inner-val ±32 subjek membuat val loss sangat bising, sehingga early stopping (patience 6) berhenti sebelum model sempat belajar.

**Usulan perbaikan (BELUM diterapkan; menunggu persetujuan):**
1. **Early stopping:** minimal 5 epoch, patience 10, pantau val loss yang dihaluskan (rata-rata 3 epoch) — mengurangi berhenti karena noise.
2. **Learning rate:** warmup 2 epoch; LR backbone 3e-5 (dari 1e-4) dan head 1e-3 — mencegah lonjakan awal & overfitting cepat.
3. **Lebih sedikit layer dilatih:** EfficientNet hanya blok terakhir; ResNet18 hanya `layer4`; ConvNeXt hanya stage terakhir — parameter latih turun, cocok untuk data kecil. Alternatif standar: *linear probe dulu, lalu fine-tune* (LP-FT).
4. **BatchNorm dibekukan** di blok yang dilatih (batch 4 subjek × 15 slice berkorelasi → statistik BN bising).
5. **Head frozen-feature (SFCN, DINOv2):** standarisasi fitur (di-fit di train) + weight decay lebih besar; pembanding linear probe (LogReg) pada embedding.
6. **Cek kewajaran input SFCN tanpa label:** prediksi usia otak SFCN pretrained harus berkorelasi kuat dengan AGE_AT_SCAN; jika tidak, adaptasi input (upsample 1 mm, orientasi, intensitas) perlu diperbaiki sebelum fine-tune.
7. **Augmentasi 2.5D:** posisi slice digeser acak ±2 voxel saat training (variasi lebih banyak tanpa mengubah preprocessing).

Perubahan hyperparameter akan mengubah config_hash → dijalankan sebagai pilot baru (`pilot200_v2`), dinilai **hanya** dari kurva train/inner-val, bukan skor test.

## Sumber daya (GPU 4 GB)
EfficientNet-B0 4,8 s/epoch, 483 MiB · ResNet18 4,2 s/epoch, 488 MiB · ConvNeXt V2-T 18,9 s/epoch, 818 MiB · SFCN 142 MiB (cache 45 s) · DINOv2 21 MiB (cache 52 s). Total run ±50 menit.

## Ditunda
TabPFN v2 (butuh scikit-learn < 1.7 → env terpisah), BrainIAC (verifikasi apakah data pretraining memuat ABIDE), 3D CNN+SE dari nol, S6, DINOv3 (gated), inner CV penuh (untuk data penuh).
