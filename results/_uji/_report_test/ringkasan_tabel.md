# Hasil eksperimen pilot200 (config_hash 1797fe5bf1d67157-ep3)

Status unit: {'done': 21}

## Fitur fenotipik — missing per grup
| feature | missing_pct_ASD | missing_pct_TD | gap_pp | used |
|---|---|---|---|---|
| AGE_AT_SCAN | 0.0 | 0.0 | 0.0 | True |
| SEX | 0.0 | 0.0 | 0.0 | True |
| FIQ | 5.26 | 7.62 | 2.36 | True |
| volume_mask_ml | 1.05 | 0.0 | 1.05 | True |

## Seleksi (AUC pooled inner validation lima fold; BUKAN skor test)
| modalitas | model | kelompok | AUC inner val (pooled) |
|---|---|---|---|
| pheno | logreg | CNN/ML klasik | 0.493 |
| pheno | xgb | CNN/ML klasik | 0.553 |
| pheno | mlp_tab | CNN/ML klasik | 0.482 |
| pheno | tabm | arsitektur modern | 0.521 |
| img25d | effnet_b0 | CNN klasik | belum ada |
| img25d | resnet18 | CNN klasik | belum ada |
| img25d | dinov2_s | foundation model | belum ada |
| img25d | convnextv2_t | arsitektur modern | belum ada |
| img3d | sfcn | CNN klasik (pretrained otak) | belum ada |

## Tabel A — fenotipik (test fold, mean ± SD; pooled OOF + 95% CI bootstrap)
★ = terbaik menurut AUC pooled inner validation lima fold (aturan seleksi, bukan skor test).

| model | kelompok | status | terbaik | AUC inner val | BA (mean±SD) | F1m (mean±SD) | AUC (mean±SD) | Sens (mean±SD) | Spes (mean±SD) | BA pooled [95% CI] | AUC pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | baseline (S0) | done |  | 0.5 | 0.500±0.000 | 0.344±0.015 | 0.500±0.000 | 0.000±0.000 | 1.000±0.000 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| logreg | CNN/ML klasik | done |  | 0.493 | 0.584±0.088 | 0.582±0.090 | 0.577±0.084 | 0.557±0.143 | 0.611±0.106 | 0.584 [0.515, 0.649] | 0.580 [0.497, 0.659] |
| xgb | CNN/ML klasik | done | ★ | 0.553 | 0.534±0.084 | 0.509±0.104 | 0.550±0.108 | 0.497±0.183 | 0.571±0.275 | 0.523 [0.457, 0.594] | 0.537 [0.461, 0.616] |
| mlp_tab | CNN/ML klasik | done |  | 0.482 | 0.526±0.063 | 0.472±0.099 | 0.605±0.122 | 0.566±0.287 | 0.487±0.374 | 0.522 [0.453, 0.592] | 0.584 [0.502, 0.660] |
| tabm | arsitektur modern | done |  | 0.521 | 0.529±0.073 | 0.506±0.087 | 0.570±0.050 | 0.646±0.174 | 0.412±0.198 | 0.526 [0.458, 0.592] | 0.546 [0.464, 0.627] |

Tidak ada model dengan BA pooled > 0.75.

## Tabel B — citra 2.5D & 3D
| model | kelompok | status |
|---|---|---|
| effnet_b0 | CNN klasik | belum ada |
| resnet18 | CNN klasik | belum ada |
| dinov2_s | foundation model | belum ada |
| convnextv2_t | arsitektur modern | belum ada |
| sfcn | CNN klasik (pretrained otak) | belum ada |

## Tabel C — fusion F1 vs F2 (utama: F1 = stacking LogReg, F2 = PCA16 + LogReg; ditetapkan sebelum hasil)
| skenario | metode |
|---|---|
| S4 | belum ada |
| S5 | belum ada |

## Tabel D — ablasi S0–S5
| skenario | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 majority | 0.500 ± 0.000 | 0.500 [0.500, 0.500] | 0.344 ± 0.015 | 0.344 [0.313, 0.373] | 0.500 ± 0.000 | 0.468 [0.392, 0.542] | 0.000 ± 0.000 | 0.000 [0.000, 0.000] | 1.000 ± 0.000 | 1.000 [1.000, 1.000] |
| S1 fenotipik (xgb) | 0.534 ± 0.084 | 0.523 [0.457, 0.594] | 0.509 ± 0.104 | 0.523 [0.455, 0.593] | 0.550 ± 0.108 | 0.537 [0.461, 0.616] | 0.497 ± 0.183 | 0.484 [0.384, 0.581] | 0.571 ± 0.275 | 0.562 [0.468, 0.663] |

## Tabel E — site learning (prediksi SITE_ID dari embedding citra)
_belum ada_

## Tabel E — performa per situs (pooled OOF)
| skenario | SITE_ID | n | BA | AUC |
|---|---|---|---|---|
| S1 fenotipik (xgb) | CALTECH | 8 | 0.625 | 0.781 |
| S1 fenotipik (xgb) | CMU | 5 | 0.667 | 0.5 |
| S1 fenotipik (xgb) | KKI | 8 | 0.7 | 0.867 |
| S1 fenotipik (xgb) | LEUVEN_1 | 6 | 0.667 | 0.556 |
| S1 fenotipik (xgb) | LEUVEN_2 | 7 | 0.375 | 0.208 |
| S1 fenotipik (xgb) | MAX_MUN | 10 | 0.5 | 0.542 |
| S1 fenotipik (xgb) | NYU | 37 | 0.543 | 0.561 |
| S1 fenotipik (xgb) | OHSU | 6 | 0.667 | 0.833 |
| S1 fenotipik (xgb) | OLIN | 6 | 0.5 | 0.556 |
| S1 fenotipik (xgb) | PITT | 11 | 0.467 | 0.6 |
| S1 fenotipik (xgb) | SBL | 6 | 0.167 | 0.0 |
| S1 fenotipik (xgb) | SDSU | 6 | 0.5 | 0.625 |
| S1 fenotipik (xgb) | STANFORD | 5 | 0.833 | 0.833 |
| S1 fenotipik (xgb) | TRINITY | 9 | 0.3 | 0.2 |
| S1 fenotipik (xgb) | UCLA_1 | 13 | 0.381 | 0.357 |
| S1 fenotipik (xgb) | UCLA_2 | 4 | 0.25 | 0.0 |
| S1 fenotipik (xgb) | UM_1 | 18 | 0.604 | 0.617 |
| S1 fenotipik (xgb) | UM_2 | 7 | 0.667 | 0.75 |
| S1 fenotipik (xgb) | USM | 18 | 0.506 | 0.61 |
| S1 fenotipik (xgb) | YALE | 10 | 0.5 | 0.56 |

## Kelompok model
- **CNN/ML klasik**: logreg, xgb, mlp_tab
- **arsitektur modern**: tabm, convnextv2_t
- **CNN klasik**: effnet_b0, resnet18
- **CNN klasik (pretrained otak)**: sfcn
- **foundation model**: dinov2_s

## Kandidat untuk data penuh (2–3 per modalitas, urut AUC inner val)
- pheno: xgb, tabm, logreg
- img25d: belum ada
- img3d: belum ada

## Ditunda
- tabpfn_v2: semua versi tabpfn v2 (2.0.x-2.2.x) menuntut scikit-learn<1.7; env dikunci 1.7.0 -> env terpisah
- dinov3_vits16: repo HF facebook/dinov3-vits16-pretrain-lvd1689m gated (manual, butuh login + persetujuan)
- brainiac: bobot ada (Dropbox, lisensi akademik) tetapi daftar data pretraining (ABIDE?) belum terverifikasi
- cnn3d_se_scratch: ditunda (rencana)
- S6: ditunda (rencana)
- inner_cv_full: untuk run data penuh

## Catatan jujur
- Pilot 20% (n=200, ±40 subjek test per fold): CI lebar; selisih antar model yang kecil belum tentu bermakna.
- Peringkat hanya indikasi; data penuh memakai inner CV penuh, bukan inner holdout tunggal.
- Head fusion dilatih pada inner holdout (±32 subjek per fold) — penyederhanaan pilot.
- SFCN & DINOv2: bagian beku di-cache tanpa augmentasi.