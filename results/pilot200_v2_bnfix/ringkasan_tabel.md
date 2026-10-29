# Hasil PILOT v2 pilot200 (config_hash 84888081837812d7)

Status unit: {'done': 10}

## Seleksi (AUC pooled inner val lima fold; bukan skor test) & kriteria sudah belajar
Kriteria: train loss terakhir < 0,98 x epoch 1 DAN (val loss halus terbaik < loss prediksi konstan ATAU AUC inner val di epoch terbaik >= 0,55)

| modalitas | model | kelompok | AUC inner val (pooled) | sudah belajar (fold) |
|---|---|---|---|---|
| pheno | logreg | CNN/ML klasik | belum ada |  |
| pheno | xgb | CNN/ML klasik | belum ada |  |
| pheno | mlp_tab | CNN/ML klasik | belum ada |  |
| pheno | tabm | arsitektur modern | belum ada |  |
| roi | roi_logreg | CNN/ML klasik (ROI) | belum ada |  |
| roi | roi_xgb | CNN/ML klasik (ROI) | belum ada |  |
| roi | roi_tabm | arsitektur modern (ROI) | belum ada |  |
| roi | roi_logreg_cb | CNN/ML klasik (ROI) | belum ada |  |
| roi | roi_xgb_cb | CNN/ML klasik (ROI) | belum ada |  |
| roi | roi_tabm_cb | arsitektur modern (ROI) | belum ada |  |
| img25d | effnet_b0 | pretrained: CNN klasik | belum ada |  |
| img25d | resnet18 | pretrained: CNN klasik | belum ada |  |
| img25d | scratch2d | CNN dari nol | 0.516 | 1/5 |
| img25d | dinov2_lp | foundation model (frozen) | belum ada |  |
| img25d | dinov2_head | foundation model (frozen) | belum ada |  |
| img25d | convnextv2_t | pretrained: arsitektur modern | belum ada |  |
| img3d | scratch3d | CNN dari nol | 0.531 | 0/5 |
| img3d | sfcn_lp | pretrained: CNN otak (frozen) | belum ada |  |
| img3d | sfcn_head | pretrained: CNN otak (frozen) | belum ada |  |

## Tabel A — fenotipik
| model | kelompok | status | terbaik | AUC inner val | BA (mean±SD) | F1m (mean±SD) | AUC (mean±SD) | Sens (mean±SD) | Spes (mean±SD) | BA pooled [95% CI] | AUC pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | baseline (S0) | done |  | 0.5 | 0.500±0.000 | 0.344±0.015 | 0.500±0.000 | 0.000±0.000 | 1.000±0.000 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| logreg | CNN/ML klasik | belum ada |  |  |  |  |  |  |  |  |  |
| xgb | CNN/ML klasik | belum ada |  |  |  |  |  |  |  |  |  |
| mlp_tab | CNN/ML klasik | belum ada |  |  |  |  |  |  |  |  |  |
| tabm | arsitektur modern | belum ada |  |  |  |  |  |  |  |  |  |

Tidak ada model dengan BA pooled > 0.75.

## Tabel v1 vs v2 (test per fold mean ± SD; pooled + 95% CI)
| versi | model | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | logreg | 0.493 | 0.584 ± 0.088 | 0.584 [0.515, 0.649] | 0.582 ± 0.090 | 0.584 [0.514, 0.649] | 0.577 ± 0.084 | 0.580 [0.497, 0.659] | 0.557 ± 0.143 | 0.558 [0.451, 0.657] | 0.611 ± 0.106 | 0.610 [0.516, 0.703] |
| v1 | xgb | 0.553 | 0.534 ± 0.084 | 0.523 [0.457, 0.594] | 0.509 ± 0.104 | 0.523 [0.455, 0.593] | 0.550 ± 0.108 | 0.537 [0.461, 0.616] | 0.497 ± 0.183 | 0.484 [0.384, 0.581] | 0.571 ± 0.275 | 0.562 [0.468, 0.663] |
| v1 | mlp_tab | 0.528 | 0.535 ± 0.043 | 0.536 [0.461, 0.605] | 0.495 ± 0.102 | 0.535 [0.459, 0.603] | 0.579 ± 0.060 | 0.569 [0.483, 0.648] | 0.551 ± 0.262 | 0.558 [0.451, 0.657] | 0.519 ± 0.301 | 0.514 [0.414, 0.608] |
| v1 | tabm | 0.533 | 0.535 ± 0.081 | 0.535 [0.464, 0.602] | 0.528 ± 0.080 | 0.534 [0.463, 0.600] | 0.563 ± 0.055 | 0.535 [0.450, 0.615] | 0.528 ± 0.182 | 0.526 [0.430, 0.625] | 0.541 ± 0.139 | 0.543 [0.447, 0.640] |
| v1 | effnet_b0 | 0.466 | 0.513 ± 0.037 | 0.526 [0.462, 0.589] | 0.446 ± 0.082 | 0.515 [0.446, 0.582] | 0.489 ± 0.101 | 0.520 [0.438, 0.598] | 0.338 ± 0.377 | 0.347 [0.250, 0.441] | 0.687 ± 0.309 | 0.705 [0.615, 0.790] |
| v1 | resnet18 | 0.48 | 0.509 ± 0.020 | 0.493 [0.428, 0.559] | 0.361 ± 0.077 | 0.485 [0.413, 0.552] | 0.479 ± 0.085 | 0.485 [0.404, 0.566] | 0.370 ± 0.509 | 0.347 [0.250, 0.442] | 0.648 ± 0.490 | 0.638 [0.544, 0.728] |
| v1 | convnextv2_t | 0.532 | 0.503 ± 0.060 | 0.505 [0.437, 0.571] | 0.394 ± 0.079 | 0.502 [0.431, 0.569] | 0.504 ± 0.151 | 0.495 [0.414, 0.576] | 0.380 ± 0.437 | 0.400 [0.302, 0.500] | 0.625 ± 0.473 | 0.610 [0.511, 0.706] |
| v1 | sfcn | 0.511 | 0.515 ± 0.035 | 0.520 [0.468, 0.578] | 0.403 ± 0.083 | 0.480 [0.413, 0.549] | 0.494 ± 0.098 | 0.501 [0.423, 0.583] | 0.210 ± 0.347 | 0.221 [0.133, 0.309] | 0.819 ± 0.354 | 0.819 [0.745, 0.888] |
| v1 | dinov2_s | 0.493 | 0.487 ± 0.018 | 0.481 [0.415, 0.549] | 0.382 ± 0.066 | 0.480 [0.414, 0.548] | 0.490 ± 0.109 | 0.479 [0.396, 0.558] | 0.415 ± 0.428 | 0.411 [0.312, 0.509] | 0.559 ± 0.438 | 0.552 [0.459, 0.649] |

## Citra v2: pretrained vs dari nol
| model | kelompok | status | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] | params latih/total | VRAM puncak (MiB) | s/epoch | waktu (menit) | epoch terbaik (median) | sudah belajar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| effnet_b0 | pretrained: CNN klasik | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| resnet18 | pretrained: CNN klasik | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| scratch2d | CNN dari nol |  | 0.516 | 0.524 ± 0.040 | 0.516 [0.448, 0.589] | 0.398 ± 0.115 | 0.515 [0.445, 0.587] | 0.559 ± 0.031 | 0.516 [0.430, 0.597] | 0.469 ± 0.489 | 0.442 [0.341, 0.545] | 0.579 ± 0.487 | 0.590 [0.495, 0.688] | 1206241/1206241 | 510.0 | 4.2 | 5.8 | 8.0 | 1/5 |
| dinov2_lp | foundation model (frozen) | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dinov2_head | foundation model (frozen) | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| convnextv2_t | pretrained: arsitektur modern | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| scratch3d | CNN dari nol |  | 0.531 | 0.464 ± 0.049 | 0.449 [0.387, 0.517] | 0.407 ± 0.048 | 0.443 [0.378, 0.511] | 0.472 ± 0.043 | 0.461 [0.384, 0.539] | 0.349 ± 0.346 | 0.326 [0.238, 0.423] | 0.579 ± 0.261 | 0.571 [0.477, 0.664] | 1791121/1791121 | 1021.0 | 9.4 | 11.0 | 1.0 | 0/5 |
| sfcn_lp | pretrained: CNN otak (frozen) | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| sfcn_head | pretrained: CNN otak (frozen) | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Fitur ROI (Harvard-Oxford, mean & SD intensitas) dengan/tanpa ComBat
| model | basis | ComBat | status |
|---|---|---|---|
| roi_logreg | logreg | tidak | belum ada |
| roi_xgb | xgb | tidak | belum ada |
| roi_tabm | tabm | tidak | belum ada |
| roi_logreg_cb | logreg | ya | belum ada |
| roi_xgb_cb | xgb | ya | belum ada |
| roi_tabm_cb | tabm | ya | belum ada |

## Fusion & ensemble
| skenario | metode |
|---|---|
| PR | belum ada |
| P25 | belum ada |
| P3 | belum ada |
| PRI | belum ada |
| ENS | belum ada |

## Site learning sebelum/sesudah ComBat (embedding citra)
_belum ada_

## Cek kewajaran SFCN (brain age vs AGE_AT_SCAN; tanpa label diagnosis)
_belum ada_

## Catatan
- Pilot n=200: CI lebar; selisih kecil belum tentu bermakna.
- ROI: registrasi afin saja -> volume per region dikeluarkan; fitur = mean & SD intensitas z-score.
- Fusion: stacking pheno+ROI dilatih pada OOF inner CV (160 subjek/fold); dengan citra dilatih pada inner holdout (±32 subjek/fold) karena model citra tidak dilatih ulang per inner fold.
- ComBat: tidak diharmonisasi & ditandai (berlaku juga untuk LOSO).