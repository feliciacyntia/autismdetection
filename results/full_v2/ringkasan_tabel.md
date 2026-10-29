# Hasil PILOT v2 all (config_hash 560c89845312e371)

Status unit: {'done': 91}

## Seleksi (AUC pooled inner val lima fold; bukan skor test) & kriteria sudah belajar
Kriteria: train loss terakhir < 0,98 x epoch 1 DAN (val loss halus terbaik < loss prediksi konstan ATAU AUC inner val di epoch terbaik >= 0,55)

| modalitas | model | kelompok | AUC inner val (pooled) | sudah belajar (fold) |
|---|---|---|---|---|
| pheno | logreg | CNN/ML klasik | 0.611 |  |
| pheno | mlp_tab | CNN/ML klasik | 0.649 |  |
| pheno | tabm | arsitektur modern | 0.638 |  |
| roi | roi_logreg_cb | CNN/ML klasik (ROI) | 0.561 |  |
| roi | roi_tabm_cb | arsitektur modern (ROI) | 0.587 |  |
| img25d | effnet_b0 | pretrained: CNN klasik | 0.577 | 5/5 |
| img25d | resnet18 | pretrained: CNN klasik | 0.533 | 4/5 |
| img25d | scratch2d | CNN dari nol | 0.503 | 0/5 |
| img3d | scratch3d | CNN dari nol | 0.587 | 2/5 |

## Tabel A — fenotipik
| model | kelompok | status | terbaik | AUC inner val | BA (mean±SD) | F1m (mean±SD) | AUC (mean±SD) | Sens (mean±SD) | Spes (mean±SD) | BA pooled [95% CI] | AUC pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | baseline (S0) | done |  | 0.5 | 0.500±0.000 | 0.348±0.007 | 0.500±0.000 | 0.000±0.000 | 1.000±0.000 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| logreg | CNN/ML klasik | done |  | 0.611 | 0.574±0.020 | 0.573±0.020 | 0.599±0.032 | 0.560±0.051 | 0.587±0.047 | 0.574 [0.541, 0.608] | 0.598 [0.560, 0.636] |
| mlp_tab | CNN/ML klasik | done | ★ | 0.649 | 0.585±0.014 | 0.580±0.018 | 0.622±0.018 | 0.480±0.100 | 0.690±0.083 | 0.585 [0.553, 0.617] | 0.615 [0.578, 0.653] |
| tabm | arsitektur modern | done |  | 0.638 | 0.595±0.030 | 0.591±0.035 | 0.633±0.035 | 0.503±0.115 | 0.687±0.066 | 0.595 [0.562, 0.628] | 0.624 [0.585, 0.661] |

Tidak ada model dengan BA pooled > 0.75.

## Tabel v1 vs v2 (test per fold mean ± SD; pooled + 95% CI)
| versi | model | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | logreg | 0.611 | 0.574 ± 0.020 | 0.574 [0.541, 0.608] | 0.573 ± 0.020 | 0.573 [0.540, 0.606] | 0.599 ± 0.032 | 0.598 [0.560, 0.636] | 0.560 ± 0.051 | 0.560 [0.512, 0.611] | 0.587 ± 0.047 | 0.587 [0.543, 0.632] |
| v2 | mlp_tab | 0.649 | 0.585 ± 0.014 | 0.585 [0.553, 0.617] | 0.580 ± 0.018 | 0.583 [0.550, 0.616] | 0.622 ± 0.018 | 0.615 [0.578, 0.653] | 0.480 ± 0.100 | 0.480 [0.431, 0.529] | 0.690 ± 0.083 | 0.690 [0.648, 0.731] |
| v2 | tabm | 0.638 | 0.595 ± 0.030 | 0.595 [0.562, 0.628] | 0.591 ± 0.035 | 0.594 [0.561, 0.627] | 0.633 ± 0.035 | 0.624 [0.585, 0.661] | 0.503 ± 0.115 | 0.503 [0.454, 0.554] | 0.687 ± 0.066 | 0.688 [0.645, 0.728] |
| v2 | effnet_b0 | 0.577 | 0.570 ± 0.042 | 0.571 [0.540, 0.605] | 0.559 ± 0.049 | 0.570 [0.539, 0.603] | 0.591 ± 0.044 | 0.591 [0.555, 0.629] | 0.576 ± 0.179 | 0.578 [0.531, 0.626] | 0.563 ± 0.169 | 0.565 [0.520, 0.611] |
| v2 | resnet18 | 0.533 | 0.499 ± 0.009 | 0.491 [0.458, 0.524] | 0.409 ± 0.087 | 0.488 [0.454, 0.520] | 0.564 ± 0.026 | 0.520 [0.479, 0.558] | 0.551 ± 0.396 | 0.543 [0.492, 0.589] | 0.447 ± 0.403 | 0.440 [0.394, 0.486] |

## Citra v2: pretrained vs dari nol
| kelompok | model | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] | params latih/total | VRAM puncak (MiB) | s/epoch | waktu (menit) | epoch terbaik (median) | sudah belajar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrained: CNN klasik | effnet_b0 | 0.577 | 0.570 ± 0.042 | 0.571 [0.540, 0.605] | 0.559 ± 0.049 | 0.570 [0.539, 0.603] | 0.591 ± 0.044 | 0.591 [0.555, 0.629] | 0.576 ± 0.179 | 0.578 [0.531, 0.626] | 0.563 ± 0.169 | 0.565 [0.520, 0.611] | 1293489/4171645 | 447 | 16.0 | 40.7 | 22 | 5/5 |
| pretrained: CNN klasik | resnet18 | 0.533 | 0.499 ± 0.009 | 0.491 [0.458, 0.524] | 0.409 ± 0.087 | 0.488 [0.454, 0.520] | 0.564 ± 0.026 | 0.520 [0.479, 0.558] | 0.551 ± 0.396 | 0.543 [0.492, 0.589] | 0.447 ± 0.403 | 0.440 [0.394, 0.486] | 8459521/11242305 | 448 | 14.2 | 35.7 | 21 | 4/5 |
| CNN dari nol | scratch2d | 0.503 | 0.515 ± 0.027 | 0.521 [0.487, 0.554] | 0.407 ± 0.094 | 0.520 [0.486, 0.552] | 0.520 ± 0.035 | 0.518 [0.480, 0.555] | 0.521 ± 0.465 | 0.525 [0.476, 0.571] | 0.510 ± 0.465 | 0.516 [0.471, 0.559] | 1206241/1206241 | 511 | 17.7 | 36.9 | 16 | 0/5 |
| CNN dari nol | scratch3d | 0.587 | 0.557 ± 0.033 | 0.556 [0.523, 0.590] | 0.543 ± 0.048 | 0.551 [0.517, 0.585] | 0.571 ± 0.043 | 0.567 [0.530, 0.606] | 0.624 ± 0.128 | 0.623 [0.575, 0.668] | 0.489 ± 0.151 | 0.488 [0.443, 0.536] | 1791121/1791121 | 1021 | 38.9 | 95.3 | 11 | 2/5 |

## Fitur ROI (Harvard-Oxford, mean & SD intensitas) dengan/tanpa ComBat
| model | basis | ComBat | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| roi_logreg_cb | logreg | ya | 0.561 | 0.573 ± 0.034 | 0.571 [0.538, 0.605] | 0.570 ± 0.034 | 0.571 [0.538, 0.605] | 0.608 ± 0.038 | 0.604 [0.567, 0.643] | 0.544 ± 0.062 | 0.543 [0.494, 0.595] | 0.602 ± 0.105 | 0.600 [0.555, 0.646] |
| roi_tabm_cb | tabm | ya | 0.587 | 0.600 ± 0.041 | 0.597 [0.565, 0.631] | 0.596 ± 0.043 | 0.597 [0.565, 0.630] | 0.627 ± 0.050 | 0.620 [0.583, 0.657] | 0.575 ± 0.089 | 0.573 [0.524, 0.624] | 0.624 ± 0.095 | 0.622 [0.578, 0.667] |

## Fusion & ensemble
| skenario | anggota | metode | stacking dilatih pada | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PR | mlp_tab+roi_tabm_cb | F1_mean_prob | inner_oof | 0.635 ± 0.036 | 0.633 [0.601, 0.664] | 0.631 ± 0.035 | 0.633 [0.601, 0.664] | 0.675 ± 0.043 | 0.671 [0.635, 0.707] | 0.584 ± 0.112 | 0.583 [0.534, 0.630] | 0.685 ± 0.090 | 0.684 [0.641, 0.724] |
| PR | mlp_tab+roi_tabm_cb | F1_stack_logreg | inner_oof | 0.625 ± 0.039 | 0.624 [0.591, 0.654] | 0.621 ± 0.040 | 0.624 [0.590, 0.654] | 0.669 ± 0.044 | 0.664 [0.628, 0.700] | 0.555 ± 0.122 | 0.553 [0.503, 0.600] | 0.695 ± 0.079 | 0.695 [0.652, 0.733] |
| PR | mlp_tab+roi_tabm_cb | F2_pca16_logreg | inner_oof | 0.618 ± 0.024 | 0.617 [0.583, 0.649] | 0.616 ± 0.023 | 0.616 [0.583, 0.649] | 0.663 ± 0.038 | 0.657 [0.621, 0.694] | 0.599 ± 0.054 | 0.598 [0.549, 0.646] | 0.636 ± 0.057 | 0.635 [0.590, 0.679] |
| P25 | mlp_tab+effnet_b0 | F1_mean_prob | inner_val | 0.613 ± 0.023 | 0.614 [0.582, 0.646] | 0.610 ± 0.027 | 0.614 [0.581, 0.645] | 0.650 ± 0.033 | 0.644 [0.607, 0.680] | 0.537 ± 0.116 | 0.538 [0.489, 0.585] | 0.690 ± 0.072 | 0.690 [0.648, 0.732] |
| P25 | mlp_tab+effnet_b0 | F1_stack_logreg | inner_val | 0.599 ± 0.024 | 0.599 [0.566, 0.631] | 0.593 ± 0.026 | 0.598 [0.564, 0.630] | 0.645 ± 0.036 | 0.634 [0.597, 0.671] | 0.498 ± 0.140 | 0.497 [0.450, 0.547] | 0.701 ± 0.097 | 0.701 [0.658, 0.742] |
| P25 | mlp_tab+effnet_b0 | F2_pca16_logreg | inner_val | 0.587 ± 0.053 | 0.586 [0.554, 0.618] | 0.585 ± 0.053 | 0.586 [0.552, 0.618] | 0.611 ± 0.073 | 0.610 [0.573, 0.647] | 0.591 ± 0.086 | 0.590 [0.542, 0.639] | 0.583 ± 0.061 | 0.582 [0.541, 0.626] |
| P3 | mlp_tab+scratch3d | F1_mean_prob | inner_val | 0.605 ± 0.034 | 0.605 [0.571, 0.636] | 0.601 ± 0.036 | 0.605 [0.570, 0.635] | 0.645 ± 0.044 | 0.644 [0.606, 0.679] | 0.533 ± 0.112 | 0.533 [0.484, 0.582] | 0.677 ± 0.098 | 0.677 [0.633, 0.719] |
| P3 | mlp_tab+scratch3d | F1_stack_logreg | inner_val | 0.599 ± 0.038 | 0.599 [0.566, 0.630] | 0.593 ± 0.040 | 0.598 [0.566, 0.629] | 0.645 ± 0.040 | 0.644 [0.607, 0.680] | 0.523 ± 0.131 | 0.523 [0.474, 0.570] | 0.675 ± 0.112 | 0.675 [0.632, 0.717] |
| P3 | mlp_tab+scratch3d | F2_pca16_logreg | inner_val | 0.572 ± 0.067 | 0.572 [0.542, 0.605] | 0.571 ± 0.067 | 0.572 [0.541, 0.605] | 0.604 ± 0.073 | 0.606 [0.570, 0.644] | 0.550 ± 0.082 | 0.550 [0.504, 0.600] | 0.593 ± 0.067 | 0.593 [0.548, 0.637] |
| PRI | mlp_tab+roi_tabm_cb+scratch3d | F1_mean_prob | inner_val | 0.638 ± 0.034 | 0.636 [0.603, 0.668] | 0.634 ± 0.034 | 0.636 [0.603, 0.668] | 0.680 ± 0.048 | 0.677 [0.642, 0.712] | 0.595 ± 0.114 | 0.593 [0.546, 0.640] | 0.680 ± 0.090 | 0.679 [0.634, 0.720] |
| PRI | mlp_tab+roi_tabm_cb+scratch3d | F1_stack_logreg | inner_val | 0.621 ± 0.050 | 0.620 [0.588, 0.651] | 0.615 ± 0.053 | 0.619 [0.587, 0.651] | 0.659 ± 0.056 | 0.659 [0.623, 0.695] | 0.531 ± 0.138 | 0.530 [0.483, 0.579] | 0.710 ± 0.118 | 0.710 [0.669, 0.751] |
| PRI | mlp_tab+roi_tabm_cb+scratch3d | F2_pca16_logreg | inner_val | 0.560 ± 0.044 | 0.560 [0.526, 0.594] | 0.560 ± 0.043 | 0.560 [0.526, 0.594] | 0.607 ± 0.057 | 0.605 [0.567, 0.642] | 0.533 ± 0.058 | 0.533 [0.483, 0.581] | 0.587 ± 0.050 | 0.587 [0.542, 0.631] |
| ENS | mlp_tab+tabm+logreg | ENS_mean_top3 | inner_val | 0.584 ± 0.021 | 0.584 [0.553, 0.616] | 0.580 ± 0.026 | 0.584 [0.551, 0.616] | 0.623 ± 0.024 | 0.616 [0.580, 0.655] | 0.503 ± 0.116 | 0.503 [0.454, 0.552] | 0.666 ± 0.083 | 0.666 [0.623, 0.707] |

## Site learning sebelum/sesudah ComBat (embedding citra)
| embedding | BA situs raw (mean fold) | BA situs ComBat (mean fold) | acak (1/n situs) | subjek test situs tak terlihat (total) | AUC probe raw [95% CI] | BA probe raw | AUC probe combat [95% CI] | BA probe combat |
|---|---|---|---|---|---|---|---|---|
| effnet_b0 | 0.713 | 0.166 | 0.05 | 0 | 0.588 [0.550, 0.626] | 0.562 | 0.582 [0.545, 0.620] | 0.568 |
| resnet18 | 0.586 | 0.122 | 0.05 | 0 | 0.572 [0.534, 0.611] | 0.547 | 0.574 [0.534, 0.613] | 0.565 |
| scratch2d | 0.513 | 0.126 | 0.05 | 0 | 0.553 [0.516, 0.592] | 0.544 | 0.556 [0.518, 0.597] | 0.531 |
| scratch3d | 0.384 | 0.099 | 0.05 | 0 | 0.582 [0.544, 0.620] | 0.552 | 0.600 [0.562, 0.640] | 0.553 |

## Cek kewajaran SFCN (brain age vs AGE_AT_SCAN; tanpa label diagnosis)
_belum ada_

## Catatan
- Pilot n=200: CI lebar; selisih kecil belum tentu bermakna.
- ROI: registrasi afin saja -> volume per region dikeluarkan; fitur = mean & SD intensitas z-score.
- Fusion: stacking pheno+ROI dilatih pada OOF inner CV (160 subjek/fold); dengan citra dilatih pada inner holdout (±32 subjek/fold) karena model citra tidak dilatih ulang per inner fold.
- ComBat: tidak diharmonisasi & ditandai (berlaku juga untuk LOSO).