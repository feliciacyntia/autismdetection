# Hasil PILOT v2 pilot200 (config_hash 27c4fa66930f2ece-ep2)

Status unit: {'done': 88}

## Seleksi (AUC pooled inner val lima fold; bukan skor test) & kriteria sudah belajar
Kriteria: train loss terakhir < 0,98 x epoch 1 DAN (val loss halus terbaik < loss prediksi konstan ATAU AUC inner val di epoch terbaik >= 0,55)

| modalitas | model | kelompok | AUC inner val (pooled) | sudah belajar (fold) |
|---|---|---|---|---|
| pheno | logreg | CNN/ML klasik | 0.52 |  |
| pheno | xgb | CNN/ML klasik | 0.541 |  |
| pheno | mlp_tab | CNN/ML klasik | 0.523 |  |
| pheno | tabm | arsitektur modern | 0.545 |  |
| roi | roi_logreg | CNN/ML klasik (ROI) | 0.579 |  |
| roi | roi_xgb | CNN/ML klasik (ROI) | belum ada |  |
| roi | roi_tabm | arsitektur modern (ROI) | belum ada |  |
| roi | roi_logreg_cb | CNN/ML klasik (ROI) | 0.595 |  |
| roi | roi_xgb_cb | CNN/ML klasik (ROI) | belum ada |  |
| roi | roi_tabm_cb | arsitektur modern (ROI) | belum ada |  |
| img25d | effnet_b0 | pretrained: CNN klasik | belum ada |  |
| img25d | resnet18 | pretrained: CNN klasik | belum ada |  |
| img25d | scratch2d | CNN dari nol | belum ada |  |
| img25d | dinov2_lp | foundation model (frozen) | 0.453 |  |
| img25d | dinov2_head | foundation model (frozen) | belum ada |  |
| img25d | convnextv2_t | pretrained: arsitektur modern | belum ada |  |
| img3d | scratch3d | CNN dari nol | 0.493 | 0/5 |
| img3d | sfcn_lp | pretrained: CNN otak (frozen) | 0.499 |  |
| img3d | sfcn_head | pretrained: CNN otak (frozen) | belum ada |  |

## Tabel A — fenotipik
| model | kelompok | status | terbaik | AUC inner val | BA (mean±SD) | F1m (mean±SD) | AUC (mean±SD) | Sens (mean±SD) | Spes (mean±SD) | BA pooled [95% CI] | AUC pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | baseline (S0) | done |  | 0.5 | 0.500±0.000 | 0.344±0.015 | 0.500±0.000 | 0.000±0.000 | 1.000±0.000 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| logreg | CNN/ML klasik | done |  | 0.52 | 0.556±0.061 | 0.553±0.063 | 0.552±0.060 | 0.524±0.132 | 0.588±0.132 | 0.558 [0.490, 0.625] | 0.564 [0.480, 0.645] |
| xgb | CNN/ML klasik | done |  | 0.541 | 0.523±0.057 | 0.505±0.051 | 0.550±0.092 | 0.436±0.160 | 0.610±0.205 | 0.516 [0.448, 0.584] | 0.512 [0.433, 0.595] |
| mlp_tab | CNN/ML klasik | done |  | 0.523 | 0.552±0.048 | 0.492±0.099 | 0.577±0.151 | 0.565±0.333 | 0.538±0.407 | 0.556 [0.485, 0.624] | 0.566 [0.484, 0.645] |
| tabm | arsitektur modern | done | ★ | 0.545 | 0.531±0.057 | 0.527±0.060 | 0.561±0.072 | 0.577±0.086 | 0.486±0.095 | 0.532 [0.462, 0.600] | 0.559 [0.476, 0.639] |

Tidak ada model dengan BA pooled > 0.75.

## Tabel v1 vs v2 (test per fold mean ± SD; pooled + 95% CI)
| versi | model | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | logreg | 0.493 | 0.584 ± 0.088 | 0.584 [0.515, 0.649] | 0.582 ± 0.090 | 0.584 [0.514, 0.649] | 0.577 ± 0.084 | 0.580 [0.497, 0.659] | 0.557 ± 0.143 | 0.558 [0.451, 0.657] | 0.611 ± 0.106 | 0.610 [0.516, 0.703] |
| v2 | logreg | 0.52 | 0.556 ± 0.061 | 0.558 [0.490, 0.625] | 0.553 ± 0.063 | 0.558 [0.489, 0.624] | 0.552 ± 0.060 | 0.564 [0.480, 0.645] | 0.524 ± 0.132 | 0.526 [0.423, 0.625] | 0.588 ± 0.132 | 0.590 [0.500, 0.684] |
| v1 | xgb | 0.553 | 0.534 ± 0.084 | 0.523 [0.457, 0.594] | 0.509 ± 0.104 | 0.523 [0.455, 0.593] | 0.550 ± 0.108 | 0.537 [0.461, 0.616] | 0.497 ± 0.183 | 0.484 [0.384, 0.581] | 0.571 ± 0.275 | 0.562 [0.468, 0.663] |
| v2 | xgb | 0.541 | 0.523 ± 0.057 | 0.516 [0.448, 0.584] | 0.505 ± 0.051 | 0.514 [0.444, 0.582] | 0.550 ± 0.092 | 0.512 [0.433, 0.595] | 0.436 ± 0.160 | 0.432 [0.333, 0.526] | 0.610 ± 0.205 | 0.600 [0.505, 0.691] |
| v1 | mlp_tab | 0.528 | 0.535 ± 0.043 | 0.536 [0.461, 0.605] | 0.495 ± 0.102 | 0.535 [0.459, 0.603] | 0.579 ± 0.060 | 0.569 [0.483, 0.648] | 0.551 ± 0.262 | 0.558 [0.451, 0.657] | 0.519 ± 0.301 | 0.514 [0.414, 0.608] |
| v2 | mlp_tab | 0.523 | 0.552 ± 0.048 | 0.556 [0.485, 0.624] | 0.492 ± 0.099 | 0.555 [0.483, 0.620] | 0.577 ± 0.151 | 0.566 [0.484, 0.645] | 0.565 ± 0.333 | 0.579 [0.478, 0.675] | 0.538 ± 0.407 | 0.533 [0.441, 0.626] |
| v1 | tabm | 0.533 | 0.535 ± 0.081 | 0.535 [0.464, 0.602] | 0.528 ± 0.080 | 0.534 [0.463, 0.600] | 0.563 ± 0.055 | 0.535 [0.450, 0.615] | 0.528 ± 0.182 | 0.526 [0.430, 0.625] | 0.541 ± 0.139 | 0.543 [0.447, 0.640] |
| v2 | tabm | 0.545 | 0.531 ± 0.057 | 0.532 [0.462, 0.600] | 0.527 ± 0.060 | 0.530 [0.458, 0.598] | 0.561 ± 0.072 | 0.559 [0.476, 0.639] | 0.577 ± 0.086 | 0.579 [0.473, 0.674] | 0.486 ± 0.095 | 0.486 [0.391, 0.583] |
| v1 | effnet_b0 | 0.466 | 0.513 ± 0.037 | 0.526 [0.462, 0.589] | 0.446 ± 0.082 | 0.515 [0.446, 0.582] | 0.489 ± 0.101 | 0.520 [0.438, 0.598] | 0.338 ± 0.377 | 0.347 [0.250, 0.441] | 0.687 ± 0.309 | 0.705 [0.615, 0.790] |
| v1 | resnet18 | 0.48 | 0.509 ± 0.020 | 0.493 [0.428, 0.559] | 0.361 ± 0.077 | 0.485 [0.413, 0.552] | 0.479 ± 0.085 | 0.485 [0.404, 0.566] | 0.370 ± 0.509 | 0.347 [0.250, 0.442] | 0.648 ± 0.490 | 0.638 [0.544, 0.728] |
| v1 | convnextv2_t | 0.532 | 0.503 ± 0.060 | 0.505 [0.437, 0.571] | 0.394 ± 0.079 | 0.502 [0.431, 0.569] | 0.504 ± 0.151 | 0.495 [0.414, 0.576] | 0.380 ± 0.437 | 0.400 [0.302, 0.500] | 0.625 ± 0.473 | 0.610 [0.511, 0.706] |
| v1 | sfcn | 0.511 | 0.515 ± 0.035 | 0.520 [0.468, 0.578] | 0.403 ± 0.083 | 0.480 [0.413, 0.549] | 0.494 ± 0.098 | 0.501 [0.423, 0.583] | 0.210 ± 0.347 | 0.221 [0.133, 0.309] | 0.819 ± 0.354 | 0.819 [0.745, 0.888] |
| v2 | sfcn_lp | 0.499 | 0.576 ± 0.049 | 0.578 [0.510, 0.645] | 0.574 ± 0.048 | 0.578 [0.509, 0.645] | 0.587 ± 0.110 | 0.594 [0.513, 0.673] | 0.533 ± 0.109 | 0.537 [0.432, 0.642] | 0.619 ± 0.067 | 0.619 [0.529, 0.708] |
| v1 | dinov2_s | 0.493 | 0.487 ± 0.018 | 0.481 [0.415, 0.549] | 0.382 ± 0.066 | 0.480 [0.414, 0.548] | 0.490 ± 0.109 | 0.479 [0.396, 0.558] | 0.415 ± 0.428 | 0.411 [0.312, 0.509] | 0.559 ± 0.438 | 0.552 [0.459, 0.649] |
| v2 | dinov2_lp | 0.453 | 0.516 ± 0.034 | 0.516 [0.446, 0.582] | 0.514 ± 0.032 | 0.515 [0.443, 0.581] | 0.457 ± 0.067 | 0.470 [0.392, 0.550] | 0.439 ± 0.046 | 0.442 [0.341, 0.543] | 0.594 ± 0.049 | 0.590 [0.495, 0.686] |

## Citra v2: pretrained vs dari nol
| model | kelompok | status | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] | C terpilih (per fold) | params latih/total | VRAM puncak (MiB) | s/epoch | waktu (menit) | epoch terbaik (median) | sudah belajar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| effnet_b0 | pretrained: CNN klasik | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| resnet18 | pretrained: CNN klasik | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| scratch2d | CNN dari nol | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dinov2_lp | foundation model (frozen) |  | 0.453 | 0.516 ± 0.034 | 0.516 [0.446, 0.582] | 0.514 ± 0.032 | 0.515 [0.443, 0.581] | 0.457 ± 0.067 | 0.470 [0.392, 0.550] | 0.439 ± 0.046 | 0.442 [0.341, 0.543] | 0.594 ± 0.049 | 0.590 [0.495, 0.686] | 0.001,0.01,0.001,0.001,1.0 |  |  |  |  |  |  |
| dinov2_head | foundation model (frozen) | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| convnextv2_t | pretrained: arsitektur modern | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| scratch3d | CNN dari nol |  | 0.493 | 0.500 ± 0.001 | 0.496 [0.441, 0.550] | 0.338 ± 0.044 | 0.432 [0.367, 0.496] | 0.540 ± 0.069 | 0.513 [0.433, 0.593] | 0.820 ± 0.402 | 0.811 [0.726, 0.889] | 0.181 ± 0.405 | 0.181 [0.109, 0.255] |  | 1791121/1791121 | 1021.0 | 7.6 | 1.3 | 1.0 | 0/5 |
| sfcn_lp | pretrained: CNN otak (frozen) |  | 0.499 | 0.576 ± 0.049 | 0.578 [0.510, 0.645] | 0.574 ± 0.048 | 0.578 [0.509, 0.645] | 0.587 ± 0.110 | 0.594 [0.513, 0.673] | 0.533 ± 0.109 | 0.537 [0.432, 0.642] | 0.619 ± 0.067 | 0.619 [0.529, 0.708] | 0.001,1.0,1.0,0.1,1.0 |  |  |  |  |  |  |
| sfcn_head | pretrained: CNN otak (frozen) | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Fitur ROI (Harvard-Oxford, mean & SD intensitas) dengan/tanpa ComBat
| model | basis | ComBat | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| roi_logreg | logreg | tidak | 0.579 | 0.525 ± 0.041 | 0.529 [0.460, 0.599] | 0.521 ± 0.043 | 0.529 [0.460, 0.598] | 0.551 ± 0.022 | 0.553 [0.475, 0.636] | 0.509 ± 0.138 | 0.516 [0.418, 0.613] | 0.542 ± 0.121 | 0.543 [0.448, 0.637] |  |
| roi_xgb | xgb | tidak |  |  |  |  |  |  |  |  |  |  |  | belum ada |
| roi_tabm | tabm | tidak |  |  |  |  |  |  |  |  |  |  |  | belum ada |
| roi_logreg_cb | logreg | ya | 0.595 | 0.546 ± 0.043 | 0.550 [0.481, 0.615] | 0.543 ± 0.044 | 0.550 [0.480, 0.614] | 0.539 ± 0.033 | 0.546 [0.470, 0.624] | 0.539 ± 0.135 | 0.547 [0.449, 0.644] | 0.552 ± 0.084 | 0.552 [0.453, 0.645] |  |
| roi_xgb_cb | xgb | ya |  |  |  |  |  |  |  |  |  |  |  | belum ada |
| roi_tabm_cb | tabm | ya |  |  |  |  |  |  |  |  |  |  |  | belum ada |

## Fusion & ensemble
| skenario | anggota | metode | stacking dilatih pada | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PR | tabm+roi_logreg_cb | F1_mean_prob | inner_oof | 0.551 ± 0.043 | 0.555 [0.486, 0.621] | 0.547 ± 0.044 | 0.555 [0.485, 0.619] | 0.542 ± 0.032 | 0.550 [0.473, 0.627] | 0.549 ± 0.146 | 0.558 [0.456, 0.650] | 0.552 ± 0.084 | 0.552 [0.453, 0.645] |
| PR | tabm+roi_logreg_cb | F1_stack_logreg | inner_oof | 0.550 ± 0.050 | 0.556 [0.488, 0.621] | 0.545 ± 0.051 | 0.555 [0.485, 0.619] | 0.544 ± 0.031 | 0.551 [0.472, 0.628] | 0.569 ± 0.156 | 0.579 [0.481, 0.674] | 0.530 ± 0.112 | 0.533 [0.432, 0.626] |
| PR | tabm+roi_logreg_cb | F2_pca16_logreg | inner_oof | 0.578 ± 0.053 | 0.578 [0.508, 0.646] | 0.578 ± 0.053 | 0.578 [0.506, 0.645] | 0.602 ± 0.077 | 0.594 [0.515, 0.673] | 0.536 ± 0.057 | 0.537 [0.430, 0.635] | 0.620 ± 0.062 | 0.619 [0.523, 0.713] |
| P25 | tabm+dinov2_lp | F1_mean_prob | inner_val | 0.514 ± 0.035 | 0.512 [0.443, 0.578] | 0.510 ± 0.032 | 0.511 [0.442, 0.577] | 0.467 ± 0.063 | 0.475 [0.397, 0.556] | 0.452 ± 0.058 | 0.453 [0.351, 0.556] | 0.575 ± 0.087 | 0.571 [0.475, 0.667] |
| P25 | tabm+dinov2_lp | F1_stack_logreg | inner_val | 0.506 ± 0.060 | 0.505 [0.437, 0.572] | 0.498 ± 0.058 | 0.504 [0.436, 0.570] | 0.486 ± 0.075 | 0.481 [0.403, 0.560] | 0.502 ± 0.176 | 0.495 [0.396, 0.593] | 0.510 ± 0.089 | 0.514 [0.417, 0.611] |
| P25 | tabm+dinov2_lp | F2_pca16_logreg | inner_val | 0.450 ± 0.045 | 0.446 [0.380, 0.510] | 0.438 ± 0.040 | 0.443 [0.376, 0.507] | 0.436 ± 0.048 | 0.432 [0.354, 0.510] | 0.369 ± 0.153 | 0.358 [0.260, 0.455] | 0.530 ± 0.109 | 0.533 [0.441, 0.626] |
| P3 | tabm+scratch3d | F1_mean_prob | inner_val | 0.510 ± 0.049 | 0.506 [0.456, 0.563] | 0.393 ± 0.105 | 0.440 [0.374, 0.511] | 0.561 ± 0.074 | 0.530 [0.448, 0.608] | 0.840 ± 0.243 | 0.832 [0.750, 0.904] | 0.181 ± 0.304 | 0.181 [0.110, 0.260] |
| P3 | tabm+scratch3d | F1_stack_logreg | inner_val | 0.489 ± 0.041 | 0.489 [0.418, 0.560] | 0.484 ± 0.045 | 0.489 [0.416, 0.559] | 0.465 ± 0.076 | 0.491 [0.411, 0.576] | 0.472 ± 0.073 | 0.474 [0.375, 0.574] | 0.505 ± 0.135 | 0.505 [0.405, 0.600] |
| P3 | tabm+scratch3d | F2_pca16_logreg | inner_val | 0.466 ± 0.055 | 0.458 [0.386, 0.523] | 0.447 ± 0.041 | 0.452 [0.378, 0.518] | 0.469 ± 0.091 | 0.461 [0.382, 0.538] | 0.332 ± 0.128 | 0.326 [0.236, 0.418] | 0.600 ± 0.178 | 0.590 [0.492, 0.680] |
| PRI | tabm+roi_logreg_cb+scratch3d | F1_mean_prob | inner_val | 0.539 ± 0.046 | 0.546 [0.478, 0.612] | 0.535 ± 0.046 | 0.545 [0.475, 0.610] | 0.544 ± 0.036 | 0.540 [0.462, 0.621] | 0.549 ± 0.146 | 0.558 [0.456, 0.650] | 0.530 ± 0.112 | 0.533 [0.432, 0.626] |
| PRI | tabm+roi_logreg_cb+scratch3d | F1_stack_logreg | inner_val | 0.498 ± 0.084 | 0.503 [0.431, 0.570] | 0.489 ± 0.093 | 0.503 [0.429, 0.569] | 0.510 ± 0.054 | 0.513 [0.434, 0.591] | 0.459 ± 0.221 | 0.463 [0.358, 0.559] | 0.536 ± 0.107 | 0.543 [0.447, 0.632] |
| PRI | tabm+roi_logreg_cb+scratch3d | F2_pca16_logreg | inner_val | 0.550 ± 0.069 | 0.545 [0.477, 0.614] | 0.536 ± 0.073 | 0.542 [0.472, 0.613] | 0.570 ± 0.081 | 0.549 [0.469, 0.629] | 0.448 ± 0.181 | 0.442 [0.347, 0.543] | 0.652 ± 0.147 | 0.648 [0.550, 0.735] |
| ENS | roi_logreg_cb+roi_logreg+tabm | ENS_mean_top3 | inner_val | 0.543 ± 0.046 | 0.549 [0.480, 0.615] | 0.540 ± 0.051 | 0.549 [0.479, 0.615] | 0.543 ± 0.028 | 0.552 [0.475, 0.632] | 0.516 ± 0.153 | 0.526 [0.427, 0.623] | 0.570 ± 0.084 | 0.571 [0.472, 0.667] |

## Site learning sebelum/sesudah ComBat (embedding citra)
| embedding | BA situs raw (mean fold) | BA situs ComBat (mean fold) | acak (1/n situs) | subjek test situs tak terlihat (total) | AUC probe raw [95% CI] | BA probe raw | AUC probe combat [95% CI] | BA probe combat |
|---|---|---|---|---|---|---|---|---|
| dinov2_lp | 0.779 | 0.03 | 0.05 | 1 | 0.470 [0.392, 0.550] | 0.516 | 0.450 [0.372, 0.528] | 0.468 |
| scratch3d | 0.392 | 0.098 | 0.05 | 1 | 0.498 [0.414, 0.578] | 0.507 | 0.475 [0.395, 0.553] | 0.484 |
| sfcn_lp | 0.38 | 0.062 | 0.05 | 1 | 0.594 [0.513, 0.673] | 0.578 | 0.552 [0.473, 0.631] | 0.573 |

## Cek kewajaran SFCN (brain age vs AGE_AT_SCAN; tanpa label diagnosis)
```
{
 "semua": {
  "n": 200,
  "pearson_r": -0.08423642831872784,
  "spearman_r": -0.06193895433768998
 },
 "usia>=18": {
  "n": 67,
  "pearson_r": -0.13139805108403535,
  "spearman_r": -0.16470236614053146
 }
}
```

## Catatan
- Pilot n=200: CI lebar; selisih kecil belum tentu bermakna.
- ROI: registrasi afin saja -> volume per region dikeluarkan; fitur = mean & SD intensitas z-score.
- Fusion: stacking pheno+ROI dilatih pada OOF inner CV (160 subjek/fold); dengan citra dilatih pada inner holdout (±32 subjek/fold) karena model citra tidak dilatih ulang per inner fold.
- ComBat: tidak diharmonisasi & ditandai (berlaku juga untuk LOSO).