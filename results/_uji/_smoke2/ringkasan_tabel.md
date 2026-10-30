# Hasil eksperimen pilot200 (config_hash 1797fe5bf1d67157-ep1)

Status unit: {'done': 28}

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
| pheno | xgb | CNN/ML klasik | belum ada |
| pheno | mlp_tab | CNN/ML klasik | belum ada |
| pheno | tabm | arsitektur modern | belum ada |
| img25d | effnet_b0 | CNN klasik | belum ada |
| img25d | resnet18 | CNN klasik | belum ada |
| img25d | dinov2_s | foundation model | 0.481 |
| img25d | convnextv2_t | arsitektur modern | belum ada |
| img3d | sfcn | CNN klasik (pretrained otak) | 0.513 |

## Tabel A — fenotipik (test fold, mean ± SD; pooled OOF + 95% CI bootstrap)
| model | kelompok | status | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] | params (latih/total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | CNN/ML klasik | done | 0.493 | 0.584 ± 0.088 | 0.584 [0.515, 0.649] | 0.582 ± 0.090 | 0.584 [0.514, 0.649] | 0.577 ± 0.084 | 0.580 [0.497, 0.659] | 0.557 ± 0.143 | 0.558 [0.451, 0.657] | 0.611 ± 0.106 | 0.610 [0.516, 0.703] | -/7 |
| xgb | CNN/ML klasik | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |
| mlp_tab | CNN/ML klasik | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |
| tabm | arsitektur modern | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |

## Tabel B — citra 2.5D & 3D
| model | kelompok | status | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] | params (latih/total) | VRAM puncak (MiB) | s/epoch | waktu total (menit) | epoch terbaik (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| effnet_b0 | CNN klasik | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| resnet18 | CNN klasik | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dinov2_s | foundation model | done | 0.481 | 0.500 ± 0.000 | 0.505 [0.451, 0.562] | 0.342 ± 0.016 | 0.465 [0.396, 0.533] | 0.481 ± 0.104 | 0.522 [0.444, 0.601] | 0.200 ± 0.447 | 0.211 [0.129, 0.297] | 0.800 ± 0.447 | 0.800 [0.723, 0.872] | 100097/100097 | 21.0 | 0.0 | 0.0 | 1.0 |
| convnextv2_t | arsitektur modern | belum ada |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| sfcn | CNN klasik (pretrained otak) | done | 0.513 | 0.500 ± 0.000 | 0.486 [0.432, 0.541] | 0.319 ± 0.014 | 0.421 [0.357, 0.487] | 0.520 ± 0.050 | 0.484 [0.407, 0.565] | 0.800 ± 0.447 | 0.800 [0.714, 0.882] | 0.200 ± 0.447 | 0.171 [0.105, 0.248] | 1795265/1795265 | 142.0 | 0.6 | 0.0 | 1.0 |

## Tabel C — fusion F1 vs F2 (utama: F1 = stacking LogReg, F2 = PCA16 + LogReg; ditetapkan sebelum hasil)
| skenario | citra | fenotipik | metode | jenis | utama | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S4 | dinov2_s | logreg | F1_mean_prob | F1 late |  | 0.587 ± 0.087 | 0.592 [0.523, 0.658] | 0.561 ± 0.106 | 0.592 [0.522, 0.657] | 0.576 ± 0.088 | 0.584 [0.502, 0.663] | 0.526 ± 0.254 | 0.537 [0.435, 0.636] | 0.648 ± 0.315 | 0.648 [0.557, 0.735] |
| S4 | dinov2_s | logreg | F1_stack_logreg | F1 late | ya | 0.529 ± 0.031 | 0.530 [0.489, 0.571] | 0.431 ± 0.091 | 0.448 [0.386, 0.512] | 0.469 ± 0.122 | 0.474 [0.395, 0.551] | 0.125 ± 0.144 | 0.126 [0.061, 0.196] | 0.934 ± 0.124 | 0.933 [0.883, 0.976] |
| S4 | dinov2_s | logreg | F2_pca16_lgbm | F2 intermediate |  | 0.517 ± 0.070 | 0.513 [0.441, 0.584] | 0.515 ± 0.070 | 0.513 [0.440, 0.583] | 0.523 ± 0.045 | 0.541 [0.457, 0.618] | 0.478 ± 0.100 | 0.474 [0.372, 0.573] | 0.556 ± 0.071 | 0.552 [0.456, 0.645] |
| S4 | dinov2_s | logreg | F2_pca16_logreg | F2 intermediate | ya | 0.460 ± 0.045 | 0.460 [0.388, 0.529] | 0.457 ± 0.046 | 0.460 [0.385, 0.527] | 0.463 ± 0.064 | 0.455 [0.370, 0.536] | 0.464 ± 0.086 | 0.463 [0.359, 0.568] | 0.456 ± 0.059 | 0.457 [0.359, 0.553] |
| S4 | dinov2_s | logreg | F2_pca16_mlp | F2 intermediate |  | 0.456 ± 0.062 | 0.453 [0.384, 0.520] | 0.452 ± 0.064 | 0.453 [0.382, 0.518] | 0.430 ± 0.038 | 0.422 [0.344, 0.500] | 0.424 ± 0.101 | 0.421 [0.323, 0.524] | 0.489 ± 0.086 | 0.486 [0.387, 0.576] |
| S4 | dinov2_s | logreg | F2_pca8_lgbm | F2 intermediate |  | 0.488 ± 0.033 | 0.484 [0.415, 0.551] | 0.479 ± 0.036 | 0.480 [0.408, 0.548] | 0.470 ± 0.081 | 0.471 [0.391, 0.555] | 0.373 ± 0.095 | 0.368 [0.270, 0.469] | 0.603 ± 0.062 | 0.600 [0.500, 0.690] |
| S4 | dinov2_s | logreg | F2_pca8_logreg | F2 intermediate |  | 0.470 ± 0.077 | 0.472 [0.402, 0.540] | 0.465 ± 0.082 | 0.471 [0.400, 0.539] | 0.473 ± 0.084 | 0.454 [0.373, 0.534] | 0.410 ± 0.160 | 0.411 [0.312, 0.516] | 0.531 ± 0.047 | 0.533 [0.438, 0.624] |
| S4 | dinov2_s | logreg | F2_pca8_mlp | F2 intermediate |  | 0.460 ± 0.023 | 0.459 [0.387, 0.528] | 0.457 ± 0.025 | 0.459 [0.385, 0.525] | 0.449 ± 0.029 | 0.446 [0.364, 0.524] | 0.441 ± 0.083 | 0.442 [0.341, 0.542] | 0.479 ± 0.052 | 0.476 [0.378, 0.570] |
| S4 | dinov2_s | logreg | F2_raw128_lgbm | F2 intermediate |  | 0.393 ± 0.067 | 0.392 [0.325, 0.458] | 0.390 ± 0.066 | 0.391 [0.323, 0.456] | 0.380 ± 0.055 | 0.392 [0.317, 0.470] | 0.326 ± 0.059 | 0.326 [0.237, 0.420] | 0.459 ± 0.102 | 0.457 [0.361, 0.550] |
| S4 | dinov2_s | logreg | F2_raw128_logreg | F2 intermediate |  | 0.532 ± 0.023 | 0.534 [0.456, 0.604] | 0.530 ± 0.022 | 0.534 [0.455, 0.601] | 0.502 ± 0.057 | 0.502 [0.417, 0.584] | 0.505 ± 0.084 | 0.505 [0.400, 0.606] | 0.560 ± 0.063 | 0.562 [0.462, 0.653] |
| S4 | dinov2_s | logreg | F2_raw128_mlp | F2 intermediate |  | 0.454 ± 0.074 | 0.450 [0.380, 0.521] | 0.447 ± 0.078 | 0.450 [0.380, 0.519] | 0.466 ± 0.061 | 0.459 [0.379, 0.541] | 0.442 ± 0.063 | 0.442 [0.343, 0.540] | 0.467 ± 0.164 | 0.457 [0.361, 0.550] |
| S5 | sfcn | logreg | F1_mean_prob | F1 late |  | 0.586 ± 0.085 | 0.584 [0.515, 0.649] | 0.583 ± 0.084 | 0.584 [0.514, 0.647] | 0.577 ± 0.084 | 0.576 [0.491, 0.654] | 0.556 ± 0.105 | 0.558 [0.455, 0.657] | 0.615 ± 0.139 | 0.610 [0.515, 0.701] |
| S5 | sfcn | logreg | F1_stack_logreg | F1 late | ya | 0.536 ± 0.037 | 0.535 [0.494, 0.578] | 0.441 ± 0.099 | 0.457 [0.390, 0.524] | 0.473 ± 0.117 | 0.475 [0.397, 0.554] | 0.138 ± 0.146 | 0.137 [0.071, 0.208] | 0.934 ± 0.124 | 0.933 [0.883, 0.976] |
| S5 | sfcn | logreg | F2_pca16_lgbm | F2 intermediate |  | 0.493 ± 0.048 | 0.489 [0.422, 0.557] | 0.484 ± 0.045 | 0.484 [0.415, 0.553] | 0.499 ± 0.057 | 0.491 [0.409, 0.568] | 0.372 ± 0.067 | 0.368 [0.267, 0.465] | 0.615 ± 0.110 | 0.610 [0.517, 0.699] |
| S5 | sfcn | logreg | F2_pca16_logreg | F2 intermediate | ya | 0.482 ± 0.075 | 0.478 [0.408, 0.545] | 0.479 ± 0.076 | 0.478 [0.405, 0.545] | 0.417 ± 0.109 | 0.425 [0.343, 0.503] | 0.447 ± 0.118 | 0.442 [0.341, 0.539] | 0.517 ± 0.088 | 0.514 [0.416, 0.608] |
| S5 | sfcn | logreg | F2_pca16_mlp | F2 intermediate |  | 0.457 ± 0.080 | 0.456 [0.388, 0.526] | 0.449 ± 0.076 | 0.455 [0.384, 0.523] | 0.456 ± 0.093 | 0.441 [0.363, 0.518] | 0.381 ± 0.085 | 0.379 [0.280, 0.475] | 0.533 ± 0.182 | 0.533 [0.436, 0.630] |
| S5 | sfcn | logreg | F2_pca8_lgbm | F2 intermediate |  | 0.505 ± 0.085 | 0.502 [0.432, 0.573] | 0.497 ± 0.092 | 0.500 [0.430, 0.570] | 0.500 ± 0.088 | 0.496 [0.418, 0.577] | 0.437 ± 0.152 | 0.432 [0.333, 0.531] | 0.574 ± 0.114 | 0.571 [0.476, 0.667] |
| S5 | sfcn | logreg | F2_pca8_logreg | F2 intermediate |  | 0.437 ± 0.076 | 0.433 [0.365, 0.503] | 0.435 ± 0.076 | 0.433 [0.364, 0.502] | 0.411 ± 0.098 | 0.413 [0.333, 0.494] | 0.403 ± 0.086 | 0.400 [0.299, 0.495] | 0.470 ± 0.087 | 0.467 [0.378, 0.566] |
| S5 | sfcn | logreg | F2_pca8_mlp | F2 intermediate |  | 0.467 ± 0.057 | 0.465 [0.400, 0.533] | 0.458 ± 0.053 | 0.462 [0.394, 0.529] | 0.485 ± 0.105 | 0.463 [0.384, 0.544] | 0.370 ± 0.098 | 0.368 [0.274, 0.462] | 0.564 ± 0.145 | 0.562 [0.462, 0.660] |
| S5 | sfcn | logreg | F2_raw128_lgbm | F2 intermediate |  | 0.496 ± 0.035 | 0.497 [0.429, 0.564] | 0.492 ± 0.035 | 0.497 [0.427, 0.561] | 0.510 ± 0.068 | 0.501 [0.421, 0.580] | 0.444 ± 0.113 | 0.442 [0.344, 0.538] | 0.548 ± 0.086 | 0.552 [0.458, 0.645] |
| S5 | sfcn | logreg | F2_raw128_logreg | F2 intermediate |  | 0.455 ± 0.079 | 0.453 [0.385, 0.524] | 0.451 ± 0.081 | 0.453 [0.383, 0.522] | 0.424 ± 0.089 | 0.432 [0.352, 0.512] | 0.412 ± 0.098 | 0.411 [0.311, 0.511] | 0.498 ± 0.120 | 0.495 [0.398, 0.588] |
| S5 | sfcn | logreg | F2_raw128_mlp | F2 intermediate |  | 0.442 ± 0.064 | 0.444 [0.378, 0.511] | 0.435 ± 0.061 | 0.439 [0.369, 0.506] | 0.448 ± 0.061 | 0.438 [0.359, 0.515] | 0.325 ± 0.052 | 0.326 [0.236, 0.420] | 0.560 ± 0.137 | 0.562 [0.465, 0.657] |

## Tabel D — ablasi S0–S5
| skenario | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 majority | 0.500 ± 0.000 | 0.500 [0.500, 0.500] | 0.344 ± 0.015 | 0.344 [0.313, 0.373] | 0.500 ± 0.000 | 0.468 [0.392, 0.542] | 0.000 ± 0.000 | 0.000 [0.000, 0.000] | 1.000 ± 0.000 | 1.000 [1.000, 1.000] |
| S1 fenotipik (logreg) | 0.584 ± 0.088 | 0.584 [0.515, 0.649] | 0.582 ± 0.090 | 0.584 [0.514, 0.649] | 0.577 ± 0.084 | 0.580 [0.497, 0.659] | 0.557 ± 0.143 | 0.558 [0.451, 0.657] | 0.611 ± 0.106 | 0.610 [0.516, 0.703] |
| S2 2.5D (dinov2_s) | 0.500 ± 0.000 | 0.505 [0.451, 0.562] | 0.342 ± 0.016 | 0.465 [0.396, 0.533] | 0.481 ± 0.104 | 0.522 [0.444, 0.601] | 0.200 ± 0.447 | 0.211 [0.129, 0.297] | 0.800 ± 0.447 | 0.800 [0.723, 0.872] |
| S3 3D (sfcn) | 0.500 ± 0.000 | 0.486 [0.432, 0.541] | 0.319 ± 0.014 | 0.421 [0.357, 0.487] | 0.520 ± 0.050 | 0.484 [0.407, 0.565] | 0.800 ± 0.447 | 0.800 [0.714, 0.882] | 0.200 ± 0.447 | 0.171 [0.105, 0.248] |
| S4 2.5D+fenotipik F1 | 0.529 ± 0.031 | 0.530 [0.489, 0.571] | 0.431 ± 0.091 | 0.448 [0.386, 0.512] | 0.469 ± 0.122 | 0.474 [0.395, 0.551] | 0.125 ± 0.144 | 0.126 [0.061, 0.196] | 0.934 ± 0.124 | 0.933 [0.883, 0.976] |
| S4 2.5D+fenotipik F2 | 0.460 ± 0.045 | 0.460 [0.388, 0.529] | 0.457 ± 0.046 | 0.460 [0.385, 0.527] | 0.463 ± 0.064 | 0.455 [0.370, 0.536] | 0.464 ± 0.086 | 0.463 [0.359, 0.568] | 0.456 ± 0.059 | 0.457 [0.359, 0.553] |
| S5 3D+fenotipik F1 | 0.536 ± 0.037 | 0.535 [0.494, 0.578] | 0.441 ± 0.099 | 0.457 [0.390, 0.524] | 0.473 ± 0.117 | 0.475 [0.397, 0.554] | 0.138 ± 0.146 | 0.137 [0.071, 0.208] | 0.934 ± 0.124 | 0.933 [0.883, 0.976] |
| S5 3D+fenotipik F2 | 0.482 ± 0.075 | 0.478 [0.408, 0.545] | 0.479 ± 0.076 | 0.478 [0.405, 0.545] | 0.417 ± 0.109 | 0.425 [0.343, 0.503] | 0.447 ± 0.118 | 0.442 [0.341, 0.539] | 0.517 ± 0.088 | 0.514 [0.416, 0.608] |

## Tabel E — site learning (prediksi SITE_ID dari embedding citra)
| embedding | BA prediksi SITE_ID (test) | peluang acak | n situs |
|---|---|---|---|
| dinov2_s | 0.635 | 0.05 | 20 |
| sfcn | 0.36 | 0.05 | 20 |

## Tabel E — performa per situs (pooled OOF)
| skenario | SITE_ID | n | BA | AUC |
|---|---|---|---|---|
| S1 fenotipik (logreg) | CALTECH | 8 | 0.75 | 0.75 |
| S1 fenotipik (logreg) | CMU | 5 | 0.417 | 0.5 |
| S1 fenotipik (logreg) | KKI | 8 | 0.9 | 0.933 |
| S1 fenotipik (logreg) | LEUVEN_1 | 6 | 0.5 | 0.444 |
| S1 fenotipik (logreg) | LEUVEN_2 | 7 | 0.375 | 0.0 |
| S1 fenotipik (logreg) | MAX_MUN | 10 | 0.667 | 0.75 |
| S1 fenotipik (logreg) | NYU | 37 | 0.591 | 0.601 |
| S1 fenotipik (logreg) | OHSU | 6 | 0.667 | 0.556 |
| S1 fenotipik (logreg) | OLIN | 6 | 0.5 | 0.667 |
| S1 fenotipik (logreg) | PITT | 11 | 0.617 | 0.833 |
| S1 fenotipik (logreg) | SBL | 6 | 0.667 | 0.778 |
| S1 fenotipik (logreg) | SDSU | 6 | 0.25 | 0.375 |
| S1 fenotipik (logreg) | STANFORD | 5 | 0.833 | 0.667 |
| S1 fenotipik (logreg) | TRINITY | 9 | 0.675 | 0.75 |
| S1 fenotipik (logreg) | UCLA_1 | 13 | 0.464 | 0.31 |
| S1 fenotipik (logreg) | UCLA_2 | 4 | 0.75 | 1.0 |
| S1 fenotipik (logreg) | UM_1 | 18 | 0.396 | 0.312 |
| S1 fenotipik (logreg) | UM_2 | 7 | 0.542 | 0.333 |
| S1 fenotipik (logreg) | USM | 18 | 0.578 | 0.494 |
| S1 fenotipik (logreg) | YALE | 10 | 0.6 | 0.72 |
| S2 2.5D (dinov2_s) | CALTECH | 8 | 0.5 | 0.562 |
| S2 2.5D (dinov2_s) | CMU | 5 | 0.833 | 0.667 |
| S2 2.5D (dinov2_s) | KKI | 8 | 0.4 | 0.2 |
| S2 2.5D (dinov2_s) | LEUVEN_1 | 6 | 0.667 | 0.667 |
| S2 2.5D (dinov2_s) | LEUVEN_2 | 7 | 0.542 | 0.5 |
| S2 2.5D (dinov2_s) | MAX_MUN | 10 | 0.542 | 0.5 |
| S2 2.5D (dinov2_s) | NYU | 37 | 0.53 | 0.521 |
| S2 2.5D (dinov2_s) | OHSU | 6 | 0.333 | 0.333 |
| S2 2.5D (dinov2_s) | OLIN | 6 | 0.333 | 0.0 |
| S2 2.5D (dinov2_s) | PITT | 11 | 0.567 | 0.533 |
| S2 2.5D (dinov2_s) | SBL | 6 | 0.667 | 0.889 |
| S2 2.5D (dinov2_s) | SDSU | 6 | 0.375 | 0.75 |
| S2 2.5D (dinov2_s) | STANFORD | 5 | 0.0 | 0.0 |
| S2 2.5D (dinov2_s) | TRINITY | 9 | 0.425 | 0.55 |
| S2 2.5D (dinov2_s) | UCLA_1 | 13 | 0.488 | 0.5 |
| S2 2.5D (dinov2_s) | UCLA_2 | 4 | 0.5 | 0.75 |
| S2 2.5D (dinov2_s) | UM_1 | 18 | 0.481 | 0.545 |
| S2 2.5D (dinov2_s) | UM_2 | 7 | 0.667 | 0.75 |
| S2 2.5D (dinov2_s) | USM | 18 | 0.519 | 0.571 |
| S2 2.5D (dinov2_s) | YALE | 10 | 0.5 | 0.52 |
| S3 3D (sfcn) | CALTECH | 8 | 0.5 | 0.375 |
| S3 3D (sfcn) | CMU | 5 | 0.75 | 1.0 |
| S3 3D (sfcn) | KKI | 8 | 0.6 | 0.667 |
| S3 3D (sfcn) | LEUVEN_1 | 6 | 0.5 | 0.444 |
| S3 3D (sfcn) | LEUVEN_2 | 7 | 0.625 | 0.583 |
| S3 3D (sfcn) | MAX_MUN | 10 | 0.583 | 0.708 |
| S3 3D (sfcn) | NYU | 37 | 0.501 | 0.543 |
| S3 3D (sfcn) | OHSU | 6 | 0.333 | 0.333 |
| S3 3D (sfcn) | OLIN | 6 | 0.667 | 1.0 |
| S3 3D (sfcn) | PITT | 11 | 0.517 | 0.483 |
| S3 3D (sfcn) | SBL | 6 | 0.333 | 0.222 |
| S3 3D (sfcn) | SDSU | 6 | 0.25 | 0.0 |
| S3 3D (sfcn) | STANFORD | 5 | 0.333 | 0.667 |
| S3 3D (sfcn) | TRINITY | 9 | 0.475 | 0.45 |
| S3 3D (sfcn) | UCLA_1 | 13 | 0.44 | 0.488 |
| S3 3D (sfcn) | UCLA_2 | 4 | 0.25 | 0.25 |
| S3 3D (sfcn) | UM_1 | 18 | 0.448 | 0.455 |
| S3 3D (sfcn) | UM_2 | 7 | 0.458 | 0.583 |
| S3 3D (sfcn) | USM | 18 | 0.481 | 0.279 |
| S3 3D (sfcn) | YALE | 10 | 0.5 | 0.32 |
| S4 2.5D+fenotipik F1 | CALTECH | 8 | 0.625 | 0.5 |
| S4 2.5D+fenotipik F1 | CMU | 5 | 0.667 | 1.0 |
| S4 2.5D+fenotipik F1 | KKI | 8 | 0.667 | 0.667 |
| S4 2.5D+fenotipik F1 | LEUVEN_1 | 6 | 0.667 | 0.333 |
| S4 2.5D+fenotipik F1 | LEUVEN_2 | 7 | 0.5 | 0.25 |
| S4 2.5D+fenotipik F1 | MAX_MUN | 10 | 0.5 | 0.458 |
| S4 2.5D+fenotipik F1 | NYU | 37 | 0.452 | 0.384 |
| S4 2.5D+fenotipik F1 | OHSU | 6 | 0.667 | 0.667 |
| S4 2.5D+fenotipik F1 | OLIN | 6 | 0.5 | 0.778 |
| S4 2.5D+fenotipik F1 | PITT | 11 | 0.567 | 0.4 |
| S4 2.5D+fenotipik F1 | SBL | 6 | 0.5 | 0.444 |
| S4 2.5D+fenotipik F1 | SDSU | 6 | 0.375 | 0.125 |
| S4 2.5D+fenotipik F1 | STANFORD | 5 | 0.5 | 0.667 |
| S4 2.5D+fenotipik F1 | TRINITY | 9 | 0.525 | 0.3 |
| S4 2.5D+fenotipik F1 | UCLA_1 | 13 | 0.571 | 0.548 |
| S4 2.5D+fenotipik F1 | UCLA_2 | 4 | 0.5 | 0.0 |
| S4 2.5D+fenotipik F1 | UM_1 | 18 | 0.526 | 0.468 |
| S4 2.5D+fenotipik F1 | UM_2 | 7 | 0.5 | 0.417 |
| S4 2.5D+fenotipik F1 | USM | 18 | 0.474 | 0.338 |
| S4 2.5D+fenotipik F1 | YALE | 10 | 0.6 | 0.68 |
| S4 2.5D+fenotipik F2 | CALTECH | 8 | 0.375 | 0.312 |
| S4 2.5D+fenotipik F2 | CMU | 5 | 0.167 | 0.333 |
| S4 2.5D+fenotipik F2 | KKI | 8 | 0.467 | 0.467 |
| S4 2.5D+fenotipik F2 | LEUVEN_1 | 6 | 0.667 | 0.778 |
| S4 2.5D+fenotipik F2 | LEUVEN_2 | 7 | 0.875 | 1.0 |
| S4 2.5D+fenotipik F2 | MAX_MUN | 10 | 0.458 | 0.375 |
| S4 2.5D+fenotipik F2 | NYU | 37 | 0.481 | 0.473 |
| S4 2.5D+fenotipik F2 | OHSU | 6 | 0.333 | 0.222 |
| S4 2.5D+fenotipik F2 | OLIN | 6 | 0.667 | 0.778 |
| S4 2.5D+fenotipik F2 | PITT | 11 | 0.25 | 0.133 |
| S4 2.5D+fenotipik F2 | SBL | 6 | 0.333 | 0.667 |
| S4 2.5D+fenotipik F2 | SDSU | 6 | 0.875 | 1.0 |
| S4 2.5D+fenotipik F2 | STANFORD | 5 | 0.583 | 0.667 |
| S4 2.5D+fenotipik F2 | TRINITY | 9 | 0.325 | 0.35 |
| S4 2.5D+fenotipik F2 | UCLA_1 | 13 | 0.476 | 0.476 |
| S4 2.5D+fenotipik F2 | UCLA_2 | 4 | 0.25 | 0.5 |
| S4 2.5D+fenotipik F2 | UM_1 | 18 | 0.351 | 0.416 |
| S4 2.5D+fenotipik F2 | UM_2 | 7 | 0.333 | 0.333 |
| S4 2.5D+fenotipik F2 | USM | 18 | 0.604 | 0.584 |
| S4 2.5D+fenotipik F2 | YALE | 10 | 0.3 | 0.24 |
| S5 3D+fenotipik F1 | CALTECH | 8 | 0.625 | 0.5 |
| S5 3D+fenotipik F1 | CMU | 5 | 0.667 | 1.0 |
| S5 3D+fenotipik F1 | KKI | 8 | 0.667 | 0.667 |
| S5 3D+fenotipik F1 | LEUVEN_1 | 6 | 0.667 | 0.333 |
| S5 3D+fenotipik F1 | LEUVEN_2 | 7 | 0.5 | 0.25 |
| S5 3D+fenotipik F1 | MAX_MUN | 10 | 0.5 | 0.458 |
| S5 3D+fenotipik F1 | NYU | 37 | 0.484 | 0.372 |
| S5 3D+fenotipik F1 | OHSU | 6 | 0.667 | 0.667 |
| S5 3D+fenotipik F1 | OLIN | 6 | 0.5 | 0.778 |
| S5 3D+fenotipik F1 | PITT | 11 | 0.567 | 0.4 |
| S5 3D+fenotipik F1 | SBL | 6 | 0.5 | 0.444 |
| S5 3D+fenotipik F1 | SDSU | 6 | 0.375 | 0.125 |
| S5 3D+fenotipik F1 | STANFORD | 5 | 0.5 | 0.667 |
| S5 3D+fenotipik F1 | TRINITY | 9 | 0.525 | 0.35 |
| S5 3D+fenotipik F1 | UCLA_1 | 13 | 0.571 | 0.548 |
| S5 3D+fenotipik F1 | UCLA_2 | 4 | 0.5 | 0.0 |
| S5 3D+fenotipik F1 | UM_1 | 18 | 0.526 | 0.468 |
| S5 3D+fenotipik F1 | UM_2 | 7 | 0.5 | 0.417 |
| S5 3D+fenotipik F1 | USM | 18 | 0.474 | 0.338 |
| S5 3D+fenotipik F1 | YALE | 10 | 0.6 | 0.68 |
| S5 3D+fenotipik F2 | CALTECH | 8 | 0.625 | 0.625 |
| S5 3D+fenotipik F2 | CMU | 5 | 0.667 | 0.333 |
| S5 3D+fenotipik F2 | KKI | 8 | 0.467 | 0.467 |
| S5 3D+fenotipik F2 | LEUVEN_1 | 6 | 0.667 | 0.444 |
| S5 3D+fenotipik F2 | LEUVEN_2 | 7 | 0.75 | 0.583 |
| S5 3D+fenotipik F2 | MAX_MUN | 10 | 0.625 | 0.542 |
| S5 3D+fenotipik F2 | NYU | 37 | 0.418 | 0.342 |
| S5 3D+fenotipik F2 | OHSU | 6 | 0.5 | 0.556 |
| S5 3D+fenotipik F2 | OLIN | 6 | 0.5 | 0.556 |
| S5 3D+fenotipik F2 | PITT | 11 | 0.667 | 0.467 |
| S5 3D+fenotipik F2 | SBL | 6 | 0.5 | 0.333 |
| S5 3D+fenotipik F2 | SDSU | 6 | 0.375 | 0.0 |
| S5 3D+fenotipik F2 | STANFORD | 5 | 0.167 | 0.167 |
| S5 3D+fenotipik F2 | TRINITY | 9 | 0.65 | 0.65 |
| S5 3D+fenotipik F2 | UCLA_1 | 13 | 0.31 | 0.333 |
| S5 3D+fenotipik F2 | UCLA_2 | 4 | 0.25 | 0.0 |
| S5 3D+fenotipik F2 | UM_1 | 18 | 0.513 | 0.442 |
| S5 3D+fenotipik F2 | UM_2 | 7 | 0.125 | 0.083 |
| S5 3D+fenotipik F2 | USM | 18 | 0.442 | 0.481 |
| S5 3D+fenotipik F2 | YALE | 10 | 0.4 | 0.28 |

## Kelompok model
- **CNN/ML klasik**: logreg, xgb, mlp_tab
- **arsitektur modern**: tabm, convnextv2_t
- **CNN klasik**: effnet_b0, resnet18
- **CNN klasik (pretrained otak)**: sfcn
- **foundation model**: dinov2_s

## Kandidat untuk data penuh (2–3 per modalitas, urut AUC inner val)
- pheno: logreg
- img25d: dinov2_s
- img3d: sfcn

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