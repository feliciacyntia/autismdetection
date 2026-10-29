# Hasil eksperimen pilot200 (config_hash 1797fe5bf1d67157)

Status unit: {'done': 57}

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
| pheno | mlp_tab | CNN/ML klasik | 0.528 |
| pheno | tabm | arsitektur modern | 0.533 |
| img25d | effnet_b0 | CNN klasik | 0.466 |
| img25d | resnet18 | CNN klasik | 0.48 |
| img25d | dinov2_s | foundation model | 0.493 |
| img25d | convnextv2_t | arsitektur modern | 0.532 |
| img3d | sfcn | CNN klasik (pretrained otak) | 0.511 |

## Tabel A — fenotipik (test fold, mean ± SD; pooled OOF + 95% CI bootstrap)
★ = terbaik menurut AUC pooled inner validation lima fold (aturan seleksi, bukan skor test).

| model | kelompok | status | terbaik | AUC inner val | BA (mean±SD) | F1m (mean±SD) | AUC (mean±SD) | Sens (mean±SD) | Spes (mean±SD) | BA pooled [95% CI] | AUC pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | baseline (S0) | done |  | 0.5 | 0.500±0.000 | 0.344±0.015 | 0.500±0.000 | 0.000±0.000 | 1.000±0.000 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| logreg | CNN/ML klasik | done |  | 0.493 | 0.584±0.088 | 0.582±0.090 | 0.577±0.084 | 0.557±0.143 | 0.611±0.106 | 0.584 [0.515, 0.649] | 0.580 [0.497, 0.659] |
| xgb | CNN/ML klasik | done | ★ | 0.553 | 0.534±0.084 | 0.509±0.104 | 0.550±0.108 | 0.497±0.183 | 0.571±0.275 | 0.523 [0.457, 0.594] | 0.537 [0.461, 0.616] |
| mlp_tab | CNN/ML klasik | done |  | 0.528 | 0.535±0.043 | 0.495±0.102 | 0.579±0.060 | 0.551±0.262 | 0.519±0.301 | 0.536 [0.461, 0.605] | 0.569 [0.483, 0.648] |
| tabm | arsitektur modern | done |  | 0.533 | 0.535±0.081 | 0.528±0.080 | 0.563±0.055 | 0.528±0.182 | 0.541±0.139 | 0.535 [0.464, 0.602] | 0.535 [0.450, 0.615] |

Tidak ada model dengan BA pooled > 0.75.

## Tabel B — citra 2.5D & 3D
| model | kelompok | status | AUC inner val | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] | params (latih/total) | VRAM puncak (MiB) | s/epoch | waktu total (menit) | epoch terbaik (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| effnet_b0 | CNN klasik | done | 0.466 | 0.513 ± 0.037 | 0.526 [0.462, 0.589] | 0.446 ± 0.082 | 0.515 [0.446, 0.582] | 0.489 ± 0.101 | 0.520 [0.438, 0.598] | 0.338 ± 0.377 | 0.347 [0.250, 0.441] | 0.687 ± 0.309 | 0.705 [0.615, 0.790] | 3319837/4171645 | 483 | 4.8 | 3.0 | 1 |
| resnet18 | CNN klasik | done | 0.48 | 0.509 ± 0.020 | 0.493 [0.428, 0.559] | 0.361 ± 0.077 | 0.485 [0.413, 0.552] | 0.479 ± 0.085 | 0.485 [0.404, 0.566] | 0.370 ± 0.509 | 0.347 [0.250, 0.442] | 0.648 ± 0.490 | 0.638 [0.544, 0.728] | 10559233/11242305 | 488 | 4.2 | 2.7 | 2 |
| dinov2_s | foundation model | done | 0.493 | 0.487 ± 0.018 | 0.481 [0.415, 0.549] | 0.382 ± 0.066 | 0.480 [0.414, 0.548] | 0.490 ± 0.109 | 0.479 [0.396, 0.558] | 0.415 ± 0.428 | 0.411 [0.312, 0.509] | 0.559 ± 0.438 | 0.552 [0.459, 0.649] | 100097/100097 | 21 | 0.0 | 0.1 | 3 |
| convnextv2_t | arsitektur modern | done | 0.532 | 0.503 ± 0.060 | 0.505 [0.437, 0.571] | 0.394 ± 0.079 | 0.502 [0.431, 0.569] | 0.504 ± 0.151 | 0.495 [0.414, 0.576] | 0.380 ± 0.437 | 0.400 [0.302, 0.500] | 0.625 ± 0.473 | 0.610 [0.511, 0.706] | 15586817/27965057 | 818 | 18.9 | 15.3 | 2 |
| sfcn | CNN klasik (pretrained otak) | done | 0.511 | 0.515 ± 0.035 | 0.520 [0.468, 0.578] | 0.403 ± 0.083 | 0.480 [0.413, 0.549] | 0.494 ± 0.098 | 0.501 [0.423, 0.583] | 0.210 ± 0.347 | 0.221 [0.133, 0.309] | 0.819 ± 0.354 | 0.819 [0.745, 0.888] | 1795265/1795265 | 142 | 0.2 | 0.4 | 6 |

## Tabel C — fusion F1 vs F2 (utama: F1 = stacking LogReg, F2 = PCA16 + LogReg; ditetapkan sebelum hasil)
| skenario | citra | fenotipik | metode | jenis | utama | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S4 | convnextv2_t | xgb | F1_mean_prob | F1 late |  | 0.525 ± 0.090 | 0.524 [0.459, 0.590] | 0.517 ± 0.082 | 0.519 [0.449, 0.585] | 0.558 ± 0.101 | 0.532 [0.454, 0.613] | 0.396 ± 0.077 | 0.400 [0.302, 0.495] | 0.654 ± 0.153 | 0.648 [0.553, 0.740] |
| S4 | convnextv2_t | xgb | F1_stack_logreg | F1 late | ya | 0.516 ± 0.036 | 0.511 [0.443, 0.581] | 0.505 ± 0.048 | 0.510 [0.439, 0.578] | 0.560 ± 0.104 | 0.538 [0.460, 0.617] | 0.542 ± 0.099 | 0.537 [0.436, 0.635] | 0.491 ± 0.162 | 0.486 [0.390, 0.587] |
| S4 | convnextv2_t | xgb | F2_pca16_lgbm | F2 intermediate |  | 0.459 ± 0.090 | 0.457 [0.389, 0.521] | 0.453 ± 0.096 | 0.456 [0.385, 0.520] | 0.449 ± 0.125 | 0.435 [0.354, 0.514] | 0.392 ± 0.115 | 0.389 [0.293, 0.489] | 0.526 ± 0.135 | 0.524 [0.426, 0.616] |
| S4 | convnextv2_t | xgb | F2_pca16_logreg | F2 intermediate | ya | 0.481 ± 0.099 | 0.475 [0.404, 0.543] | 0.474 ± 0.094 | 0.475 [0.403, 0.540] | 0.464 ± 0.083 | 0.455 [0.375, 0.532] | 0.494 ± 0.182 | 0.484 [0.387, 0.580] | 0.469 ± 0.073 | 0.467 [0.365, 0.556] |
| S4 | convnextv2_t | xgb | F2_pca16_mlp | F2 intermediate |  | 0.491 ± 0.034 | 0.486 [0.415, 0.556] | 0.483 ± 0.027 | 0.485 [0.410, 0.554] | 0.483 ± 0.068 | 0.476 [0.394, 0.557] | 0.522 ± 0.121 | 0.516 [0.416, 0.616] | 0.461 ± 0.086 | 0.457 [0.356, 0.549] |
| S4 | convnextv2_t | xgb | F2_pca8_lgbm | F2 intermediate |  | 0.496 ± 0.100 | 0.491 [0.420, 0.564] | 0.484 ± 0.107 | 0.490 [0.417, 0.562] | 0.483 ± 0.116 | 0.469 [0.389, 0.551] | 0.427 ± 0.182 | 0.421 [0.322, 0.524] | 0.565 ± 0.161 | 0.562 [0.462, 0.657] |
| S4 | convnextv2_t | xgb | F2_pca8_logreg | F2 intermediate |  | 0.470 ± 0.073 | 0.467 [0.394, 0.536] | 0.460 ± 0.078 | 0.465 [0.390, 0.530] | 0.463 ± 0.100 | 0.459 [0.378, 0.541] | 0.508 ± 0.141 | 0.505 [0.404, 0.604] | 0.433 ± 0.130 | 0.429 [0.333, 0.523] |
| S4 | convnextv2_t | xgb | F2_pca8_mlp | F2 intermediate |  | 0.529 ± 0.087 | 0.527 [0.456, 0.600] | 0.518 ± 0.090 | 0.525 [0.455, 0.595] | 0.484 ± 0.131 | 0.496 [0.414, 0.579] | 0.558 ± 0.173 | 0.558 [0.461, 0.659] | 0.500 ± 0.124 | 0.495 [0.394, 0.590] |
| S4 | convnextv2_t | xgb | F2_raw128_lgbm | F2 intermediate |  | 0.507 ± 0.117 | 0.501 [0.433, 0.570] | 0.499 ± 0.115 | 0.500 [0.430, 0.568] | 0.508 ± 0.099 | 0.500 [0.423, 0.582] | 0.535 ± 0.187 | 0.526 [0.425, 0.626] | 0.480 ± 0.118 | 0.476 [0.381, 0.571] |
| S4 | convnextv2_t | xgb | F2_raw128_logreg | F2 intermediate |  | 0.520 ± 0.061 | 0.513 [0.441, 0.579] | 0.505 ± 0.057 | 0.510 [0.435, 0.576] | 0.470 ± 0.088 | 0.464 [0.382, 0.540] | 0.576 ± 0.167 | 0.568 [0.462, 0.670] | 0.464 ± 0.131 | 0.457 [0.361, 0.548] |
| S4 | convnextv2_t | xgb | F2_raw128_mlp | F2 intermediate |  | 0.498 ± 0.109 | 0.495 [0.425, 0.566] | 0.486 ± 0.119 | 0.495 [0.424, 0.564] | 0.483 ± 0.094 | 0.478 [0.393, 0.561] | 0.498 ± 0.155 | 0.495 [0.393, 0.598] | 0.499 ± 0.200 | 0.495 [0.397, 0.590] |
| S5 | sfcn | xgb | F1_mean_prob | F1 late |  | 0.536 ± 0.081 | 0.532 [0.469, 0.599] | 0.524 ± 0.071 | 0.525 [0.456, 0.595] | 0.539 ± 0.116 | 0.535 [0.459, 0.615] | 0.382 ± 0.056 | 0.379 [0.280, 0.478] | 0.690 ± 0.177 | 0.686 [0.598, 0.774] |
| S5 | sfcn | xgb | F1_stack_logreg | F1 late | ya | 0.514 ± 0.053 | 0.508 [0.439, 0.576] | 0.492 ± 0.080 | 0.505 [0.434, 0.574] | 0.551 ± 0.102 | 0.536 [0.458, 0.616] | 0.567 ± 0.149 | 0.558 [0.456, 0.656] | 0.462 ± 0.229 | 0.457 [0.362, 0.557] |
| S5 | sfcn | xgb | F2_pca16_lgbm | F2 intermediate |  | 0.548 ± 0.090 | 0.541 [0.471, 0.610] | 0.535 ± 0.092 | 0.540 [0.469, 0.610] | 0.517 ± 0.093 | 0.512 [0.431, 0.594] | 0.564 ± 0.200 | 0.558 [0.455, 0.656] | 0.531 ± 0.128 | 0.524 [0.430, 0.618] |
| S5 | sfcn | xgb | F2_pca16_logreg | F2 intermediate | ya | 0.515 ± 0.073 | 0.515 [0.438, 0.585] | 0.511 ± 0.073 | 0.515 [0.436, 0.584] | 0.540 ± 0.079 | 0.533 [0.447, 0.613] | 0.515 ± 0.111 | 0.516 [0.412, 0.615] | 0.515 ± 0.139 | 0.514 [0.413, 0.600] |
| S5 | sfcn | xgb | F2_pca16_mlp | F2 intermediate |  | 0.535 ± 0.117 | 0.533 [0.459, 0.599] | 0.529 ± 0.120 | 0.532 [0.459, 0.597] | 0.555 ± 0.111 | 0.544 [0.463, 0.622] | 0.486 ± 0.145 | 0.484 [0.382, 0.582] | 0.584 ± 0.141 | 0.581 [0.486, 0.670] |
| S5 | sfcn | xgb | F2_pca8_lgbm | F2 intermediate |  | 0.493 ± 0.137 | 0.486 [0.416, 0.556] | 0.482 ± 0.142 | 0.485 [0.412, 0.553] | 0.525 ± 0.128 | 0.527 [0.441, 0.605] | 0.420 ± 0.235 | 0.411 [0.310, 0.509] | 0.566 ± 0.088 | 0.562 [0.465, 0.659] |
| S5 | sfcn | xgb | F2_pca8_logreg | F2 intermediate |  | 0.519 ± 0.066 | 0.520 [0.445, 0.589] | 0.517 ± 0.065 | 0.520 [0.443, 0.585] | 0.533 ± 0.090 | 0.531 [0.445, 0.608] | 0.525 ± 0.109 | 0.526 [0.424, 0.624] | 0.514 ± 0.117 | 0.514 [0.415, 0.604] |
| S5 | sfcn | xgb | F2_pca8_mlp | F2 intermediate |  | 0.548 ± 0.117 | 0.547 [0.477, 0.614] | 0.540 ± 0.123 | 0.547 [0.476, 0.613] | 0.579 ± 0.142 | 0.571 [0.489, 0.647] | 0.495 ± 0.148 | 0.495 [0.393, 0.596] | 0.602 ± 0.179 | 0.600 [0.505, 0.686] |
| S5 | sfcn | xgb | F2_raw128_lgbm | F2 intermediate |  | 0.507 ± 0.122 | 0.504 [0.432, 0.573] | 0.500 ± 0.127 | 0.503 [0.430, 0.572] | 0.524 ± 0.131 | 0.528 [0.444, 0.610] | 0.482 ± 0.226 | 0.474 [0.374, 0.574] | 0.532 ± 0.034 | 0.533 [0.435, 0.630] |
| S5 | sfcn | xgb | F2_raw128_logreg | F2 intermediate |  | 0.506 ± 0.064 | 0.506 [0.430, 0.577] | 0.502 ± 0.063 | 0.505 [0.429, 0.575] | 0.531 ± 0.066 | 0.529 [0.445, 0.610] | 0.515 ± 0.111 | 0.516 [0.412, 0.615] | 0.497 ± 0.116 | 0.495 [0.396, 0.587] |
| S5 | sfcn | xgb | F2_raw128_mlp | F2 intermediate |  | 0.484 ± 0.076 | 0.482 [0.414, 0.548] | 0.479 ± 0.079 | 0.482 [0.412, 0.545] | 0.541 ± 0.117 | 0.538 [0.455, 0.616] | 0.433 ± 0.104 | 0.432 [0.333, 0.531] | 0.536 ± 0.116 | 0.533 [0.435, 0.623] |

## Tabel D — ablasi S0–S5
| skenario | BA fold | BA pooled [95% CI] | F1m fold | F1m pooled [95% CI] | AUC fold | AUC pooled [95% CI] | Sens fold | Sens pooled [95% CI] | Spes fold | Spes pooled [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 majority | 0.500 ± 0.000 | 0.500 [0.500, 0.500] | 0.344 ± 0.015 | 0.344 [0.313, 0.373] | 0.500 ± 0.000 | 0.500 [0.500, 0.500] | 0.000 ± 0.000 | 0.000 [0.000, 0.000] | 1.000 ± 0.000 | 1.000 [1.000, 1.000] |
| S1 fenotipik (xgb) | 0.534 ± 0.084 | 0.523 [0.457, 0.594] | 0.509 ± 0.104 | 0.523 [0.455, 0.593] | 0.550 ± 0.108 | 0.537 [0.461, 0.616] | 0.497 ± 0.183 | 0.484 [0.384, 0.581] | 0.571 ± 0.275 | 0.562 [0.468, 0.663] |
| S2 2.5D (convnextv2_t) | 0.503 ± 0.060 | 0.505 [0.437, 0.571] | 0.394 ± 0.079 | 0.502 [0.431, 0.569] | 0.504 ± 0.151 | 0.495 [0.414, 0.576] | 0.380 ± 0.437 | 0.400 [0.302, 0.500] | 0.625 ± 0.473 | 0.610 [0.511, 0.706] |
| S3 3D (sfcn) | 0.515 ± 0.035 | 0.520 [0.468, 0.578] | 0.403 ± 0.083 | 0.480 [0.413, 0.549] | 0.494 ± 0.098 | 0.501 [0.423, 0.583] | 0.210 ± 0.347 | 0.221 [0.133, 0.309] | 0.819 ± 0.354 | 0.819 [0.745, 0.888] |
| S4 2.5D+fenotipik F1 | 0.516 ± 0.036 | 0.511 [0.443, 0.581] | 0.505 ± 0.048 | 0.510 [0.439, 0.578] | 0.560 ± 0.104 | 0.538 [0.460, 0.617] | 0.542 ± 0.099 | 0.537 [0.436, 0.635] | 0.491 ± 0.162 | 0.486 [0.390, 0.587] |
| S4 2.5D+fenotipik F2 | 0.481 ± 0.099 | 0.475 [0.404, 0.543] | 0.474 ± 0.094 | 0.475 [0.403, 0.540] | 0.464 ± 0.083 | 0.455 [0.375, 0.532] | 0.494 ± 0.182 | 0.484 [0.387, 0.580] | 0.469 ± 0.073 | 0.467 [0.365, 0.556] |
| S5 3D+fenotipik F1 | 0.514 ± 0.053 | 0.508 [0.439, 0.576] | 0.492 ± 0.080 | 0.505 [0.434, 0.574] | 0.551 ± 0.102 | 0.536 [0.458, 0.616] | 0.567 ± 0.149 | 0.558 [0.456, 0.656] | 0.462 ± 0.229 | 0.457 [0.362, 0.557] |
| S5 3D+fenotipik F2 | 0.515 ± 0.073 | 0.515 [0.438, 0.585] | 0.511 ± 0.073 | 0.515 [0.436, 0.584] | 0.540 ± 0.079 | 0.533 [0.447, 0.613] | 0.515 ± 0.111 | 0.516 [0.412, 0.615] | 0.515 ± 0.139 | 0.514 [0.413, 0.600] |

## Tabel E — site learning (prediksi SITE_ID dari embedding citra)
| embedding | BA prediksi SITE_ID (test) | peluang acak | n situs |
|---|---|---|---|
| effnet_b0 | 0.586 | 0.05 | 20 |
| resnet18 | 0.48 | 0.05 | 20 |
| dinov2_s | 0.625 | 0.05 | 20 |
| convnextv2_t | 0.518 | 0.05 | 20 |
| sfcn | 0.303 | 0.05 | 20 |

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
| S2 2.5D (convnextv2_t) | CALTECH | 8 | 0.625 | 0.625 |
| S2 2.5D (convnextv2_t) | CMU | 5 | 0.583 | 0.833 |
| S2 2.5D (convnextv2_t) | KKI | 8 | 0.4 | 0.6 |
| S2 2.5D (convnextv2_t) | LEUVEN_1 | 6 | 0.833 | 0.667 |
| S2 2.5D (convnextv2_t) | LEUVEN_2 | 7 | 0.708 | 0.75 |
| S2 2.5D (convnextv2_t) | MAX_MUN | 10 | 0.5 | 0.625 |
| S2 2.5D (convnextv2_t) | NYU | 37 | 0.497 | 0.491 |
| S2 2.5D (convnextv2_t) | OHSU | 6 | 0.333 | 0.333 |
| S2 2.5D (convnextv2_t) | OLIN | 6 | 0.167 | 0.333 |
| S2 2.5D (convnextv2_t) | PITT | 11 | 0.55 | 0.667 |
| S2 2.5D (convnextv2_t) | SBL | 6 | 0.667 | 0.667 |
| S2 2.5D (convnextv2_t) | SDSU | 6 | 0.625 | 0.375 |
| S2 2.5D (convnextv2_t) | STANFORD | 5 | 0.25 | 0.0 |
| S2 2.5D (convnextv2_t) | TRINITY | 9 | 0.55 | 0.45 |
| S2 2.5D (convnextv2_t) | UCLA_1 | 13 | 0.393 | 0.381 |
| S2 2.5D (convnextv2_t) | UCLA_2 | 4 | 0.25 | 0.0 |
| S2 2.5D (convnextv2_t) | UM_1 | 18 | 0.487 | 0.545 |
| S2 2.5D (convnextv2_t) | UM_2 | 7 | 0.417 | 0.5 |
| S2 2.5D (convnextv2_t) | USM | 18 | 0.513 | 0.403 |
| S2 2.5D (convnextv2_t) | YALE | 10 | 0.6 | 0.52 |
| S3 3D (sfcn) | CALTECH | 8 | 0.5 | 0.438 |
| S3 3D (sfcn) | CMU | 5 | 0.667 | 1.0 |
| S3 3D (sfcn) | KKI | 8 | 0.567 | 0.467 |
| S3 3D (sfcn) | LEUVEN_1 | 6 | 0.5 | 0.667 |
| S3 3D (sfcn) | LEUVEN_2 | 7 | 0.708 | 0.667 |
| S3 3D (sfcn) | MAX_MUN | 10 | 0.625 | 0.708 |
| S3 3D (sfcn) | NYU | 37 | 0.561 | 0.533 |
| S3 3D (sfcn) | OHSU | 6 | 0.333 | 0.222 |
| S3 3D (sfcn) | OLIN | 6 | 0.333 | 0.667 |
| S3 3D (sfcn) | PITT | 11 | 0.567 | 0.6 |
| S3 3D (sfcn) | SBL | 6 | 0.5 | 0.556 |
| S3 3D (sfcn) | SDSU | 6 | 0.375 | 0.0 |
| S3 3D (sfcn) | STANFORD | 5 | 0.167 | 0.167 |
| S3 3D (sfcn) | TRINITY | 9 | 0.525 | 0.5 |
| S3 3D (sfcn) | UCLA_1 | 13 | 0.488 | 0.524 |
| S3 3D (sfcn) | UCLA_2 | 4 | 0.5 | 0.5 |
| S3 3D (sfcn) | UM_1 | 18 | 0.526 | 0.468 |
| S3 3D (sfcn) | UM_2 | 7 | 0.5 | 0.583 |
| S3 3D (sfcn) | USM | 18 | 0.519 | 0.39 |
| S3 3D (sfcn) | YALE | 10 | 0.5 | 0.4 |
| S4 2.5D+fenotipik F1 | CALTECH | 8 | 0.625 | 0.812 |
| S4 2.5D+fenotipik F1 | CMU | 5 | 0.417 | 0.333 |
| S4 2.5D+fenotipik F1 | KKI | 8 | 0.8 | 0.8 |
| S4 2.5D+fenotipik F1 | LEUVEN_1 | 6 | 0.5 | 0.556 |
| S4 2.5D+fenotipik F1 | LEUVEN_2 | 7 | 0.292 | 0.417 |
| S4 2.5D+fenotipik F1 | MAX_MUN | 10 | 0.542 | 0.583 |
| S4 2.5D+fenotipik F1 | NYU | 37 | 0.488 | 0.554 |
| S4 2.5D+fenotipik F1 | OHSU | 6 | 0.833 | 0.889 |
| S4 2.5D+fenotipik F1 | OLIN | 6 | 0.5 | 0.667 |
| S4 2.5D+fenotipik F1 | PITT | 11 | 0.55 | 0.533 |
| S4 2.5D+fenotipik F1 | SBL | 6 | 0.167 | 0.111 |
| S4 2.5D+fenotipik F1 | SDSU | 6 | 0.625 | 0.75 |
| S4 2.5D+fenotipik F1 | STANFORD | 5 | 0.5 | 0.667 |
| S4 2.5D+fenotipik F1 | TRINITY | 9 | 0.3 | 0.15 |
| S4 2.5D+fenotipik F1 | UCLA_1 | 13 | 0.381 | 0.381 |
| S4 2.5D+fenotipik F1 | UCLA_2 | 4 | 0.25 | 0.0 |
| S4 2.5D+fenotipik F1 | UM_1 | 18 | 0.513 | 0.584 |
| S4 2.5D+fenotipik F1 | UM_2 | 7 | 0.667 | 0.667 |
| S4 2.5D+fenotipik F1 | USM | 18 | 0.552 | 0.623 |
| S4 2.5D+fenotipik F1 | YALE | 10 | 0.5 | 0.56 |
| S4 2.5D+fenotipik F2 | CALTECH | 8 | 0.5 | 0.375 |
| S4 2.5D+fenotipik F2 | CMU | 5 | 0.417 | 0.167 |
| S4 2.5D+fenotipik F2 | KKI | 8 | 0.467 | 0.6 |
| S4 2.5D+fenotipik F2 | LEUVEN_1 | 6 | 0.667 | 1.0 |
| S4 2.5D+fenotipik F2 | LEUVEN_2 | 7 | 0.417 | 0.583 |
| S4 2.5D+fenotipik F2 | MAX_MUN | 10 | 0.417 | 0.458 |
| S4 2.5D+fenotipik F2 | NYU | 37 | 0.481 | 0.432 |
| S4 2.5D+fenotipik F2 | OHSU | 6 | 0.667 | 0.778 |
| S4 2.5D+fenotipik F2 | OLIN | 6 | 0.667 | 0.778 |
| S4 2.5D+fenotipik F2 | PITT | 11 | 0.483 | 0.433 |
| S4 2.5D+fenotipik F2 | SBL | 6 | 0.667 | 0.889 |
| S4 2.5D+fenotipik F2 | SDSU | 6 | 0.5 | 0.5 |
| S4 2.5D+fenotipik F2 | STANFORD | 5 | 0.417 | 0.5 |
| S4 2.5D+fenotipik F2 | TRINITY | 9 | 0.325 | 0.45 |
| S4 2.5D+fenotipik F2 | UCLA_1 | 13 | 0.536 | 0.452 |
| S4 2.5D+fenotipik F2 | UCLA_2 | 4 | 0.25 | 0.0 |
| S4 2.5D+fenotipik F2 | UM_1 | 18 | 0.416 | 0.338 |
| S4 2.5D+fenotipik F2 | UM_2 | 7 | 0.333 | 0.333 |
| S4 2.5D+fenotipik F2 | USM | 18 | 0.435 | 0.26 |
| S4 2.5D+fenotipik F2 | YALE | 10 | 0.4 | 0.4 |
| S5 3D+fenotipik F1 | CALTECH | 8 | 0.625 | 0.875 |
| S5 3D+fenotipik F1 | CMU | 5 | 0.417 | 0.5 |
| S5 3D+fenotipik F1 | KKI | 8 | 0.7 | 0.8 |
| S5 3D+fenotipik F1 | LEUVEN_1 | 6 | 0.667 | 0.556 |
| S5 3D+fenotipik F1 | LEUVEN_2 | 7 | 0.292 | 0.25 |
| S5 3D+fenotipik F1 | MAX_MUN | 10 | 0.542 | 0.625 |
| S5 3D+fenotipik F1 | NYU | 37 | 0.543 | 0.562 |
| S5 3D+fenotipik F1 | OHSU | 6 | 0.667 | 0.778 |
| S5 3D+fenotipik F1 | OLIN | 6 | 0.667 | 0.667 |
| S5 3D+fenotipik F1 | PITT | 11 | 0.45 | 0.533 |
| S5 3D+fenotipik F1 | SBL | 6 | 0.167 | 0.0 |
| S5 3D+fenotipik F1 | SDSU | 6 | 0.5 | 0.75 |
| S5 3D+fenotipik F1 | STANFORD | 5 | 0.5 | 0.667 |
| S5 3D+fenotipik F1 | TRINITY | 9 | 0.2 | 0.15 |
| S5 3D+fenotipik F1 | UCLA_1 | 13 | 0.381 | 0.381 |
| S5 3D+fenotipik F1 | UCLA_2 | 4 | 0.25 | 0.0 |
| S5 3D+fenotipik F1 | UM_1 | 18 | 0.513 | 0.584 |
| S5 3D+fenotipik F1 | UM_2 | 7 | 0.667 | 0.667 |
| S5 3D+fenotipik F1 | USM | 18 | 0.552 | 0.636 |
| S5 3D+fenotipik F1 | YALE | 10 | 0.5 | 0.56 |
| S5 3D+fenotipik F2 | CALTECH | 8 | 0.5 | 0.25 |
| S5 3D+fenotipik F2 | CMU | 5 | 0.833 | 1.0 |
| S5 3D+fenotipik F2 | KKI | 8 | 0.633 | 0.667 |
| S5 3D+fenotipik F2 | LEUVEN_1 | 6 | 0.333 | 0.333 |
| S5 3D+fenotipik F2 | LEUVEN_2 | 7 | 0.75 | 0.583 |
| S5 3D+fenotipik F2 | MAX_MUN | 10 | 0.625 | 0.583 |
| S5 3D+fenotipik F2 | NYU | 37 | 0.504 | 0.577 |
| S5 3D+fenotipik F2 | OHSU | 6 | 0.5 | 0.444 |
| S5 3D+fenotipik F2 | OLIN | 6 | 0.5 | 0.444 |
| S5 3D+fenotipik F2 | PITT | 11 | 0.467 | 0.433 |
| S5 3D+fenotipik F2 | SBL | 6 | 0.333 | 0.333 |
| S5 3D+fenotipik F2 | SDSU | 6 | 0.25 | 0.375 |
| S5 3D+fenotipik F2 | STANFORD | 5 | 0.417 | 0.5 |
| S5 3D+fenotipik F2 | TRINITY | 9 | 0.45 | 0.5 |
| S5 3D+fenotipik F2 | UCLA_1 | 13 | 0.536 | 0.5 |
| S5 3D+fenotipik F2 | UCLA_2 | 4 | 0.75 | 1.0 |
| S5 3D+fenotipik F2 | UM_1 | 18 | 0.604 | 0.532 |
| S5 3D+fenotipik F2 | UM_2 | 7 | 0.375 | 0.167 |
| S5 3D+fenotipik F2 | USM | 18 | 0.461 | 0.468 |
| S5 3D+fenotipik F2 | YALE | 10 | 0.4 | 0.48 |

## Kelompok model
- **CNN/ML klasik**: logreg, xgb, mlp_tab
- **arsitektur modern**: tabm, convnextv2_t
- **CNN klasik**: effnet_b0, resnet18
- **CNN klasik (pretrained otak)**: sfcn
- **foundation model**: dinov2_s

## Kandidat untuk data penuh (2–3 per modalitas, urut AUC inner val)
- pheno: logreg, xgb, tabm
- img25d: convnextv2_t, dinov2_s, resnet18
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