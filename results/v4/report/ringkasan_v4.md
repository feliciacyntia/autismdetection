# Analisis v4 — morfometri FreeSurfer PCP (853 subjek dev, 5 fold)

Model morfometri terpilih (skor inner_oof): **E1 · morfometri relatif ICV + ComBat · TabPFN v2**, AUC inner 0.605; masuk ensemble: True.

## Ablasi (threshold 0,5)

| komponen | Accuracy | BA | F1 macro | AUC | Sens | Spes |
|---|---|---|---|---|---|---|
| fenotipik (E3 · fenotipik · TabM (5 seed)) | 0.596 [0.562, 0.628] | 0.590 [0.556, 0.621] | 0.589 [0.555, 0.620] | 0.627 [0.588, 0.664] | 0.500 | 0.679 |
| ROI (E3 · ROI · LogReg) | 0.593 [0.560, 0.625] | 0.591 [0.558, 0.623] | 0.591 [0.558, 0.623] | 0.635 [0.598, 0.671] | 0.565 | 0.618 |
| morfometri (E1 · morfometri relatif ICV + ComBat · TabPFN v2) | 0.571 [0.537, 0.603] | 0.567 [0.533, 0.600] | 0.567 [0.532, 0.599] | 0.608 [0.569, 0.643] | 0.515 | 0.620 |
| citra 2.5D (full_v2 · effnet_b0) | 0.571 [0.539, 0.605] | 0.571 [0.539, 0.605] | 0.570 [0.538, 0.605] | 0.591 [0.552, 0.629] | 0.578 | 0.565 |
| citra 3D (full_v2 · scratch3d) | 0.551 [0.518, 0.585] | 0.556 [0.524, 0.588] | 0.551 [0.518, 0.584] | 0.567 [0.529, 0.606] | 0.623 | 0.488 |
| fenotipik + ROI | 0.657 [0.624, 0.688] | 0.654 [0.621, 0.685] | 0.654 [0.622, 0.686] | 0.686 [0.650, 0.723] | 0.613 | 0.695 |
| fenotipik + morfometri | 0.628 [0.596, 0.660] | 0.625 [0.592, 0.657] | 0.625 [0.592, 0.657] | 0.665 [0.628, 0.700] | 0.573 | 0.677 |
| fenotipik + ROI + morfometri | 0.642 [0.611, 0.674] | 0.639 [0.607, 0.670] | 0.639 [0.607, 0.670] | 0.693 [0.658, 0.727] | 0.588 | 0.690 |
| semua modalitas tanpa morfometri (ENS_utama v3) | 0.660 [0.628, 0.691] | 0.657 [0.626, 0.688] | 0.658 [0.626, 0.688] | 0.692 [0.656, 0.727] | 0.618 | 0.697 |
| semua modalitas + morfometri (ENS_v4) | 0.652 [0.620, 0.682] | 0.649 [0.617, 0.680] | 0.649 [0.616, 0.680] | 0.697 [0.661, 0.732] | 0.606 | 0.692 |
| pembanding: full_v2 · fusion PR (F1_stack_logreg) | 0.628 [0.596, 0.659] | 0.624 [0.591, 0.654] | 0.624 [0.590, 0.654] | 0.664 [0.628, 0.700] | 0.553 | 0.695 |


## Uji utama (DeLong AUC, McNemar accuracy; Holm mencakup semua uji v4)

| model | pembanding | AUC model | AUC pembanding | ΔAUC | p DeLong | Acc model | Acc pembanding | ΔAcc | hanya pembanding benar | hanya model benar | p McNemar | p DeLong (Holm) | p McNemar (Holm) | signifikan (Holm < 0,05) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 · morfometri relatif ICV + ComBat · TabPFN v2 | ENS_utama v3 (model final) | 0.608 | 0.692 | -0.084 | 0.0003 | 0.571 | 0.66 | -0.089 | 205 | 129 | 0.0 | 0.0073 | 0.0012 | ya |
| E1 · morfometri relatif ICV + ComBat · TabPFN v2 | full_v2 · fusion PR (F1_stack_logreg) | 0.608 | 0.664 | -0.057 | 0.0215 | 0.571 | 0.628 | -0.057 | 193 | 144 | 0.0088 | 0.2146 | 0.106 | tidak |
| ENS_v4 (ENS_utama v3 + morfometri) | ENS_utama v3 (model final) | 0.697 | 0.692 | 0.006 | 0.6075 | 0.652 | 0.66 | -0.008 | 78 | 71 | 0.6232 | 1.0 | 1.0 | tidak |
| ENS_v4 (ENS_utama v3 + morfometri) | full_v2 · fusion PR (F1_stack_logreg) | 0.697 | 0.664 | 0.033 | 0.0382 | 0.652 | 0.628 | 0.023 | 96 | 116 | 0.1918 | 0.3438 | 1.0 | tidak |
| fenotipik + morfometri | ENS_utama v3 (model final) | 0.665 | 0.692 | -0.027 | 0.1485 | 0.628 | 0.66 | -0.032 | 140 | 113 | 0.1019 | 0.9494 | 0.8155 | tidak |
| fenotipik + morfometri | full_v2 · fusion PR (F1_stack_logreg) | 0.665 | 0.664 | 0.001 | 0.9761 | 0.628 | 0.628 | 0.0 | 120 | 120 | 1.0 | 1.0 | 1.0 | tidak |
| fenotipik + ROI + morfometri | ENS_utama v3 (model final) | 0.693 | 0.692 | 0.001 | 0.9291 | 0.642 | 0.66 | -0.018 | 108 | 93 | 0.3234 | 1.0 | 1.0 | tidak |
| fenotipik + ROI + morfometri | full_v2 · fusion PR (F1_stack_logreg) | 0.693 | 0.664 | 0.029 | 0.0655 | 0.642 | 0.628 | 0.014 | 98 | 110 | 0.4457 | 0.524 | 1.0 | tidak |
| E1 · morfometri relatif ICV + ComBat · TabPFN v2 | E3 · ROI · LogReg | 0.608 | 0.635 | -0.027 | 0.2511 | 0.571 | 0.593 | -0.022 | 190 | 171 | 0.3435 | 1.0 | 1.0 | tidak |
| E1 · morfometri relatif ICV + ComBat · TabPFN v2 | E3 · fenotipik · TabM (5 seed) | 0.608 | 0.627 | -0.019 | 0.46 | 0.571 | 0.596 | -0.025 | 203 | 182 | 0.3081 | 1.0 | 1.0 | tidak |


## LOSO

| model LOSO | padanan 5-fold | n subjek | n site | Accuracy LOSO [95% CI] | BA LOSO [95% CI] | AUC LOSO [95% CI] | Accuracy 5-fold (subjek sama) | BA 5-fold (subjek sama) | AUC 5-fold (subjek sama) |
|---|---|---|---|---|---|---|---|---|---|
| loso_morph_cb_tabpfn | E1 · morfometri relatif ICV + ComBat · TabPFN v2 | 812 | 18 | 0.544 [0.510, 0.580] | 0.545 [0.510, 0.581] | 0.584 [0.544, 0.624] | 0.574 | 0.57 | 0.608 |


## Konfounder

| model | ρ AGE_AT_SCAN (semua) | ρ AGE_AT_SCAN (TD saja) | ρ AGE_AT_SCAN (ASD saja) | ρ FIQ (semua) | ρ FIQ (TD saja) | ρ FIQ (ASD saja) | ρ volume_mask_ml (semua) | ρ volume_mask_ml (TD saja) | ρ volume_mask_ml (ASD saja) | ρ ICV (semua) | ρ ICV (TD saja) | ρ ICV (ASD saja) | Kruskal prob ~ site |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 · morfometri relatif ICV + ComBat · TabPFN v2 | -0.09 (p 0.010) | -0.07 (p 0.154) | -0.11 (p 0.028) | +0.00 (p 0.947) | +0.07 (p 0.150) | +0.01 (p 0.858) | +0.10 (p 0.003) | +0.06 (p 0.175) | +0.11 (p 0.022) | +0.06 (p 0.066) | -0.03 (p 0.558) | +0.14 (p 0.004) | H 14.1 (p 0.78) |
| ENS_v4 (ENS_utama v3 + morfometri) | -0.03 (p 0.346) | -0.02 (p 0.663) | -0.04 (p 0.375) | -0.23 (p 0.000) | -0.06 (p 0.205) | -0.33 (p 0.000) | +0.21 (p 0.000) | +0.16 (p 0.001) | +0.20 (p 0.000) | +0.20 (p 0.000) | +0.15 (p 0.001) | +0.25 (p 0.000) | H 26.9 (p 0.11) |
| ENS_utama v3 (model final) | +0.03 (p 0.333) | +0.04 (p 0.440) | +0.04 (p 0.447) | -0.33 (p 0.000) | -0.16 (p 0.001) | -0.44 (p 0.000) | +0.22 (p 0.000) | +0.17 (p 0.000) | +0.21 (p 0.000) | +0.25 (p 0.000) | +0.24 (p 0.000) | +0.25 (p 0.000) | H 35.5 (p 0.012) |
| E3 · ROI · LogReg | +0.02 (p 0.588) | -0.05 (p 0.244) | +0.10 (p 0.048) | +0.02 (p 0.649) | +0.08 (p 0.100) | +0.04 (p 0.416) | +0.11 (p 0.001) | +0.08 (p 0.073) | +0.10 (p 0.053) | +0.10 (p 0.004) | +0.10 (p 0.031) | +0.07 (p 0.162) | H 6.6 (p 1) |


## Semua model

| model | Accuracy pooled [95% CI] | BA pooled [95% CI] | F1 macro pooled [95% CI] | AUC pooled [95% CI] |
|---|---|---|---|---|
| E1 · morfometri mentah · LogReg (C dari inner CV) | 0.555 [0.519, 0.587] | 0.554 [0.519, 0.586] | 0.553 [0.518, 0.585] | 0.583 [0.545, 0.620] |
| E1 · morfometri mentah · TabM (5 seed) | 0.558 [0.525, 0.591] | 0.560 [0.527, 0.594] | 0.558 [0.525, 0.591] | 0.599 [0.560, 0.634] |
| E1 · morfometri mentah · CatBoost | 0.559 [0.525, 0.593] | 0.556 [0.522, 0.590] | 0.556 [0.521, 0.590] | 0.578 [0.540, 0.615] |
| E1 · morfometri mentah · TabPFN v2 | 0.586 [0.552, 0.619] | 0.582 [0.549, 0.614] | 0.582 [0.549, 0.614] | 0.628 [0.591, 0.664] |
| E1 · morfometri relatif ICV · LogReg (C dari inner CV) | 0.543 [0.509, 0.574] | 0.542 [0.507, 0.574] | 0.542 [0.507, 0.573] | 0.568 [0.529, 0.604] |
| E1 · morfometri relatif ICV · TabM (5 seed) | 0.538 [0.504, 0.570] | 0.540 [0.506, 0.571] | 0.538 [0.504, 0.569] | 0.576 [0.538, 0.610] |
| E1 · morfometri relatif ICV · CatBoost | 0.522 [0.489, 0.555] | 0.518 [0.485, 0.550] | 0.518 [0.484, 0.550] | 0.525 [0.488, 0.562] |
| E1 · morfometri relatif ICV · TabPFN v2 | 0.580 [0.547, 0.612] | 0.576 [0.543, 0.608] | 0.576 [0.543, 0.608] | 0.604 [0.567, 0.640] |
| E1 · morfometri relatif ICV + ComBat · LogReg (C dari inner CV) | 0.563 [0.529, 0.594] | 0.563 [0.528, 0.594] | 0.562 [0.528, 0.594] | 0.584 [0.545, 0.620] |
| E1 · morfometri relatif ICV + ComBat · TabM (5 seed) | 0.562 [0.528, 0.594] | 0.563 [0.529, 0.596] | 0.561 [0.527, 0.594] | 0.589 [0.550, 0.626] |
| E1 · morfometri relatif ICV + ComBat · CatBoost | 0.571 [0.538, 0.603] | 0.567 [0.534, 0.598] | 0.567 [0.533, 0.598] | 0.576 [0.538, 0.612] |
| E1 · morfometri relatif ICV + ComBat · TabPFN v2 | 0.571 [0.537, 0.603] | 0.567 [0.533, 0.600] | 0.567 [0.532, 0.599] | 0.608 [0.569, 0.643] |
| E3 · morfometri mentah · LogReg | 0.553 [0.519, 0.586] | 0.552 [0.518, 0.584] | 0.552 [0.518, 0.584] | 0.585 [0.546, 0.621] |
| E3 · morfometri mentah · TabM (5 seed) | 0.569 [0.536, 0.601] | 0.571 [0.537, 0.604] | 0.569 [0.536, 0.601] | 0.610 [0.572, 0.645] |
| E3 · morfometri mentah · CatBoost | 0.551 [0.517, 0.586] | 0.549 [0.515, 0.584] | 0.549 [0.515, 0.584] | 0.575 [0.537, 0.613] |
| E3 · morfometri relatif ICV · LogReg | 0.547 [0.512, 0.580] | 0.547 [0.511, 0.580] | 0.546 [0.511, 0.579] | 0.570 [0.531, 0.607] |
| E3 · morfometri relatif ICV · TabM (5 seed) | 0.553 [0.519, 0.585] | 0.552 [0.518, 0.584] | 0.552 [0.518, 0.583] | 0.585 [0.546, 0.620] |
| E3 · morfometri relatif ICV · CatBoost | 0.543 [0.510, 0.576] | 0.537 [0.504, 0.569] | 0.535 [0.502, 0.567] | 0.532 [0.494, 0.570] |
| E3 · morfometri relatif ICV + ComBat · LogReg | 0.553 [0.519, 0.586] | 0.553 [0.518, 0.586] | 0.552 [0.518, 0.584] | 0.580 [0.541, 0.616] |
| E3 · morfometri relatif ICV + ComBat · TabM (5 seed) | 0.563 [0.529, 0.597] | 0.566 [0.531, 0.599] | 0.563 [0.528, 0.596] | 0.594 [0.554, 0.630] |
| E3 · morfometri relatif ICV + ComBat · CatBoost | 0.540 [0.506, 0.572] | 0.539 [0.505, 0.571] | 0.539 [0.505, 0.571] | 0.579 [0.541, 0.614] |
| E3 · fenotipik · TabM (5 seed) | 0.596 [0.562, 0.628] | 0.590 [0.556, 0.621] | 0.589 [0.555, 0.620] | 0.627 [0.588, 0.664] |
| E3 · ROI · LogReg | 0.593 [0.560, 0.625] | 0.591 [0.558, 0.623] | 0.591 [0.558, 0.623] | 0.635 [0.598, 0.671] |
| full_v2 · fusion PR (F1_stack_logreg) | 0.628 [0.596, 0.659] | 0.624 [0.591, 0.654] | 0.624 [0.590, 0.654] | 0.664 [0.628, 0.700] |
| ENS_utama v3 (model final) | 0.660 [0.628, 0.691] | 0.657 [0.626, 0.688] | 0.658 [0.626, 0.688] | 0.692 [0.656, 0.727] |
| ENS_tabel v3 (fenotipik + ROI) | 0.657 [0.624, 0.688] | 0.654 [0.621, 0.685] | 0.654 [0.622, 0.686] | 0.686 [0.650, 0.723] |
| ENS_v4 (ENS_utama v3 + morfometri) | 0.652 [0.620, 0.682] | 0.649 [0.617, 0.680] | 0.649 [0.616, 0.680] | 0.697 [0.661, 0.732] |
| fenotipik + morfometri | 0.628 [0.596, 0.660] | 0.625 [0.592, 0.657] | 0.625 [0.592, 0.657] | 0.665 [0.628, 0.700] |
| fenotipik + ROI + morfometri | 0.642 [0.611, 0.674] | 0.639 [0.607, 0.670] | 0.639 [0.607, 0.670] | 0.693 [0.658, 0.727] |


## Seleksi inner morfometri

| model | skor_seleksi |
|---|---|
| E1 · morfometri relatif ICV + ComBat · TabPFN v2 | 0.6047 |
| E1 · morfometri mentah · TabPFN v2 | 0.5994 |
| E1 · morfometri relatif ICV + ComBat · LogReg (C dari inner CV) | 0.5985 |
| E3 · morfometri relatif ICV + ComBat · LogReg | 0.5975 |
| E3 · morfometri mentah · LogReg | 0.5948 |
| E3 · morfometri relatif ICV + ComBat · TabM (5 seed) | 0.5948 |
| E1 · morfometri mentah · TabM (5 seed) | 0.5929 |
| E3 · morfometri mentah · TabM (5 seed) | 0.5925 |
| E1 · morfometri relatif ICV + ComBat · TabM (5 seed) | 0.5916 |
| E1 · morfometri relatif ICV · TabPFN v2 | 0.5866 |
| E3 · morfometri relatif ICV · LogReg | 0.5844 |
| E1 · morfometri mentah · LogReg (C dari inner CV) | 0.5714 |
| E1 · morfometri relatif ICV + ComBat · CatBoost | 0.5706 |
| E1 · morfometri relatif ICV · LogReg (C dari inner CV) | 0.5671 |
| E3 · morfometri mentah · CatBoost | 0.563 |
| E1 · morfometri relatif ICV · TabM (5 seed) | 0.5586 |
| E3 · morfometri relatif ICV · TabM (5 seed) | 0.5581 |
| E1 · morfometri mentah · CatBoost | 0.5562 |
| E3 · morfometri relatif ICV + ComBat · CatBoost | 0.5553 |
| E3 · morfometri relatif ICV · CatBoost | 0.5243 |
| E1 · morfometri relatif ICV · CatBoost | 0.5218 |
