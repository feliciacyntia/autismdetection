---
license: cc-by-nc-sa-3.0
pretty_name: ABIDE I preprocessed T1w (05c pipeline, development set)
tags:
  - neuroimaging
  - mri
  - autism
  - abide
size_categories:
  - n<1K
---

# ABIDE I — preprocessed T1-weighted MRI (pipeline 05c, development set, n = 853)

Skull-stripped, affinely registered T1w volumes of the **853 development subjects** of a bachelor-thesis project
(ASD vs TD classification, ABIDE I). Code, folds, phenotypic table, QC documentation and all modelling results are in
the GitHub repository: **https://github.com/feliciacyntia/autismdetection**.

> **Held-out test set not included.** 149 subjects are reserved for one final evaluation and are deliberately absent here.
> Do not reconstruct or add them for model development.

## Pipeline (05c-v2.1, config fingerprint `682404ae0f5d2624`)
Reorient → N4 bias correction → HD-BET skull stripping (GPU, TTA, deterministic) → ANTs **affine** registration to MNI152
(1 mm, seed 42, 1 thread) → resample to 2 mm → crop to **96 × 112 × 96** → z-score inside the brain mask.
Exact versions, parameters and template hashes: `data/smri_preproc/preprocess_config.json` and `reference_space.json`
in the GitHub repo. Every subject was produced by the same fingerprint; a full consistency audit passed
(`data/smri_preproc/audit_full/` in the GitHub repo).

## Files
| folder | per subject | content |
|---|---|---|
| `vol/` | `{SUB_ID}.npy` | float32 `(1, 96, 112, 96)`, z-scored inside mask, 0 outside — model input |
| `mask/` | `{SUB_ID}.npy` | uint8 `(1, 96, 112, 96)` brain mask |
| `nii/` | `{SUB_ID}.nii.gz`, `{SUB_ID}_mask.nii.gz` | same volume + mask as NIfTI (RAS, 2 mm, affine in header) |
| `mask_native/` | `{SUB_ID}_mask_native.nii.gz` | HD-BET mask in native space |
| `xfm/` | `{SUB_ID}_0GenericAffine.mat` | ANTs affine (native → MNI) |
| `qc_figures/` | `{SUB_ID}.png` | QC montage |
| `templates/` | — | MNI152 2009 templates used (from nilearn) |
| `repro/` | 4 dev subjects | reproducibility check: subjects re-run from scratch; `repro_*.json` records the hash comparison (outputs bitwise identical to the main run, so the identical re-run files are not duplicated here — only logs/meta) |
| `_superseded/d21d49b87afd8c11/` | 3 dev subjects | outputs of an older config (HD-BET non-deterministic) that was replaced — kept for the record, **do not use** |

`SHA256SUMS` lists a checksum for every file. This repo holds **only the large image files**; everything small lives in
the GitHub repo (no duplicates): subject list, labels and folds (`data/folds.csv`, rows with `split == "dev"`),
phenotypic table (`dataset/cleaned_dataset/`), QC report & decisions and per-subject logs (`data/smri_preproc/`).

```python
import numpy as np, pandas as pd
folds = pd.read_csv("data/folds.csv")                 # from the GitHub repo
dev = folds[folds.split == "dev"]                     # 853 subjects = the files in this repo
x = np.load(f"vol/{dev.SUB_ID.iloc[0]}.npy")          # (1, 96, 112, 96) float32
```

## Rules used in the project (please keep them when comparing results)
- Use `fold_strat` (5-fold, stratified by site × label) from `data/folds.csv`; fit every transform (scaling, imputation,
  ComBat, feature selection) on training folds only.
- `SITE_ID` and clinical scores (ADOS, ADI-R, SRS, SCQ, AQ, DSM) are never model features.
- QC: all 853 images kept; subject 50404 has an unreliable brain-mask volume (see `data/smri_preproc/qc_decisions.csv` in the GitHub repo).

## License and attribution
Derived from the **Autism Brain Imaging Data Exchange (ABIDE I)**, shared under CC BY-NC-SA 3.0; this derivative uses
the same license (non-commercial, share-alike, attribution). Please cite:
- Di Martino A. et al. (2014). The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic
  brain architecture in autism. *Molecular Psychiatry*, 19, 659–667.
- Isensee F. et al. (2019). Automated brain extraction of multisequence MRI using artificial neural networks (HD-BET).
  *Human Brain Mapping*, 40(17), 4952–4964.
- Avants B. B. et al. (2011). A reproducible evaluation of ANTs similarity metric performance in brain image registration.
  *NeuroImage*, 54(3), 2033–2044.
- Fonov V. et al. (2011). Unbiased average age-appropriate atlases for pediatric studies. *NeuroImage*, 54(1), 313–327
  (MNI152 2009 template; Copyright (C) 1993–2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological
  Institute, McGill University — redistributed with this notice).
