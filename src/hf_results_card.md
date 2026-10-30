---
license: cc-by-nc-sa-3.0
tags:
  - neuroimaging
  - autism
  - abide
---

# ABIDE I ASD vs TD — model checkpoints & embedding caches

Binary outputs of the modelling runs of the bachelor-thesis project **https://github.com/feliciacyntia/autismdetection**
(predictions, tables, configs and logs are in that GitHub repo; preprocessed images are in the dataset
**https://huggingface.co/datasets/feliciacyntia/abide1-smri-prepro**). Folder layout mirrors the GitHub repo, so download
straight into the cloned repo root:

```bash
hf download feliciacyntia/abide1-asd-results --local-dir . --include "results/*" --include "pretrained/*"
```

| path | content |
|---|---|
| `results/full_v2/units/<model>/fold<k>/ckpt/{best,last}.pt` | **main run** (853 dev subjects, 5 folds, locked config): image models effnet_b0, resnet18, scratch2d, scratch3d |
| `results/pilot200_v2_bnfix/...` | pilot (200 subjects) rerun of scratch2d/scratch3d after the BatchNorm bugfix |
| `results/pilot200_v2/...` | pilot v2 (200 subjects) — trial run; scratch2d/scratch3d here are **invalid (BN bug)** |
| `results/pilot200/...` | pilot v1 (200 subjects) — trial run, includes models later excluded (ConvNeXt, DINOv2, SFCN) |
| `results/pilot200*/cache/*/{SUB_ID}.npy` | frozen-feature caches (DINOv2, SFCN) of the pilot runs |
| `pretrained/atlas/` | Harvard-Oxford atlas resampled to the 2 mm 96 × 112 × 96 grid + labels (used for ROI features) |

All models were trained on development folds only; the 149 held-out test subjects were never used.
`best.pt` = model `state_dict` at the best epoch (inner-validation loss); `last.pt` = full resume state
(`model`, `opt`, `sched`, AMP `scaler`, `rng`, `state`). Use the loaders in `src/pilot_models.py` with the hyper-parameters in `results/<run>/config.json`.
`SHA256SUMS` lists a checksum for every file.

**Atlas:** Harvard-Oxford cortical/subcortical atlases (Desikan et al. 2006; Makris et al. 2006; Frazier et al. 2005;
Goldstein et al. 2007), distributed with FSL and fetched via nilearn — non-commercial use, cite the original papers.
**Data:** derived from ABIDE I (Di Martino et al., 2014, *Molecular Psychiatry* 19, 659–667), CC BY-NC-SA 3.0.
