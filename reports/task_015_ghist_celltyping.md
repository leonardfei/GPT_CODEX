# Task 015 — GHIST classification-only HCC cell typing

## Status

**IN_PROGRESS** (2026-10-08). The official-source blocker was resolved by the user's authorized local download and upload. Canonical contour, fold, overlap geometry, and raster-mask QC passed. The required five-epoch fold-0 pilot passed its technical gate. The first fold-0 CV pair stalled after epoch 25 because root `/tmp` filled; those outputs were preserved and the pair restarted from epoch 1 with temporary files on `/data` and unchanged scientific settings. All ten CV runs have now finished successfully. Five-fold aggregation, OOF checks, paired Midnight comparison and figures are complete. Full outer-fold morphology-head QC and gate-eligible full-data model training remain in progress; no production model has been replaced.

## Official source and environment

The official [SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST) GPL-3.0 source is now at `/data/lf_data/models/GHIST`. Server GitHub cloning failed twice because of interrupted connections. A local shallow clone was uploaded as a tar archive; local/server archive SHA-256 both equaled `15eaad78a595eba4f85d8026743bbcb4756f9ac32225d226b01e046094f897cd`. Both checkouts resolve to commit `917456be305fc82e92293ea272812e79675e821c`. The upstream clone was not edited. The HCC adapter imports official `Backbone`, `Embed`, and `MLP` modules without instantiating expression heads. Provenance is in `config/task015/ghist_source_provenance.json`.

Runtime: `/data/lf_data/task010_env/bin/python`, Python 3.10.14, PyTorch 2.7.1+cu128, NumPy 1.23.5, pandas 1.4.3, SciPy 1.8.1, OpenCV 4.5.4-dev, pyvips 2.2.3, scikit-learn 1.3.0. Official `stainlib` is absent, so HED augmentation is disabled under the task's conditional rule; no alternative stain method is introduced. The initial pilot uses float32 to establish numerical stability.

## Input observations

The canonical cohort `/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz` contains 96,044 unique cell IDs, seven frozen classes and eight batches, with no missing essential labels/coordinates. The H&E TIFF is 50,000 × 23,451 pixels. Generic OpenSlide MPP is implausible (~352.783 μm/px), so the project-validated native 0.2125 μm/px was explicitly used for binary CellViT detection. That whole-WSI detection produced 620,205 retained nuclei at `/data/lf_data/result/task015_ghist_celltyping/work/detector/ID0060276_cells.json`, but is **not** the supervised contour source: direct 15-pixel nearest-centroid matching reached only 93,126/96,044 canonical cells and 88,179 distinct nearest nuclei. We rejected that mapping rather than changing the threshold. The binary-detection command/log remain in the previous commit's report history and server log `/data/lf_data/result/task015_ghist_celltyping/logs/cellvit_binary_detection.log`.

## Canonical contour and fold QC

The Task010 patch-level CellViT-SAM-H-x40-AMP inference was repeated on 5,005 original 256×256 patches using the identical 15-pixel one-to-one pairing and seed `20260923`. Its 252,921 detections include valid unique contours for **96,044/96,044 canonical cells**. All matched centroid coordinates reproduce the frozen Task010 token manifest to within `1e-6` pixel (maximum `6.36e-14`). However, **33,988 (35.39%)** canonical cells had multiple candidate nuclei within 15 pixels; the original one-to-one pairing is reproducible but not biologically unambiguous. This is a limitation, not a claim of perfect label correspondence.

The canonical cells occupy 4,814 patches; no canonical-bearing patch has more than one biological batch. Exact five grouped outer folds remain intact. The corrected training-only inner-validation batches are `s93` (folds 0–3) and `s22` (fold 4). The 30-pixel-overlap geometry audit generated 136,052 candidate patch appearances and selected one largest-visible-area patch per cell, yielding exactly 96,044 unique cells; per-fold validation counts are unchanged. Machine-readable server outputs: `metrics/canonical_instance_manifest.csv.gz`, `metrics/patch_manifest.csv.gz`, `metrics/overlap_patch_candidates.csv.gz`, `metrics/overlap_largest_area_manifest.csv.gz`, plus `qc/patch_matching_summary.json`, `qc/patch_manifest_audit.json`, and `qc/overlap_geometry_audit.json` under `/data/lf_data/result/task015_ghist_celltyping`. Large contours remain server-side. See `qc/task015/canonical_instance_alignment.md`.

## Model and completed pilot

`scripts/python/task015_ghist_celltype_model.py` uses official GHIST UNet3+ pixel head (eight channels), per-instance means of the first and last feature maps plus patch means, official 256-d `Embed`, and seven-class `MLP`. It has no expression tensors or heads. A batch-size-8 forward/backward smoke test passed. Split-safe rasterization retains unlabeled nuclei as context, assigns `ignore_index=-100` to unsupervised nucleus pixels, and excludes inactive cells from cell loss. A 194-cell fold-0 training-batch loss test gave finite pixel CE `2.1089` and cell CE `1.9537`, with a successful optimizer step.

Fold-0 official-variant pilot completed five epochs on GPU 0 using training batches `s01A,s01B,s04B,s11,s22`; inner validation `s93` had 9,607 cells on 617 overlapping patches. Outer validation `s02A,s06A` was untouched. Mean training loss fell `1.8726→1.5575`. Every epoch produced exactly 9,607 unique finite predictions; peak allocated GPU memory was stable at about 13.696 GiB. The inner-validation Macro-F1 sequence was `0.0695, 0.1864, 0.2207, 0.2455, 0.2366`; these are **not** outer-fold estimates. See `qc/task015/pilot_audit.md` and server log `/data/lf_data/result/task015_ghist_celltyping/logs/pilot_fold0_official.log`.

## Required benchmark status and next gate

The gated CV launcher `/data/lf_data/result/task015_ghist_celltyping/code/task015_launch_cv.sh` completed all ten GHIST folds with status 0. Python/PyTorch temporary files used `/data/lf_data/result/task015_ghist_celltyping/work/tmp`; the root filesystem remains nearly full and was not cleaned. The previous failed logs and 25 model-only checkpoints per variant were preserved. Because optimizer and RNG states were not saved, exact epoch-26 continuation was not reproducible; see `qc/task015/cv_restart_audit.md`. Per-fold logs and top-level `logs/cv_launcher.log` are server-side.

### Five-fold outer-validation results

Means and sample SDs below are across the five exact Task009 grouped outer folds, not pooled-cell point estimates. The corrected Task011 Midnight `F1_FINAL_BLOCK` is the paired baseline.

| Model | Macro-F1 | Macro-AUPRC | Neutrophil F1 | Neutrophil one-vs-rest AUPRC |
|---|---:|---:|---:|---:|
| Midnight | 0.4522 | 0.4893 | 0.2352 | 0.1976 |
| GHIST_CT_OFFICIAL | 0.4345 | 0.4528 | 0.2436 | 0.2359 |
| GHIST_CT_BALANCED | 0.4225 | 0.4499 | 0.2774 | 0.2323 |

The five selected training-only epochs were OFFICIAL `30, 28, 25, 25, 42` and BALANCED `35, 25, 25, 24, 43`. Each fold was refit from its saved initialization for the selected count before one untouched outer-validation evaluation. Both models yielded exactly 96,044 unique OOF cell IDs across five folds, with the frozen class IDs, correct per-fold batches, finite probabilities summing to one, and reproduced Macro-F1/Macro-AUPRC from the saved OOF tables. Corrected Midnight per-class supports match GHIST in every fold. Full audit code: `scripts/python/task015_finalize_ghist.py`; server outputs: `metrics/ghist_*.csv` and `qc/oof_coverage_qc.csv`.

The true binary N-vs-Myeloid AUROC/AUPRC means were `0.7369/0.4659` (OFFICIAL), `0.7371/0.4702` (BALANCED), versus `0.7202/0.4385` (Midnight). N-vs-T/B means were `0.7693/0.4574` (OFFICIAL), `0.7588/0.4441` (BALANCED), versus `0.7069/0.3930` (Midnight). These are independently trained training-only probes on 256-d embeddings, not seven-class probability ratios. Fold-level metrics, per-class metrics, binary metrics and paired deltas are in the corresponding CSV files.

Per-class F1 changes versus Midnight are mixed. OFFICIAL improves Myeloid by `+0.0439` and T/B by `+0.1004`, but lowers Endothelial by `-0.1325`, Plasma by `-0.0792`, and Tumor by `-0.0700`. BALANCED improves Neutrophil by `+0.0422` and T/B by `+0.0617`, but lowers Endothelial by `-0.1271`, Plasma by `-0.0887`, and Tumor by `-0.0672`. These are quantitative comparisons, not evidence of a biological mechanism.

### Prespecified decision gate and remaining work

OFFICIAL has the higher primary endpoint, mean cell-level Macro-F1 (`0.4345` versus `0.4225`), and is the selected GHIST variant. Relative to Midnight, its paired mean Macro-F1 delta is `-0.0177` (positive in 2/5 folds), while Neutrophil AUPRC is `+0.0382` (positive in 5/5). BALANCED's paired mean Macro-F1 delta is `-0.0297` (positive in 1/5), while Neutrophil F1 is `+0.0422` (positive in 5/5). Neither meets the strong gate. Both meet the task's moderate gate via consistent Neutrophil improvement; improvements above +0.03 are treated as at least moderate when the strong *conjunction* fails. This gate permits an auditable full-data GHIST model export, **not** replacement of the Task012 Midnight production checkpoint. The optional neighborhood-composition stage is eligible by its technical threshold but is not required for the primary benchmark and has not been run.

The full-data OFFICIAL model is training for the median selected epoch count, `28`, on all 96,044 frozen canonical cells. The outer-fold pixel-head morphology QC is running independently on the unused GPU. Until both finish and their outputs are checked, Task015 remains `IN_PROGRESS`. No P84/P169 inference or production replacement has been performed. An independently assessed external test would require a separate supervisory scientific decision.

Environment: `/data/lf_data/task010_env/bin/python`, Python 3.10.14, PyTorch 2.7.1+cu128, NumPy 1.23.5, pandas 1.4.3, SciPy 1.8.1, scikit-learn 1.3.0, Matplotlib 3.7.1. Fixed seed `20260923`; AdamW, batch 8, linear epoch-wise LR schedule, 256×256 native patches at 0.2125 μm/px, 30-pixel validation overlap, and training-only RGB standardization were preserved. Commands: `python code/task015_finalize_ghist.py`, `python code/task015_morphology_qc.py --gpu 1`, and `python code/task015_full_data_train.py` with the Task015 `/data/.../work/tmp` directory exported as `TMPDIR`. Large OOF tables, checkpoints and instance data remain on the server. The official GHIST source was not edited.
