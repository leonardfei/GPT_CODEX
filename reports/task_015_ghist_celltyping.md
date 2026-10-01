# Task 015 — GHIST classification-only HCC cell typing

## Status

**IN_PROGRESS** (2026-10-01). The official-source blocker was resolved by the user's authorized local download and upload. Canonical contour, fold, overlap geometry, and raster-mask QC passed. The required five-epoch fold-0 pilot passed its technical gate. A gated two-GPU, two-variant five-fold CV launcher is running; no completed cross-validation or GHIST-versus-Midnight performance claim exists yet.

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

The gated CV launcher `/data/lf_data/result/task015_ghist_celltyping/code/task015_launch_cv.sh` has started GHIST_CT_OFFICIAL and GHIST_CT_BALANCED fold 0 on separate GPUs. It will advance through all five folds only when each pair succeeds, and stop on failure. Per-fold logs and top-level `logs/cv_launcher.log` are server-side. The corrected-Task011-style true binary probe implementation is present but has not yet processed GHIST fold embeddings. Five-fold results, morphology-head QC, paired Midnight comparison, optional neighborhood stage, decision gate, figures and full-data model export remain **pending**. No performance or model-promotion decision is possible. Do not train on P84/P169 or touch Task012 Midnight/production models.
