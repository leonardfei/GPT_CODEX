# Task 004 — High-Confidence CellViT–Xenium Label Reconstruction and Retraining

## Outcome

Task 004 primary experiment completed with the official CellViT++ classifier-head entry point and the fixed five grouped folds. The task is marked **PARTIAL** because the requested secondary threshold-sensitivity candidates were summarized for retention/ambiguity, but were not each retrained with official grouped CV. No test evaluation or production promotion was performed.

Remote artifacts are under `/data/lf_data/result/task004_high_confidence`.

## Matching and geometry

Task 2 used the installed CellViT++ `pair_coordinates` implementation: Euclidean distances, SciPy Hungarian/Munkres global minimum-cost one-to-one assignment, followed by a 15 px post-assignment radius filter. It was not greedy nearest-neighbor matching.

Available geometry was centroid-only. Source annotations provided x/y/class rows and the archived pair table provided matched CellViT centroids. Xenium polygons, CellViT polygons, an affine transform, and a global-coordinate mapping were unavailable. The complete unmatched detection coordinate pool was not archived, so reciprocal competitor diagnostics are explicitly limited to the archived matched-detection pool.

## Retention

| Class | ALL_MATCHED | HIGH_CONFIDENCE | ULTRA_HIGH_CONFIDENCE |
|---|---:|---:|---:|
| Endothelial | 13,183 | 5,061 (38.4%) | 3,418 (25.9%) |
| Mesenchymal | 16,950 | 6,918 (40.8%) | 5,194 (30.6%) |
| Myeloid | 15,742 | 5,937 (37.7%) | 3,551 (22.6%) |
| Neutrophil | 6,482 | 2,167 (33.4%) | 1,548 (23.9%) |
| Plasma cell | 8,597 | 3,690 (42.9%) | 3,470 (40.4%) |
| T and B | 20,829 | 7,325 (35.2%) | 5,353 (25.7%) |
| Tumor | 12,879 | 5,931 (46.1%) | 2,532 (19.7%) |
| **Total matched cells** | **94,662** | **37,029** | **25,066** |

All eight batches remained represented in each tier.

## Frozen embedding separability

| Tier | Class silhouette | Class NN purity | Batch silhouette | Batch NN purity | Cache coverage |
|---|---:|---:|---:|---:|---:|
| ALL_MATCHED | -0.0174 | 0.2687 | -0.0022 | 0.4764 | 99.60% |
| HIGH_CONFIDENCE | -0.0150 | 0.2787 | -0.0082 | 0.5101 | 99.62% |
| ULTRA_HIGH_CONFIDENCE | -0.0198 | 0.2825 | -0.0217 | 0.5732 | 99.64% |

The older Task 2 token archive had no image/coordinate keys and was shorter than the pair table. For this analysis, official CellViT cache tokens were joined by image, detection centroid and class; unmatched cache keys were excluded and coverage is reported above. No positional truncation was used.

## Grouped-CV results

| Tier | Macro F1 | Balanced accuracy | Macro AUROC | Macro AUPRC | Lowest-three F1 |
|---|---:|---:|---:|---:|---:|
| ALL_MATCHED | 0.3324 ± 0.0134 | 0.3455 ± 0.0195 | 0.7314 ± 0.0167 | 0.3413 ± 0.0281 | 0.1748 ± 0.0213 |
| HIGH_CONFIDENCE | 0.3401 ± 0.0164 | 0.3524 ± 0.0256 | 0.7476 ± 0.0232 | 0.3565 ± 0.0332 | 0.1671 ± 0.0169 |
| ULTRA_HIGH_CONFIDENCE | 0.3226 ± 0.0199 | 0.3367 ± 0.0256 | 0.7316 ± 0.0253 | 0.3403 ± 0.0344 | 0.1565 ± 0.0298 |

HIGH_CONFIDENCE improved macro-F1 by only +0.0077 versus ALL_MATCHED. Its lowest-three-class F1 declined by -0.0077, and mean Neutrophil F1 declined by -0.0348. ULTRA_HIGH_CONFIDENCE declined in both macro-F1 (-0.0098) and lowest-three-class F1 (-0.0183). The weak-class details are preserved in `metrics/weak_class_cv_by_tier.csv`.

## Promotion and production

Neither tier met the predefined promotion rule requiring at least +0.03 macro-F1, at least +0.03 lowest-three F1, improvement in at least 4/5 folds, no strong class loss, no depletion and no batch-leakage increase. The best primary tier by macro-F1 was HIGH_CONFIDENCE, but it was not eligible for promotion.

Production remains:

```text
/data/lf_data/result/model_best.pth
SHA256 f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164
```

No new candidate was promoted; fold runs and checkpoints remain under the Task 004 remote result root for audit. The Task 1 and Task 3 protected model files were not overwritten. Task 005 was not started.

## Recommended Task 005 direction

Before another head-only tuning round, archive the complete detection-to-Xenium candidate graph including unmatched detections, verify registration/coordinate transforms, and manually review borderline matches across batches. If a secondary sensitivity study is needed, run candidate-specific official grouped CV only for a pre-registered, small set of rules.

## Reproducibility

- Official entry point: `python3 ./cellvit/train_cell_classifier_head.py --config <YAML>`.
- Fixed seed: 42; five grouped folds; seven classes unchanged.
- Remote report: `/data/lf_data/result/task004_high_confidence/TASK004_REPORT.md`.
- Remote figures: `/data/lf_data/result/task004_high_confidence/figures/`.
- Environment, checksums and run manifest: `/data/lf_data/result/task004_high_confidence/config/`.
