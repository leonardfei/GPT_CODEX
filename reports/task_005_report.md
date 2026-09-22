# Task 005 Report — Multi-Backbone, Stain-Domain, and Neutrophil Detection Benchmark

Status: **COMPLETED**

## Scope and reproducibility

The benchmark was executed after updating from `origin/main`. It used the official CellViT++ classifier path, the seven-class taxonomy, seed 42, the shared grouped batch 5-fold splits, and the fixed AdamW head recipe. The remote output root is `/data/lf_data/result/task005_backbone_domain`.

The checkpoint inventory found only `CellViT-SAM-H-x40-AMP.pth` (officially loadable, 1280-dimensional embeddings, 256-pixel input). UNI, Virchow, Virchow2, and ViT256 checkpoints were unavailable locally and were not downloaded. Both official stain conditions were run: RAW and `STAIN_NORMALIZED`.

## Main results

| Condition | Macro-F1 | Macro-AUPRC | Lowest-three F1 | Primary score |
|---|---:|---:|---:|---:|
| SAM-H RAW | 0.3324 ± 0.0134 | 0.3413 ± 0.0281 | 0.1748 ± 0.0213 | 0.2455 |
| SAM-H STAIN_NORMALIZED | 0.3295 ± 0.0121 | 0.3467 ± 0.0310 | 0.1603 ± 0.0185 | 0.2439 |

Neutrophil metrics:

| Condition | Precision | CV recall | F1 | AUPRC | Detection recall | Conditional classifier recall | End-to-end recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAW | 0.2004 | 0.0953 | 0.1145 | 0.1366 | 0.5467 | 0.0824 | 0.0450 |
| STAIN_NORMALIZED | 0.2115 | 0.0869 | 0.1060 | 0.1469 | 0.5870 | 0.0678 | 0.0398 |

Stain normalization increased detector Neutrophil recall by 0.0402 and reduced batch nearest-neighbor purity from 0.4735 to 0.4235, but reduced macro-F1, lowest-three F1, Neutrophil F1, conditional recall, and end-to-end recall. Its higher Neutrophil AUPRC did not compensate for those losses.

Embedding diagnostics were class silhouette / batch silhouette / class NN purity / batch NN purity: RAW `-0.0166 / -0.0032 / 0.2605 / 0.4735`; normalized `-0.0137 / -0.0081 / 0.2668 / 0.4235`.

## Decision

The best overall and best Neutrophil-F1 condition was SAM-H RAW. No condition satisfied the promotion rule: neither reached the required macro-F1 improvement, Neutrophil-F1 improvement, and associated guardrails. Therefore no final refit or one-time test evaluation was run, and production remains unchanged.

The production model is `/data/lf_data/result/model_best.pth` with SHA256 `f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`.

Per-class classification, detection recall, confusion flows, end-to-end metrics, domain diagnostics, ranking, and promotion JSON are in the remote `metrics/` directory. The ten required vector PDFs and their source CSVs are in `figures/` and `figure_data/`. Task 006 was not started; the evidence-supported next direction is detection-model adaptation or a context-aware Neutrophil-specialist representation, retaining grouped batch validation.
