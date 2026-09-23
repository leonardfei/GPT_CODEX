# Task 010 — Corrected CellViT alignment and local Midnight-12k integration

Status: **PARTIAL-CORRECTED-FIVEWAY-COMPLETE**. The corrected shared-cohort five-representation benchmark is complete. The secondary Midnight GT-centered upper bound remains pending.

## Executive result

The previous Phase A CellViT baseline was misaligned. All 96,044 token rows changed positional index after reindexing by the canonical composite key. After correction, CellViT_TOKEN_ALIGNED is no longer near-random and is essentially tied with Phikon-v2 SMALL. Midnight-12k SMALL is the strongest representation in this diagnostic benchmark.

| representation | macro-F1 mean ± SD | macro-AUPRC mean ± SD | Neutrophil F1 | Neutrophil AUPRC |
|---|---:|---:|---:|---:|
| CELLVIT_TOKEN_ALIGNED | 0.3364 ± 0.0263 | 0.3418 ± 0.0260 | 0.1885 | 0.1327 |
| PHIKON_V2_SMALL | 0.3366 ± 0.0243 | 0.3434 ± 0.0339 | 0.1885 | 0.1349 |
| PHIKON_V2_CONTEXT | 0.2726 ± 0.0253 | 0.2834 ± 0.0315 | 0.1414 | 0.0998 |
| MIDNIGHT12K_SMALL | **0.4116 ± 0.0312** | **0.4327 ± 0.0393** | **0.2175** | **0.1620** |
| MIDNIGHT12K_CONTEXT | 0.2806 ± 0.0221 | 0.3072 ± 0.0305 | 0.1399 | 0.1045 |

The prior claim of a +0.226 macro-F1 Phikon gain over CellViT is superseded. The corrected Phikon SMALL−CellViT macro-F1 delta is +0.0001; Midnight SMALL−CellViT is +0.0751.

## A–D. Alignment correction and canonical cohort

The frozen `SHARED_DETECTED_CORE` manifest was reused unchanged: 96,044 cells, the same V3_CORE labels, matched H&E centers, and exact Task009 five held-out batch folds. A canonical order file was created with `cell_id`, composite key, class, batch, coordinates, and canonical index.

The old CellViT token tensor and manifest each had 96,044 rows. The tensor was reindexed using `image|local_x|local_y|class_id`; key uniqueness, one-to-one coverage, no missing/extra cells, and exact post-reindex cell_id equality were asserted. **96,044/96,044 rows (100%) changed positional index.** The corrected token dimension is 1,280.

The corrected CellViT macro-F1 (0.3364 ± 0.0263) is close to the Task009 V3_CORE classifier result (0.3330 ± 0.0222), although the universes and probe recipes differ. This strongly indicates that the former 0.1102 baseline was primarily caused by alignment, not by an intrinsically unusable CellViT token representation.

Phikon SMALL and CONTEXT payloads were revalidated against canonical cell IDs and saved as aligned copies. Both were exact-ID aligned before reuse, finite, and dimension 1,024.

## E–G. Midnight-12k provenance and extraction

The user-uploaded archive was found at `/data/lf_data/models/midnight-12k.tar.gz`, passed `gzip -t`, and was extracted to `/data/lf_data/models/midnight-12k`. Only the public Midnight-12k checkpoint was used; no Midnight-92k variant, network fallback, mirror, or credential was used.

- model weight SHA256: `52c14f20386ca17c2af8a7bf32c31c352668a8fbf6aefc88d86be6eaa0c72ca1`
- runtime architecture: local `Dinov2Model`, hidden size 1,536, patch size 14
- input: official README transform resize 224, center crop 224, mean/std `(0.5, 0.5, 0.5)`
- runtime output: 257 tokens for 224×224 input
- classification feature: concatenate CLS token and mean patch tokens, dimension 3,072
- loading: `AutoModel.from_pretrained(..., local_files_only=True)` under `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`

Midnight used the exact Phikon native crop centers and fields: 75×75 SMALL and 263×263 CONTEXT from the original OME-TIFF, followed by Midnight-specific preprocessing.

## H–L. True binary probes and context value

The old seven-class probability-ratio diagnostics were archived and are not treated as binary specialist results. Corrected results use separately trained class-balanced binary probes on each outer training fold.

Mean AUROC / AUPRC:

| representation | Neutrophil vs Myeloid | Neutrophil vs T and B |
|---|---:|---:|
| CELLVIT_TOKEN_ALIGNED | 0.644 / 0.351 | 0.673 / 0.331 |
| PHIKON_V2_SMALL | 0.617 / 0.339 | 0.683 / 0.349 |
| PHIKON_V2_CONTEXT | 0.539 / 0.271 | 0.627 / 0.294 |
| MIDNIGHT12K_SMALL | **0.671 / 0.380** | **0.707 / 0.371** |
| MIDNIGHT12K_CONTEXT | 0.567 / 0.291 | 0.660 / 0.307 |

Both encoders independently show SMALL > CONTEXT. Corrected paired macro-F1 deltas were Phikon CONTEXT−SMALL −0.0640, Midnight CONTEXT−SMALL −0.1310, Midnight SMALL−Phikon SMALL +0.0750, and Phikon SMALL−corrected CellViT +0.0001.

## M–N. Nested MLP, geometry, and cross-encoder agreement

The corrected MLP uses training-only grouped inner validation: the prespecified last outer-training batch is the inner validation batch, the selected epoch is then refit from scratch on all outer-training cells, and the outer validation is evaluated once.

| representation | nested MLP macro-F1 | nested MLP macro-AUPRC |
|---|---:|---:|
| CELLVIT_TOKEN_ALIGNED | 0.3239 ± 0.0354 | 0.3438 ± 0.0327 |
| PHIKON_V2_SMALL | 0.3083 ± 0.0192 | 0.3349 ± 0.0343 |
| PHIKON_V2_CONTEXT | 0.2593 ± 0.0155 | 0.2854 ± 0.0291 |
| MIDNIGHT12K_SMALL | **0.3813 ± 0.0238** | **0.4151 ± 0.0329** |
| MIDNIGHT12K_CONTEXT | 0.2624 ± 0.0218 | 0.2989 ± 0.0295 |

On the same deterministic 10,000-cell subset, class-separation ratios were CellViT 1.015, Phikon SMALL 1.015, Phikon CONTEXT 1.019, Midnight SMALL 1.033, and Midnight CONTEXT 1.041. Batch-separation ratios were 1.034, 1.130, 1.263, 1.127, and 1.306. Batch kNN purity was 0.501, 0.841, 0.987, 0.731, and 0.957, respectively. These batch/spatial signals require caution when ranking pathology encoders.

Phikon/Midnight prediction agreement was 0.526 (Cohen κ 0.420) for SMALL and 0.510 (κ 0.380) for CONTEXT. SMALL correct/correct overlap was 0.294 and error/error overlap 0.445; the low-to-moderate agreement indicates that Midnight's gain is not simply identical to Phikon's errors.

## O–Q. Interpretation and Task011 recommendation

The alignment correction changes the interpretation materially: CellViT token representation is not the dominant bottleneck suggested by the previous invalid comparison. Phikon-v2 SMALL is approximately equal to corrected CellViT under this linear probe, while Midnight-12k SMALL provides a reproducible independent gain. Both encoders perform worse with the larger CONTEXT crop, and both show elevated batch/spatial separation relative to corrected CellViT.

The evidence supports a Task011 focused on local-morphology representation with explicit batch/spatial robustness and specialist immune classification. A multiscale/contextual model should not be prioritized until the SMALL-versus-CONTEXT and batch-separation effects are controlled. These are computational representation findings, not biological conclusions.

The Phikon GT-centered upper bound remains available from the prior run (SMALL macro-F1 0.3379; CONTEXT 0.2746). Midnight GT-centered extraction is pending as a secondary analysis and does not invalidate the completed corrected shared-cohort benchmark.

Production `/data/lf_data/result/model_best.pth`, Task007 labels, Task008 eligibility, and Task009 datasets/metrics were not modified.

## Reproducibility and artefacts

Corrected script: `scripts/python/task010_corrected_midnight.py`. The remote output root is `/data/lf_data/result/task010_representation_benchmark`. Corrected outputs include canonical alignment files, aligned tensors, five-way fold/per-class/Neutrophil metrics, true binary specialist metrics, corrected paired deltas, deterministic geometry, cross-encoder agreement, nested MLP metrics, corrected figures, model provenance, and the pre-alignment-fix archive. Large feature tensors, crops, checkpoints, and credentials are not committed to Git.
