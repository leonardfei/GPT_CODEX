# Task 010 — Phikon-first local pathology foundation-model representation benchmark

Status: **PARTIAL-PHIKON-COMPLETE**. Phase A completed with the locally uploaded Phikon-v2. Midnight-12k Phase B remains pending and does not block this result.

## Executive result

On the frozen Task009 V3_CORE training cells and exact five held-out batch folds, the frozen Phikon-v2 SMALL representation outperformed the official CellViT token representation in the identical linear probe:

| representation | macro-F1 mean ± SD | macro-AUPRC mean ± SD | Neutrophil F1 | Neutrophil AUPRC |
|---|---:|---:|---:|---:|
| CellViT_TOKEN | 0.1102 ± 0.0187 | 0.1388 ± 0.0068 | 0.0631 | 0.0500 |
| PHIKON_V2_SMALL | **0.3366 ± 0.0243** | **0.3434 ± 0.0339** | **0.1885** | **0.1349** |
| PHIKON_V2_CONTEXT | 0.2726 ± 0.0253 | 0.2834 ± 0.0315 | 0.1414 | 0.0998 |

The SMALL gain versus CellViT was positive in all five paired folds. It satisfies the prespecified strong-gain thresholds, but Phikon features also showed substantially higher batch separation, so the result is evidence for a stronger frozen morphology representation—not yet evidence of a production-ready or batch-robust model.

## A–C. Local model provenance and offline loading

The uploaded model was found at `/data/lf_data/models/phikon-v2`. The archive was locally validated (`gzip -t`) and extracted without changing source data. The directory contained `model.safetensors`, `config.json`, `preprocessor_config.json`, `README.md`, `LICENSE.pdf`, and `.gitattributes`.

- weight SHA256: `261ae680fa699b3b951597fd57aa19c02ef735805acb104b93af69b36d928569`
- architecture: `Dinov2Model`, model type `dinov2`, patch size 16, configured hidden size 1024
- runtime embedding: `last_hidden_state[:, 0, :]`, dimension 1024, finite float32 output
- processor: resize shortest edge 224, center crop 224×224, RGB, rescale 1/255, ImageNet mean/std, bicubic resampling
- loading: `AutoImageProcessor.from_pretrained(..., local_files_only=True)` and `AutoModel.from_pretrained(..., local_files_only=True)`
- environment: Python 3.10.14, torch 2.7.1+cu128, NVIDIA RTX PRO 5000 72GB, isolated offline Transformers 4.45.2 / Hugging Face Hub 0.25.2 / safetensors 0.4.5
- offline flags: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`

No fallback model, mirror, credential, token, or online download was used. Full provenance is in `config/phikon_v2_provenance.json` and `qc/phikon_local_provenance.md`.

## D–F. Frozen data, crop geometry, and cohort

The source was the original `/data/lf_data/xenium_data/ID0060276.ome.tif` (50,000 × 23,451 × 3, uint8 RGB), not the forbidden historical CellViT dataset and not Task009 PNGs. The validated project scale was 0.2125 µm/px; inconsistent OME PhysicalSize metadata was not used.

- SMALL: 16 × 16 µm, 75 × 75 native pixels
- CONTEXT: 56 × 56 µm, 263 × 263 native pixels
- both centered on the matched CellViT H&E nucleus centroid

The model-independent `SHARED_DETECTED_CORE` was frozen before Phikon extraction. It contains 96,044 cells after V3_CORE label, Task009 training-batch, CellViT-detection/token, and both-crop in-bounds checks.

| class | cells |
|---|---:|
| Endothelial | 13,339 |
| Mesenchymal | 16,988 |
| Myeloid | 15,994 |
| Neutrophil | 6,139 |
| Plasma cell | 8,539 |
| T and B | 21,018 |
| Tumor | 14,027 |

Batch counts were s01A 10,359; s01B 6,714; s02A 2,170; s04B 19,124; s06A 17,062; s11 14,524; s22 16,484; s93 9,607. The exact Task009 V3_CORE held-out identities were reused:

| fold | train batches | validation batches |
|---:|---|---|
| 0 | s01A;s01B;s04B;s11;s22;s93 | s02A;s06A |
| 1 | s01A;s01B;s02A;s04B;s06A;s11;s93 | s22 |
| 2 | s01B;s02A;s04B;s06A;s22;s93 | s01A;s11 |
| 3 | s01A;s02A;s04B;s06A;s11;s22;s93 | s01B |
| 4 | s01A;s01B;s02A;s06A;s11;s22 | s04B;s93 |

No train/validation batch overlap was present.

## G–L. Linear probe, confusion, and geometry

All three representations used the same standardization, class-balanced multinomial softmax linear probe, fixed C=1, fixed seed, and the same fold assignments. The GPU implementation is mathematically equivalent to multinomial logistic regression and was used to avoid an impractically slow CPU L-BFGS fit.

Phikon SMALL improved over CellViT by +0.2263 macro-F1 and +0.2047 macro-AUPRC. CONTEXT was worse than SMALL by −0.0640 macro-F1 and −0.0601 macro-AUPRC in paired folds. Neutrophil metrics were:

| representation | precision | recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| CellViT_TOKEN | 0.0453 | 0.1691 | 0.0631 | 0.4659 | 0.0500 |
| PHIKON_V2_SMALL | 0.1491 | 0.2810 | 0.1885 | 0.6874 | 0.1349 |
| PHIKON_V2_CONTEXT | 0.1169 | 0.1831 | 0.1414 | 0.6125 | 0.0998 |

Binary specialist AUROC / AUPRC means were: CellViT 0.5002 / 0.2407 for Neutrophil-vs-Myeloid and 0.4605 / 0.1919 for Neutrophil-vs-T/B; Phikon SMALL 0.6240 / 0.3440 and 0.6932 / 0.3526; CONTEXT 0.5472 / 0.2811 and 0.6349 / 0.3045, respectively.

The main confusion rates, averaged across folds, were N→Myeloid approximately 0.100 for CellViT, 0.143 for SMALL, and 0.158 for CONTEXT; N→T/B was approximately 0.134, 0.139, and 0.167. Thus the specialist AUROC/AUPRC improved, but the thresholded multiclass confusion flow did not uniformly decrease; this distinction is retained as an observed result.

Geometry showed class-separation ratios of 1.000 (CellViT), 1.014 (SMALL), and 1.019 (CONTEXT), while batch-separation ratios were 1.020, 1.130, and 1.265. kNN batch purity was 0.340, 0.841, and 0.987, respectively. The Phikon gain is therefore accompanied by a major batch-structure signal, especially for CONTEXT.

## M. MLP probe

Using the same `Linear256 → ReLU → Dropout0.5 → Linear128 → ReLU → Dropout0.5 → Linear7` architecture and class-balanced AdamW training:

| representation | MLP macro-F1 | MLP macro-AUPRC |
|---|---:|---:|
| CellViT_TOKEN | 0.1095 ± 0.0155 | 0.1398 ± 0.0046 |
| PHIKON_V2_SMALL | 0.3008 ± 0.0213 | 0.3189 ± 0.0344 |
| PHIKON_V2_CONTEXT | 0.2489 ± 0.0257 | 0.2700 ± 0.0319 |

The MLP did not reverse the representation ranking and did not exceed the linear probe on this frozen cohort.

## N. GT-centered morphology upper bound

An additional diagnostic used all 156,300 Task009 V3_CORE training cells with valid registered centers and both crop bounds, without requiring a CellViT detection. This is not the deployable shared-cell benchmark and does not change the frozen shared manifest.

- GT-centered SMALL: macro-F1 0.3379 ± 0.0231; macro-AUPRC 0.3431 ± 0.0265
- GT-centered CONTEXT: macro-F1 0.2746 ± 0.0216; macro-AUPRC 0.2871 ± 0.0260

The GT-centered upper bound is nearly identical to the shared-detected SMALL result, so the observed representation gain is not explained solely by excluding CellViT-undetected cells. CONTEXT remains lower than SMALL.

## O–Q. Interpretation and next direction

Observed evidence is most consistent with a meaningful frozen local morphology representation gain over the CellViT token baseline, with little support for CONTEXT helping under the current preprocessing and batches. However, the large increase in batch separation means that the next experiment should prioritize batch-robust validation/normalization and possibly local-morphology-focused or hierarchical immune classification. A multiscale model should be considered only after controlling the batch signal. No biological hypothesis is asserted from these representation metrics alone.

Midnight-12k is still pending; Phase B has not been run. Production `/data/lf_data/result/model_best.pth`, Task007 labels, Task008 eligibility, and Task009 outputs were not modified.

## Reproducibility and artefacts

Primary scripts: `scripts/python/task010_phikon_phase_a.py`, `scripts/python/task010_gt_centered_upper_bound.py`, and `scripts/python/task010_finalize_figures.py`. The remote command used the isolated offline environment and the output root `/data/lf_data/result/task010_representation_benchmark`.

Remote outputs include the frozen manifests, crop manifests, CellViT and Phikon feature tensors, fold-wise metrics, binary diagnostics, geometry, MLP and GT-centered metrics, all eleven Phase A figures, crop montages, and QC/provenance files. Large embeddings, crops, checkpoints, and server credentials are intentionally not committed to Git. Previous MUSK and public-download blocked-run artefacts remain in the remote output root as provenance.
