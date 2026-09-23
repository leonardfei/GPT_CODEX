# Task 011 Report — Midnight Local-Morphology Optimization and Hierarchical Immune Specialist

## Status and scope

Task 011 is complete. The analysis used the Task010 canonical cohort of 96,044 cells, the exact Task009 V3_CORE five held-out batch folds, the frozen Task007/Task008 biological ground truth, and the frozen Midnight-12k encoder. No labels, eligibility rules, Task009 outputs, or production checkpoint were changed.

The remote output root was `/data/lf_data/result/task011_midnight_local_optimization`. Small metrics, configurations, QC tables, and figures were copied to `metrics/task011/`, `config/task011/`, `qc/task011/`, and `figures/task011/` for review. Large crops and embedding tensors remain remote and were not committed.

Seed: `20260923`. Primary identity: `cell_id`. H&E scale: `0.2125 μm/px`. The local F1 gate used the exact outer folds and selected epoch from a training-only inner batch; outer validation labels were not used during refitting.

## A–B. Local FOV sweep

| FOV | Native crop | Macro-F1 | Macro-AUPRC | Lowest-three F1 |
|---|---:|---:|---:|---:|
| 12 μm | 57 px | 0.4204 ± 0.0330 | 0.4403 ± 0.0415 | 0.2591 |
| 16 μm | 75 px | 0.4088 ± 0.0283 | 0.4303 ± 0.0396 | 0.2428 |
| 20 μm | 95 px | 0.3794 ± 0.0255 | 0.3975 ± 0.0376 | 0.2186 |
| 24 μm | 113 px | 0.3520 ± 0.0245 | 0.3735 ± 0.0390 | 0.1895 |
| 32 μm | 151 px | 0.3231 ± 0.0225 | 0.3433 ± 0.0364 | 0.1662 |

Observed result: FOV12 was selected. The Task010 16 μm choice was not optimal in this focused sweep; FOV12 improved macro-F1 by 0.0116 absolute over FOV16. This is a development benchmark selection, not a production claim.

## C–D. Frozen feature fusion

| Candidate | Macro-F1 | Delta vs B0 raw | Decision |
|---|---:|---:|---|
| B0 raw Midnight FOV12 | 0.4204 | — | Preferred frozen baseline |
| B1 raw + CellViT | 0.4344 | +0.0140 | Positive in 5/5 folds but below +0.015 criterion |
| B2 raw + Phikon | 0.4161 | −0.0043 | Not useful |
| B3 raw + CellViT + Phikon | 0.4299 | +0.0095 | Below criterion |

The dimension-controlled PCA branches did not reproduce the raw-concatenation gain: B1 PCA-controlled reached 0.4138 macro-F1 and B3 PCA-controlled 0.4160. Therefore `B0_raw` remains the selected frozen fusion under the prespecified rule. CellViT showed some complementary signal, but not enough for the Task011 promotion threshold; Phikon did not add useful signal in this setting.

## E–G. Hierarchy, specialist, and TTA

| Stage | Macro-F1 | Neutrophil F1 | Neutrophil AUPRC | Decision |
|---|---:|---:|---:|---|
| Flat BEST_FOV | 0.4204 | 0.2138 | 0.1671 | Baseline |
| Soft hierarchy | 0.4065 | 0.2092 | 0.1678 | Worse overall; not promoted |
| Hierarchy + N specialist | 0.3799 | 0.1879 | 0.1511 | No rescue; not promoted |
| TTA4 | 0.4268 | — | — | +0.0064; below keep threshold |
| TTA8 | 0.4267 | — | — | +0.0063; below keep threshold |

The hierarchy did not improve the four immune classes without sacrificing overall performance. The specialist decreased both overall macro-F1 and Neutrophil F1/AUPRC, so the specialist rescue criterion was not met. TTA4 and TTA8 gave small improvements, below the required +0.01 macro-F1 or +0.02 Neutrophil-AUPRC threshold; the selected TTA configuration is `SINGLE`.

## H–I. Fine-tuning gate

Stages A–E plateaued with frozen BEST_FOV macro-F1 0.4204, below 0.50, so F1 fine-tuning was triggered. The final transformer block and classification head were trained with class-balanced loss, mild rotations/flips/brightness/contrast, and inner grouped epoch selection. LoRA/adapters were not run because no local implementation was available without external downloads.

Outer-fold F1 macro-F1 values were 0.4663, 0.4617, 0.3960, 0.4540, and 0.4842; mean 0.4524 ± 0.0335. Relative to the frozen BEST_FOV model:

- macro-F1 gain: +0.0320;
- Neutrophil F1 gain: +0.0185;
- Neutrophil AUPRC gain: +0.1195;
- macro-F1 improved in 5/5 outer folds;
- batch kNN purity increased by 0.0455, within the 0.05 guardrail;
- spatial kNN purity changed by −0.0011.

Thus the Task011 fine-tuning promotion criteria were met through macro-F1 and Neutrophil-AUPRC improvement, five-fold improvement, and the purity guardrail. The preferred Task012 candidate is `F1_FINAL_BLOCK`, still subject to independent-slide/patient validation.

## J. Model ladder and final metrics

| Model | Macro-F1 | Macro-AUPRC | Lowest-three F1 | N F1 | N AUPRC | N-vs-Myeloid AUROC/AUPRC | N-vs-T/B AUROC/AUPRC |
|---|---:|---:|---:|---:|---:|---|---|
| Task010 corrected CellViT | 0.3364 | 0.3418 | 0.1897 | 0.1885 | 0.1327 | 0.644 / 0.3507 | 0.673 / 0.3306 |
| Task010 Midnight 16 μm | 0.4116 | 0.4327 | 0.2444 | 0.2175 | 0.1620 | 0.671 / 0.3801 | 0.707 / 0.3706 |
| Task011 BEST_FOV | 0.4204 | 0.4403 | 0.2591 | 0.2138 | 0.1671 | 0.688 / 0.3953 | 0.698 / 0.3603 |
| Task011 BEST_FUSION | 0.4204 | 0.4403 | 0.2591 | 0.2138 | 0.1671 | 0.689 / 0.3980 | 0.696 / 0.3603 |
| Task011 HIERARCHICAL | 0.4065 | 0.4396 | 0.2378 | 0.2092 | 0.1678 | 0.688 / 0.3953 | 0.698 / 0.3603 |
| Task011 HIERARCHICAL + N specialist | 0.3799 | 0.4227 | 0.1849 | 0.1879 | 0.1511 | 0.688 / 0.3953 | 0.696 / 0.3603 |
| Task011 TTA best (SINGLE) | 0.4204 | 0.4403 | 0.2591 | 0.2138 | 0.1671 | 0.688 / 0.3953 | 0.698 / 0.3603 |
| Task011 F1 final block | 0.4524 | 0.4883 | 0.2870 | 0.2323 | 0.1959 | 0.714 / 0.5148 | 0.748 / 0.3678 |

The largest incremental gain came from the gated F1 final-block fine-tuning (+0.0320 macro-F1 over frozen BEST_FOV). The final seven-class accuracy was 0.4964 and balanced accuracy 0.4787.

Final per-class F1/AUPRC:

| Class | F1 | AUPRC |
|---|---:|---:|
| Endothelial | 0.5578 | 0.5991 |
| Mesenchymal | 0.3965 | 0.4295 |
| Myeloid | 0.2833 | 0.3249 |
| Neutrophil | 0.2323 | 0.1959 |
| Plasma cell | 0.4582 | 0.4750 |
| T and B | 0.4569 | 0.5373 |
| Tumor | 0.7820 | 0.8562 |

## K–M. Error profile and binary specialist endpoints

Observed errors were concentrated in Neutrophil and Myeloid, with Mesenchymal the next-lowest seven-class F1. Neutrophil remained the limiting class despite the large improvement in probability-ranking performance. The final true binary endpoints were:

- Neutrophil vs Myeloid: AUROC 0.7139, AUPRC 0.5148, F1 0.4552 mean across folds;
- Neutrophil vs T/B: AUROC 0.7481, AUPRC 0.3678, F1 0.3937 mean across folds.

These are held-out outer-fold diagnostic measurements, not biological claims about patient-level separability.

## N. Spatial and batch robustness

The final F1 model had batch kNN purity 0.6637 versus 0.6182 for frozen BEST_FOV (+0.0455), and spatial kNN purity 0.0340 versus 0.0351 (−0.0011). The batch change remained within the prespecified 0.05 guardrail; the five outer-fold macro-F1 range was 0.3960–0.4842. The spatial proxy is `patch_id` on a single paired specimen, so batch purity must not be interpreted as a pure technical-batch effect. Independent slides and patients are still required.

## O. Success targets and Task012 recommendation

The final F1 model meets the minimum meaningful internal target because macro-F1 is 0.4524 ≥ 0.43. It does not meet the strong target (macro-F1 < 0.46, Neutrophil F1 < 0.27, and Neutrophil AUPRC < 0.20) or the excellent target (macro-F1 < 0.50, Neutrophil F1 < 0.30, and N-vs-Myeloid AUROC < 0.75). These targets are internal diagnostic benchmarks only.

Carry `F1_FINAL_BLOCK` at FOV12 into Task012 as a validation candidate, not as a production replacement. Task012 should pre-register an untouched independent-slide/patient holdout, keep the ground-truth labels and production model frozen, use patient/slide-level grouping, report calibration and per-class uncertainty, and repeat batch/spatial robustness checks. The final model should only be considered for production after that independent validation.

## Reproducibility and audit trail

- Scripts: `scripts/python/task011_midnight_optimization.py`, `scripts/python/task011_finetune_f1.py`, and `scripts/python/task011_finalize.py`.
- Remote environment: `/data/lf_data/task010_env/bin/python` with Python 3.10.14, PyTorch 2.7.1+cu128, pandas 1.4.3, and scikit-learn 1.3.0; remote commands and logs are under the Task011 output root.
- Inputs: Task010 canonical cell order, Task009 V3_CORE fold assignments, original H&E OME-TIFF, frozen Midnight-12k, and aligned Task010 CellViT/Phikon features.
- Outputs: remote `/data/lf_data/result/task011_midnight_local_optimization`; reviewable local copies under `metrics/task011/`, `config/task011/`, `qc/task011/`, and `figures/task011/`.
- Fine-tuning QC: `qc/task011/fine_tune_inner_validation.csv` records the training-only inner batch and selected epoch for all five folds; outer validation was evaluated only after refitting.
- Production guardrail: `/data/lf_data/result/model_best.pth` was not modified.
