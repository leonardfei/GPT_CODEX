# Project Status

## Current task

Task 011 — Midnight local-morphology optimization and hierarchical immune specialist — COMPLETED

## Last completed benchmark

Task 010 — Corrected CellViT alignment + local Midnight-12k integration — corrected five-way shared-cohort benchmark COMPLETE; secondary Midnight GT-centered upper bound remains optional/pending.

## Frozen biological ground truth

Task007 panel-aware Xenium annotation and Task008 H&E eligibility remain frozen.

Task009 regenerated V3_CORE data and established the prior CellViT benchmark.

Do not change labels or eligibility based on model performance.

## Task010 corrected canonical results

Canonical shared cohort:
`96,044` cells

Exact Task009 V3_CORE held-out batch folds reused.

Corrected linear macro-F1:
- CELLVIT_TOKEN_ALIGNED 0.3364 ± 0.0263
- PHIKON_V2_SMALL 0.3366 ± 0.0243
- PHIKON_V2_CONTEXT 0.2726 ± 0.0253
- MIDNIGHT12K_SMALL 0.4116 ± 0.0312
- MIDNIGHT12K_CONTEXT 0.2806 ± 0.0221

Neutrophil:
- corrected CellViT F1 0.1885, AUPRC 0.1327
- Phikon SMALL F1 0.1885, AUPRC 0.1349
- Midnight SMALL F1 0.2175, AUPRC 0.1620

True binary Midnight SMALL:
- Neutrophil vs Myeloid AUROC/AUPRC 0.671/0.380
- Neutrophil vs T/B AUROC/AUPRC 0.707/0.371

Key conclusions:
- the prior Task010 CellViT near-random result was an alignment bug and is superseded;
- Phikon SMALL is approximately equal to corrected CellViT;
- Midnight-12k SMALL is the strongest current frozen representation;
- both Phikon and Midnight show SMALL >> CONTEXT;
- large CONTEXT strongly encodes spatial/batch structure;
- the next optimization should focus on local target-cell morphology and immune-class specialization.

## Task011 strategy

Task011 proceeds in staged order:

### Stage A — local FOV sweep
Frozen Midnight-12k:
- 12 μm
- 16 μm
- 20 μm
- 24 μm
- 32 μm

All centered on the matched H&E nucleus.

### Stage B — frozen feature fusion
Compare:
- Midnight best FOV
- Midnight + aligned CellViT
- Midnight + Phikon SMALL
- Midnight + aligned CellViT + Phikon SMALL

Use raw concatenation and a training-only PCA-controlled secondary analysis.

### Stage C — soft hierarchical classifier
Coarse:
- Tumor
- Structural
- Immune

Structural:
- Endothelial
- Mesenchymal

Immune:
- Myeloid
- Neutrophil
- Plasma
- T/B

Final seven-class probabilities are composed softly from the hierarchy.

### Stage D — Neutrophil specialist rescue
Training-only specialist heads:
- N vs Myeloid
- N vs T/B
- N vs all other immune cells

Combine specialist evidence only through training-only inner calibration.

### Stage E — orientation-invariant TTA
Compare:
- single view
- TTA4 rotations
- TTA8 rotations + flips

Keep only if performance improves without worsening spatial/batch purity excessively.

### Stage F — partial Midnight fine-tuning gate
Do not fine-tune until Stages A–E are complete.

Fine-tuning only if frozen optimization plateaus:
- final block unfreeze
- optional LoRA/adapters if locally available

Use strict inner grouped validation and untouched outer folds.

## Task011 internal targets

Minimum meaningful:
- macro-F1 >=0.43
OR
- Neutrophil F1 >=0.25

Strong:
- macro-F1 >=0.46
AND
- Neutrophil F1 >=0.27
AND
- Neutrophil AUPRC >=0.20

Excellent within-specimen:
- macro-F1 >=0.50
AND
- Neutrophil F1 >=0.30
AND
- true N-vs-Myeloid AUROC >=0.75

These are internal development targets, not claims of patient-level generalization.

## Task011 completed results

Canonical cohort: `96,044` cells; exact Task009 V3_CORE five held-out batch folds; seed `20260923`.

- BEST_FOV: Midnight FOV12, 57 native px, macro-F1 `0.4204 ± 0.0330`.
- Task010 FOV16 was not optimal; FOV12 improved macro-F1 by `+0.0116`.
- BEST_FUSION: `B0_raw`; CellViT raw fusion reached `0.4344` (+0.0140) but did not meet the +0.015 promotion criterion. PCA-controlled fusion did not reproduce the gain.
- Hierarchy: macro-F1 `0.4065`; N specialist: `0.3799`; neither promoted.
- TTA4/TTA8 macro-F1: `0.4268/0.4267`; both below the keep threshold, so `SINGLE` retained.
- Fine-tuning gate triggered after frozen optimization plateaued. F1 final-block macro-F1 `0.4524 ± 0.0335`, Neutrophil F1 `0.2323`, Neutrophil AUPRC `0.1959`; macro-F1 improved in 5/5 folds, batch purity delta `+0.0455`, spatial purity delta `−0.0011`.
- Final preferred validation candidate: `F1_FINAL_BLOCK` at FOV12. It meets the minimum meaningful internal target, but not the strong or excellent target.

No frozen biological ground truth, Task009 output, or `/data/lf_data/result/model_best.pth` was modified. Task011 remains diagnostic and requires independent-slide/patient validation before production consideration.

## Canonical data

Task010 canonical order:
`/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz`

Original H&E:
`/data/lf_data/xenium_data/ID0060276.ome.tif`

Validated scale:
`0.2125 μm/px`

Midnight-12k:
`/data/lf_data/models/midnight-12k`

Phikon-v2:
`/data/lf_data/models/phikon-v2`

Corrected CellViT aligned feature:
`/data/lf_data/result/task010_representation_benchmark/features/cellvit_tokens_aligned.pt`

Task011 output root:
`/data/lf_data/result/task011_midnight_local_optimization`

## Current production model

`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Production remains unchanged.

## Pending tasks

- Design and execute Task012 independent-slide/patient validation for `F1_FINAL_BLOCK`.
- Preserve strict canonical cell alignment and exact grouped folds.
- Do not change frozen biological labels.
- Do not promote Task011 model to production without independent-slide/patient validation.

## Latest workflow files

- tasks/task_010.md
- reports/task_010_report.md
- tasks/task_011.md
- reports/task_011_report.md
- scripts/python/task011_midnight_optimization.py
- scripts/python/task011_finetune_f1.py
- scripts/python/task011_finalize.py
- PROJECT_STATUS.md

## Next execution command

```text
Prepare Task012 independent-slide/patient validation.
```

Codex must pull origin/main and follow AGENTS.md plus tasks/task_011.md.

## Last update

2026-09-24
