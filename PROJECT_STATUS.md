# Project Status

## Current task

Task 009 — CellViT retraining with frozen panel-aware Xenium labels and recalibrated Neutrophil eligibility — PENDING

## Last completed task

Task 008 — Neutrophil nucleus-centered H&E QC recalibration — COMPLETED after manual review acceptance

## Repository status

Tasks 001–008 are complete.

Task 007 corrected the Xenium 5K panel-awareness problem and produced a conservative panel-aware biological annotation:
- KEEP 987,846
- RELABEL 0
- REVIEW 3,004
- broad original Neutrophil 24,167
- biologically retained Neutrophil 24,107

Task 008 corrected the Task 007 whole-crop H&E debris over-call using target-centered nuclear evidence:
- TARGET_NUCLEUS_PRESENT 22,630
- NO_TARGET_NUCLEUS 13
- FRAGMENTED_TARGET_SUSPECT 11
- MANUAL_REVIEW 1,736
- TRAINABLE_CORE Neutrophil 21,639
- TRAINABLE_EXTENDED Neutrophil 22,107

The supervising user manually reviewed the Task 008 H&E montage and reported that the large majority of candidate nuclei are normal. The Task 008 review gate is therefore closed and its target-centered eligibility is accepted for downstream model testing.

## Frozen annotation decision

The ground truth for Task 009 is frozen BEFORE training:

### Biological identity
Use Task 007 panel-aware v3 biological labels.

### Neutrophil H&E eligibility
Primary:
`Task008 TRAINABLE_CORE`

Sensitivity:
`Task008 TRAINABLE_EXTENDED`

Do not change annotation or QC thresholds in response to Task 009 model performance.

## Task 009 objective

Rebuild the CellViT seven-class training dataset using:
- Task 007 frozen panel-aware biological labels for all classes;
- Task 008 accepted target-centered Neutrophil eligibility;
- historical patch geometry and batch split;
- the same official SAM-H RAW fixed training recipe used in prior controlled benchmarks.

Compare:
```text
OLD_LABELS
V3_CORE
V3_EXTENDED
```

Primary endpoints:
- macro-F1
- macro-AUPRC
- lowest-three-class F1
- Neutrophil precision / recall / F1 / AUPRC
- per-class detection recall
- Neutrophil end-to-end recall

## Current production model

`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Production remains unchanged until Task 009 predefined promotion criteria are satisfied.

## Task 009 inputs

Task 007 annotation:
`/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`

Task 008 Neutrophil eligibility:
`/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`

H&E:
`/data/lf_data/xenium_data/ID0060276.ome.tif`

Registration:
`/data/lf_data/xenium_data/matrix.csv`

Historical dataset notebook:
`/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb`

Task 009 output root:
`/data/lf_data/result/task009_v3_retraining`

## Pending tasks

- Execute Task 009.
- Do not modify frozen v3 annotation or Task008 eligibility based on model results.
- Do not start a context-aware/specialist model task until Task009 is completed and reviewed.

## Latest workflow files

- `tasks/task_007.md`
- `reports/task_007_report.md`
- `tasks/task_008.md`
- `reports/task_008_report.md`
- `tasks/task_009.md`
- `PROJECT_STATUS.md`

## Next execution command

```text
Execute task_009.
```

Codex must pull `origin/main` before execution and follow `AGENTS.md` plus `tasks/task_009.md`.

## Last update

2026-09-23
