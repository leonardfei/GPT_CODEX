# Project Status

## Current task

Task 008 — Neutrophil nucleus-centered H&E QC recalibration — PENDING

## Last completed task

Task 007 — Xenium 5K panel-aware reannotation and training-label QC — COMPLETED; biological annotation retained, Neutrophil H&E eligibility requires recalibration

## Repository status

Tasks 001–007 are complete. Task 007 successfully corrected the panel-awareness problem from Task 006 and preserved nearly all biological identities:
- KEEP 987,846
- RELABEL 0
- REVIEW 3,004
- broad original Neutrophil 24,167
- biologically retained Neutrophil 24,107

However, Task 007 H&E triage flagged 22,164 Neutrophils as debris-suspect and retained only 1,969 as trainable. Review of the implementation showed that the debris rule depended on connected-component counts across the entire 128×128 H&E crop, which is likely confounded by neighboring nuclei in immune-dense tissue.

Task 008 has therefore been created to recalibrate Neutrophil H&E eligibility using target-centered nuclear evidence.

## Fixed biological annotation

Task 007 v3 panel-aware biological labels are frozen for Task 008.

Task 008 must not change:
- original_cl1
- Neutrophil / Neutrophil_CXCR4 subtype labels
- seven-class biological mapping
- Task 007 panel-aware biological identity

Task 008 changes only:
- H&E nuclear-quality status
- Neutrophil training eligibility

## Task 008 principles

- Use level-0 H&E.
- Calibrate empirical centroid-to-nucleus offset using technically valid non-Neutrophil reference cells.
- Judge the target cell from a center-associated nuclear component/group, not the total number of components in the 128×128 context crop.
- Allow multilobulated Neutrophil nuclei.
- Separate target-nucleus evidence from necrosis/debris context.
- Integrate Xenium nucleus_count/nucleus_area as independent supporting evidence.
- Ambiguous cells go to MANUAL_REVIEW rather than hard exclusion.
- Do not use CellViT predictions.
- Do not train CellViT in Task 008.

## Inputs

- Xenium source:
  `/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad`
- H&E:
  `/data/lf_data/xenium_data/ID0060276.ome.tif`
- registration:
  `/data/lf_data/xenium_data/matrix.csv`
- Task 007 v3 annotation:
  `/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`

Task 008 output root:

`/data/lf_data/result/task008_neutrophil_he_recalibration`

## Task 007 result retained as biological baseline

Panel:
- 5,001 genes
- CD3D absent
- CD3E present

Biological annotation:
- KEEP 987,846
- RELABEL 0
- REVIEW 3,004

Broad Neutrophil:
- original 24,167
- Neutrophil_CXCR4 9,520
- biologically retained 24,107

Task 007 H&E triage:
- no-visible-nucleus 0
- debris-suspect 22,164
- trainable 1,969

The Task 007 H&E debris status must not be treated as final training ground truth.

## Current production model

`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Production remains unchanged.

## Pending tasks

- Task 008 — execute target-centered Neutrophil H&E QC recalibration.
- Do not start CellViT retraining until Task 008 is completed and reviewed.

## Latest workflow files

- `tasks/task_007.md`
- `reports/task_007_report.md`
- `tasks/task_008.md`
- `PROJECT_STATUS.md`

## Next execution command

```text
Execute task_008.
```

Codex must pull `origin/main` before execution and follow `AGENTS.md` plus `tasks/task_008.md`.

## Last update

2026-09-23
