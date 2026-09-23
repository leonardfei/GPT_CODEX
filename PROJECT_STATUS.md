# Project Status

## Current task

Task 007 — Xenium 5K panel-aware reannotation and training-label QC — PENDING

## Last completed task

Task 006 — Xenium re-annotation, H&E nuclear-integrity QC, and high-quality CellViT retraining — COMPLETED; v2 labels not adopted

## Repository status

Tasks 001–006 are complete. Task 006 demonstrated that the upstream ground-truth problem is real, but its reannotation policy was not appropriate for a targeted Xenium 5K panel and caused severe over-filtering. The v2 labels must not be used as production ground truth.

Task 007 has been created to rebuild the annotation using the actual Xenium panel, original `cl1` as a prior, batch-aware data-driven class signatures, conservative KEEP/RELABEL/REVIEW decisions, and independent H&E/segmentation quality control.

Task 007 must stop before CellViT retraining.

## Fixed source inputs

- Xenium AnnData:
  `/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad`
- H&E:
  `/data/lf_data/xenium_data/ID0060276.ome.tif`
- Registration matrix:
  `/data/lf_data/xenium_data/matrix.csv`
- Historical preprocessing notebook:
  `/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb`

Task 007 output root:

`/data/lf_data/result/task007_xenium5k_panelaware`

Source files must remain unchanged.

## Why Task 006 v2 is not accepted

Task 006 produced:
- old→new label change rate: ~89.7%
- HQ_CORE: 73,960 / 990,850 cells
- T and B HQ_CORE: 1,386
- Neutrophil HQ_CORE: 445
- V2_CORE macro-F1 lower than OLD_LABELS

The main methodological issue was that annotation confidence depended on external/canonical marker logic and uniform marker-support rules that are inappropriate for a targeted ~5,000-gene panel.

Task 007 corrects this by:
- using only genes actually present in `adata.var_names`;
- deriving panel-aware class signatures from the data;
- preserving original `cl1` as a prior;
- separating biological identity from technical cell quality;
- using batch-held-out cross-fitting for label consistency;
- requiring high-specificity evidence for automatic relabeling;
- treating ambiguity as REVIEW rather than Low_quality;
- using full-resolution H&E review for all original/proposed Neutrophils.

## Seven-class taxonomy

- Endothelial
- Mesenchymal
- Myeloid
- Neutrophil
- Plasma cell
- T and B
- Tumor

Original finer labels such as `Neutrophil_CXCR4` must be preserved in separate audit fields.

## Task 007 anti-overfiltering safety rails

Before any v3 annotation is called FINAL, automatically flag review if:
- >30% of non-Low-quality original cells are biologically relabeled;
- >40% of a major class is sent to identity REVIEW solely for transcriptional ambiguity;
- TRAIN_EXTENDED retains <50% of technically valid cells for a major class without a documented artifact;
- T/B or Neutrophil is depleted >50% by annotation evidence alone;
- one batch is disproportionately depleted.

If triggered, Task 007 must be marked PARTIAL and the v3 annotation PROVISIONAL.

## Anti-circularity rule

Do not use CellViT predictions or CellViT performance to define or modify v3 annotations.

Task 007 must not train CellViT.

## Current production model

`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Production remains unchanged.

## Pending tasks

- Task 007 — execute Xenium 5K panel-aware reannotation and training-label QC.
- Do not start CellViT retraining until Task 007 is reviewed and v3 labels are accepted.

## Latest workflow files

- `tasks/task_001.md`
- `reports/task_001_report.md`
- `tasks/task_002.md`
- `reports/task_002_report.md`
- `tasks/task_003.md`
- `reports/task_003_report.md`
- `tasks/task_004.md`
- `reports/task_004_report.md`
- `tasks/task_005.md`
- `reports/task_005_report.md`
- `tasks/task_006.md`
- `reports/task_006_report.md`
- `tasks/task_007.md`
- `PROJECT_STATUS.md`

## Next execution command

```text
Execute task_007.
```

Codex must pull `origin/main` before execution and follow `AGENTS.md` plus `tasks/task_007.md`.

## Last update

2026-09-23
