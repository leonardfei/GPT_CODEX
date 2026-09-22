# Project Status

## Current task

Task 006 — Xenium re-annotation, H&E nuclear-integrity QC, and high-quality CellViT retraining — PENDING

## Last completed task

Task 005 — multi-backbone, stain-domain, and Neutrophil detection benchmark — COMPLETED; no condition promoted

## Repository status

Tasks 001–005 are complete. Task 006 has been created because the working hypothesis has shifted upstream: the original Xenium-derived training labels may contain substantial biological annotation error, low-quality cells, and necrosis-associated/no-nucleus objects, including false Neutrophil labels from residual RNA.

Task 006 will rebuild the Xenium ground truth before further CellViT model development.

## Task 006 input files

- Xenium AnnData:
  `/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad`
- Original H&E:
  `/data/lf_data/xenium_data/ID0060276.ome.tif`
- Registration matrix:
  `/data/lf_data/xenium_data/matrix.csv`
- Previous preprocessing notebook:
  `/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb`

Task 006 outputs must be written under:

`/data/lf_data/result/task006_xenium_reannotation`

Source inputs must not be modified.

## Scientific rationale

Task 005 showed:
- SAM-H RAW grouped-CV macro-F1: 0.3324 ± 0.0134
- Neutrophil F1: 0.1145
- Neutrophil detection recall: 0.5467
- Neutrophil conditional classifier recall: 0.0824
- Neutrophil end-to-end recall: 0.0450
- stain normalization reduced batch structure but did not materially improve classification

Earlier tasks also showed that:
- classifier-head optimization did not materially improve performance;
- strict official CellViT++ training did not improve performance;
- stricter centroid-only matching did not improve weak-class performance.

The next hypothesis is therefore that ground-truth quality itself is limiting performance.

## Task 006 design

Task 006 will independently combine:

1. Xenium transcriptomic QC;
2. seven-class biological re-annotation;
3. H&E nuclear-integrity QC after registration;
4. explicit Neutrophil debris/no-nucleus safeguards;
5. frozen Xenium-v2 labels;
6. controlled old-label vs Xenium-v2 CellViT retraining.

The seven training classes remain:
- Endothelial
- Mesenchymal
- Myeloid
- Neutrophil
- Plasma cell
- T and B
- Tumor

Non-training states are allowed:
- Uncertain
- Mixed_lineage
- Low_quality
- Artifact_or_no_nucleus

Cells must not be forced into one of the seven classes.

## Anti-circularity rule

CellViT classifier predictions must not be used to decide which Xenium cells are high quality.

The Xenium-v2 annotation/QC rules must be frozen before viewing new CellViT cross-validation results.

## Current production model

`/data/lf_data/result/model_best.pth`

SHA256:

`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Task 001 remains production until a future model satisfies predefined grouped-CV promotion criteria.

## Pending tasks

- Task 006 — rebuild Xenium annotations and high-quality training labels, then perform controlled retraining.
- Do not start Task 007 until Task 006 is completed and reviewed by Web GPT.

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
- `PROJECT_STATUS.md`

## Next execution command

```text
Execute task_006.
```

Codex must pull `origin/main` before execution and follow `AGENTS.md` plus `tasks/task_006.md`.

## Last update

2026-09-22
