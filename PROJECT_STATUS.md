# Project Status

## Current task

Task 010 — Frozen representation benchmark: CellViT token vs MUSK nuclear/contextual H&E embeddings — PENDING

## Last completed task

Task 009 — CellViT retraining with frozen panel-aware Xenium labels and recalibrated Neutrophil eligibility — COMPLETED; no production promotion

## Repository status

Tasks 001–009 are complete.

Task007 established the frozen Xenium 5K panel-aware biological annotation.
Task008 established and manually validated target-centered Neutrophil H&E eligibility.
Task009 regenerated all V3 datasets from the original OME-TIFF and showed that corrected ground truth alone did not materially improve the official SAM-H classifier:
- OLD_LABELS macro-F1 0.3324 ± 0.0134
- V3_CORE macro-F1 0.3330 ± 0.0222
- V3_EXTENDED macro-F1 0.3340 ± 0.0217
- V3_EXTENDED Neutrophil F1 0.1084
- Neutrophil end-to-end recall ~0.055

The current hypothesis is therefore that the dominant limitation is representation/classification rather than label eligibility alone.

## Task 010 objective

Task010 is a minimal, controlled representation benchmark inspired by the CANVAS design principle:

- retain CellViT for nucleus detection/segmentation/localization;
- test whether a pathology foundation model (MUSK) provides better H&E morphology/context embeddings than the CellViT cell token.

Primary representations:

1. CellViT_TOKEN
2. MUSK_SMALL — ~16 μm nucleus/local-cell field of view
3. MUSK_CONTEXT — ~56 μm local microenvironment field of view

All primary conditions must use the same:
- V3_CORE labels;
- detected/matched cell cohort;
- nucleus center;
- five held-out batch folds;
- linear-probe classifier.

Task010 must not fine-tune MUSK in the primary comparison.

## Frozen ground truth

Task007 annotation:
 /data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz

Task008 Neutrophil eligibility:
 /data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz

Primary label set:
 V3_CORE

Do not change annotation/QC rules based on Task010 performance.

## Data sources

Original H&E:
 /data/lf_data/xenium_data/ID0060276.ome.tif

Registration:
 /data/lf_data/xenium_data/matrix.csv

Task009 regenerated CORE dataset:
 /data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE

Historical dataset:
 /data/lf_data/xenium_data/CellViT_dataset

The historical dataset is forbidden as an image/label/split/embedding source.

## MUSK gate

Task010 must first verify an official MUSK pathology model and record model provenance and SHA256.

If official MUSK is unavailable without credentials, gated approval, private tokens, or manual license acceptance:
- mark Task010 BLOCKED_MUSK_ACCESS;
- do not silently substitute another foundation model.

## Primary benchmark

Shared cohort:
 SHARED_DETECTED_CORE

Each included cell must have:
- frozen V3_CORE label;
- valid CellViT matched nucleus/token;
- valid MUSK_SMALL crop;
- valid MUSK_CONTEXT crop.

MUSK crops are cut directly from the original OME-TIFF and centered on the matched H&E nucleus.

Primary classifier:
 identical multinomial linear probe.

Secondary:
 identical MLP probe.

## Key endpoints

Overall:
- macro-F1
- macro-AUPRC
- lowest-three F1

Neutrophil:
- precision
- recall
- F1
- AUROC
- AUPRC
- Neutrophil→Myeloid
- Neutrophil→T/B

Binary diagnostics:
- Neutrophil vs Myeloid
- Neutrophil vs T/B

Secondary upper bound:
- MUSK_SMALL and MUSK_CONTEXT on all GT-centered V3_CORE cells, including cells not detected by CellViT.

## Interpretation

If MUSK_CONTEXT strongly outperforms MUSK_SMALL and CellViT:
- proceed to multiscale/context-aware model.

If MUSK_SMALL outperforms CellViT but context adds little:
- proceed to local morphology specialist / limited MUSK fine-tuning.

If binary immune comparisons improve but seven-class does not:
- proceed to hierarchical immune classifier.

If frozen MUSK does not improve:
- test limited fine-tuning before building a complex architecture.

## Current production model

 /data/lf_data/result/model_best.pth

SHA256:
 f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164

Production remains unchanged.

## Task 010 output root

 /data/lf_data/result/task010_representation_benchmark

## Pending tasks

- Execute Task010.
- Do not build the final multiscale model until Task010 identifies which representation/scale carries useful signal.
- Do not modify production.
- Do not modify frozen ground truth.

## Latest workflow files

- tasks/task_009.md
- reports/task_009_report.md
- tasks/task_010.md
- PROJECT_STATUS.md

## Next execution command

Execute task_010.

Codex must pull origin/main before execution and follow AGENTS.md plus tasks/task_010.md.

## Last update

2026-09-23
