# Project Status

## Current task

Task 010 — Phikon-first local pathology foundation model representation benchmark — PENDING / READY TO RUN

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

The current hypothesis is that the dominant limitation is representation/classification rather than label eligibility alone.

## Task010 history

The MUSK-based plan was blocked by gated model access.

The first public-encoder revision was then blocked because the server could not reach Hugging Face.

The user has now manually uploaded Phikon-v2 to the server. Task010 is therefore revised to a local-only Phikon-first execution.

Midnight-12k is still uploading and must NOT block Phase A.

## Task010 Phase A primary representations

1. CellViT_TOKEN
2. PHIKON_V2_SMALL
3. PHIKON_V2_CONTEXT

All three use:
- V3_CORE frozen labels;
- the same model-independent SHARED_DETECTED_CORE cell IDs;
- the same matched H&E nucleus center;
- the exact Task009 V3_CORE held-out batch folds;
- identical linear-probe and secondary MLP-probe procedures.

No encoder fine-tuning is allowed in Phase A.

## Local Phikon rule

Codex must search local storage first and must NOT attempt network download before local discovery.

Preferred local path:
`/data/lf_data/models/phikon-v2`

If not present there, search under `/data/lf_data` and the user model/cache directories.

Load fully offline with `local_files_only=True`.

If the local upload is incomplete, stop and report the exact missing files.

## Frozen ground truth

Task007 annotation:
`/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`

Task008 Neutrophil eligibility:
`/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`

Primary label set:
`V3_CORE`

Do not modify ground truth based on model results.

## Data sources

Original H&E:
`/data/lf_data/xenium_data/ID0060276.ome.tif`

Registration:
`/data/lf_data/xenium_data/matrix.csv`

Task009 regenerated CORE dataset:
`/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE`

Task009 split manifest:
`/data/lf_data/result/task009_v3_retraining/metrics/split_manifest.csv`

Forbidden historical dataset:
`/data/lf_data/xenium_data/CellViT_dataset`

No forbidden historical dataset file may be used.

## Physical crop conditions

Use validated project scale, approximately:
`0.2125 μm/px`

SMALL:
~16 μm FOV, preferred 75×75 native px.

CONTEXT:
~56 μm FOV, preferred 263×263 native px.

Both crops are centered on the matched H&E CellViT nucleus centroid, then processed by the official local Phikon preprocessing.

## Phase A endpoints

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
- N→Myeloid
- N→T/B

Binary diagnostics:
- Neutrophil vs Myeloid
- Neutrophil vs T/B

Context value:
- PHIKON_V2_CONTEXT − PHIKON_V2_SMALL

Representation geometry:
- class separation vs batch separation

Secondary:
- identical MLP probe
- Phikon GT-centered morphology upper bound

## Midnight Phase B

When Midnight-12k finishes uploading:
- verify it locally;
- reuse the frozen SHARED_DETECTED_CORE manifest;
- reuse exact SMALL/CONTEXT crops and folds;
- add Midnight SMALL/CONTEXT as independent confirmation.

Do not alter the shared cohort after Phase A.

## Current production model

`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Production remains unchanged.

## Task010 output root

`/data/lf_data/result/task010_representation_benchmark`

## Pending tasks

- Execute Task010 Phase A now using the locally uploaded Phikon-v2.
- Do not wait for Midnight-12k.
- Preserve earlier blocked-run artefacts as provenance.
- Do not start Task011 until Phase A has been reviewed.
- Do not modify production.
- Do not modify frozen ground truth.

## Latest workflow files

- `tasks/task_009.md`
- `reports/task_009_report.md`
- `tasks/task_010.md`
- `PROJECT_STATUS.md`

## Next execution command

```text
Execute task_010.
```

Codex must pull `origin/main` before execution and follow `AGENTS.md` plus the revised `tasks/task_010.md`.

## Last update

2026-09-23
