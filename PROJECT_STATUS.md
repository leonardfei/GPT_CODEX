# Project Status

## Current task

Task 010 — Public pathology foundation model representation benchmark: CellViT token vs Phikon-v2 vs Midnight-12k — BLOCKED (official public model downloads unreachable)

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

## Task010 revision after MUSK block

The first Task010 attempt correctly stopped at BLOCKED_MUSK_ACCESS because the official MUSK checkpoint required gated Hugging Face authorization.

The MUSK access block is retained as provenance but no longer blocks the project.

Task010 has been revised to use two publicly accessible pathology foundation models:

### Primary encoder
Phikon-v2
Official model ID:
`owkin/phikon-v2`

### Independent confirmation encoder
Midnight-12k
Official model ID:
`kaiko-ai/midnight`

Use only the public Midnight-12k weights, not restricted Midnight-92k variants.

## Task010 primary representations

1. CellViT_TOKEN
2. PHIKON_V2_SMALL
3. PHIKON_V2_CONTEXT
4. MIDNIGHT12K_SMALL
5. MIDNIGHT12K_CONTEXT

All five must use exactly the same:
- V3_CORE labels;
- SHARED_DETECTED_CORE cell IDs;
- matched H&E nucleus centers;
- Task009 V3_CORE held-out batch folds;
- linear-probe procedure;
- secondary MLP-probe procedure.

No foundation-model fine-tuning is allowed in the primary Task010 comparison.

The revised execution audited the frozen inputs and physical H&E scale, but the remote server and local diagnostic environment both timed out when connecting to the official Hugging Face host for Phikon-v2 and Midnight-12k. No official checkpoint was loaded, no substitute encoder was used, and no benchmark metrics were generated. Production and frozen ground truth remain unchanged.

## Frozen ground truth

Task007 annotation:
`/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`

Task008 Neutrophil eligibility:
`/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`

Primary label set:
`V3_CORE`

Do not change annotation/QC rules based on Task010 results.

## Data sources

Original H&E:
`/data/lf_data/xenium_data/ID0060276.ome.tif`

Registration:
`/data/lf_data/xenium_data/matrix.csv`

Task009 regenerated CORE dataset:
`/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE`

Forbidden historical dataset:
`/data/lf_data/xenium_data/CellViT_dataset`

The forbidden historical dataset may not be used as an image, label, split, metadata, or embedding source.

## Physical crop conditions

SMALL:
approximately 16 × 16 μm field of view.

CONTEXT:
approximately 56 × 56 μm field of view.

The native H&E crop size must be calculated from the validated physical pixel scale before encoder-specific resizing.

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

Context-value comparisons:
- PHIKON_V2_CONTEXT vs PHIKON_V2_SMALL
- MIDNIGHT12K_CONTEXT vs MIDNIGHT12K_SMALL

Cross-encoder confirmation:
- determine whether Phikon-v2 and Midnight-12k independently support the same conclusion.

Secondary morphology upper bound:
- Phikon-v2 and Midnight-12k on ALL_V3_CORE_GT_CENTERED cells, including cells not detected by CellViT.

## Interpretation

If both public encoders show CONTEXT > SMALL > CellViT:
- proceed to multiscale pathology-FM fusion and neighborhood auxiliary modeling.

If SMALL > CellViT but context adds little:
- proceed to local morphology specialist / limited encoder fine-tuning.

If immune binary tasks improve but seven-class remains weak:
- proceed to hierarchical immune classification.

If only one encoder improves:
- verify encoder-specific preprocessing and representation effects first.

If neither improves:
- reassess H&E separability ceiling and consider limited fine-tuning rather than immediately building a complex model.

## Current production model

`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Production remains unchanged.

## Task010 output root

`/data/lf_data/result/task010_representation_benchmark`

## Pending tasks

- Restore access to the official public Phikon-v2 and Midnight-12k weights, then rerun revised Task010 from model provenance.
- Preserve the previous MUSK access-block artefacts as provenance.
- Do not build Task011 until Task010 identifies the useful encoder/scale.
- Do not modify production.
- Do not modify frozen ground truth.

## Latest workflow files

- `tasks/task_009.md`
- `reports/task_009_report.md`
- `tasks/task_010.md`
- `reports/task_010_report.md`
- `config/phikon_v2_provenance.json`
- `config/midnight12k_provenance.json`
- `config/foundation_model_environment.txt`
- `qc/public_model_provenance.md`
- `qc/pixel_scale_audit.md`
- `PROJECT_STATUS.md`

## Task 010 status artefacts

Blocked-run artefacts are recorded under `/data/lf_data/result/task010_representation_benchmark/`. No embeddings, crops, model checkpoints, benchmark metrics, or figures were generated.

After official public weights are made reachable, Codex must pull `origin/main` before rerunning `Execute task_010.` and follow `AGENTS.md` plus the revised `tasks/task_010.md`.

## Last update

2026-09-23
