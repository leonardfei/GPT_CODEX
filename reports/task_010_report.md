# Task 010 — Public pathology foundation model representation benchmark

Status: **BLOCKED** — official public model downloads were unreachable; no benchmark interpretation was performed.

## Executive result

The revised Task010 correctly moved from the prior MUSK-only access block to the public Phikon-v2 and Midnight-12k comparison. Both official model repositories were identified and their documented extraction rules were reviewed, but the execution environment could not connect to `huggingface.co` to download either checkpoint. No unofficial mirror, random initialization, or substitute encoder was used. The task stopped before shared-cohort feature extraction and classification.

The previous MUSK access-block artefacts remain under `/data/lf_data/result/task010_representation_benchmark/config/` and are retained as provenance; they are not the reason for this revised block.

## A–B. Public model provenance and access

### Phikon-v2

- official model: `owkin/phikon-v2`
- official source: [Owkin Phikon-v2](https://huggingface.co/owkin/phikon-v2)
- architecture documented by the model card: ViT-L/16 via DINOv2
- official feature rule: CLS token from `last_hidden_state[:, 0, :]`
- expected feature dimension: 1024; no runtime dimension was obtained
- license: Owkin non-commercial licence; no downstream licensing assumption was made
- revision, local checkpoint, and SHA256: unavailable because download did not complete

### Midnight-12k

- official model: `kaiko-ai/midnight`
- official source: [Kaiko Midnight](https://huggingface.co/kaiko-ai/midnight) and [official repository](https://github.com/kaiko-ai/Midnight)
- model variant: Midnight-12k only; Midnight-92k and Midnight-92k/392 were not used
- official preprocessing: 224×224, mean `(0.5, 0.5, 0.5)`, standard deviation `(0.5, 0.5, 0.5)`
- official classification feature: concatenated CLS token and mean patch-token embedding
- expected feature dimension: 3072 from the documented 1536-wide representation; no runtime dimension was obtained
- license: MIT
- revision, local checkpoint, and SHA256: unavailable because download did not complete

Access attempts were made against the official Hugging Face host only. The remote server timed out on both official model URLs after 20 seconds. A local diagnostic request to the same official host also timed out after 30 seconds. The remote environment had no model cache and did not contain either encoder. `transformers` was not installed in the existing CellViT environment, but dependency installation was not started because the required public checkpoints were unreachable. Exact gate records are in `qc/public_model_provenance.md` and the two JSON files under `config/`.

## C. H&E scale and crop geometry audit

The original H&E exists and was metadata-audited without reading the full pixel array. It is a 50,000 × 23,451 RGB uint8 OME-TIFF. Its OME metadata reports `PhysicalSizeX = PhysicalSizeY = 352.77777777777777 µm`, which would imply an implausible scale of approximately 0.00706 µm/px and is inconsistent with the validated project registration workflow.

The historical validated notebook records `PIXEL_SIZE = 0.2125` µm/px. Task010 therefore preserves that validated scale for the planned geometry audit. The intended physical fields are 16 × 16 µm (SMALL) and 56 × 56 µm (CONTEXT), corresponding to approximately 75.29 and 263.53 native pixels per side before encoder-specific resizing. No crop was generated while the model access gate was blocked. See `qc/pixel_scale_audit.md` and `config/crop_geometry.json`.

## D–E. Frozen inputs and fold audit

The required frozen inputs exist and their schemas were inspected:

- Task007 annotation: `/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`
- Task008 eligibility: `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`
- original H&E: `/data/lf_data/xenium_data/ID0060276.ome.tif`
- registration matrix: `/data/lf_data/xenium_data/matrix.csv`
- Task009 V3_CORE dataset: `/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE`
- Task009 split manifest: `/data/lf_data/result/task009_v3_retraining/metrics/split_manifest.csv`

The exact Task009 V3_CORE batch folds were recovered and would have been reused:

| fold | train batches | validation batches |
|---:|---|---|
| 0 | s01A, s01B, s04B, s11, s22, s93 | s02A, s06A |
| 1 | s01A, s01B, s02A, s04B, s06A, s11, s93 | s22 |
| 2 | s01B, s02A, s04B, s06A, s22, s93 | s01A, s11 |
| 3 | s01A, s02A, s04B, s06A, s11, s22, s93 | s01B |
| 4 | s01A, s01B, s02A, s06A, s11, s22 | s04B, s93 |

No historical `CellViT_dataset` file was used. Frozen labels, raw H&E, registration, Task009 data, and production checkpoint were not modified.

## F–Q. Benchmark endpoints not estimable

Because neither public encoder could be loaded, `SHARED_DETECTED_CORE` was not constructed and no cells entered the revised benchmark. Consequently there are no class/batch counts, crop manifests, feature matrices, linear-probe or MLP results, Neutrophil metrics, confusion flows, binary diagnostics, context deltas, cross-encoder agreement, geometry metrics, GT-centered upper-bound results, paired fold deltas, or best-representation decision.

The following questions therefore remain unanswered by this run: whether Phikon-v2 or Midnight-12k improves over CellViT_TOKEN; whether CONTEXT improves over SMALL; whether either encoder reduces Neutrophil→Myeloid or Neutrophil→T/B confusion; whether gains replicate across encoders; and which Task011 branch is justified. No biological or model-ranking conclusion is made.

## Safety and next step

Production `/data/lf_data/result/model_best.pth` was not changed. Task007/Task008 ground truth was not changed. No foundation-model checkpoint, embedding, crop, credential, or token was committed to Git.

Next step: restore outbound access to the official Hugging Face repositories or stage the exact official Phikon-v2 and Midnight-12k files under `/data/lf_data` with their revisions and SHA256 values, then rerun Task010 from model provenance. Do not use a mirror, restricted Midnight variant, or another encoder as a substitute.

## Remote artefacts

Blocked-run audit files are under `/data/lf_data/result/task010_representation_benchmark/`, including `TASK010_REPORT.md`, `qc/public_model_provenance.md`, `qc/pixel_scale_audit.md`, the model provenance JSON files, `config/foundation_model_environment.txt`, and `config/crop_geometry.json`. No benchmark metrics or figures were generated.
