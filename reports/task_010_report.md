# Task 010 — Frozen representation benchmark

Status: **BLOCKED_MUSK_ACCESS**

## Executive result

Task010 stopped at the required official-MUSK access gate. The server has no existing official MUSK installation, checkpoint, or cache, and the official Hugging Face model requires gated-term acceptance plus a Hugging Face write token. Those credentials and manual authorization were not available. No substitute encoder was used and no representation benchmark was started.

## A. MUSK availability and provenance

The official code source is [lilab-stanford/MUSK](https://github.com/lilab-stanford/MUSK). Its documented model reference is `hf_hub:xiangjx/musk` with model ID `xiangjx/musk`. The official README requires accepting the Hugging Face model terms and logging in with a Hugging Face write token before model access.

Server checks under `/data/lf_data` found no MUSK installation, checkpoint, or cache. `musk` and `MUSK` were not importable; `transformers` and `open_clip` were also unavailable in the existing CellViT environment. A no-credential Hugging Face metadata request could not complete because the server could not establish the external connection; in any event, the official README's gated-access requirement is sufficient to trigger the task-defined blocker.

No checkpoint was downloaded or loaded:

- checkpoint path: none
- checkpoint SHA256: none
- model verification: not performed because access was blocked
- blocker: Hugging Face gated terms plus write-token login and manual authorization

The CANVAS repository also states that MUSK must be installed first: [lilab-stanford/CANVAS](https://github.com/lilab-stanford/CANVAS).

## B. Benchmark scope not executed

Because the access gate blocked the task before benchmarking, the following were intentionally not generated:

- `SHARED_DETECTED_CORE` manifest and class/batch counts;
- MUSK_SMALL and MUSK_CONTEXT crops, embeddings, and crop montages;
- CellViT token re-extraction and representation comparison;
- Task009-equivalent five-fold linear or MLP probes;
- binary Neutrophil-vs-Myeloid and Neutrophil-vs-T/B diagnostics;
- MUSK-only GT-centered upper-bound analysis;
- representation geometry, UMAP, figures, and benchmark metrics.

Therefore, macro-F1/AUPRC, Neutrophil metrics, confusion flows, binary metrics, paired fold deltas, and upper-bound results are **not estimable from this blocked run**. No performance claim or next-model promotion decision is made.

## C. Frozen inputs and safety checks

The task was started only after `git pull --ff-only origin main`. The intended frozen inputs were not modified:

- Task007 annotation: `/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`
- Task008 eligibility: `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`
- original H&E: `/data/lf_data/xenium_data/ID0060276.ome.tif`
- registration: `/data/lf_data/xenium_data/matrix.csv`
- Task009 regenerated CORE dataset: `/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE`
- historical `/data/lf_data/xenium_data/CellViT_dataset`: not accessed as an input

Production `/data/lf_data/result/model_best.pth` was not modified. Frozen Task007/Task008 ground truth was not modified.

## D. Environment observed at the gate

The existing remote environment was `cellvit_env`; no separate MUSK environment was created. Observed versions were recorded in `config/task010_musk_environment.txt`: Python 3.10.14, PyTorch 2.7.1+cu128, timm 1.0.8, huggingface_hub 0.22.2, Pillow 10.3.0, NumPy 1.23.5, pandas 1.4.3, scikit-learn 1.3.0, CUDA available. Installing or changing dependencies was avoided because the official weights were inaccessible.

## E. Physical scale and crop geometry

No crops were generated because Task010 is blocked before the benchmark. The requested geometry remains pending: 0.25 µm/pixel equivalent, MUSK_SMALL approximately 16 × 16 µm, MUSK_CONTEXT approximately 56 × 56 µm, with native-scale conversion and official 384 × 384 MUSK resizing to be audited only after access is authorized. No physical-scale claim is made from this blocked run.

## F. Required interpretation and next step

No representation ranking can be inferred. The Task009 finding that CellViT detection/representation is a bottleneck remains the last completed evidence, but Task010 cannot distinguish CellViT-token limitation from MUSK morphology/context signal without the official encoder.

Recommended next action: obtain authorized access to the official `xiangjx/musk` model by completing the Hugging Face terms and providing an approved token through the secure server environment. Then rerun Task010 from the access gate, preserving V3_CORE labels, Task009 batch folds, and the shared-cell fairness design. Do not substitute another encoder and do not update production.

## Remote artefacts

The blocked-run gate artefacts are stored under:

`/data/lf_data/result/task010_representation_benchmark/`

including `TASK010_REPORT.md`, `config/musk_model_provenance.md`, `config/musk_environment.txt`, `config/musk_checkpoint_sha256.json`, and `config/crop_geometry.json`. No embeddings, checkpoints, crops, or large datasets were created.
