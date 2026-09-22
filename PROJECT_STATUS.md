# Project Status

## Current task

Task 003 — strict official CellViT++ classifier retraining — COMPLETED (candidate not promoted)

## Last completed task

Task 003 — strict official CellViT++ classifier retraining — COMPLETED (candidate not promoted)

## Repository status

Tasks 001–003 are complete. Task 003 retrained the seven-class classifier strictly through the official CellViT++ training stack and compared it directly with the Task 001 hybrid/custom-head training result.

Scientific source data and the pretrained CellViT-SAM-H-x40-AMP backbone must remain unchanged.

Task 003 remote outputs were written under:

`/data/lf_data/result/task003_official`

## GitHub synchronization

- Repository: `leonardfei/GPT_CODEX`
- Branch: `main`
- Task 001 workflow/report synchronized.
- Task 002 workflow/report synchronized.
- Task 003 specification added at `tasks/task_003.md`.
- Task 003 workflow/report and reproducible helper scripts synchronized after completion.
- No force-push should be used.
- Credentials and tokens must not be committed.

## Task 001 baseline

- Production model: `/data/lf_data/result/model_best.pth`
- Grouped 5-fold CV macro-F1: approximately 0.342.
- Independent-test macro-F1: 0.3440.
- Balanced accuracy: 0.3514.
- Macro-AUROC: 0.7400.
- Macro-AUPRC: 0.3441.

## Task 002 diagnostic result

- GT cells: 159,349.
- Detected cells: 252,899.
- Matched cells: 94,662.
- GT match rate: 0.594.
- Ambiguous GT assignments: 36,775.
- Frozen-embedding class silhouette: -0.0209.
- Nearest-neighbor class purity: 0.2492.
- Best Task 002 grouped-CV macro-F1: 0.3440 versus Task 001 baseline 0.3417.
- Lowest-three-class mean F1: 0.2054 versus baseline 0.1929.
- Task 002 candidate was not promoted.
- Task 001 production model remains unchanged.

## Rationale for Task 003

Task 001 used official CellViT++ token extraction, official LinearClassifier architecture, and native official inference/evaluation, but the classifier optimization/training loop itself was custom.

Task 003 will therefore test whether this custom training loop materially limited performance by retraining through the strict official workflow:

`train_cell_classifier_head.py → ExperimentCellVitClassifier → CellViTHeadTrainer → official checkpoint/evaluation`

## Task 003 result

- Strict official CellViT++ CLI was used for all five grouped folds and the final all-training run; no custom PyTorch training loop was used.
- Grouped 5-fold CV: macro-F1 `0.3324±0.0134`, balanced accuracy `0.3455±0.0195`, macro-AUROC `0.7314±0.0167`, macro-AUPRC `0.3413±0.0281`, lowest-three-class F1 `0.1748±0.0213`.
- Official native test: classifier-global F1 `0.3986`, AUROC `0.7439`, AP `0.3495`; CellViT detection F1 `0.3563`; TIA binary detection F1 `0.3472`.
- Final official candidate: `/data/lf_data/result/task003_official/model_official_best.pth`.
- Final configuration used the baseline official head (`hidden_dim=100`, `drop_rate=0`, AdamW `lr=0.001`, weight decay `0.0001`, batch size `2048`, cosine scheduler) for 7 fixed epochs, selected by the median official AUROC-selected epoch across folds.
- WandB was not used because the remote environment had no configured API key; official CLI fallback configs were used and documented.
- Promotion thresholds were not met: official grouped-CV macro-F1 and lowest-three-class F1 did not improve by `+0.03`. The production model remains `/data/lf_data/result/model_best.pth`.
- Full remote report, metrics, native JSON, source hashes, environment records, run manifest, and seven vector figures are under `/data/lf_data/result/task003_official`.

The leakage-safe grouped folds from Task 001 will be retained only as split definitions.

## Pending tasks

- Do not start Task 004 until Task 003 is reviewed by Web GPT.

## Latest workflow files

- `tasks/task_001.md`
- `reports/task_001_report.md`
- `tasks/task_002.md`
- `reports/task_002_report.md`
- `tasks/task_003.md`
- `reports/task_003_report.md`
- `scripts/python/task003_prepare.py`
- `scripts/python/task003_configure_winner.py`
- `scripts/python/task003_aggregate.py`
- `scripts/python/task003_finalize.py`
- `PROJECT_STATUS.md`

## Next action

Review Task 003 before starting any later task.

Codex must pull `origin/main` before any later execution and follow `AGENTS.md` plus the relevant task specification.

## Last update

2026-09-22
