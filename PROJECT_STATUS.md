# Project Status

## Current task

Task 003 — strict official CellViT++ classifier retraining — PENDING

## Last completed task

Task 002 — CellViT++ diagnostic and targeted classifier-head optimization — COMPLETED (candidate not promoted)

## Repository status

Tasks 001 and 002 are complete. Task 003 has been created to retrain the seven-class classifier strictly through the official CellViT++ training stack and compare it directly with the Task 001 hybrid/custom-head training result.

Scientific source data and the pretrained CellViT-SAM-H-x40-AMP backbone must remain unchanged.

Task 003 remote outputs must be written under:

`/data/lf_data/result/task003_official`

## GitHub synchronization

- Repository: `leonardfei/GPT_CODEX`
- Branch: `main`
- Task 001 workflow/report synchronized.
- Task 002 workflow/report synchronized.
- Task 003 specification added at `tasks/task_003.md`.
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

The leakage-safe grouped folds from Task 001 will be retained only as split definitions.

## Pending tasks

- Task 003 — strict official CellViT++ retraining and Task 001 comparison.
- Do not start Task 004 until Task 003 is completed and reviewed by Web GPT.

## Latest workflow files

- `tasks/task_001.md`
- `reports/task_001_report.md`
- `tasks/task_002.md`
- `reports/task_002_report.md`
- `tasks/task_003.md`
- `PROJECT_STATUS.md`

## Next execution command

```text
Execute task_003.
```

Codex must pull `origin/main` before execution and follow `AGENTS.md` plus `tasks/task_003.md`.

## Last update

2026-09-22
