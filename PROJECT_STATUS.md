# Project Status

## Current task

Task 004 — high-confidence CellViT–Xenium label reconstruction and retraining — PENDING

## Last completed task

Task 003 — strict official CellViT++ classifier retraining — COMPLETED (candidate not promoted)

## Repository status

Tasks 001–003 are complete. Task 004 has been added to test whether stricter CellViT↔Xenium one-to-one matching and higher-confidence training labels can improve seven-class CellViT++ classification performance.

Scientific source data and the pretrained CellViT-SAM-H-x40-AMP backbone must remain unchanged.

Task 004 remote outputs must be written under:

`/data/lf_data/result/task004_high_confidence`

## GitHub synchronization

- Repository: `leonardfei/GPT_CODEX`
- Branch: `main`
- Tasks 001–003 workflow reports are synchronized.
- Task 004 specification added at `tasks/task_004.md`.
- No force-push should be used.
- Credentials and tokens must not be committed.

## Current production model

`/data/lf_data/result/model_best.pth`

Task 001 remains the production model.

## Evidence motivating Task 004

### Task 001
- grouped-CV macro-F1: approximately 0.342
- independent-test macro-F1: 0.3440

### Task 002
- GT cells: 159,349
- detected cells: 252,899
- matched cells: 94,662
- GT match rate: 0.594
- ambiguous GT assignments: 36,775
- frozen-embedding class silhouette: -0.0209
- nearest-neighbor class purity: 0.2492
- nearest-neighbor batch purity: 0.4530

### Task 003
- strict official grouped-CV macro-F1: 0.3324 ± 0.0134
- strict official pipeline did not outperform Task 001
- official candidate was not promoted

These results suggest the training-loop implementation is unlikely to be the dominant limitation. Upstream detection-to-Xenium matching quality, label ambiguity, class imbalance, and weak frozen representation are now the main suspected bottlenecks.

## Task 004 primary experiment

Compare three training-label tiers while keeping the backbone, seven classes, grouped folds and official classifier training recipe fixed:

1. ALL_MATCHED
2. HIGH_CONFIDENCE
3. ULTRA_HIGH_CONFIDENCE

Task 004 must test whether stricter one-to-one geometric matching improves:
- frozen-embedding class separability;
- grouped-CV macro-F1;
- lowest-three-class F1;
- Myeloid, Neutrophil and Plasma-cell performance.

## Pending tasks

- Task 004 — reconstruct high-confidence CellViT↔Xenium labels and retrain with the official CellViT++ trainer.
- Do not start Task 005 until Task 004 is completed and reviewed by Web GPT.

## Latest workflow files

- `tasks/task_001.md`
- `reports/task_001_report.md`
- `tasks/task_002.md`
- `reports/task_002_report.md`
- `tasks/task_003.md`
- `reports/task_003_report.md`
- `tasks/task_004.md`
- `PROJECT_STATUS.md`

## Next execution command

```text
Execute task_004.
```

Codex must pull `origin/main` before execution and follow `AGENTS.md` plus `tasks/task_004.md`.

## Last update

2026-09-22
