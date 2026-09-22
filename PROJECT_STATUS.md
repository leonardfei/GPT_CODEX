# Project Status

## Current task

Task 004 — high-confidence CellViT–Xenium label reconstruction and retraining — PARTIAL

## Last completed task

Task 003 — strict official CellViT++ classifier retraining — COMPLETED (candidate not promoted)

## Repository status

Tasks 001–003 are complete. Task 004 primary three-tier retraining is complete, but the requested candidate-specific CV for every secondary threshold-sensitivity rule was not run; therefore Task 004 is marked PARTIAL.

Scientific source data and the pretrained CellViT-SAM-H-x40-AMP backbone must remain unchanged.

Task 004 remote outputs must be written under:

`/data/lf_data/result/task004_high_confidence`

Task 004 primary outcome: HIGH_CONFIDENCE improved macro-F1 by only +0.0077 versus ALL_MATCHED while lowering lowest-three-class F1 by -0.0077 and reducing Neutrophil F1 by more than 0.03; ULTRA_HIGH_CONFIDENCE did not improve. No tier met promotion criteria, no test evaluation was run, and production was unchanged.

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

## Task 004 result summary

- Matching: official CellViT++ Hungarian/Munkres global one-to-one assignment with a post-assignment 15 px radius filter; available geometry was centroid-only.
- Retained cells: ALL_MATCHED 94,662; HIGH_CONFIDENCE 37,029; ULTRA_HIGH_CONFIDENCE 25,066.
- Grouped-CV macro-F1: 0.3324±0.0134, 0.3401±0.0164, and 0.3226±0.0199 respectively.
- Lowest-three-class F1: 0.1748±0.0213, 0.1671±0.0169, and 0.1565±0.0298 respectively.
- Promotion: not promoted; production remains `/data/lf_data/result/model_best.pth` with SHA256 `f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`.
- Remote report and artifacts: `/data/lf_data/result/task004_high_confidence/TASK004_REPORT.md`.
- Limitation: threshold-sensitivity retention/ambiguity counts were generated, but candidate-specific official CV was not run; the complete unmatched detection pool was also not archived.

## Pending tasks

- Review whether to complete the secondary sensitivity-CV matrix and archive the full detection candidate graph before any further model selection.
- Do not start Task 005 until Task 004 is completed and reviewed by Web GPT.

## Latest workflow files

- `tasks/task_001.md`
- `reports/task_001_report.md`
- `tasks/task_002.md`
- `reports/task_002_report.md`
- `tasks/task_003.md`
- `reports/task_003_report.md`
- `tasks/task_004.md`
- `reports/task_004_report.md`
- `scripts/python/task004_prepare.py`
- `scripts/python/task004_configs.py`
- `scripts/python/task004_aggregate.py`
- `scripts/python/task004_figures.py`
- `scripts/python/task004_report.py`
- `PROJECT_STATUS.md`

## Next execution command

```text
Review Task 004 partial result before deciding whether to complete secondary sensitivity CV.
```

Codex must pull `origin/main` before execution and follow `AGENTS.md` plus `tasks/task_004.md`.

## Last update

2026-09-22
