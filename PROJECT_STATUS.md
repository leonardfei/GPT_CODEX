# Project Status

## Current task

Task 002 — CellViT++ diagnostic and targeted classifier-head optimization — PENDING

## Last completed task

Task 001 — CellViT++ classifier-head optimization

## Repository status

Task 001 has been completed and synchronized. Task 002 has now been added to the repository for diagnostic bottleneck analysis and targeted second-round classifier-head optimization. Scientific source data and the pretrained CellViT-SAM-H-x40-AMP backbone must remain unchanged. Remote Task 002 outputs should be written under `/data/lf_data/result/task002`.

## GitHub synchronization

- Remote: `https://github.com/leonardfei/GPT_CODEX.git`
- Remote name: `origin`
- Branch: `main`
- Task 001 workflow and reports are synchronized.
- Task 002 specification has been added to `tasks/task_002.md`.
- No force-push should be used.
- Credentials and tokens must not be committed.

## Recent completed analyses

- Task 001: CellViT++ classifier-head optimization.
- Final Task 001 production checkpoint: `/data/lf_data/result/model_best.pth`.
- Grouped 5-fold CV macro-F1: approximately 0.342.
- Independent-test macro-F1: 0.3440.
- Balanced accuracy: 0.3514.
- Macro-AUROC: 0.7400.
- Macro-AUPRC: 0.3441.

## Important findings

- Train/test exact image-hash overlap was zero.
- Supplied folds mixed batches between train and validation; leakage-safe derived folds using `StratifiedGroupKFold` grouped by batch were used instead.
- Task 001 CV and test macro-F1 were nearly identical, suggesting that the main limitation is unlikely to be simple overfitting.
- Performance was heterogeneous across classes; Plasma cell, Myeloid and Neutrophil were the weakest classes, while Tumor was the strongest.
- Native CellViT++ classifier evaluation is based on detected cells paired to ground truth, so detector/matching quality and classifier quality must be interpreted separately.

## Outstanding QC issues

Task 002 must determine the dominant bottleneck among:

- annotation/registration noise;
- detection-to-ground-truth matching;
- batch/domain shift;
- weak frozen-embedding separability;
- classifier-head optimization;
- class imbalance;
- mixed causes.

## Open scientific questions

1. Are the weak-class errors primarily due to label/registration or matching noise?
2. Do frozen SAM-H CellViT embeddings adequately separate the seven classes?
3. Is batch signal stronger than cell-type signal in the frozen embedding space?
4. Can targeted loss/sampling/head optimization materially improve grouped-CV macro-F1 and weak-class performance?

## Pending tasks

- Task 002 — Diagnose the dominant CellViT++ bottleneck and perform targeted classifier-head optimization.
- Do not start Task 003 until Task 002 is completed and reviewed by Web GPT.

## Latest workflow files

- `tasks/task_001.md`
- `reports/task_001_report.md`
- `tasks/task_002.md`
- `PROJECT_STATUS.md`

## Next execution command

```text
Execute task_002.
```

Codex should pull `origin/main` before execution and follow `AGENTS.md` and `tasks/task_002.md`.

## Last update

2026-09-21
