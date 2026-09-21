# Project Status

## Current task

Task 002 — CellViT++ diagnostic and targeted classifier-head optimization — COMPLETED (candidate not promoted)

## Last completed task

Task 001 — CellViT++ classifier-head optimization

## Repository status

Task 001 and Task 002 have been completed and the workflow summaries/scripts are synchronized. Task 002 outputs are under `/data/lf_data/result/task002`. Scientific source data, workflow rules, and the pretrained CellViT-SAM-H-x40-AMP backbone were left unchanged.

## GitHub synchronization

- Remote: `git@github.com:leonardfei/GPT_CODEX.git`
- Remote name: `origin`
- Branch: `main`
- Task 001 and Task 002 workflow code/reports are synchronized.
- Before Task 002 execution, `origin/main` was fetched and fast-forwarded locally.
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

## Task 002 result

- GT cells: 159,349; detected cells: 252,899; matched cells: 94,662; GT match rate: 0.594.
- Matching showed 36,775 ambiguous GT assignments, with no duplicate assignments or out-of-bounds coordinates.
- Frozen-embedding geometry was weak: class silhouette -0.0209 and nearest-neighbor class purity 0.2492; UMAP was unavailable and a deterministic t-SNE fallback was recorded.
- Best grouped-CV candidate: macro-F1 0.3440 versus baseline 0.3417; lowest-three-class mean F1 0.2054 versus 0.1929.
- Promotion was rejected because the improvements (+0.0024 macro-F1 and +0.0125 lowest-three-class F1) did not meet the predefined +0.03 thresholds. Fold variance and strong-class loss checks were acceptable.
- The independent test set was not used for Task 002 selection, and `/data/lf_data/result/model_best.pth` was preserved with SHA256 `f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`.
- Detailed report: `reports/task_002_report.md`; remote artifacts: `/data/lf_data/result/task002`.

## Outstanding QC issues

Task 002 identified mixed causes dominated by incomplete/ambiguous detection-to-ground-truth matching and weak frozen-embedding separability, with class imbalance contributing.

## Open scientific questions

1. Are the weak-class errors primarily due to label/registration or matching noise?
2. Do frozen SAM-H CellViT embeddings adequately separate the seven classes?
3. Is batch signal stronger than cell-type signal in the frozen embedding space?
4. Can targeted loss/sampling/head optimization materially improve grouped-CV macro-F1 and weak-class performance?

## Pending tasks

- Review Task 002 findings before starting Task 003.
- Do not start Task 003 until Task 002 is completed and reviewed by Web GPT.

## Latest workflow files

- `tasks/task_001.md`
- `reports/task_001_report.md`
- `tasks/task_002.md`
- `scripts/python/task002_diagnostics.py`
- `scripts/python/task002_optimize.py`
- `scripts/python/task002_finalize.py`
- `reports/task_002_report.md`
- `PROJECT_STATUS.md`

## Next execution command

```text
Review Task 002, then decide whether to define Task 003.
```

Codex should pull `origin/main` before execution and follow `AGENTS.md` and `tasks/task_002.md`.

## Last update

2026-09-21
