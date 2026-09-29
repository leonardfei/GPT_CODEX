# Project Status

## Current task

Task 012 — Final Model Export and Publication Figure Package — IN_PROGRESS

Execution started 2026-09-29. Canonical inputs and exact Task009 CORE folds passed input QC. Corrected Task011 fold refits and checkpoint export are running on the analysis server under `/data/lf_data/result/final_model`. The final checkpoint, OOF results, figure package, and report remain pending until training and QC complete.

## Selected model recipe

Current best development candidate:
- Midnight-12k
- target-centered FOV12
- 57 × 57 native px at 0.2125 μm/px
- CLS + mean patch-token feature, 3072 dim
- final transformer block fine-tuning
- linear 3072 → 7 head
- 3 epochs
- backbone LR 1e-5
- head LR 1e-3
- AdamW, weight decay 1e-4
- class-balanced weighted cross entropy

Corrected Task011 grouped-CV performance:
- Macro-F1 0.4522 ± 0.0294
- Macro-AUPRC 0.4893 ± 0.0428
- Neutrophil F1 0.2352 ± 0.1214
- Neutrophil one-vs-rest AUPRC 0.1976 ± 0.1181
- true N-vs-Myeloid AUROC/AUPRC 0.7202 ± 0.0221 / 0.4385 ± 0.1787
- true N-vs-T/B AUROC/AUPRC 0.7069 ± 0.0359 / 0.3930 ± 0.2118

MUSK frozen benchmark did not replace Midnight.

## Task012 objectives

1. Reproduce and save the five fold-specific Midnight final-block checkpoints.
2. Save corrected OOF probabilities/predictions for publication QC.
3. Train one full-data 96,044-cell final checkpoint for deployment.
4. Save model/preprocessing/class metadata and SHA256.
5. Export single-model and ensemble inference scripts.
6. Generate publication-grade editable vector PDF figures directly from corrected server metrics.
7. Keep the existing production model unchanged.

Important:
The full-data final checkpoint must not be evaluated on its own training cells as a performance estimate. Report the corrected Task011 grouped-CV performance until independent validation is available.

## Output root

`/data/lf_data/result/final_model`

Primary final checkpoint:
`/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`

Task file:
`tasks/task_012.md`

## Production guardrail

Current production remains:
`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Do not overwrite it.

## Next execution command

```text
Execute task_012.
```

Codex must pull origin/main and follow AGENTS.md plus `tasks/task_012.md`.

## Last update

2026-09-29
