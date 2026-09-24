# Project Status

## Current task

Task 011 correction — fine-tuning promotion gate and true binary probes — PENDING / READY TO RUN

## Background

Task011 primary optimization stages completed on the frozen 96,044-cell canonical cohort with exact Task009 V3_CORE held-out batch folds.

Observed development results before correction:
- BEST_FOV: Midnight FOV12, macro-F1 0.4204 ± 0.0330
- raw Midnight+CellViT fusion: 0.4344, but below the predeclared +0.015 fusion threshold
- hierarchy and specialist rescue did not improve performance
- TTA4/TTA8 produced only small sub-threshold gains
- final-block Midnight fine-tuning produced the highest observed macro-F1: 0.4524 ± 0.0335
- observed Neutrophil F1: 0.2323
- observed Neutrophil one-vs-rest AUPRC: 0.1959
- macro-F1 improved in 5/5 outer folds
- batch kNN purity delta: +0.0455
- spatial kNN purity delta: -0.0011

## Methodological correction required

Audit identified three issues in the final Task011 fine-tuning interpretation:

1. The strict promotion gate incorrectly used N-vs-Myeloid AUPRC instead of Neutrophil one-vs-rest AUPRC.
2. The fine-tuned "true binary" endpoints were derived from seven-class probability ratios rather than independently trained binary probes.
3. The final binary AUROC/AUPRC summary used the first/fold-0 row instead of five-fold means.

Therefore:
- the statement that F1_FINAL_BLOCK "passed the Task011 promotion gate" is superseded pending correction;
- the seven-class fine-tuning gain remains a valid observed signal but will be rerun/re-audited;
- Task012 must not begin until Task011 correction is complete.

## Correct promotion criterion

Relative to frozen FOV12:

- macro-F1 gain >= 0.03
AND
- Neutrophil F1 gain >= 0.03 OR Neutrophil one-vs-rest AUPRC gain >= 0.03
AND
- macro-F1 improves in >=4/5 outer folds
AND
- batch purity increase <=0.05
AND
- spatial purity increase <=0.05

Do not substitute pairwise N-vs-Myeloid AUPRC for Neutrophil one-vs-rest AUPRC.

## Correct true binary procedure

For each outer fold:
- use that fold's own fine-tuned Midnight encoder;
- extract relevant outer-training and outer-validation embeddings from the SAME encoder;
- fit StandardScaler only on outer training;
- train an independent class-balanced binary linear probe on outer training;
- evaluate once on outer validation.

Required:
- Neutrophil vs Myeloid
- Neutrophil vs T/B

Probability-ratio diagnostics remain separate and must not be called true binary probes.

## Correction scripts

- `scripts/python/task011_finetune_correction.py`
- `scripts/python/task011_finalize_correction.py`

Correction task:
- `tasks/task_011_correction.md`

## Canonical data

Task010 canonical order:
`/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz`

Task011 root:
`/data/lf_data/result/task011_midnight_local_optimization`

Midnight:
`/data/lf_data/models/midnight-12k`

Environment:
`/data/lf_data/task010_env/bin/python`

## Production guardrail

Current production model remains:
`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

No production model, Task007 label, Task008 eligibility, or Task009 output may be modified.

## Next execution command

```text
Execute task_011_correction.
```

Codex must pull origin/main and follow AGENTS.md plus `tasks/task_011_correction.md`.

## Last update

2026-09-24
