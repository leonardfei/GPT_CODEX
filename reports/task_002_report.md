# Task 002 Report — CellViT++ diagnostic and targeted optimization

## Outcome

Task 002 completed on the remote server. The backbone remained frozen, source images/labels and workflow rules were not modified, and the candidate was not promoted. The independent test set was not used for Task 002 selection.

## Bottleneck diagnosis

- Ground-truth cells: 159,349; detected cells: 252,899; matched cells: 94,662.
- Ground-truth match rate: 0.594; median matching distance: 7.23 px at a 15 px threshold.
- Ambiguous GT assignments: 36,775; duplicate assignments: 0; out-of-bounds assignments: 0.
- Frozen-embedding class silhouette: -0.0209; batch silhouette: -0.0066.
- Nearest-neighbor class purity: 0.2492; batch purity: 0.4530.

These results support mixed causes, dominated by incomplete/ambiguous detection-to-ground-truth matching and weak frozen-embedding separability, with class imbalance contributing. UMAP was unavailable in the remote environment; the deterministic t-SNE fallback is recorded in the remote figure-data directory.

## Grouped-CV optimization

| Metric | Task 001 baseline | Best Task 002 candidate |
|---|---:|---:|
| Macro-F1 | 0.3417 ± 0.0218 | 0.3440 ± 0.0293 |
| Balanced accuracy | 0.3552 | 0.3616 |
| Macro-AUPRC | 0.3425 | 0.3431 |
| Lowest-three-class mean F1 | 0.1929 | 0.2054 |

The best candidate used AdamW, a 384-unit head, 0.4 dropout, label smoothing 0.02, and class-aware sampling. It failed the required minimum improvements of +0.03 macro-F1 and +0.03 lowest-three-class F1, so no final candidate test evaluation or model promotion was performed.

## Production model and outputs

Task 001 remains the production model at `/data/lf_data/result/model_best.pth`. Its protected baseline copy has the same SHA256: `f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`.

Detailed remote artifacts and the complete report are in `/data/lf_data/result/task002`. The tracked workflow implementation is in `scripts/python/task002_diagnostics.py`, `scripts/python/task002_optimize.py`, and `scripts/python/task002_finalize.py`.

## Recommendation

Review detection-to-Xenium registration and matching upstream before another head-only tuning round, then validate any future change on an untouched external cohort.
