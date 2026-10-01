# Project Status

## Current task

Task 015 — GHIST Classification-Only Retraining for HCC H&E Cell Typing — PENDING / READY TO RUN

## Goal

Train an HCC-specific GHIST cell-type model using H&E morphology and the existing frozen seven-class Xenium-derived labels, while excluding gene-expression prediction entirely.

Primary model:

`H&E 256×256 patch + CellViT nucleus instance masks -> GHIST UNet3+ morphology backbone -> official GHIST per-nucleus feature pooling -> 256-d nucleus embedding -> 7-class cell-type MLP`

Primary losses:
- GHIST morphology/pixel classification loss;
- GHIST direct cell-type classification loss.

No gene-expression heads or gene-expression losses.

## Official GHIST reference

Use the official:
`SydneyBioX/GHIST`

Reuse/adapt:
- UNet3+ Backbone;
- Embed;
- MLP;
- official nucleus-level + patch-level feature pooling.

Do not modify the upstream clone in place.

## Training data

H&E:
`/data/lf_data/xenium_data/ID0060276.ome.tif`

Canonical cell cohort:
`/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz`

Expected n:
`96,044`

Frozen classes:
- Endothelial
- Mesenchymal
- Myeloid
- Neutrophil
- Plasma cell
- T and B
- Tumor

Use CellViT binary contours as the external nucleus instance masks required by GHIST.

## Validation

Reuse exact Task009 V3_CORE five outer folds.

No random cell-level split.

Checkpoint selection is training-only inner grouped validation.

Validation predictions must be deduplicated across overlapping GHIST patches by retaining the prediction from the patch containing the largest nucleus area.

## Primary variants

1. `GHIST_CT_OFFICIAL`
   - L_Morph + L_CT
   - unweighted classification losses

2. `GHIST_CT_BALANCED`
   - same architecture
   - training-fold-only class-balanced weighting for rare classes

Optional GHIST neighborhood-composition auxiliary loss is allowed only after primary cell-typing benchmarks and still without gene expression.

## Official-like starting hyperparameters

- patch size 256×256
- training overlap 0
- validation overlap 30 px
- batch size 8
- max epochs 50
- AdamW
- LR 1e-3
- betas 0.9 / 0.999
- weight decay 1e-4
- horizontal/vertical flips
- 90/180/270° rotations
- train-fold-only RGB standardization

## Main comparison

Corrected Midnight final-block reference:
- Macro-F1 0.4522 ± 0.0294
- Macro-AUPRC 0.4893 ± 0.0428
- Neutrophil F1 0.2352 ± 0.1214
- Neutrophil AUPRC 0.1976 ± 0.1181
- N-vs-Myeloid AUROC/AUPRC 0.7202 / 0.4385
- N-vs-T/B AUROC/AUPRC 0.7069 / 0.3930

GHIST is not promoted unless it improves consistently under the predefined Task015 gate.

## Output root

`/data/lf_data/result/task015_ghist_celltyping`

## Task file

`tasks/task_015.md`

## Next execution command

```text
Execute task_015.
```

Codex must pull origin/main and follow AGENTS.md plus `tasks/task_015.md`.

## Production guardrail

Midnight final model, production model, frozen biological labels and P84/P169 outputs remain unchanged.

## Last update

2026-10-01
