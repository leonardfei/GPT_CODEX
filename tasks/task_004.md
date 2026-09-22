# Task 004 — High-Confidence CellViT–Xenium Label Reconstruction and Retraining

## Status
PARTIAL

## Objective

Improve seven-class CellViT++ classification by rebuilding the training labels from higher-confidence CellViT↔Xenium cell matches rather than continuing classifier-head hyperparameter tuning.

Tasks 001–003 showed:

- Task 001 grouped-CV macro-F1: ~0.342
- Task 003 strict official grouped-CV macro-F1: ~0.332
- Task 002 GT cells: 159,349
- detected cells: 252,899
- matched cells: 94,662
- GT match rate: 0.594
- ambiguous GT assignments: 36,775
- frozen-embedding class silhouette: -0.0209
- nearest-neighbor class purity: 0.2492
- nearest-neighbor batch purity: 0.4530

These findings suggest that imperfect detection-to-Xenium matching and label noise may be major performance bottlenecks.

This task must test that hypothesis directly.

Do not fine-tune the CellViT-SAM-H backbone.

Do not redefine the seven biological classes.

---

## 1. Fixed environment and paths

Activate:

```bash
conda activate cellvit_env
```

CellViT++ repository:

```text
/data/lf_data/CellViT-plus-plus
```

Backbone:

```text
/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth
```

Source dataset:

```text
/data/lf_data/xenium_data/CellViT_dataset
```

Current production model:

```text
/data/lf_data/result/model_best.pth
```

Task 004 output root:

```text
/data/lf_data/result/task004_high_confidence
```

Do not modify Task 001–003 outputs.

---

## 2. Scientific question

Does classifier performance improve materially when the CellViT classifier is trained only on high-confidence CellViT↔Xenium matched cells?

The central comparison is:

```text
A. all currently matched cells
B. high-confidence matched cells
C. ultra-high-confidence matched cells
```

All three arms must use:

- the same CellViT-SAM-H backbone;
- the same seven classes;
- the same leakage-safe grouped 5-fold splits;
- the same official CellViT++ classifier architecture;
- the same fixed training recipe within the primary comparison.

The only intended difference is label/matching confidence.

---

## 3. Preserve all existing artifacts

Before starting:

1. calculate SHA256 of:
   `/data/lf_data/result/model_best.pth`

2. preserve:
   `/data/lf_data/result/task001_model_best_baseline.pth`

3. preserve Task 003 official candidate:
   `/data/lf_data/result/task003_official/model_official_best.pth`

4. never overwrite any existing model until the predefined promotion rule is met.

---

## 4. Reconstruct the current matching process

Determine exactly how current CellViT detections are paired to Xenium/GT annotations.

Document:

- coordinate system;
- patch coordinate origin;
- image pixel resolution;
- CellViT centroid definition;
- Xenium/GT centroid definition;
- current pairing algorithm;
- current matching threshold;
- whether matching is greedy, Hungarian, nearest-neighbor, or repository-native `pair_coordinates`;
- whether a matched GT cell can have multiple nearby CellViT candidates;
- whether a detected CellViT cell can have multiple nearby GT candidates.

The current official evaluation path appears to use a 15 px pairing radius. Verify this from the installed code and the actual preprocessing path.

Do not assume that the training-label generation used exactly the same rule until verified.

Create:

```text
task004_high_confidence/qc/current_matching_definition.md
```

---

## 5. Detect what geometric information is available

Inspect the existing Xenium-derived labels and preprocessing intermediates.

Determine whether the available information includes any of:

- Xenium nucleus polygons;
- Xenium cell polygons;
- Xenium nucleus centroids;
- Xenium cell centroids;
- CellViT nucleus polygons;
- CellViT cell/nucleus masks;
- CellViT centroids;
- affine/alignment transforms;
- patch-to-global coordinate mappings.

Record available geometry in:

```text
task004_high_confidence/qc/geometry_inventory.csv
```

Do not invent polygon-based criteria if polygons are not available.

---

## 6. Matching diagnostics

For each CellViT detection and nearby GT/Xenium cell candidate, where computationally feasible calculate:

- nearest GT distance d1;
- second-nearest GT distance d2;
- distance margin = d2 - d1;
- distance ratio = d1 / d2;
- reciprocal/mutual nearest-neighbor status;
- number of GT candidates within 15 px;
- number of CellViT detections within 15 px of the same GT cell;
- batch;
- class;
- patch/image;
- current match status.

If polygons are available, also calculate where meaningful:

- predicted centroid inside Xenium nucleus polygon;
- predicted centroid inside Xenium cell polygon;
- Xenium centroid inside CellViT nucleus polygon;
- polygon intersection;
- IoU or overlap fraction;
- centroid-to-polygon-boundary distance.

Create:

```text
task004_high_confidence/metrics/matching_pair_features.csv
```

This file may be large; keep it on the server only if necessary.

---

## 7. Define confidence tiers without using the test set

Use TRAINING data only to define and compare confidence rules.

### Tier A — ALL_MATCHED

Reproduce the current baseline matched-cell rule exactly.

This is the internal control.

### Tier B — HIGH_CONFIDENCE

Preferred generic rule if centroid-only geometry is available:

A match must satisfy all of:

1. one-to-one assignment;
2. mutual nearest-neighbor between GT and CellViT detection;
3. d1 <= 8 px;
4. second-nearest ambiguity margin >= 4 px OR d1/d2 <= 0.70;
5. no competing GT assignment within the same confidence boundary.

If one of these quantities cannot be computed, document the missing component and use the strongest available subset of the rule.

### Tier C — ULTRA_HIGH_CONFIDENCE

If centroid-only geometry:

1. one-to-one assignment;
2. mutual nearest-neighbor;
3. d1 <= 5 px;
4. second-nearest ambiguity margin >= 5 px OR d1/d2 <= 0.60;
5. no competing GT or CellViT candidate within 10 px.

If polygon information is available, strengthen Tier C by requiring at least one strong geometric consistency condition, preferably:

- CellViT centroid inside Xenium nucleus polygon; or
- Xenium centroid inside CellViT nucleus polygon; or
- meaningful nucleus polygon overlap.

Do not use a weak cell-polygon overlap alone if very large Xenium cell polygons would make it uninformative.

---

## 8. Sensitivity analysis of confidence thresholds

Because 5 px / 8 px thresholds are prespecified but still somewhat arbitrary, perform a TRAINING-only sensitivity analysis.

Evaluate candidate distance thresholds:

```text
5 px
6 px
8 px
10 px
12 px
15 px
```

and ambiguity conditions such as:

```text
mutual-NN only
margin >= 3 px
margin >= 5 px
distance ratio <= 0.6
distance ratio <= 0.7
distance ratio <= 0.8
```

Do not choose a rule based on test performance.

For every candidate rule report:

- retained matched cells;
- retained fraction;
- class-specific retained counts;
- class-specific retention rate;
- batch-specific retained counts;
- ambiguity rate;
- grouped-CV performance using the fixed training recipe defined below.

The predefined Tier B and Tier C remain the primary analysis. Sensitivity rules are secondary.

---

## 9. Protect against class depletion

High-confidence filtering may preferentially remove difficult classes.

For every tier, report counts for all seven classes.

No class may be silently dropped.

If any class has insufficient retained cells to support grouped 5-fold CV:

- do not oversample before reporting the true count;
- mark that tier as not fully evaluable;
- retain the class;
- consider a slightly relaxed confidence threshold in the sensitivity analysis.

Create:

```text
task004_high_confidence/metrics/class_retention_by_tier.csv
task004_high_confidence/metrics/batch_retention_by_tier.csv
```

---

## 10. Build derivative official-compatible datasets

Do not modify:

```text
/data/lf_data/xenium_data/CellViT_dataset
```

Create derivative datasets under:

```text
/data/lf_data/result/task004_high_confidence/work/
```

Suggested structure:

```text
work/
├── ALL_MATCHED/
├── HIGH_CONFIDENCE/
└── ULTRA_HIGH_CONFIDENCE/
```

Each must remain compatible with CellViT++ `DetectionDataset`.

Reuse source images via symlinks if appropriate.

Create new label CSVs containing only the retained matched cells for that confidence tier.

Preserve exact seven-class IDs and label map.

---

## 11. Grouped cross-validation

Use exactly the same leakage-safe grouped folds from Task 001/003:

- n_splits = 5;
- grouping = batch;
- random seed = 42;
- no train/validation batch overlap.

The patch/image allocation must remain identical across tiers wherever possible.

Only cell labels within each patch should change according to confidence filtering.

Do not create a different fold split per confidence tier unless technically unavoidable.

---

## 12. Primary training recipe

For the primary A/B/C comparison, use a fixed official CellViT++ training recipe for all tiers.

Use the strict official training stack:

```text
train_cell_classifier_head.py
→ ExperimentCellVitClassifier
→ CellViTHeadTrainer
→ official LinearClassifier
```

Use one fixed configuration across all confidence tiers.

Preferred fixed recipe:

```text
optimizer: AdamW
learning rate: 0.001
weight decay: 0.0001
hidden_dim: 100
drop_rate: 0
batch size: 2048
scheduler: cosine
```

Use the official Task 003 recipe unless the installed configuration requires a minor compatibility adjustment.

Do not tune separate hyperparameters for each confidence tier in the primary experiment.

This is essential to isolate the effect of label quality.

---

## 13. Epoch selection

Use one consistent strategy across all tiers.

Preferred:

- perform official early stopping per fold using AUROC/Validation;
- report best epoch;
- for final refit use the median best epoch across grouped folds.

Do not use test data.

---

## 14. Primary evaluation metrics

For each confidence tier calculate grouped 5-fold:

- macro-F1;
- balanced accuracy;
- macro-AUROC;
- macro-AUPRC;
- MCC;
- weighted F1;
- lowest-three-class mean F1;
- per-class precision;
- per-class recall;
- per-class F1;
- per-class AUROC;
- per-class AUPRC.

Report mean ± SD across folds.

---

## 15. Main statistical comparison

The primary comparison is paired across the same five grouped folds.

Compare:

```text
HIGH_CONFIDENCE vs ALL_MATCHED
ULTRA_HIGH_CONFIDENCE vs ALL_MATCHED
```

For each metric report the five paired fold differences.

Because n=5 folds is small, emphasize:

- absolute effect size;
- fold consistency;
- confidence intervals or bootstrap where defensible;

rather than relying on p-values alone.

Do not claim significance from pseudo-replicated cell-level tests.

---

## 16. Embedding separability after filtering

Using the same frozen CellViT-SAM-H embeddings, recalculate for each tier:

- class silhouette;
- nearest-neighbor class purity;
- batch silhouette;
- nearest-neighbor batch purity.

The key question is whether high-confidence filtering makes class geometry cleaner without materially amplifying batch structure.

Create:

```text
task004_high_confidence/metrics/embedding_separability_by_tier.csv
```

---

## 17. Weak-class focus

Pay special attention to:

```text
Myeloid
Neutrophil
Plasma cell
```

For each tier report:

- retained training-cell count;
- retention percentage;
- mean matching distance;
- ambiguous-match removal rate;
- grouped-CV F1;
- grouped-CV recall;
- grouped-CV AUPRC.

Do not merge these classes.

Do not create manual morphology rules.

---

## 18. Registration QC figures

Generate representative overlays from:

- ALL_MATCHED cells that are excluded by HIGH_CONFIDENCE;
- HIGH_CONFIDENCE retained matches;
- ULTRA_HIGH_CONFIDENCE retained matches.

Include all seven classes across multiple batches.

The purpose is to visually verify that stricter matching actually removes questionable labels.

Save:

```text
task004_high_confidence/qc/overlay_examples/
task004_high_confidence/figures/Fig1_matching_confidence_examples.pdf
```

---

## 19. Required figures

All figures must be editable vector PDF with source-data CSV.

Create at minimum:

```text
Fig1_matching_confidence_examples.pdf
Fig2_matching_distance_distribution.pdf
Fig3_class_retention_by_tier.pdf
Fig4_embedding_separability_by_tier.pdf
Fig5_CV_macroF1_by_tier.pdf
Fig6_per_class_F1_by_tier.pdf
Fig7_weak_class_recall_by_tier.pdf
Fig8_confusion_matrix_ALL_MATCHED.pdf
Fig9_confusion_matrix_HIGH_CONFIDENCE.pdf
Fig10_confusion_matrix_ULTRA_HIGH_CONFIDENCE.pdf
```

Use:
- white background;
- vector text/lines;
- individual fold points;
- mean ± uncertainty;
- consistent class order;
- restrained colorblind-safe palette;
- no 3D effects;
- no decorative gradients.

Save all source data under:

```text
task004_high_confidence/figure_data/
```

---

## 20. Do not evaluate the test set during tier selection

Do not use the existing independent test set to select:

- confidence thresholds;
- Tier B/Tier C rules;
- distance cutoffs;
- ambiguity margins;
- final tier;
- training recipe.

All decisions must come from training-only grouped CV.

---

## 21. Promotion rule for a high-confidence model

A high-confidence tier becomes eligible for final training only if grouped CV shows:

1. macro-F1 improvement >= +0.03 absolute over ALL_MATCHED;
2. lowest-three-class mean F1 improvement >= +0.03;
3. at least 4 of 5 folds improve in macro-F1;
4. no strong class loses >0.03 F1 without clear overall benefit;
5. no severe class depletion;
6. no increased evidence of batch leakage;
7. matching QC supports improved label confidence.

If neither HIGH_CONFIDENCE nor ULTRA_HIGH_CONFIDENCE meets these criteria:

- do not replace production model;
- conclude that simple label filtering is insufficient;
- recommend the next evidence-supported strategy.

---

## 22. Final model training

If one tier satisfies the promotion criteria:

1. freeze the tier definition;
2. freeze the training recipe;
3. train the final classifier using all TRAINING images with only retained labels from the selected tier;
4. use the official CellViT++ training stack;
5. save:

```text
/data/lf_data/result/task004_high_confidence/model_high_confidence_best.pth
```

6. verify official native inference loading.

Do not overwrite production model yet.

---

## 23. One-time test evaluation

Only if a high-confidence tier satisfies the grouped-CV promotion rule:

evaluate that single final candidate once on the test dataset.

Important:

- apply the same predefined matching-confidence logic only where appropriate for classifier-only analysis;
- preserve the native official whole-pipeline evaluation separately;
- clearly distinguish classifier-only paired-cell metrics from detection+classification pipeline metrics.

Because the test set has already been used in earlier methodological comparisons, state that it is no longer a pristine lockbox.

Do not optimize after viewing this result.

---

## 24. Production promotion

Only if the selected high-confidence tier met the CV promotion criteria before test evaluation:

- preserve the current production model;
- copy the new model to:
  `/data/lf_data/result/model_best.pth`
- verify SHA256;
- document the promotion.

Do not reverse the decision based on test performance.

---

## 25. Required outputs

Create:

```text
/data/lf_data/result/task004_high_confidence/
├── TASK004_REPORT.md
├── config/
├── code/
├── work/
├── qc/
├── metrics/
├── figures/
├── figure_data/
├── logs/
└── models/
```

Required tables:

```text
metrics/matching_summary_by_tier.csv
metrics/class_retention_by_tier.csv
metrics/batch_retention_by_tier.csv
metrics/cv_fold_metrics_by_tier.csv
metrics/cv_summary_by_tier.csv
metrics/cv_per_class_metrics_by_tier.csv
metrics/embedding_separability_by_tier.csv
metrics/threshold_sensitivity.csv
```

If a final high-confidence candidate is evaluated:

```text
metrics/test_summary.csv
metrics/test_per_class_metrics.csv
```

---

## 26. Required helper code

Save reproducible helper scripts under:

```text
task004_high_confidence/code/
```

Suggested:

```text
01_inventory_geometry.py
02_reconstruct_matching.py
03_compute_match_confidence.py
04_build_tier_datasets.py
05_validate_tier_datasets.py
06_generate_official_configs.py
07_run_official_cv.py
08_aggregate_cv.py
09_embedding_separability.py
10_generate_figures.py
11_train_final_candidate.py
12_evaluate_final_candidate.py
run_task004.sh
```

Training itself must call the official CellViT++ CLI rather than implement a custom training loop.

---

## 27. Required report

Create:

```text
/data/lf_data/result/task004_high_confidence/TASK004_REPORT.md
```

It must answer explicitly:

### A. How was the original matching performed?

### B. What geometric information was available?

### C. How many cells/classes/batches were retained at each tier?

### D. Did stricter matching improve frozen-embedding class separability?

### E. Did HIGH_CONFIDENCE improve grouped-CV performance?

### F. Did ULTRA_HIGH_CONFIDENCE improve grouped-CV performance?

### G. What happened specifically to Myeloid, Neutrophil, and Plasma cell?

### H. Was any model promoted?

### I. What is the final production model path and SHA256?

### J. What should Task 005 do next?

Do not start Task 005.

---

## 28. GitHub synchronization

After completion:

1. update `tasks/task_004.md` to COMPLETED or PARTIAL;
2. create `reports/task_004_report.md`;
3. update `PROJECT_STATUS.md`;
4. commit only small workflow code/report/config summaries;
5. do not commit large server datasets, embeddings, images, or checkpoints;
6. push to `origin/main`;
7. do not force-push.

---

## 29. Final handoff

Return:

1. original matching algorithm;
2. available geometry;
3. ALL_MATCHED cell count;
4. HIGH_CONFIDENCE cell count;
5. ULTRA_HIGH_CONFIDENCE cell count;
6. per-class retention;
7. ALL_MATCHED grouped-CV macro-F1;
8. HIGH_CONFIDENCE grouped-CV macro-F1;
9. ULTRA_HIGH_CONFIDENCE grouped-CV macro-F1;
10. lowest-three-class F1 for each tier;
11. embedding class purity/silhouette for each tier;
12. best tier;
13. promotion decision;
14. final production model path;
15. new candidate path if any;
16. report path;
17. figures path;
18. unresolved issue;
19. recommended Task 005 direction.

Do not start Task 005.
