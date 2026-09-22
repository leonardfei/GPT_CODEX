# Task 005 — Multi-Backbone, Stain-Domain, and Neutrophil Detection Benchmark

## Status
COMPLETED

## Execution result

Task 005 completed on 2026-09-22 under `/data/lf_data/result/task005_backbone_domain`.

- Only the installed official-compatible SAM-H checkpoint was available and tested; UNI, Virchow, Virchow2, and ViT256 checkpoints were not present and were not downloaded.
- RAW and official `STAIN_NORMALIZED` conditions were evaluated with the fixed official CellViT++ recipe and grouped batch 5-fold splits.
- RAW was the best overall and Neutrophil-F1 condition, but neither condition met the promotion guardrails.
- The production model was not changed, no test evaluation was run, and Task 006 was not started.
- Detailed results are in `/data/lf_data/result/task005_backbone_domain/TASK005_REPORT.md` and `reports/task_005_report.md`.

## Objective

Improve overall seven-class HCC cell recognition while explicitly prioritizing Neutrophil performance.

Tasks 001–004 indicate that:
- classifier-head tuning alone has plateaued;
- strict official CellViT++ training did not outperform the Task 001 model;
- stricter centroid-only CellViT↔Xenium filtering did not materially improve performance;
- frozen SAM-H embeddings show weak class separation and substantial batch/domain structure;
- CellViT detection performance is itself limited and may constrain end-to-end Neutrophil recognition.

Task 005 therefore evaluates whether a better foundation-model backbone and/or stain-domain handling improves:
1. overall seven-class classification;
2. Neutrophil classification;
3. Neutrophil detection recall;
4. end-to-end Neutrophil recognition;
5. batch/domain robustness.

Do not start Task 006.

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

Source dataset:

```text
/data/lf_data/xenium_data/CellViT_dataset
```

Current production model:

```text
/data/lf_data/result/model_best.pth
```

Task 005 output root:

```text
/data/lf_data/result/task005_backbone_domain
```

Do not modify source data or Tasks 001–004 outputs.

---

## 2. First step — Inventory available official backbones

Inspect:

```text
/data/lf_data/CellViT-plus-plus/checkpoints/
```

Identify which compatible pretrained CellViT++ checkpoints are already present and usable among:

- CellViT-SAM-H
- CellViT-UNI
- CellViT-Virchow
- CellViT-Virchow2
- CellViT256 / ViT256

Do not download new weights automatically.
Do not request credentials or gated-model tokens.

Record:

```text
metrics/backbone_inventory.csv
```

For every available checkpoint record:
- backbone name;
- checkpoint path;
- SHA256;
- file size;
- inferred backbone type;
- embedding dimension;
- compatible input resolution;
- whether official CellViT++ can load it successfully.

Backbone embedding dimensions expected from the installed official code, to be verified:
- ViT256: 384
- SAM-H: 1280
- UNI: 1024
- Virchow: 1280
- Virchow2: 1280

---

## 3. Benchmark design

The primary benchmark should compare all available major backbones using identical:
- source dataset;
- seven-class taxonomy;
- leakage-safe grouped folds;
- official CellViT++ training stack;
- evaluation definitions;
- model-selection rules.

Preferred backbones if available:

```text
SAM-H
UNI
Virchow2
```

Optional:
- Virchow
- ViT256 as a lower-capacity reference

Do not delay the task if one or more checkpoints are unavailable.
Run only models already available and compatible.

---

## 4. Stain/domain conditions

For each primary backbone, compare two conditions where officially supported:

### Condition A — RAW
```yaml
normalize_stains_train: false
normalize_stains_val: false
```

### Condition B — STAIN_NORMALIZED
Use the official CellViT++ stain-normalization pathway only.

```yaml
normalize_stains_train: true
normalize_stains_val: true
```

Do not implement a new custom stain-normalization algorithm during Task 005.

If the official pathway requires additional resources unavailable on the server, document the limitation and skip that condition rather than modifying the framework.

---

## 5. Data split policy

Use the same leakage-safe grouped 5-fold split definitions used in Tasks 001–004:

- n_splits = 5
- grouping variable = batch
- random seed = 42
- no train/validation batch overlap

Do not use the original leaking folds.

Do not redesign fold membership separately by backbone or stain condition.

---

## 6. Training stack

Use the official CellViT++ classifier training path:

```text
train_cell_classifier_head.py
→ ExperimentCellVitClassifier
→ CellViTHeadTrainer
→ official token caching/data loading
→ official LinearClassifier
```

Training command:

```bash
python3 ./cellvit/train_cell_classifier_head.py --config <CONFIG>
```

Do not use a custom PyTorch training loop.

---

## 7. Fixed classifier recipe for primary backbone comparison

To isolate backbone/stain effects, the primary comparison must use one fixed classifier-head recipe across all conditions.

Use the Task 003 official recipe unless a backbone requires a documented compatibility change:

```text
optimizer: AdamW
learning rate: 0.001
weight decay: 0.0001
hidden_dim: 100
drop_rate: 0
batch size: 2048
scheduler: cosine
early stopping/model selection: AUROC/Validation
```

Do not tune separate hyperparameters for each backbone in the primary benchmark.

A limited secondary tuning round is allowed only after the primary benchmark identifies the best backbone/condition, and only if Task 005 explicitly records the primary untuned result first.

---

## 8. Required classification metrics

For every backbone × stain condition × fold, calculate:

- accuracy
- balanced accuracy
- macro-F1
- weighted F1
- macro-AUROC
- macro-AUPRC
- MCC
- lowest-three-class mean F1
- per-class precision
- per-class recall
- per-class F1
- per-class AUROC
- per-class AUPRC
- support

Report mean ± SD over five grouped folds.

---

## 9. Neutrophil-focused classification metrics

Neutrophil is a key co-primary endpoint.

For every condition report:

- Neutrophil precision
- Neutrophil recall
- Neutrophil F1
- Neutrophil AUROC
- Neutrophil AUPRC
- Neutrophil support

Also report confusion flows:

```text
Neutrophil → Myeloid
Myeloid → Neutrophil
Neutrophil → T and B
T and B → Neutrophil
Neutrophil → Plasma cell
Plasma cell → Neutrophil
```

Do not merge classes.

---

## 10. Per-class detection audit

This is mandatory.

Using the official CellViT detection output before classifier-head prediction, calculate per-class GT detection recall.

For every GT class:

```text
GT class cell
→ was a CellViT detection paired within the official matching threshold?
```

Report:

- GT count
- matched detection count
- class-specific detection recall
- class-specific unmatched GT count
- median pairing distance
- pairing-distance distribution

Create:

```text
metrics/per_class_detection_recall.csv
```

This must be reported separately from classification conditional on successful detection.

---

## 11. Neutrophil end-to-end recognition

For each backbone/condition, calculate three distinct Neutrophil quantities:

### A. Detection recall
```text
GT Neutrophil
→ any valid CellViT detection
```

### B. Conditional classifier recall
```text
detected + paired GT Neutrophil
→ predicted as Neutrophil
```

### C. End-to-end Neutrophil recall
```text
all GT Neutrophils
→ correctly detected and classified as Neutrophil
```

Also report end-to-end Neutrophil precision where the denominator is well-defined from the full pipeline.

Create:

```text
metrics/neutrophil_end_to_end.csv
```

Do not conflate conditional classifier recall with end-to-end recall.

---

## 12. Batch/domain diagnostics

For every backbone × stain condition:

extract or reuse frozen training embeddings and calculate:

- class silhouette
- batch silhouette
- nearest-neighbor class purity
- nearest-neighbor batch purity
- optional batch-classifier accuracy using grouped CV if computationally reasonable

The desired direction is:

```text
class separation ↑
batch separation ↓
```

Create:

```text
metrics/embedding_domain_diagnostics.csv
```

---

## 13. Stain-normalization effectiveness

For each backbone compare RAW vs STAIN_NORMALIZED on:

- macro-F1
- macro-AUPRC
- Neutrophil F1
- Neutrophil recall
- Neutrophil AUPRC
- lowest-three-class mean F1
- batch NN purity
- batch silhouette

A stain-normalized model is useful only if biological class performance improves or remains stable while batch structure decreases.

Do not prefer stain normalization merely because batch metrics improve if class performance deteriorates.

---

## 14. Primary model-selection score

The main model-selection score must balance overall and Neutrophil performance:

```text
Score =
0.35 × macro_F1
+ 0.20 × macro_AUPRC
+ 0.20 × Neutrophil_F1
+ 0.15 × Neutrophil_AUPRC
+ 0.10 × lowest_three_class_mean_F1
```

This score is secondary to reporting the raw metrics but may be used to rank conditions.

Guardrails:

1. no previously strong class should lose >0.05 F1 unless clearly justified;
2. Neutrophil recall must not decrease materially;
3. batch leakage must remain absent;
4. batch-domain structure should preferably decrease, not increase.

---

## 15. Primary benchmark promotion criterion

A backbone/condition becomes eligible for final validation if it meets all of:

1. grouped-CV macro-F1 improvement >= +0.03 over the Task 001 grouped-CV baseline;
2. Neutrophil F1 improvement >= +0.05 over the Task 001/Task 004 comparable grouped-CV reference;
3. Neutrophil recall improvement >= +0.05 OR Neutrophil AUPRC improvement >= +0.05;
4. lowest-three-class mean F1 does not decrease;
5. no strong class loses >0.05 F1 without compelling balanced benefit;
6. batch-separated folds remain leakage-free;
7. official inference loads successfully.

If no model satisfies all criteria:
- do not overwrite production;
- report the best condition and identify the next bottleneck.

---

## 16. Optional secondary tuning

Only for the best one or at most two backbone/stain conditions from the primary fixed-recipe benchmark:

perform a limited official classifier-head tuning using only supported official config parameters.

Allowed:
- learning rate
- weight decay
- hidden_dim
- dropout
- official weighted sampling/loss options
- scheduler

Do not exceed ~12–20 informative configurations total.

Do not run separate exhaustive sweeps for every backbone.

Use grouped-CV only.

---

## 17. Final candidate training

If a condition meets the promotion rule:

1. freeze backbone;
2. freeze stain condition;
3. freeze head settings;
4. train on all training data using official CellViT++ training;
5. use median best epoch guidance across folds;
6. save under:

```text
/data/lf_data/result/task005_backbone_domain/models/model_task005_best.pth
```

Do not overwrite the production model yet.

---

## 18. One-time test evaluation

Only if the candidate meets predefined grouped-CV promotion criteria:

run the official native test evaluation once.

Report separately:

- classifier-only paired-cell metrics;
- per-class detection recall;
- Neutrophil end-to-end performance;
- whole-pipeline detection/classification metrics.

Because the test set has already been inspected in prior tasks, state that it is a comparative evaluation rather than a pristine unseen benchmark.

Do not optimize after viewing test performance.

---

## 19. Production promotion

If and only if the candidate met CV promotion criteria before test evaluation:

- preserve current production model;
- copy selected candidate to:

```text
/data/lf_data/result/model_best.pth
```

- verify SHA256;
- record provenance.

Otherwise keep the Task 001 production model unchanged.

---

## 20. Required figures

All figures must be editable vector PDF with source-data CSV.

Create:

```text
Fig1_backbone_macroF1.pdf
Fig2_backbone_macroAUPRC.pdf
Fig3_backbone_per_class_F1.pdf
Fig4_neutrophil_metrics.pdf
Fig5_per_class_detection_recall.pdf
Fig6_neutrophil_end_to_end.pdf
Fig7_embedding_class_vs_batch.pdf
Fig8_stain_normalization_effect.pdf
Fig9_confusion_best_conditions.pdf
Fig10_primary_score_ranking.pdf
```

Use:
- white background;
- vector text/lines;
- same class order;
- individual fold points;
- mean ± uncertainty;
- restrained colorblind-safe palette;
- no 3D;
- no gradients.

---

## 21. Required output structure

Create:

```text
/data/lf_data/result/task005_backbone_domain/
├── TASK005_REPORT.md
├── config/
├── code/
├── metrics/
├── models/
├── cache/
├── figures/
├── figure_data/
├── logs/
└── qc/
```

Required metric files:

```text
metrics/backbone_inventory.csv
metrics/cv_fold_metrics.csv
metrics/cv_summary.csv
metrics/cv_per_class_metrics.csv
metrics/neutrophil_metrics.csv
metrics/per_class_detection_recall.csv
metrics/neutrophil_end_to_end.csv
metrics/embedding_domain_diagnostics.csv
metrics/stain_effect_summary.csv
metrics/model_ranking.csv
```

---

## 22. Required report

Create:

```text
/data/lf_data/result/task005_backbone_domain/TASK005_REPORT.md
```

It must answer explicitly:

### A. Which backbones were available and tested?

### B. Did stain normalization reduce batch/domain signal?

### C. Which backbone/condition achieved the best overall seven-class performance?

### D. Which backbone/condition achieved the best Neutrophil performance?

### E. What was Neutrophil detection recall?

### F. What was Neutrophil conditional classification recall?

### G. What was Neutrophil end-to-end recall?

### H. Did any model satisfy the promotion rule?

### I. What is the final production model path and SHA256?

### J. What should Task 006 do?
Choose based on evidence, especially among:
- context-aware representation;
- Neutrophil specialist head;
- limited backbone adaptation;
- detection-model adaptation;
- domain/stain harmonization;
- another evidence-supported direction.

Do not start Task 006.

---

## 23. GitHub synchronization

After execution:

1. update `tasks/task_005.md` to COMPLETED or PARTIAL;
2. create `reports/task_005_report.md`;
3. update `PROJECT_STATUS.md`;
4. commit only small workflow scripts/config/report summaries;
5. do not commit checkpoints, caches or large images;
6. push to `origin/main`;
7. do not force-push.

---

## 24. Final handoff

Return:

1. backbones found;
2. backbones tested;
3. stain-normalization conditions tested;
4. grouped-CV macro-F1 for every condition;
5. macro-AUPRC for every condition;
6. lowest-three-class mean F1;
7. Neutrophil precision/recall/F1/AUPRC;
8. per-class detection recall;
9. Neutrophil detection recall;
10. Neutrophil conditional classifier recall;
11. Neutrophil end-to-end recall;
12. embedding class/batch purity;
13. best overall condition;
14. best Neutrophil condition;
15. promotion decision;
16. production model path;
17. new candidate path if any;
18. report path;
19. figure directory;
20. recommended Task 006 direction.

Do not start Task 006.
