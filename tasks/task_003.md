# Task 003 — Strict Official CellViT++ Classifier Retraining

## Status
COMPLETED

## Objective

Retrain the seven-class CellViT++ classifier strictly through the official CellViT++ training workflow from:

https://github.com/TIO-IKIM/CellViT-plus-plus

This task is a methodological control experiment. Its purpose is to determine whether the relatively low performance observed in Tasks 001–002 is caused by our previous custom classifier-training loop or instead reflects limitations in the data, matching quality, frozen CellViT-SAM-H representation, or class separability.

The official CellViT++ training/evaluation code must be used directly.

Do NOT implement a custom PyTorch training loop in this task.

---

## 1. Server environment and fixed paths

Remote environment:

```bash
conda activate cellvit_env
```

CellViT++ repository:

```text
/data/lf_data/CellViT-plus-plus
```

Official pretrained CellViT-SAM-H backbone:

```text
/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth
```

Dataset:

```text
/data/lf_data/xenium_data/CellViT_dataset
```

Existing production model from Task 001:

```text
/data/lf_data/result/model_best.pth
```

Task 003 output root:

```text
/data/lf_data/result/task003_official
```

Do not overwrite Task 001 or Task 002 outputs.

---

## 2. Strict official-code requirement

The classifier training itself must be executed using the official CellViT++ entry point:

```bash
python3 ./cellvit/train_cell_classifier_head.py --config <CONFIG>
```

For sweep mode, use the official entry point exactly as documented:

```bash
python3 ./cellvit/train_cell_classifier_head.py --config <SWEEP_CONFIG> --sweep
```

The official training stack must remain intact:

```text
train_cell_classifier_head.py
→ ExperimentCellVitClassifier
→ CellViTHeadTrainer
→ official CellViT token caching/data loading
→ official LinearClassifier
→ official optimizer/scheduler/loss handling
→ official checkpoint serialization
```

Do NOT replace the training engine with:
- a custom DataLoader loop;
- a custom optimizer loop;
- a custom checkpoint writer;
- direct manual training on cached HDF5 tokens.

Custom code may be used only for:
- preparing official YAML configs;
- validating split files;
- aggregating official output metrics;
- generating comparison tables/figures;
- copying the selected official checkpoint;
- reporting.

---

## 3. Official tutorial requirements to follow

Follow the current official README section:

```text
Re-training your own classifier on new data: Workflow
```

The official dataset requirements are:

```text
CellViT_dataset/
├── label_map.yaml
├── splits/
│   ├── fold_0/
│   │   ├── train.csv
│   │   └── val.csv
│   ├── ...
├── train/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── train_configs/
```

For DetectionDataset:
- annotations must remain CSV files containing x, y and integer class label;
- labels start at 0;
- correct input shape must be supplied;
- test data remain separated.

Use the official evaluation entry point:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir <OFFICIAL_RUN_LOGDIR> \
  --dataset_path /data/lf_data/xenium_data/CellViT_dataset \
  --cellvit_path /data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --input_shape 256 256
```

If the actual validated input shape differs from 256 × 256, use the actual supported shape and document it.

---

## 4. Preserve existing models

Before starting:

1. Calculate SHA256 of:

```text
/data/lf_data/result/model_best.pth
```

2. Confirm the preserved Task 001 baseline exists:

```text
/data/lf_data/result/task001_model_best_baseline.pth
```

3. If not present, create a protected copy.

4. Never overwrite the current production model during training.

Task 003 official candidate must initially be stored under:

```text
/data/lf_data/result/task003_official/model_official_best.pth
```

---

## 5. Record exact official source version

The directory `/data/lf_data/CellViT-plus-plus` was previously found not to contain Git metadata.

Therefore:

1. record file SHA256 hashes for at least:
   - `README.md`
   - `cellvit/train_cell_classifier_head.py`
   - `cellvit/training/experiments/experiment_cell_classifier.py`
   - `cellvit/training/trainer/trainer_cell_classifier.py`
   - `cellvit/models/classifier/linear_classifier.py`
   - `cellvit/training/evaluate/inference_cellvit_experiment_detection.py`

2. save:

```text
/data/lf_data/result/task003_official/config/official_source_hashes.csv
```

3. record Python, PyTorch, CUDA, cuDNN, CellViT package/environment information.

Do not update or replace the installed CellViT++ repository during this task.

---

## 6. Dataset QC

Verify:

- train images: expected 5,002;
- test images: expected 6,107;
- label_map.yaml;
- seven classes;
- image-label pairing;
- image size;
- supported input shape;
- no exact train/test duplicate images;
- no missing annotations;
- class counts;
- batch identifiers.

The actual class map must be read from the dataset.

Expected current taxonomy, to be verified rather than assumed:

```text
Endothelial
Mesenchymal
Myeloid
Neutrophil
Plasma cell
T and B
Tumor
```

Do not modify source labels.

---

## 7. Cross-validation split policy

The original supplied folds were shown in Task 001 to mix the same batches between training and validation.

Do NOT use those leaking source folds for model selection.

Use the leakage-safe grouped folds already produced in Task 001, where:
- splitter: StratifiedGroupKFold;
- grouping variable: batch;
- n_splits: 5;
- random seed: 42;
- train/validation batches do not overlap.

For strict official training compatibility:

create an official-compatible derivative dataset/split structure under:

```text
/data/lf_data/result/task003_official/work/CellViT_dataset_official
```

You may use symlinks for images/labels and copy only small split/config files.

Do not modify:

```text
/data/lf_data/xenium_data/CellViT_dataset
```

The derivative dataset should point to the same read-only train/test images and labels while using the leakage-safe split CSV files.

---

## 8. Official config generation

Inspect the official example configs under the installed repository, especially:

```text
test_database/training_database/Example-Detection/train_configs/
logs/Classifiers/
```

Generate official-style YAML configs for CellViT-SAM-H.

The config must use:

```yaml
data:
  dataset: DetectionDataset
  dataset_path: /data/lf_data/result/task003_official/work/CellViT_dataset_official
  num_classes: 7
  input_shape: 256
```

or the correct actual shape if different.

Use the official CellViT-SAM-H checkpoint:

```yaml
cellvit_path: /data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth
```

Use the dataset's real label_map.

Do not invent unsupported YAML keys.

Save all configs under:

```text
/data/lf_data/result/task003_official/config/
```

---

## 9. Phase 1 — Official default/reference baseline

First run a single official training job using the closest official/example configuration compatible with DetectionDataset + CellViT-SAM-H.

Purpose:
- verify official caching;
- verify official CellViTHeadTrainer;
- verify official checkpoint output;
- establish a pure official baseline.

Command must be of the form:

```bash
cd /data/lf_data/CellViT-plus-plus

python3 ./cellvit/train_cell_classifier_head.py \
  --config /data/lf_data/result/task003_official/config/fold_0_official_baseline.yaml
```

Do not replace this with custom training code.

Record:
- official logdir;
- best epoch;
- official validation metrics;
- checkpoint path;
- runtime;
- GPU usage;
- any warnings.

---

## 10. Phase 2 — Official hyperparameter search

Preferred approach: use the official WandB sweep workflow exactly as documented.

Official command:

```bash
python3 ./cellvit/train_cell_classifier_head.py \
  --config /data/lf_data/result/task003_official/config/fold_0_official_sweep.yaml \
  --sweep
```

### WandB rule

If WandB is already authenticated and functional on the remote server:
- use the official sweep workflow.

If WandB authentication is not available:
- do not request, expose, store, or commit a WandB API token;
- do not implement a custom PyTorch training loop;
- instead create a small set of official YAML configs and execute each one separately through `train_cell_classifier_head.py`.

This fallback still uses the official training stack and is acceptable.

Document which route was used.

---

## 11. Hyperparameter scope

Use only parameters supported by the installed official config system.

Start from official example ranges and the previous Task 001 winner.

At minimum consider, where officially supported:

```text
optimizer
learning rate
weight decay
hidden_dim
drop_rate
scheduler
weighted_sampling / loss weighting options exposed by official config
early_stopping_patience
```

Do not introduce focal loss or custom loss functions unless they already exist in the installed official CellViT++ config/trainer.

The goal of this task is official reproduction, not extending official code.

---

## 12. Official hyperparameter selection

If an official WandB sweep is used, use:

```bash
python3 ./scripts/find_best_hyperparameter.py <SWEEP_FOLDER> --metric AUROC/Validation
```

as documented by the official tutorial.

Also independently extract:
- F1/Validation if available;
- Accuracy/Validation if available;
- loss;
- class-specific metrics if available.

The official winner must be identified according to the official tutorial metric:

```text
AUROC/Validation
```

Do not substitute macro-F1 for the official winner in the primary official-reproduction analysis.

For scientific comparison, macro-F1 may be reported secondarily.

---

## 13. Phase 3 — Official grouped 5-fold training

After determining the official best configuration from fold 0:

apply the same official configuration to each of the five leakage-safe grouped folds.

For every fold:

```text
fold_0
fold_1
fold_2
fold_3
fold_4
```

execute via:

```bash
python3 ./cellvit/train_cell_classifier_head.py --config <FOLD_CONFIG>
```

Do not use a custom trainer.

Collect the official best checkpoint and metrics for every fold.

---

## 14. Cross-validation reporting

Report across the five official runs:

- AUROC/Validation;
- accuracy;
- F1/macro-F1 if available or calculable from official validation predictions;
- balanced accuracy;
- macro-AUPRC where probabilities are available;
- per-class precision;
- per-class recall;
- per-class F1.

For metrics not directly emitted by the trainer:
- calculate them from official model predictions/checkpoints;
- do not alter training.

Create:

```text
/data/lf_data/result/task003_official/metrics/cv_fold_metrics.csv
/data/lf_data/result/task003_official/metrics/cv_summary.csv
/data/lf_data/result/task003_official/metrics/cv_per_class_metrics.csv
```

---

## 15. Final official model

After grouped CV:

1. use the official-selected hyperparameters;
2. determine final epoch guidance from the five official folds;
3. create a final official config using all 5,002 training images;
4. run final training through:

```bash
python3 ./cellvit/train_cell_classifier_head.py --config <FINAL_CONFIG>
```

Do not directly train a classifier from cached tokens with custom code.

Copy the official best checkpoint to:

```text
/data/lf_data/result/task003_official/model_official_best.pth
```

Preserve the original official run directory/checkpoint as well.

---

## 16. Native official test evaluation

Only after the final official model is frozen, run the official evaluation path:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir <FINAL_OFFICIAL_LOGDIR> \
  --dataset_path /data/lf_data/xenium_data/CellViT_dataset \
  --cellvit_path /data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --input_shape 256 256
```

Use the correct actual input shape if different.

Do not tune after seeing test performance.

Because the same test set has already been inspected in Task 001, explicitly state that this is a comparative methodological evaluation rather than a pristine unseen benchmark.

---

## 17. Metrics required

For the final official model report:

- Accuracy
- Balanced accuracy
- Macro-F1
- Weighted F1
- Macro-AUROC
- Macro-AUPRC
- MCC
- native detection F1

Per class:
- precision
- recall
- F1
- AUROC
- AUPRC
- support

Use the native CellViT++ paired-cell evaluation definition.

---

## 18. Required comparison

Create a direct comparison between:

### Task 001
Previous hybrid/custom training loop using official CellViT tokens + official LinearClassifier.

### Task 003
Strict official CellViT++ training stack.

Create:

```text
/data/lf_data/result/task003_official/metrics/task001_vs_official.csv
```

Include:

- grouped-CV macro-F1;
- balanced accuracy;
- macro-AUROC;
- macro-AUPRC;
- lowest-three-class mean F1;
- per-class F1;
- test macro-F1;
- test per-class F1.

Do not compare Task 002 candidate as production because it was not promoted.

---

## 19. Interpretation rule

Task 003 is primarily a methodological comparison.

If the strict official pipeline substantially outperforms Task 001:

```text
Δ grouped-CV macro-F1 >= +0.03
```

then conclude that the previous custom training implementation likely contributed materially to performance loss.

If the official pipeline remains within approximately ±0.02–0.03 macro-F1 of Task 001:

conclude that the training-loop implementation is unlikely to be the dominant bottleneck, and upstream matching/label quality/frozen-representation limitations become the leading explanation.

Do not claim causality beyond the evidence.

---

## 20. Production-model promotion rule

Do not automatically overwrite:

```text
/data/lf_data/result/model_best.pth
```

The strict official model may be promoted only if:

1. grouped-CV macro-F1 improves by at least +0.03 over Task 001;
2. lowest-three-class mean F1 improves by at least +0.03;
3. no strong class drops >0.03 F1 without a compelling balanced gain;
4. CV variance is acceptable;
5. no leakage is detected;
6. native official inference loads successfully.

If promotion criteria are met:

- preserve current model;
- copy official model to `/data/lf_data/result/model_best.pth`;
- verify SHA256;
- record the decision.

Otherwise leave Task 001 production model untouched.

---

## 21. Required figures

All figures must be editable vector PDF with source-data CSV.

Create:

```text
/data/lf_data/result/task003_official/figures/Fig1_official_CV_performance.pdf
/data/lf_data/result/task003_official/figures/Fig2_task001_vs_official.pdf
/data/lf_data/result/task003_official/figures/Fig3_official_per_class_performance.pdf
/data/lf_data/result/task003_official/figures/Fig4_official_confusion_matrix.pdf
/data/lf_data/result/task003_official/figures/Fig5_official_ROC_curves.pdf
/data/lf_data/result/task003_official/figures/Fig6_official_PR_curves.pdf
/data/lf_data/result/task003_official/figures/Fig7_official_training_curves.pdf
```

Use Nature-family styling:
- white background;
- editable vector text/lines;
- individual CV fold points;
- restrained colorblind-safe palette;
- no 3D;
- no gradients;
- consistent class order.

Save source data under:

```text
/data/lf_data/result/task003_official/figure_data/
```

---

## 22. Required code and configs

Save only helper/orchestration scripts under:

```text
/data/lf_data/result/task003_official/code/
```

Permitted helper code:
- dataset/split validation;
- YAML generation;
- running official CLI commands;
- parsing official logs;
- metric aggregation;
- figure generation;
- comparison reporting.

Training itself must stay inside the official CellViT++ entry point.

Save all YAML files under:

```text
/data/lf_data/result/task003_official/config/
```

Save full command history to:

```text
/data/lf_data/result/task003_official/logs/commands.sh
```

---

## 23. Required report

Create:

```text
/data/lf_data/result/task003_official/TASK003_REPORT.md
```

It must explicitly answer:

### A. Was training performed strictly through the official CellViT++ trainer?
YES/NO, with the exact official commands.

### B. Was WandB official sweep used?
YES/NO. If not, state why and describe the official-CLI fallback.

### C. What official hyperparameters were selected?

### D. What was the grouped 5-fold official CV performance?

### E. What was the native official test performance?

### F. Did the official pipeline materially outperform Task 001?

### G. Was the official model promoted?

### H. What does this imply for the next task?
Choose evidence-supported next direction:
- keep official model and proceed;
- focus on CellViT↔Xenium matching/registration;
- rebuild high-confidence training labels;
- investigate frozen-representation/domain shift;
- another direction supported by results.

---

## 24. GitHub workflow update

After server execution:

1. update `tasks/task_003.md` status to COMPLETED or PARTIAL;
2. create `reports/task_003_report.md` with concise results;
3. update `PROJECT_STATUS.md`;
4. add only small workflow scripts/config summaries if useful;
5. do not add remote model checkpoints or large artifacts to GitHub;
6. commit;
7. push to `origin/main`;
8. do not force-push;
9. do not start Task 004.

---

## 25. Completion criteria

Task 003 is COMPLETED only when:

1. official source files are hashed;
2. dataset/splits are validated;
3. official YAML configs are generated;
4. at least one pure official baseline run completes;
5. official hyperparameter selection completes;
6. all five grouped folds are trained through `train_cell_classifier_head.py`;
7. final official model is trained through `train_cell_classifier_head.py`;
8. native official evaluation completes;
9. Task 001 vs official comparison is generated;
10. editable PDF figures are generated;
11. TASK003_REPORT.md exists;
12. production promotion rule is applied;
13. GitHub summaries are pushed.

---

## Final Codex handoff

Return:

1. official training route used;
2. exact official training command;
3. WandB sweep used YES/NO;
4. selected official hyperparameters;
5. 5-fold CV macro-F1 mean ± SD;
6. balanced accuracy;
7. macro-AUROC;
8. macro-AUPRC;
9. lowest-three-class mean F1;
10. official test macro-F1;
11. per-class F1/recall;
12. Task 001 vs official delta;
13. promotion decision;
14. final production model path;
15. official candidate path;
16. report path;
17. figure directory;
18. unresolved issues.

Do not start Task 004.
