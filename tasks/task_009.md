# Task 009 — CellViT Retraining with Frozen Panel-Aware Xenium Labels and Recalibrated Neutrophil Eligibility

## Status
COMPLETED (V3 promotion criteria not met; production unchanged)

## Objective

Test whether the corrected Xenium ground truth improves CellViT seven-class H&E classification, with special emphasis on Neutrophil recognition.

The biological annotation and H&E eligibility rules are now frozen before model training:

- biological identity: Task 007 Xenium 5K panel-aware v3 annotation;
- Neutrophil H&E eligibility: Task 008 target-centered H&E QC, manually reviewed and accepted;
- no annotation rule may be changed after viewing Task 009 model performance.

Task 009 must isolate the effect of improved ground truth before introducing new backbones, context models, or specialist architectures.

---

## 1. Frozen source labels

### Task 007 biological annotation

```text
/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz
```

Derived AnnData:
```text
/data/lf_data/result/task007_xenium5k_panelaware/adata_xenium_v3_panelaware.h5ad
```

Use:
- original_cl1;
- original_cl1_7class;
- v3_action;
- v3_label;
- cell_quality_status;
- technical_fail;
- Task 007 training-quality fields.

Do not rerun or modify Task 007 reannotation.

### Task 008 Neutrophil eligibility

```text
/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz
```

Use the manually accepted Task 008 target-centered H&E eligibility.

Primary Neutrophil definition:
```text
TRAINABLE_CORE
```

Sensitivity Neutrophil definition:
```text
TRAINABLE_EXTENDED
```

Do not reuse the Task 007 whole-crop debris status.

---

## 2. Source H&E and registration

H&E:
```text
/data/lf_data/xenium_data/ID0060276.ome.tif
```

Registration:
```text
/data/lf_data/xenium_data/matrix.csv
```

Historical dataset-building notebook:
```text
/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb
```

CellViT++:
```text
/data/lf_data/CellViT-plus-plus
```

Current production checkpoint:
```text
/data/lf_data/result/model_best.pth
```

Task 009 output root:
```text
/data/lf_data/result/task009_v3_retraining
```

Do not modify source files.

---

## 3. Seven-class training taxonomy

Use exactly:

```text
0 Endothelial
1 Mesenchymal
2 Myeloid
3 Neutrophil
4 Plasma cell
5 T and B
6 Tumor
```

Mapping:
- Neutrophil → Neutrophil
- Neutrophil_CXCR4 → Neutrophil

Preserve subtype in metadata for audit.

Exclude from training:
- original Low quality;
- Task 007 REVIEW identity cells;
- Task 007 technical failures;
- Task 007 Artifact_or_no_nucleus;
- Task 008 Neutrophil MANUAL_REVIEW from the PRIMARY condition;
- Task 008 NO_TARGET_NUCLEUS;
- Task 008 FRAGMENTED_TARGET_SUSPECT;
- registration-uncertain Neutrophils.

---

## 4. Primary and sensitivity label sets

Construct two new label sets.

### V3_CORE — primary

For six non-Neutrophil classes:
- Task 007 biologically retained seven-class labels;
- cell_quality_status == Pass;
- v3_action != REVIEW;
- valid registration;
- use Task 007 trainable-quality rules.

For Neutrophil:
- frozen Task 007 biological Neutrophil identity;
- Task 008 TRAINABLE_CORE only.

### V3_EXTENDED — sensitivity

Same six non-Neutrophil classes.

For Neutrophil:
- Task 008 TRAINABLE_EXTENDED.

Do not include Task 008 MANUAL_REVIEW in CORE.

If EXTENDED includes any review-like status by implementation, document the exact rule and counts.

---

## 5. Historical OLD_LABELS baseline

The comparison baseline must remain the same OLD_LABELS dataset/metrics used in Tasks 003/005 where possible.

Do not rebuild the baseline differently unless necessary.

Primary old-label grouped-CV reference:
- SAM-H RAW
- macro-F1 0.3324 ± 0.0134
- Neutrophil F1 0.1145
- Neutrophil recall 0.0953
- Neutrophil AUPRC 0.1366

If exact fold-wise old predictions are available, reuse them.

---

## 6. Dataset reconstruction audit

Before training, reproduce the historical patch-building logic from the notebook.

Record exactly:
- coordinate transform;
- 256×256 patch geometry;
- stride;
- train/test batch assignments;
- patch naming;
- label CSV format;
- historical per-class cap behavior;
- how cells near patch edges were handled;
- whether multiple labels per patch were allowed;
- whether patches with multiple batches were excluded.

Create:
```text
qc/dataset_reconstruction_audit.md
```

Do not silently change historical geometry.


---

## 6A. Mandatory full dataset regeneration from source

This requirement is mandatory and overrides any temptation to reuse the historical prepared dataset.

### Forbidden source dataset

Do NOT reuse any images, labels, split files, manifests, cached patch metadata, or other prepared artifacts from:

```text
/data/lf_data/xenium_data/CellViT_dataset
```

The old `CellViT_dataset` was generated from the historical annotation/QC policy and therefore must not be treated as a valid source for Task 009.

It may be inspected only for format/audit purposes if absolutely necessary, but no file from that directory may be copied, symlinked, hard-linked, read as a training image source, or used as a label/split source in the new datasets.

### Required regeneration inputs

Regenerate all Task 009 patch images and labels directly from:

```text
/data/lf_data/xenium_data/ID0060276.ome.tif
/data/lf_data/xenium_data/matrix.csv
/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz
/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz
```

Use the historical preprocessing notebook only to reconstruct and document:
- coordinate transformation;
- patch size;
- stride;
- patch naming;
- train/test batch assignment;
- label-file schema;
- patch-edge handling;
- historical filtering logic that remains intentionally preserved.

Do NOT reuse the notebook-generated old patch files themselves.

### New output datasets

Create entirely new datasets under Task 009:

```text
/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE
/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_EXTENDED
```

Each dataset must contain newly generated:
- H&E patch image files;
- label CSV files;
- train/validation/test file lists;
- grouped-CV split files;
- dataset manifests;
- cell-to-patch mapping tables;
- per-cell frozen v3 biological labels;
- Neutrophil CORE/EXTENDED eligibility metadata.

### No image reuse

Even when the new patch geometry is numerically identical to the historical geometry, Task 009 must re-export the image patches from the original OME-TIFF.

Do not:
- copy old PNG/TIFF/JPEG patches;
- symlink old patch directories;
- hard-link old patch files;
- use old patch hashes as image inputs.

The purpose is to guarantee that the new dataset is fully regenerated from the accepted ground truth and original source image.

### Provenance audit

Create:

```text
qc/dataset_regeneration_provenance.md
metrics/dataset_regeneration_manifest.csv
```

The provenance report must explicitly state:

1. whether any file from `/data/lf_data/xenium_data/CellViT_dataset` was used;
2. exact source H&E path;
3. exact registration-matrix path;
4. exact Task 007 annotation-table path;
5. exact Task 008 eligibility-table path;
6. number of newly generated image patches;
7. number of newly generated label files;
8. number of cells assigned to patches;
9. number of cells excluded and reason;
10. SHA256 or another reproducible hash summary for the newly generated dataset manifests.

The expected answer to item 1 must be:

```text
No old CellViT_dataset files were reused.
```

### Hard failure condition

If the implementation detects that Task 009 training is reading patch images, labels, split lists, or metadata from:

```text
/data/lf_data/xenium_data/CellViT_dataset
```

stop execution and mark Task 009 BLOCKED rather than proceeding.

Do not silently fall back to the historical dataset.


---

## 7. Primary training dataset composition

The goal is to test label quality, not class-frequency artifacts.

### 7.1 Training pool

Build:
```text
work/CellViT_dataset_v3_CORE
work/CellViT_dataset_v3_EXTENDED
```

Preserve the historical train/test batch split.

### 7.2 Class cap / sampling

First reproduce the historical per-class cap if the notebook applied one.

If the old benchmark used a 50,000-cell per-class cap:
- keep that cap for the primary controlled comparison;
- include all Neutrophils because they are below 50,000;
- sample capped classes reproducibly with the same seed where possible.

Also report the uncapped eligible counts separately.

Do not use a new balancing strategy in the primary comparison.

### 7.3 Expected uncapped primary eligible counts

Task 007 / 008 currently support approximately:
- Tumor 354,569
- T and B 120,909
- Myeloid 103,566
- Mesenchymal 88,182
- Endothelial 61,488
- Plasma cell 27,264
- Neutrophil CORE 21,639
- Neutrophil EXTENDED 22,107

Recompute exact counts from source tables and report any discrepancy.

---

## 8. Leakage-safe splits

Use the same leakage-safe grouped 5-fold logic used in prior tasks.

Grouping variable:
```text
batch
```

No train/validation batch overlap.

Preserve the historical frozen external test batches.

Do not use the external test set for model selection.

Record:
```text
metrics/split_manifest.csv
```

---

## 9. Fixed model and training recipe

Primary controlled benchmark:

Backbone:
```text
CellViT-SAM-H-x40-AMP
```

Use RAW / no stain normalization.

Use the same official CellViT++ classifier training stack as Task 003/005:

```text
python3 ./cellvit/train_cell_classifier_head.py --config <CONFIG>
```

Fixed head recipe:
- AdamW
- lr 0.001
- weight decay 0.0001
- hidden_dim 100
- dropout 0
- batch size 2048
- cosine scheduler
- same epoch/early-stopping logic as Task 003/005

Do not tune per condition in the primary comparison.

Conditions:

```text
OLD_LABELS
V3_CORE
V3_EXTENDED
```

---

## 10. Primary grouped-CV metrics

For every condition and fold:

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

Report mean ± SD.

---

## 11. Neutrophil-specific metrics

Primary co-endpoints:

- Neutrophil precision
- recall
- F1
- AUROC
- AUPRC

Confusion flows:
- Neutrophil → Myeloid
- Myeloid → Neutrophil
- Neutrophil → T and B
- T and B → Neutrophil
- Neutrophil → Plasma
- Plasma → Neutrophil

Also stratify Neutrophil evaluation metadata by original subtype where feasible:
- conventional Neutrophil
- Neutrophil_CXCR4

The classifier output remains one broad Neutrophil class.

---

## 12. Detection vs classification

Because prior tasks showed limited detection recall, separate:

### Detection recall
GT trainable cell → CellViT nucleus detection matched?

### Conditional classification
Matched GT cell → correct class?

### End-to-end
GT cell → detected and correctly classified?

Calculate these for all seven classes where feasible.

For Neutrophil report:
- detection recall;
- conditional recall;
- end-to-end recall;
- end-to-end precision.

Use the NEW frozen ground truth for V3 conditions.

Do not compare old vs new end-to-end metrics without explicitly noting that ground truth changed.

---

## 13. Main scientific comparison

The key question is:

Does corrected ground truth improve H&E classification?

Primary comparisons:

```text
V3_CORE - OLD_LABELS
V3_EXTENDED - OLD_LABELS
```

Report fold-paired deltas for:
- macro-F1;
- macro-AUPRC;
- lowest-three F1;
- Neutrophil F1;
- Neutrophil recall;
- Neutrophil AUPRC.

Count how many of 5 folds improve.

---

## 14. Promotion rule

A V3 condition qualifies for a final refit if all are met:

1. macro-F1 improves >= +0.03 absolute over OLD_LABELS;
2. Neutrophil F1 improves >= +0.05 OR Neutrophil AUPRC improves >= +0.05;
3. lowest-three-class mean F1 does not decrease;
4. at least 4/5 folds improve in macro-F1;
5. no previously strong class loses >0.05 F1 without a clear balanced benefit;
6. no batch leakage;
7. dataset reconstruction/QC passes.

If neither qualifies:
- do not overwrite production;
- report whether the remaining bottleneck is representation/detection rather than label quality.

---

## 15. Secondary limited optimization

Only if V3_CORE or V3_EXTENDED clearly improves the fixed-recipe benchmark:

perform a limited second-stage head optimization on the better V3 condition.

Allowed:
- Task 001 head recipe;
- modest learning-rate / hidden-dim / dropout alternatives;
- previously validated official options.

Do not exceed approximately 12–20 informative configs.

Use grouped CV only.

Do not tune the annotation.

---

## 16. Final refit and test

Only if a V3 model meets the predefined grouped-CV promotion criteria:

1. freeze label set;
2. freeze training recipe;
3. train on all training batches;
4. run one final evaluation on the frozen external test batches.

The test labels must be rebuilt from the same frozen V3 annotation/QC policy.

Because previous tasks have inspected historical test data and because ground truth has changed, state explicitly that this is a comparative final evaluation, not a pristine unseen benchmark.

Do not optimize after viewing test results.

---

## 17. Production promotion

Only if the V3 candidate met CV promotion criteria before test evaluation and final test performance is not grossly inconsistent:

- preserve current production checkpoint;
- save selected candidate under Task 009 first;
- record SHA256 and provenance;
- only then update:
  ```text
  /data/lf_data/result/model_best.pth
  ```

Otherwise production remains Task 001.

---

## 18. Required outputs

Create:
```text
/data/lf_data/result/task009_v3_retraining/
├── TASK009_REPORT.md
├── config/
├── code/
├── qc/
├── metrics/
├── work/
├── models/
├── cache/
├── figures/
├── figure_data/
└── logs/
```

Required metrics:
```text
metrics/dataset_composition.csv
metrics/dataset_regeneration_manifest.csv
metrics/split_manifest.csv
metrics/cv_fold_metrics.csv
metrics/cv_summary.csv
metrics/cv_per_class_metrics.csv
metrics/neutrophil_metrics.csv
metrics/detection_recall_by_class.csv
metrics/neutrophil_end_to_end.csv
metrics/paired_fold_deltas.csv
metrics/promotion_decision.json
```

---

## 19. Required figures

```text
Fig1_dataset_composition.pdf
Fig2_macroF1_old_vs_v3.pdf
Fig3_macroAUPRC_old_vs_v3.pdf
Fig4_per_class_F1_old_vs_v3.pdf
Fig5_neutrophil_metrics_old_vs_v3.pdf
Fig6_neutrophil_confusion_old_vs_v3.pdf
Fig7_detection_recall_by_class.pdf
Fig8_neutrophil_end_to_end.pdf
Fig9_fold_paired_deltas.pdf
Fig10_best_condition_confusion.pdf
```

Use editable vector PDF and source CSV.

---

## 20. Required report questions

A. Confirm that all Task 009 patch images, labels, split files, and manifests were newly regenerated from the original OME-TIFF and frozen Task007/008 labels, and confirm that no file from `/data/lf_data/xenium_data/CellViT_dataset` was reused.

B. What exact labels and QC rules defined V3_CORE and V3_EXTENDED?

C. How many eligible cells per class and batch were available before capping?

D. How many cells per class were actually used for training?

E. Did the corrected labels improve grouped-CV macro-F1?

F. Did Neutrophil precision/recall/F1/AUPRC improve?

G. Did Myeloid↔Neutrophil confusion decrease?

H. What were per-class detection recalls under the new ground truth?

I. What was Neutrophil end-to-end recall?

J. Did CORE or EXTENDED perform better?

K. Did any V3 condition satisfy the promotion rule?

L. Was a final external test run performed?

M. Was production model updated?

N. If not, what is the next dominant bottleneck: detector, representation, context, or another issue?

---

## 21. GitHub synchronization

After execution:

1. update `tasks/task_009.md`;
2. create `reports/task_009_report.md`;
3. update `PROJECT_STATUS.md`;
4. commit code/config/report/small metrics only;
5. do not commit checkpoints/caches/large patches;
6. push `main`;
7. no force push.

---

## 22. Final handoff

Return:

1. uncapped eligible counts by class;
2. actual training counts by class;
3. OLD_LABELS macro-F1;
4. V3_CORE macro-F1;
5. V3_EXTENDED macro-F1;
6. macro-AUPRC for all conditions;
7. lowest-three F1 for all conditions;
8. Neutrophil precision/recall/F1/AUPRC for all conditions;
9. Myeloid↔Neutrophil confusion;
10. detection recall by class;
11. Neutrophil end-to-end recall;
12. fold-paired deltas;
13. best V3 condition;
14. promotion decision;
15. candidate model path;
16. production model path/SHA;
17. report path;
18. figures path;
19. unresolved issues;
20. recommended next task.

Do not modify frozen annotation rules based on Task 009 performance.
