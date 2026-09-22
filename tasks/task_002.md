# Task 002 — Diagnose and Improve CellViT++ Classification Head

## Status
COMPLETED

## Objective
Diagnose the dominant limitation of the Task 001 CellViT++ classifier and perform a targeted second optimization round while keeping the CellViT-SAM-H-x40-AMP backbone frozen.

Task 001 baseline:
- grouped 5-fold CV macro-F1: ~0.342
- independent test macro-F1: 0.3440
- balanced accuracy: 0.3514
- macro-AUROC: 0.7400
- macro-AUPRC: 0.3441
- weak test classes: Plasma cell (~0.203 F1), Myeloid (~0.231), Neutrophil (~0.246)
- strongest test class: Tumor (~0.695)

The goal is balanced multiclass improvement, not maximum overall accuracy.

## Paths
- conda environment: cellvit_env
- CellViT++: /data/lf_data/CellViT-plus-plus
- backbone: /data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth
- dataset root: /data/lf_data/xenium_data
- Task 001 results: /data/lf_data/result
- Task 002 outputs: /data/lf_data/result/task002

## Non-negotiable rules
1. Preserve /data/lf_data/result/model_best.pth before any new work as /data/lf_data/result/task001_model_best_baseline.pth and verify SHA256.
2. Do not modify source images, labels, or the pretrained backbone.
3. Keep the backbone frozen.
4. Do not use the test set for search, tuning, early stopping, architecture selection, loss selection, sampling selection, or promotion decisions.
5. Use only grouped training CV for optimization decisions.
6. Evaluate only one final Task 002 candidate on the test set after all choices are frozen.
7. Save all new outputs under /data/lf_data/result/task002.
8. Do not silently redefine or remove any of the seven classes.

## Phase A — Diagnose the bottleneck before tuning

### A1. Audit cell matching / label assignment
Inspect how Xenium/ground-truth cell labels were assigned to CellViT-detected cells.

Quantify where possible:
- GT cell count
- detected-cell count
- matched-cell count
- unmatched GT cells
- unmatched detections
- match-distance distribution
- match distance by class
- match distance by batch/slide/ROI
- ambiguous matching
- duplicate assignments
- out-of-bounds coordinates

Create:
- task002/qc/cell_matching_summary.csv
- task002/qc/cell_matching_by_class.csv
- task002/qc/cell_matching_by_batch.csv

Do not change matching thresholds before reporting their current behavior.

### A2. Registration/label visual QC
Generate representative overlays across all seven classes and multiple batches/slides where source information permits.

Include examples of:
- correctly classified cells
- misclassified cells
- high-confidence errors
- detected cell/nucleus
- assigned class
- matched GT/Xenium target
- matching displacement/distance when available

Save:
- task002/qc/registration_overlays/
- task002/figures/FigA_registration_QC.pdf

Do not modify labels during this task.

### A3. Training-only OOF baseline analysis
Using the leakage-safe grouped folds from Task 001, generate out-of-fold predictions for the Task 001 baseline.

Create:
- task002/diagnostics/oof_predictions.csv
- task002/diagnostics/oof_confusion_matrix.csv
- task002/diagnostics/oof_per_class_metrics.csv

Report:
- most common confusion pairs
- per-class F1/recall/precision
- class confidence distributions
- batch-specific performance
- slide/ROI-specific performance where available

### A4. Frozen embedding separability
Extract/cache frozen CellViT-SAM-H embeddings for matched TRAINING cells only.

Evaluate grouped-CV performance of:
- multinomial logistic regression
- linear classifier
- kNN
- current native MLP classifier head

Report macro-F1, balanced accuracy, macro-AUPRC and per-class F1.

Also calculate:
- class silhouette score
- batch silhouette score
- nearest-neighbor class purity
- nearest-neighbor batch purity

Generate vector PDFs:
- FigB_embedding_PCA.pdf
- FigC_embedding_UMAP_class.pdf
- FigD_embedding_UMAP_batch.pdf

Do not treat UMAP appearance as quantitative proof.

### A5. Bottleneck decision
Determine whether the dominant limitation is:
- annotation/registration noise
- detection-to-GT matching
- batch/domain shift
- weak frozen-embedding separability
- classifier-head optimization
- class imbalance
- mixed causes

If severe label/registration problems make the training labels unreliable, stop model promotion, mark Task 002 PARTIAL, preserve diagnostics, and report what must be corrected. Otherwise continue to Phase B.

## Phase B — Targeted classifier-head optimization

### B1. Optimization target
Rank candidates using grouped-CV only with:
score = 0.50*macro_F1 + 0.20*balanced_accuracy + 0.15*macro_AUPRC + 0.15*mean_F1_of_lowest_3_classes

Also record:
- macro-AUROC
- MCC
- every class F1
- every class recall
- between-fold SD
- batch-level performance

### B2. Compare loss strategies
Where compatible with the installed native classifier implementation, compare:
- unweighted cross-entropy
- inverse-square-root weighted cross-entropy
- effective-number weighted cross-entropy
- focal loss
- class-balanced focal loss
- weighted cross-entropy with modest label smoothing

Candidate focal gamma: 1.0, 1.5, 2.0, 3.0
Candidate label smoothing: 0, 0.02, 0.05, 0.10

Calculate all weights from the training fold only.

### B3. Compare sampling
Where supported, compare:
- ordinary shuffled sampling
- weighted random sampling
- class-aware/balanced mini-batches

Never rebalance validation/test data.

### B4. Focused head search
Search around Task 001 winner:
- AdamW
- lr around 1e-4 to 7e-4
- weight decay 1e-5 to 1e-3
- hidden_dim 128, 256, 384, 512
- dropout 0.1, 0.2, 0.3, 0.4

Use Optuna/TPE or efficient random search where practical.
Target roughly 30–60 informative trials, not a brute-force Cartesian grid.

If native architecture safely supports another hidden layer without breaking classifier checkpoint compatibility, test a limited one-layer vs two-layer comparison. Otherwise retain native architecture.

### B5. Hard-class strategy
Confirm weak classes using TRAINING OOF results, not test metrics.

If Plasma cell, Myeloid and Neutrophil remain weak in OOF results, test:
- balanced loss
- focal/class-balanced focal
- balanced sampling
- training-fold-only hard-example mining if compatible

Do not construct rules from test errors.

### B6. Full validation
Take the top 3–5 configurations and run full leakage-safe grouped 5-fold CV.

For every candidate report:
- mean ± SD macro-F1
- balanced accuracy
- macro-AUPRC
- macro-AUROC
- MCC
- lowest-3-class mean F1
- per-class F1 and recall
- batch-level results

## Model promotion rule
Do NOT replace the Task 001 production model unless grouped CV shows all of the following:
1. macro-F1 improves by at least +0.03 absolute over the Task 001 baseline;
2. lowest-3-class mean F1 improves by at least +0.03 absolute;
3. no previously strong class loses >0.03 F1 unless clearly justified by a substantial overall gain;
4. fold variance is not materially worse;
5. no leakage or QC failure is present.

Preferred target: macro-F1 >= 0.40, but do not force or fabricate this target.

If criteria are not met:
- keep the Task 001 model as production model;
- do not overwrite /data/lf_data/result/model_best.pth;
- conclude that head-only optimization has likely plateaued.

## Final candidate and test evaluation
If promotion criteria are met:
1. freeze all choices;
2. train the final head on all training data using CV-derived epoch guidance;
3. save candidate as /data/lf_data/result/task002/model_best_v2.pth;
4. verify native CellViT++ loading/inference;
5. evaluate this single frozen candidate once on the test set;
6. do not optimize again after viewing the Task 002 test result.

Because Task 001 test results are already known, explicitly note that the same test set is no longer a fully pristine iterative-development lockbox. For manuscript-grade unbiased estimation, emphasize grouped/nested CV and/or a future untouched external cohort.

If the model met the predefined CV promotion rule before test evaluation, preserve Task 001 baseline and promote the Task 002 model to:
- /data/lf_data/result/model_best.pth

Do not reverse the promotion decision based on test performance.

## Required Task 002 outputs
Create at minimum:
- task002/metrics/embedding_separability.csv
- task002/metrics/cv_by_batch.csv
- task002/metrics/candidate_cv_summary.csv
- task002/metrics/candidate_per_class_metrics.csv
- task002/metrics/test_summary.csv if a promoted candidate is tested
- task002/metrics/test_per_class_metrics.csv if tested
- task002/metrics/test_bootstrap_ci.csv if tested
- task002/figure_data/
- task002/qc/
- task002/diagnostics/
- task002/embeddings/
- task002/code/
- task002/figures/
- task002/logs/
- task002/TASK002_REPORT.md

All primary plots must be editable vector PDF with source-data CSV.

Required figures:
- Fig1_baseline_vs_optimized_CV.pdf
- Fig2_per_class_CV_comparison.pdf
- Fig3_OOF_confusion_matrix.pdf
- Fig4_loss_sampling_comparison.pdf
- Fig5_hyperparameter_optimization.pdf
- Fig6_CV_by_batch.pdf
- Fig7_test_per_class_performance.pdf if final test performed
- Fig8_test_confusion_matrix.pdf if final test performed
- plus the registration and embedding diagnostic figures above

Use clean Nature-family style: white background, vector text/lines, individual fold points, uncertainty, consistent class order, restrained colorblind-safe palette, no 3D effects or decorative gradients.

## Required report
Create /data/lf_data/result/task002/TASK002_REPORT.md and answer explicitly:
A. What is the dominant bottleneck?
B. Did head-only optimization materially improve grouped-CV performance?
C. Was the Task 002 model promoted?
D. What is the final production model path and SHA256?
E. Which classes remain weak and why?
F. What should Task 003 do?

## Completion and GitHub synchronization
After server execution:
1. update the workflow repository's reports/task_002_report.md with a concise audit and key metrics;
2. update PROJECT_STATUS.md;
3. commit only workflow text/code summaries, not large server artifacts;
4. push to origin/main using ordinary push;
5. do not force-push;
6. do not start Task 003.

## Final handoff
Return:
- dominant bottleneck
- baseline grouped-CV macro-F1
- optimized grouped-CV macro-F1
- baseline and optimized lowest-3-class mean F1
- best loss/sampler/hyperparameters
- promotion decision
- final production model path and SHA256
- Task 002 test macro-F1 if tested
- per-class F1/recall
- report path
- figure directory
- unresolved issues
