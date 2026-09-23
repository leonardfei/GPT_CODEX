
# Task 010 — Frozen Representation Benchmark: CellViT Token vs MUSK Nuclear and Contextual H&E Embeddings

## Status
PENDING

## Objective

Determine whether the main bottleneck identified in Task 009 is the CellViT cell-token representation, and whether a pathology foundation model using nucleus-centered H&E context provides substantially better cell-type separability.

This is deliberately a MINIMAL representation benchmark.

Do NOT build the final multiscale model yet.
Do NOT add cellular-neighborhood auxiliary losses yet.
Do NOT modify Task007/Task008 ground truth.
Do NOT tune labels based on model performance.
Do NOT update the production model.

Primary comparison:
A. CellViT-SAM-H cell token
B. MUSK small/nuclear-context crop
C. MUSK large/local-context crop

All three primary conditions must:
- use the same frozen V3_CORE biological labels;
- use the same cells whenever technically possible;
- use the same held-out batch folds;
- use the same linear-probe evaluation;
- differ only in representation.

## 1. Scientific rationale

Task009 showed:
- V3_CORE macro-F1: 0.3330 ± 0.0222
- V3_EXTENDED macro-F1: 0.3340 ± 0.0217
- V3_EXTENDED Neutrophil recall: ~0.1006
- V3_EXTENDED Neutrophil F1: ~0.1084
- V3_EXTENDED Neutrophil AUPRC: ~0.1379
- Neutrophil detector recall: ~0.55
- conditional Neutrophil classification recall: ~0.10

Thus, even when a Neutrophil nucleus is detected, the current CellViT-derived representation usually fails to separate it from other classes.

The CANVAS study motivates separating:
- CellViT for nuclear detection/segmentation/localization;
- pathology foundation-model embeddings for H&E morphology/context.

Task010 tests that hypothesis directly.

## 2. Frozen ground truth

Biological labels:
 /data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz

Neutrophil eligibility:
 /data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz

Primary label condition:
Use ONLY V3_CORE.

Seven classes:
0 Endothelial
1 Mesenchymal
2 Myeloid
3 Neutrophil
4 Plasma cell
5 T and B
6 Tumor

Neutrophil and Neutrophil_CXCR4 are merged into the broad Neutrophil output class, while subtype metadata must be preserved.

Do not use V3_EXTENDED in the primary benchmark.

## 3. H&E and registration sources

Original H&E:
 /data/lf_data/xenium_data/ID0060276.ome.tif

Registration:
 /data/lf_data/xenium_data/matrix.csv

Task009 regenerated CORE dataset:
 /data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE

Task009 result root:
 /data/lf_data/result/task009_v3_retraining

Historical old dataset:
 /data/lf_data/xenium_data/CellViT_dataset

is FORBIDDEN as an image, label, metadata, split, or embedding source.

No old CellViT_dataset file may be used.

## 4. Output root

 /data/lf_data/result/task010_representation_benchmark

Create:
TASK010_REPORT.md
config/
code/
metrics/
features/
models/
figures/
figure_data/
qc/
logs/
work/

Do not overwrite Task009.

## 5. MUSK availability and provenance gate

Before any benchmark:

1. inspect the server for an existing official MUSK installation/checkpoint/cache;
2. record package version, checkpoint path, model identifier, source, and SHA256;
3. verify that the loaded model is the pathology MUSK model described in the CANVAS-style workflow, not a different model with the same acronym.

If MUSK is not already available:

- if the official model can be downloaded programmatically from its official/public source without credentials, private tokens, click-through license acceptance, or manual authorization, download it to a dedicated model directory under /data/lf_data/ and record provenance + SHA256;
- install dependencies in a dedicated environment or without breaking the existing CellViT environment;
- do not overwrite existing CellViT dependencies.

If official MUSK access requires credentials, gated approval, private tokens, or manual license acceptance:

- mark Task010 BLOCKED_MUSK_ACCESS;
- document the exact missing prerequisite;
- do NOT silently substitute UNI, Virchow, ResNet, or another encoder.

Do not start performance comparisons with a substituted encoder under the MUSK label.

Create:
qc/musk_model_provenance.md
config/musk_environment.txt
config/musk_checkpoint_sha256.json

## 6. Physical field-of-view normalization

The CANVAS workflow standardized H&E analysis to approximately 40× / 0.25 μm per pixel.

Our source H&E must not trust obviously inconsistent OME physical-size metadata blindly.

First audit:
- actual WSI dimensions;
- historical validated registration geometry;
- historical native H&E pixel size used in prior paired Xenium/H&E analyses;
- any discrepancy with OME metadata.

Use the validated native H&E scale from the historical project/registration workflow.

Define two MUSK conditions by PHYSICAL FIELD OF VIEW:

MUSK_SMALL:
64 × 64 pixels at 0.25 μm/pixel equivalent
approximately 16 × 16 μm field of view.

MUSK_CONTEXT:
224 × 224 pixels at 0.25 μm/pixel equivalent
approximately 56 × 56 μm field of view.

If native H&E is not exactly 0.25 μm/pixel:
- crop the equivalent physical field of view in native pixels;
- then resize the crop to the official MUSK input size, expected 384 × 384 if confirmed by the loaded implementation.

Do not distort the physical field of view simply by taking arbitrary 64/224 native pixels.

Create:
qc/pixel_scale_audit.md
config/crop_geometry.json

## 7. Shared-cell cohort — critical fairness constraint

The primary representation comparison must be performed on the SAME cells.

Construct:
SHARED_DETECTED_CORE

Eligibility:
1. frozen V3_CORE training-eligible biological label;
2. located in the Task009 training batches;
3. successfully matched to a CellViT-detected nucleus;
4. valid CellViT token available;
5. valid H&E small crop available;
6. valid H&E context crop available;
7. crop does not cross WSI boundary;
8. no forbidden historical dataset dependency.

Use the matched H&E CellViT nucleus centroid as the crop center for BOTH MUSK conditions.

This ensures:
same cell
same center
same label
same fold
different representation

Save:
metrics/shared_cell_manifest.csv.gz
metrics/shared_cell_counts_by_class.csv
metrics/shared_cell_counts_by_batch.csv

Report how many V3_CORE cells are lost when restricting to the shared detected cohort.

## 8. CellViT token baseline

Backbone:
 /data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth

Use the same cell-token extraction semantics as Task009 / official CellViT++.

Preferred:
- reuse Task009 regenerated V3_CORE images/metadata and Task009 detector matches if provenance is exact;
- re-extract tokens if needed.

Do NOT use embeddings from the historical /data/lf_data/xenium_data/CellViT_dataset.

For every cell record:
- cell_id;
- batch;
- class;
- original subtype;
- matched nucleus coordinate;
- embedding dimension;
- detector match distance.

Save:
features/cellvit_tokens.npy or .pt
features/cellvit_token_manifest.csv.gz

## 9. MUSK crop generation

Generate crops DIRECTLY from the original OME-TIFF.

Do not crop from Task009 PNG patches if doing so would truncate the desired field of view or introduce double-resampling.

Center each crop on the matched H&E nucleus centroid used for the CellViT token.

Generate:
MUSK_SMALL: approximately 16 μm FOV.
MUSK_CONTEXT: approximately 56 μm FOV.

For QC, save only a stratified sample of crop montages, not every image to Git.

Create:
qc/musk_small_crop_montage.pdf
qc/musk_context_crop_montage.pdf
metrics/crop_manifest.csv.gz

The montage must include all seven classes and multiple batches, with dedicated Neutrophil and Neutrophil_CXCR4 examples.

## 10. Frozen MUSK feature extraction

PRIMARY Task010 must use a frozen MUSK encoder.

Do NOT fine-tune MUSK in the primary benchmark.

For each crop:
- apply official preprocessing;
- resize to official input shape;
- extract the model's standard visual embedding;
- record embedding dimension and pooling strategy.

Conditions:
MUSK_SMALL
MUSK_CONTEXT

Save:
features/musk_small_embeddings.npy or .pt
features/musk_context_embeddings.npy or .pt

Check:
- no NaN/Inf;
- identical cell ordering across representations;
- exact shared cell IDs.

## 11. Fold definition — reuse exact Task009 held-out batches

Do not create a new random batch partition.

Read the Task009 split manifest and recover the exact validation batch identities for folds 0–4.

Every Task010 cell inherits fold assignment from its batch.

Required:
train_batches(fold i) == Task009 train_batches(fold i)
val_batches(fold i) == Task009 val_batches(fold i)

No batch overlap is allowed.

Create:
metrics/fold_manifest.csv
qc/fold_equivalence_task009.md

If Task009 OLD_LABELS folds were not batch-identical to the V3 folds, document this. Task010 comparisons are only between the three Task010 representations and therefore remain internally paired.

## 12. Primary classifier — identical linear probe

The main representation benchmark is a LINEAR PROBE.

For each representation:
CellViT_TOKEN
MUSK_SMALL
MUSK_CONTEXT

Use:
- multinomial logistic regression;
- standardized features where appropriate;
- class-balanced weights;
- same solver;
- same regularization-selection procedure;
- same training and validation cells;
- no feature-specific hyperparameter tuning beyond a small common prespecified C grid.

Common C grid:
0.01, 0.1, 1, 10

Select C using only training batches via inner grouped validation, or use fixed C=1 if inner grouping is impractical.

The SAME selection rule must be applied to all representations.

Primary purpose:
How linearly separable are the frozen representations?

Save fold-wise predictions and probabilities.

## 13. Secondary identical MLP probe

Only after the linear probe completes.

Use the SAME MLP architecture for all representations after an input projection if required:

embedding
→ Linear 256
→ ReLU
→ Dropout 0.5
→ Linear 128
→ ReLU
→ Dropout 0.5
→ Linear 7

Use:
- Adam;
- identical learning rate;
- identical epoch budget;
- identical early-stopping rule;
- identical class-imbalance strategy.

Do not fine-tune either CellViT or MUSK.

This secondary probe asks whether a modest nonlinear head changes the representation ranking.

## 14. Primary 7-class metrics

For both linear and MLP probes report:
- accuracy
- balanced accuracy
- macro-F1
- macro-AUPRC
- macro-AUROC
- weighted F1
- MCC
- lowest-three-class F1

Per class:
- precision
- recall
- F1
- AUROC
- AUPRC
- support

Report mean ± SD across the same five folds.

## 15. Neutrophil-specific endpoints

For each representation:
- Neutrophil precision
- recall
- F1
- AUROC
- AUPRC

Confusions:
- Neutrophil → Myeloid
- Myeloid → Neutrophil
- Neutrophil → T and B
- T and B → Neutrophil
- Neutrophil → Plasma
- Plasma → Neutrophil

Also report metrics separately for source subtype metadata where possible:
Neutrophil
Neutrophil_CXCR4

Do not make them separate output classes in the main seven-class model.

## 16. Binary specialist diagnostic benchmarks

Use the same frozen representations and batch folds.

Binary benchmark 1:
Neutrophil vs Myeloid

Binary benchmark 2:
Neutrophil vs T and B

Use the same linear-probe method.

Report:
- AUROC
- AUPRC
- balanced accuracy
- F1
- sensitivity
- specificity

These are diagnostic representation tests, not production classifiers.

Interpretation:
- if MUSK_CONTEXT strongly improves these binaries, local context contains useful discriminatory signal;
- if MUSK_SMALL improves but MUSK_CONTEXT does not, nucleus/local-cell morphology dominates;
- if neither improves over CellViT, H&E separability may be intrinsically limited for these Xenium-defined identities.

## 17. Secondary MUSK morphology upper-bound cohort

After the fair shared-cell benchmark, run a SECONDARY analysis:
ALL_V3_CORE_GT_CENTERED

This cohort may include V3_CORE cells not detected by CellViT.

For MUSK only:
- center the crop using the accepted registered H&E target location from Task007/008;
- extract SMALL and CONTEXT embeddings;
- run the same grouped-batch linear probe.

Purpose:
If the detector were perfect, how separable are the H&E crops?

This analysis must be reported separately from the shared-cell primary benchmark.

Do not compare its metrics directly against CellViT_TOKEN as if they used the same cell universe.

## 18. Representation geometry diagnostics

For SHARED_DETECTED_CORE, compute for each representation:
- class centroid distances;
- within-class vs between-class cosine distance;
- kNN class purity;
- silhouette score;
- batch kNN purity;
- class-vs-batch neighborhood mixing.

Generate UMAP only as visualization; do not use UMAP for model training.

Determine whether a representation improves biological class separation at the cost of stronger batch separation.

## 19. Predefined interpretation thresholds

Task010 is exploratory and must NOT promote a production model.

Strong gain:
MUSK versus CellViT_TOKEN:
- macro-F1 +0.05 absolute or more;
AND
- Neutrophil F1 +0.10 absolute or Neutrophil AUPRC +0.10 absolute;
AND
- improvement in at least 4/5 folds for macro-F1;
AND
- no major increase in batch separation.

Moderate gain:
- macro-F1 +0.02 to +0.05;
OR
- Neutrophil F1/AUPRC +0.05 to +0.10;
with consistent fold direction.

No meaningful gain:
Changes below these ranges or highly inconsistent across folds.

These thresholds guide the next experiment only; they do not authorize production replacement.

## 20. Next-step decision tree

If MUSK_CONTEXT >> MUSK_SMALL > CellViT:
- next task: multi-scale MUSK fusion + cellular-neighborhood auxiliary head.

If MUSK_SMALL >> CellViT and MUSK_CONTEXT adds little:
- next task: nucleus/local morphology specialist + limited MUSK fine-tuning.

If MUSK_CONTEXT improves Neutrophil binaries but not seven-class:
- next task: hierarchical immune classifier and specialist Neutrophil/Myeloid/T-B branch.

If frozen MUSK approximately equals CellViT:
- next task: test limited MUSK fine-tuning or conclude H&E morphology has a low ceiling;
- do not immediately build a complex multi-scale network.

If MUSK cannot be accessed:
- stop at model-access gate and report BLOCKED.

## 21. Required metrics

Create:
metrics/shared_cell_manifest.csv.gz
metrics/shared_cell_counts_by_class.csv
metrics/shared_cell_counts_by_batch.csv
metrics/fold_manifest.csv
metrics/linear_probe_fold_metrics.csv
metrics/linear_probe_summary.csv
metrics/linear_probe_per_class.csv
metrics/mlp_probe_fold_metrics.csv
metrics/mlp_probe_summary.csv
metrics/mlp_probe_per_class.csv
metrics/neutrophil_metrics.csv
metrics/confusion_flows.csv
metrics/binary_neutrophil_vs_myeloid.csv
metrics/binary_neutrophil_vs_tb.csv
metrics/musk_upper_bound_summary.csv
metrics/representation_geometry.csv
metrics/representation_paired_deltas.csv
metrics/decision_summary.json

## 22. Required figures

Generate vector PDFs plus source CSV where relevant:
Fig1_task010_design.pdf
Fig2_shared_cohort_composition.pdf
Fig3_linear_probe_macroF1.pdf
Fig4_linear_probe_per_class_F1.pdf
Fig5_neutrophil_metrics.pdf
Fig6_neutrophil_confusions.pdf
Fig7_binary_specialist_results.pdf
Fig8_representation_geometry.pdf
Fig9_fold_paired_deltas.pdf
Fig10_musk_gt_centered_upper_bound.pdf

Also save crop QC montages.

## 23. Required report questions

TASK010_REPORT.md must answer:

A. Was the official MUSK model successfully obtained and verified? What exact checkpoint/model ID/SHA was used?
B. What H&E pixel scale and physical crop sizes were used?
C. How many V3_CORE cells entered SHARED_DETECTED_CORE by class and batch?
D. Were the exact Task009 held-out batch folds reused?
E. What were linear-probe macro-F1/AUPRC for CellViT_TOKEN, MUSK_SMALL, and MUSK_CONTEXT?
F. What were MLP-probe results?
G. Which representation had the best Neutrophil precision/recall/F1/AUPRC?
H. Did MUSK reduce Neutrophil→Myeloid and Neutrophil→T/B confusion?
I. How well did each representation separate Neutrophil vs Myeloid?
J. How well did each representation separate Neutrophil vs T/B?
K. Did larger context outperform the small nuclear-context crop?
L. Did class separation improve without increasing batch/domain separation?
M. What was the MUSK-only GT-centered upper-bound performance on all V3_CORE cells?
N. Is the dominant bottleneck more consistent with CellViT representation, lack of local context, detector limitation, or intrinsic H&E ambiguity?
O. Which next experiment is justified by the predefined decision tree?

Do not update production.

## 24. GitHub synchronization

After execution:
1. update tasks/task_010.md to COMPLETED / PARTIAL / BLOCKED;
2. create reports/task_010_report.md;
3. update PROJECT_STATUS.md;
4. commit scripts, configs, small metrics, and reports only;
5. do not commit large embeddings/checkpoints/crops;
6. do not commit credentials/tokens;
7. push main;
8. no force push.

## 25. Final handoff

Return:
1. MUSK availability/provenance;
2. shared cohort size;
3. class counts;
4. fold/batch manifest;
5. CellViT_TOKEN linear macro-F1/AUPRC;
6. MUSK_SMALL linear macro-F1/AUPRC;
7. MUSK_CONTEXT linear macro-F1/AUPRC;
8. corresponding Neutrophil metrics;
9. MLP results for all three;
10. Neutrophil↔Myeloid confusion;
11. Neutrophil↔T/B confusion;
12. binary specialist metrics;
13. representation geometry metrics;
14. MUSK GT-centered upper-bound results;
15. fold-paired deltas;
16. best representation;
17. strength of evidence;
18. dominant bottleneck interpretation;
19. report/figures paths;
20. recommended Task011.

Do not modify production and do not modify frozen ground truth.
