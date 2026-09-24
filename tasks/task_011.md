# Task 011 — Midnight Local-Morphology Optimization and Hierarchical Immune Specialist

## Status
CORRECTION-PENDING — primary Task011 stages completed, but final fine-tuning promotion/binary summary is under methodological correction. See `tasks/task_011_correction.md`.

Completion note (superseded in part): Task011 primary stages completed on the canonical 96,044-cell cohort with exact Task009 V3_CORE folds. FOV12 was selected and the final-block fine-tuning produced the highest observed macro-F1. However, subsequent audit found that the promotion gate used N-vs-Myeloid AUPRC instead of Neutrophil one-vs-rest AUPRC, and the fine-tuned binary summary was not based on independently trained binary probes. The strict promotion decision and final binary endpoints are therefore pending correction under `tasks/task_011_correction.md`. No biological ground truth or production model was changed.

## Goal

Improve H&E single-cell classification performance beyond Task010 while preserving strict grouped validation and frozen biological ground truth.

Task010 corrected benchmark established:

- CELLVIT_TOKEN_ALIGNED macro-F1 0.3364 ± 0.0263
- PHIKON_V2_SMALL macro-F1 0.3366 ± 0.0243
- MIDNIGHT12K_SMALL macro-F1 0.4116 ± 0.0312
- MIDNIGHT12K_CONTEXT macro-F1 0.2806 ± 0.0221
- MIDNIGHT12K_SMALL Neutrophil F1 0.2175
- MIDNIGHT12K_SMALL Neutrophil AUPRC 0.1620
- true N-vs-Myeloid AUROC/AUPRC 0.671/0.380
- true N-vs-T/B AUROC/AUPRC 0.707/0.371

Both Phikon and Midnight showed SMALL > CONTEXT. Therefore Task011 prioritizes target-cell-local morphology, not large-context modeling.

Do not modify Task007/Task008 labels, Task009 datasets, or production model unless explicit promotion criteria are met in a later task.

---

## 1. Canonical data and folds

Reuse Task010 canonical cohort exactly:

`/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz`

Expected n:
`96,044`

Reuse exact Task009 V3_CORE five held-out batch folds.

No new random split.

Primary identity:
`cell_id`

All new feature matrices must be element-wise aligned to canonical cell_id order.

---

## 2. Output root

`/data/lf_data/result/task011_midnight_local_optimization`

Create:
- config/
- code/
- features/
- metrics/
- models/
- figures/
- qc/
- logs/
- work/

Do not overwrite Task010.

---

## 3. Stage A — local physical-FOV sweep

Task010 used:
- SMALL 16 μm = 75 native px
- CONTEXT 56 μm = 263 native px

The best result was at 16 μm, while 56 μm was substantially worse.

Test a focused LOCAL FOV sweep around the target cell using frozen Midnight-12k.

Required physical FOVs:

### FOV12
12 μm per side
At 0.2125 μm/px use nearest odd native crop:
57 × 57 px

### FOV16
16 μm per side
75 × 75 px
Reuse Task010 MIDNIGHT12K_SMALL features as the reference if exact IDs/provenance match.

### FOV20
20 μm per side
95 × 95 px

### FOV24
24 μm per side
113 × 113 px

### FOV32
32 μm per side
151 × 151 px

Do not retest 56 μm as a candidate; retain Task010 CONTEXT only as a negative reference.

All crops:
- same matched H&E nucleus center;
- source directly from original OME-TIFF;
- Midnight official preprocessing after native crop;
- same CLS + mean-patch-token feature rule;
- frozen encoder.

For each FOV run the identical corrected linear probe on the same five folds.

Primary Stage A endpoints:
- macro-F1
- macro-AUPRC
- lowest-three F1
- Neutrophil F1/AUPRC
- true N-vs-Myeloid AUROC/AUPRC
- true N-vs-T/B AUROC/AUPRC
- batch/spatial kNN purity on the same deterministic geometry subset

Select BEST_FOV using:
1. highest mean macro-F1;
2. if within 0.01 macro-F1, prefer higher Neutrophil AUPRC;
3. if still tied, prefer lower batch/spatial kNN purity;
4. selection uses outer-fold summary only for Task011 model-design benchmarking; no production claim.

Save:
- metrics/fov_sweep_summary.csv
- metrics/fov_sweep_fold_metrics.csv
- metrics/fov_sweep_neutrophil.csv
- metrics/fov_sweep_true_binary.csv
- metrics/fov_sweep_geometry.csv
- config/best_fov.json

---

## 4. Stage B — complementary frozen-feature fusion

Task010 showed different error structures across encoders. Test whether complementary representations improve classification.

Use BEST_FOV Midnight representation and exact aligned Task010 features.

Candidates:

### B0
MIDNIGHT_BEST_FOV alone

### B1
MIDNIGHT_BEST_FOV + CELLVIT_TOKEN_ALIGNED

### B2
MIDNIGHT_BEST_FOV + PHIKON_V2_SMALL

### B3
MIDNIGHT_BEST_FOV + CELLVIT_TOKEN_ALIGNED + PHIKON_V2_SMALL

Fusion rule for the PRIMARY benchmark:
- concatenate raw frozen embeddings;
- fit StandardScaler separately on each outer-training fold;
- no validation-derived feature selection;
- same class-balanced multinomial linear probe.

Because dimensions differ, also run a SECONDARY dimension-controlled fusion:
- training-only PCA independently per encoder branch;
- retain either 256 dimensions per branch or 95% variance capped at 256, whichever is smaller;
- concatenate projected branches;
- fit the same linear probe.

Never fit scaler/PCA on outer validation.

Evaluate:
- seven-class metrics
- per-class F1/AUPRC
- Neutrophil metrics
- true binary specialist metrics
- batch/spatial geometry

A fused representation is considered worthwhile only if:
- macro-F1 improves by >=0.015 absolute over MIDNIGHT_BEST_FOV;
OR
- Neutrophil F1/AUPRC improves by >=0.03,
and the gain is positive in >=4/5 folds.

Save:
- metrics/fusion_linear_summary.csv
- metrics/fusion_linear_fold_metrics.csv
- metrics/fusion_per_class.csv
- metrics/fusion_true_binary.csv
- metrics/fusion_geometry.csv
- config/best_fusion.json

---

## 5. Stage C — hierarchical classifier

The current hardest boundary is within the immune compartment.

Define coarse groups:

### TUMOR
- Tumor

### STRUCTURAL
- Endothelial
- Mesenchymal

### IMMUNE
- Myeloid
- Neutrophil
- Plasma cell
- T and B

Build a soft hierarchical classifier using BEST_REPRESENTATION from Stage B.

Train entirely within each outer training fold.

### Coarse head
3-class:
TUMOR / STRUCTURAL / IMMUNE

### Structural head
2-class:
Endothelial / Mesenchymal

### Immune head
4-class:
Myeloid / Neutrophil / Plasma / T and B

Final seven-class probabilities must be composed softly:

- P(Tumor) = P(coarse=Tumor)
- P(Endothelial) = P(coarse=Structural) × P(structural=Endothelial)
- P(Mesenchymal) = P(coarse=Structural) × P(structural=Mesenchymal)
- P(Myeloid) = P(coarse=Immune) × P(immune=Myeloid)
- P(Neutrophil) = P(coarse=Immune) × P(immune=Neutrophil)
- P(Plasma) = P(coarse=Immune) × P(immune=Plasma)
- P(T/B) = P(coarse=Immune) × P(immune=T/B)

Renormalize final probabilities numerically.

Primary heads:
class-balanced logistic/softmax linear probes.

Secondary heads:
small MLP only if linear hierarchy improves or is within 0.01 macro-F1 of flat BEST_REPRESENTATION.

Compare:
- flat best model
- hierarchical model

Primary question:
Does hierarchy improve immune-class discrimination without sacrificing Tumor/Endothelial performance?

Save:
- metrics/hierarchical_fold_metrics.csv
- metrics/hierarchical_per_class.csv
- metrics/hierarchical_neutrophil.csv
- metrics/hierarchical_confusion.csv
- metrics/hierarchical_true_binary.csv
- metrics/hierarchical_vs_flat_deltas.csv

---

## 6. Stage D — Neutrophil specialist rescue head

Only run after Stage C.

Train specialist binary heads on BEST_REPRESENTATION:

1. Neutrophil vs Myeloid
2. Neutrophil vs T and B
3. Neutrophil vs all non-Neutrophil immune cells

Use only outer-training cells for fitting.

Integrate specialist evidence into the hierarchical immune head by training a training-only meta-calibration layer on inner grouped validation folds.

The meta layer may use:
- immune-head logits/probabilities
- N-vs-Myeloid specialist score
- N-vs-T/B specialist score
- N-vs-all-immune specialist score

Restrictions:
- no outer-validation labels in calibration;
- no threshold tuning on outer validation;
- use the same calibration rule across folds.

Compare:
- hierarchical baseline
- hierarchical + specialist rescue

Primary endpoint:
Neutrophil F1 and AUPRC.

Secondary:
overall macro-F1 and lowest-three F1.

Promotion within Task011 requires:
- Neutrophil F1 +>=0.03 OR AUPRC +>=0.03 over Stage C;
- no macro-F1 loss >0.01;
- positive N F1 delta in >=4/5 folds.

---

## 7. Stage E — orientation-invariant TTA feature averaging

If BEST_FOV is established, test orientation invariance using frozen Midnight without fine-tuning.

For each crop generate deterministic views:

Primary TTA4:
- 0°
- 90°
- 180°
- 270°

Secondary TTA8:
- TTA4
- horizontal flip of each TTA4 view

For every view:
- extract Midnight classification embedding;
- L2-normalize each embedding;
- average embeddings across views;
- L2-normalize the averaged embedding.

Run the same linear probe.

Compare:
- single-view BEST_FOV
- TTA4
- TTA8

Do not use color jitter for the primary TTA benchmark.

Keep TTA only if:
- macro-F1 +>=0.01 OR Neutrophil AUPRC +>=0.02;
- no batch/spatial kNN purity increase >0.05 absolute.

---

## 8. Stage F — partial Midnight fine-tuning gate

Do NOT fine-tune Midnight unless Stages A–E are complete.

Fine-tuning is justified only if:
- frozen BEST_REPRESENTATION macro-F1 remains <0.50;
AND
- local FOV/fusion/hierarchy/TTA gains have plateaued.

If triggered, run a limited comparison:

### F0
Frozen encoder baseline

### F1
Unfreeze only final transformer block + classification head

### F2
LoRA/adapters on final 2 transformer blocks if implementation is available locally without external downloads

Training:
- BEST_FOV crop
- exact outer grouped folds
- augmentation: rotations/flips, mild brightness/contrast only
- no aggressive stain normalization in the primary run
- weighted sampler or class-balanced loss
- early stopping via inner grouped training-only validation
- outer validation untouched

Fine-tuning promotion criteria:
- macro-F1 +>=0.03 vs best frozen Task011 model;
AND
- Neutrophil F1/AUPRC +>=0.03;
AND
- >=4/5 folds improve macro-F1;
AND
- no batch/spatial purity worsening >0.05 absolute.

If criteria not met, keep frozen model as preferred representation.

---

## 9. Spatial/batch robustness guardrail

Because all current data derive from a single paired specimen with spatial batches, do not interpret batch purity purely as technical batch effect.

For every candidate record:
- kNN class purity
- kNN batch/spatial purity
- batch-separation ratio
- held-out batch fold performance dispersion

A candidate with higher mean performance but extreme dependence on one spatial fold must not be declared robust.

Create:
- metrics/spatial_robustness_summary.csv
- figures/Fig_spatial_fold_robustness.pdf

---

## 10. Required model ladder

The final Task011 report must include a single ordered model ladder, all on the same canonical shared cohort and folds:

1. Task010 corrected CellViT
2. Task010 Midnight 16 μm
3. Task011 BEST_FOV
4. Task011 BEST_FUSION
5. Task011 HIERARCHICAL
6. Task011 HIERARCHICAL + N specialist
7. Task011 TTA best
8. Task011 fine-tuned model if Stage F triggered

For every rung report:
- macro-F1
- macro-AUPRC
- lowest-three F1
- Neutrophil F1
- Neutrophil AUPRC
- N-vs-Myeloid AUROC/AUPRC
- N-vs-T/B AUROC/AUPRC
- batch/spatial kNN purity

Save:
- metrics/model_ladder.csv

---

## 11. Predefined success targets

Task011 target levels:

### Minimum meaningful improvement
- macro-F1 >=0.43
OR
- Neutrophil F1 >=0.25

### Strong improvement
- macro-F1 >=0.46
AND
- Neutrophil F1 >=0.27
AND
- Neutrophil AUPRC >=0.20

### Excellent within-specimen benchmark
- macro-F1 >=0.50
AND
- Neutrophil F1 >=0.30
AND
- true N-vs-Myeloid AUROC >=0.75

These are internal diagnostic targets, not claims of patient-level generalization.

---

## 12. Required report questions

Task011 report must answer:

A. What physical FOV is optimal for Midnight?
B. Is the 16 μm Task010 choice actually optimal?
C. Does CellViT or Phikon add complementary information to Midnight?
D. Does dimension-controlled fusion reproduce the raw-concatenation result?
E. Does hierarchy improve the four immune classes?
F. Does the Neutrophil specialist rescue improve N without harming overall macro-F1?
G. Does orientation TTA help?
H. Was partial fine-tuning triggered?
I. If fine-tuned, did it improve >=4/5 held-out spatial folds?
J. Which stage contributes the largest incremental gain?
K. Which cell classes remain the dominant errors?
L. What is the final N-vs-Myeloid performance?
M. What is the final N-vs-T/B performance?
N. Did spatial/batch dependence worsen?
O. What exact model should be carried into Task012 / external-patient validation?

---

## 13. Production guardrail

Task011 is still development/diagnostic.

Do NOT replace:
`/data/lf_data/result/model_best.pth`

Do not promote any Task011 model to production until external-patient or independent-slide validation is designed and explicitly approved.

---

## 14. GitHub synchronization

After execution:
1. create `reports/task_011_report.md`;
2. update `tasks/task_011.md`;
3. update `PROJECT_STATUS.md`;
4. commit scripts/configs/small metrics/reports only;
5. do not commit large feature tensors/crops/checkpoints;
6. do not commit credentials;
7. ordinary push only.

## 15. Final handoff

Return:
1. best FOV;
2. FOV sweep table;
3. best fusion and delta over Midnight baseline;
4. hierarchy results;
5. specialist rescue results;
6. TTA results;
7. fine-tuning status/results if triggered;
8. final seven-class metrics;
9. final per-class F1/AUPRC;
10. final Neutrophil metrics;
11. final true binary specialist metrics;
12. spatial/batch robustness;
13. model ladder;
14. best model configuration;
15. whether Task011 meets minimum/strong/excellent internal targets;
16. recommended Task012 validation design.

Do not modify frozen biological ground truth or production.
