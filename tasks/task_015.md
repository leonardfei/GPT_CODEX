# Task 015 — GHIST Classification-Only Retraining for HCC H&E Cell Typing

## Status
IN_PROGRESS — official GHIST uploaded and verified; canonical contour mapping and fold/overlap QC passed; fold-0 five-epoch pilot running. See `reports/task_015_ghist_celltyping.md`.

## Goal

Retrain an HCC-specific **GHIST classification-only model** using the existing paired H&E–Xenium ground truth, while **completely excluding gene-expression prediction from the model and loss**.

The task is focused only on:

> H&E + nucleus instance mask -> seven-class cell-type prediction

Primary scientific question:

> Does the GHIST morphology/cell-type architecture improve H&E cell classification, especially Neutrophil vs Myeloid, compared with the current Midnight final-block model?

Do not use P84 or P169 for training or hyperparameter selection.

Do not modify Task007/008 labels or Task009/010/011/012 historical outputs.

---

## 1. Official GHIST provenance

Official source:

`SydneyBioX/GHIST`

Use the official GHIST implementation as the architectural reference.

Required GHIST components to reuse/adapt:

- `model.backbone.Backbone` — UNet3+ backbone;
- `model.modules.Embed`;
- `model.modules.MLP`;
- the official nucleus-level + patch-level feature extraction logic from `model/model.py`.

The classification-only implementation must preserve the GHIST morphology feature construction:

1. run H&E patch through the GHIST UNet3+ backbone;
2. use the externally supplied nucleus instance masks;
3. extract nucleus-level features from the first and last backbone feature volumes;
4. average those features within each nucleus;
5. concatenate nucleus-level and patch-level features;
6. pass through the GHIST `Embed` module to obtain a 256-dimensional per-nucleus embedding;
7. classify each nucleus with the GHIST cell-type MLP.

Official GHIST uses a joint pixel map with cell-type classes + background and a separate cell-level classifier. The classification-only adaptation should retain both:

- morphology / pixel-map loss `L_Morph`;
- cell-level classification loss `L_CT,class`.

Do **not** instantiate or use:

- gene-expression prediction heads;
- averaged-expression reference heads;
- expression-derived cell-type heads;
- expression consistency losses;
- expression MSE losses;
- cross-attention expression refinement;
- any gene-expression CSV.

This is an H&E cell-typing model, not a gene-expression model.

### Local code discovery

First search:

- `/data/lf_data/models/GHIST`
- `/data/lf_data/GHIST`
- `/data/lf_data/models/ghist`

If an official GHIST clone is not present, attempt to clone:

`https://github.com/SydneyBioX/GHIST.git`

to:

`/data/lf_data/models/GHIST`

If server networking prevents GitHub access, stop with:

`BLOCKED_NEEDS_LOCAL_GHIST_UPLOAD`

and do not substitute an unofficial implementation.

Record:
- source path;
- source commit SHA;
- GPL-3.0 license;
- environment versions.

Do not modify the upstream clone in place. Put all HCC-specific adapter code under this project's Task015 working/code directory.

---

## 2. Canonical biological labels

Use the same frozen seven-class label system:

0. Endothelial
1. Mesenchymal
2. Myeloid
3. Neutrophil
4. Plasma cell
5. T and B
6. Tumor

Primary canonical cohort:

`/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz`

Expected:
- 96,044 unique canonical cells;
- exact frozen Task007/Task008 CORE biological labels;
- same Task009 CORE fold identities.

Do not relabel any cell based on GHIST predictions.

Do not change Neutrophil eligibility.

---

## 3. H&E source and resolution

Training H&E:

`/data/lf_data/xenium_data/ID0060276.ome.tif`

Validated scale:

`0.2125 μm/px`

GHIST primary patch geometry:

- model patch: 256 × 256 pixels;
- native physical field: approximately 54.4 μm × 54.4 μm;
- training patches: non-overlapping;
- validation/test inference patches: 30-pixel overlap.

This follows the GHIST single-cell patch logic.

Do not resize the entire training WSI to a different MPP for the primary benchmark.

For later external WSI inference, physical-scale normalization will be handled separately.

---

## 4. Nucleus instance masks

GHIST requires nucleus instance masks as an input for per-nucleus feature pooling.

Use CellViT binary segmentation as the nucleus-instance source, not Xenium cell boundaries.

Detector:

`/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth`

### Preferred mapping order

1. Search existing Task009/010/011 server outputs for an exact canonical-cell-to-CellViT-instance mapping.
2. If an exact mapping exists and provenance matches the canonical cohort, reuse it.
3. Otherwise rerun CellViT binary detection on `ID0060276.ome.tif` and construct a one-to-one mapping to canonical cells.

### Matching guardrail

Prefer exact stored identifiers over geometric rematching.

If geometric rematching is required:
- use the same coordinate system and alignment recipe already established in Task008–Task010;
- perform one-to-one matching;
- record distance distribution and ambiguous candidates;
- do not silently assign multiple canonical labels to one nucleus;
- do not invent a new permissive distance threshold without audit.

If a robust one-to-one canonical-to-contour mapping cannot be established, stop before training.

### Patch-local masks

Do not construct a giant dense full-slide uint32 instance image unless necessary.

Prefer patch-local instance masks:
- for each 256×256 patch, rasterize CellViT contours into local positive instance IDs;
- keep a mapping from local instance ID -> canonical cell_id -> class_id;
- invalid/unlabeled nuclei are retained as visual context but excluded from supervised cell loss;
- pixels belonging to nuclei outside the active split must use an `ignore_index` for morphology loss, **not background**.

This avoids train/validation label leakage at spatial boundaries.

---

## 5. Split strategy

Reuse the **exact Task009 V3_CORE five outer grouped folds**.

Do not create random cell-level train/test splits.

For each outer fold:

- outer validation cells = exact Task009 validation batches;
- outer training cells = exact Task009 training batches.

For checkpoint/epoch selection, reuse the corrected Task011 training-only inner-group rule:

- choose one whole outer-training batch as inner validation;
- never use outer validation labels for epoch selection.

### Split-safe patch supervision

A 256×256 patch may contain cells belonging to different fold roles near boundaries.

Therefore:
- H&E context may remain visible;
- cell-level loss is computed only on cells belonging to the active training split;
- morphology pixels for non-training labeled cells are `ignore_index`;
- validation/test predictions are scored only for cells belonging to the current validation fold.

No validation labels may contribute to a training loss.

---

## 6. GHIST classification-only model

Create:

`scripts/python/task015_ghist_celltype_model.py`

Implement a minimal GHIST classification model using official modules:

### Backbone

`Backbone(..., n_classes=8)`

Output:
- background + seven cell types.

### Cell feature embedding

Replicate official GHIST `Framework.forward` nucleus feature extraction:

- first backbone feature volume;
- last backbone feature volume;
- per-nucleus mean feature pooling;
- patch-level feature means;
- concatenate;
- GHIST `Embed(..., 256)`.

### Cell head

Official GHIST-style:

`MLP(256, 256, 7)`

Output:
- seven-class logits per nucleus.

### Forward return

Return at minimum:
- pixel-map logits;
- cell-type logits;
- ordered cell IDs;
- 256-d cell embeddings.

No expression tensors or dummy gene dimensions.

---

## 7. Losses

### Primary model: GHIST_CT

Use:

`L_total = L_Morph + L_CT`

where:

- `L_Morph` = pixel-wise cross entropy over background + seven types;
- `L_CT` = cell-level seven-class cross entropy.

Use `ignore_index` for pixels/cells that are not supervised in the active split.

### Class imbalance

Run two predefined variants:

#### A. GHIST_CT_OFFICIAL
- unweighted `L_Morph`;
- unweighted `L_CT`;
- closest classification-only interpretation of the published GHIST losses.

#### B. GHIST_CT_BALANCED
Same architecture, but:
- compute cell-type weights only from the current outer-training cells;
- apply class-balanced weights to `L_CT`;
- use foreground-class weighting for `L_Morph`, with background weight recorded explicitly;
- never compute weights from outer validation.

Do not tune class weights on outer validation.

Record the exact formula and resulting weights per fold.

No focal loss in the primary benchmark.

---

## 8. Training hyperparameters

Start from official GHIST settings:

- batch size: 8;
- max epochs: 50;
- optimizer: AdamW;
- LR: 1e-3;
- beta1: 0.9;
- beta2: 0.999;
- weight decay: 1e-4;
- patch size: 256×256;
- augmentation:
  - horizontal flip;
  - vertical flip;
  - 90/180/270° rotations;
- training RGB standardization:
  - mean/std computed from training patches only.

The current official repository applies an epoch-wise linear LR decay; record whether this repository behavior is used.

HED/stain augmentation:
- primary benchmark: ENABLED only if the official installed GHIST/stainlib pipeline works reproducibly;
- record the exact setting;
- do not introduce Macenko normalization into the primary training benchmark.

Use mixed precision if numerically stable.

Hardware may use the two available GPUs to run different folds/variants in parallel.

---

## 9. Pilot gate

Before launching the full benchmark:

Run **fold 0 only** for GHIST_CT_OFFICIAL through at least 5 epochs.

Verify:
- loss decreases;
- finite logits/probabilities;
- instance IDs remain aligned after augmentation;
- predicted cell count equals supervised validation cell count after deduplication;
- cell IDs are unique in final validation output;
- no train/validation cell leakage;
- GPU memory is stable.

Create:

`qc/task015/pilot_audit.md`

Only proceed after pilot QC passes.

---

## 10. Checkpoint selection

For each outer fold and variant:

1. train using outer-training minus inner-validation batch;
2. save per-epoch checkpoint;
3. evaluate inner validation after each epoch;
4. select checkpoint by **inner-validation Macro-F1**;
5. tie-break using:
   - higher Neutrophil AUPRC;
   - then lower validation cross-entropy;
6. restart from the same initialization and refit on the full outer-training set for the selected epoch count;
7. evaluate the untouched outer validation once.

Do not choose checkpoints using outer-validation performance.

---

## 11. Validation-cell deduplication

GHIST validation/inference uses overlapping 256×256 patches with 30-pixel overlap.

A nucleus may appear in multiple patches.

Follow the published GHIST strategy:

- for each distinct validation nucleus, retain the prediction from the patch that contains the **largest area of that nucleus**;
- ensure exactly one final prediction per canonical validation cell.

Save the cell-level OOF predictions.

---

## 12. Primary metrics

Use the same metrics as Task011:

### Seven-class
- accuracy;
- balanced accuracy;
- Macro-F1;
- Macro-AUPRC;
- Macro-AUROC;
- weighted F1;
- lowest-three F1;
- per-class precision / recall / F1 / AUROC / AUPRC.

### Neutrophil
- precision;
- recall;
- F1;
- AUROC;
- one-vs-rest AUPRC.

### True binary probes

Using the 256-d GHIST cell embeddings, independently train fold-specific training-only linear probes for:

- Neutrophil vs Myeloid;
- Neutrophil vs T/B.

Report five-fold:
- AUROC;
- AUPRC;
- F1;
- sensitivity;
- specificity;
- precision.

Do not use seven-class probability ratios as true binary probes.

---

## 13. Morphology-head QC

Also report, separately from cell-typing metrics:

- foreground-vs-background Dice / IoU;
- pixel-level class accuracy on supervised nuclei pixels;
- per-class pixel IoU where feasible.

These are diagnostic only.

The main endpoint remains cell-level classification.

---

## 14. Comparison baseline

Compare both GHIST variants directly against corrected Midnight final-block:

Midnight baseline:
- Macro-F1 = 0.4522 ± 0.0294
- Macro-AUPRC = 0.4893 ± 0.0428
- Neutrophil F1 = 0.2352 ± 0.1214
- Neutrophil AUPRC = 0.1976 ± 0.1181
- N-vs-Myeloid AUROC/AUPRC = 0.7202 ± 0.0221 / 0.4385 ± 0.1787
- N-vs-T/B AUROC/AUPRC = 0.7069 ± 0.0359 / 0.3930 ± 0.2118

Use the exact same outer folds.

Create paired fold deltas.

---

## 15. Decision criteria

### Strong GHIST improvement

GHIST is considered a strong candidate if:

- Macro-F1 improves by >= 0.03 over Midnight;
AND
- Neutrophil F1 or AUPRC improves by >= 0.03;
AND
- Macro-F1 is improved in >=4/5 folds.

### Moderate improvement

- Macro-F1 +0.01 to +0.03;
OR
- Neutrophil F1/AUPRC +0.02 to +0.03;
with positive direction in >=3/5 folds.

### No meaningful improvement

Below these thresholds or inconsistent folds.

Do not replace the Task012 Midnight final checkpoint yet.

---

## 16. Optional neighborhood module — not in primary result

The current task is classification-only.

Do **not** use gene expression.

After GHIST_CT_OFFICIAL and GHIST_CT_BALANCED finish, a secondary `GHIST_CT_NC` run is permitted only if:

- the classification-only model is technically sound;
AND
- either Macro-F1 >= 0.42 or Neutrophil AUPRC >= 0.18.

If run:
- use only the GHIST H&E neighborhood-composition auxiliary head;
- derive patch ground-truth composition from the same seven labels;
- do not use expression heads;
- do not use the paper's breast-specific stochastic B/Myeloid/T recovery rule;
- evaluate whether the NC auxiliary loss improves the direct seven-class cell head.

This secondary model must be reported separately.

---

## 17. Final full-data model export

After five-fold CV, identify the best GHIST variant.

Only train/export a full-data GHIST model if the best GHIST variant meets at least the **moderate improvement** criterion.

If eligible:
- choose final epoch count from the median of the five training-only selected epochs;
- train on all 96,044 canonical cells;
- save:

`/data/lf_data/result/task015_ghist_celltyping/models/ghist_hcc_7class_final.pth`

Also save:
- class mapping;
- input patch geometry;
- training normalization;
- source GHIST commit;
- SHA256;
- inference configuration.

If GHIST does not meet the gate, save CV checkpoints for audit but do not designate a final replacement model.

---

## 18. Output root

`/data/lf_data/result/task015_ghist_celltyping`

Create:
- code/
- config/
- data_manifest/
- models/
- metrics/
- qc/
- figures/
- logs/
- work/

Large instance masks, patches and checkpoints remain server-side.

Do not commit them.

---

## 19. Required outputs

### Data/QC

- `qc/canonical_instance_alignment.md`
- `qc/pilot_audit.md`
- `metrics/patch_manifest.csv.gz`
- `metrics/canonical_instance_manifest.csv.gz`
- `config/ghist_source_provenance.json`
- `config/training_config_official.json`
- `config/training_config_balanced.json`

### Metrics

- `metrics/ghist_fold_metrics.csv`
- `metrics/ghist_summary.csv`
- `metrics/ghist_per_class.csv`
- `metrics/ghist_neutrophil.csv`
- `metrics/ghist_true_binary.csv`
- `metrics/ghist_true_binary_summary.csv`
- `metrics/ghist_morphology_qc.csv`
- `metrics/ghist_vs_midnight_paired.csv`
- `metrics/decision_summary.json`

### Figures

At minimum:
- model comparison Macro-F1;
- Macro-AUPRC;
- per-class F1;
- Neutrophil F1/AUPRC;
- paired fold deltas;
- N-vs-Myeloid / N-vs-TB binary performance;
- normalized confusion matrix.

### Report

`reports/task_015_ghist_celltyping.md`

---

## 20. Required scripts

Create reusable project scripts:

- `scripts/python/task015_prepare_ghist_data.py`
- `scripts/python/task015_ghist_celltype_model.py`
- `scripts/python/task015_train_ghist_celltype.py`
- `scripts/python/task015_evaluate_ghist_celltype.py`
- `scripts/python/task015_finalize_ghist.py`

Do not edit the official GHIST clone in place.

---

## 21. Report questions

The final report must answer:

A. Was the official GHIST code acquired and what commit was used?
B. How were CellViT nucleus instances mapped to the 96,044 canonical cells?
C. What proportion of canonical cells obtained valid nucleus masks?
D. Were exact Task009 folds preserved without leakage?
E. What epoch was selected in each fold?
F. How did GHIST_CT_OFFICIAL perform?
G. How did GHIST_CT_BALANCED perform?
H. Which seven-class types improved/worsened versus Midnight?
I. Did Neutrophil improve?
J. Did N-vs-Myeloid improve?
K. Did N-vs-T/B improve?
L. How well did the morphology/pixel head perform?
M. Was the optional neighborhood-only auxiliary stage triggered?
N. Which GHIST variant is best?
O. Does GHIST meet strong/moderate/no-gain criteria?
P. Should a full-data GHIST checkpoint be exported?
Q. Should GHIST proceed to P84/P169 testing?

---

## 22. Production guardrails

Do not modify:

- `/data/lf_data/result/model_best.pth`
- `/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`
- existing P84/P169 GeoJSON outputs;
- Task007/008 biological labels;
- Task009–Task014 historical outputs.

No P84/P169 hyperparameter tuning.

No gene-expression prediction in Task015.

GitHub should contain only:
- adapter scripts;
- configs;
- small metrics;
- QC reports;
- final report;
- publication figures if size permits.

Do not commit:
- GHIST upstream clone;
- large masks/patches;
- model checkpoints;
- raw H&E;
- credentials.

## 23. Next execution command

```text
Execute task_015.
```
