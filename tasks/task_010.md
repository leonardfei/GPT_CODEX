# Task 010 — Public Pathology Foundation Model Representation Benchmark

## Status
PENDING (revised after MUSK access block)

## Objective

Determine whether the main bottleneck identified in Task009 is the CellViT cell-token representation, and whether publicly accessible pathology foundation models using nucleus-centered H&E context provide substantially better cell-type separability.

The previous MUSK-only Task010 attempt stopped correctly at BLOCKED_MUSK_ACCESS because the official MUSK weights required gated Hugging Face authorization. That blocked attempt is retained for provenance but is superseded by this revised executable benchmark.

This remains a MINIMAL representation benchmark.

Do NOT build the final multiscale model yet.
Do NOT add cellular-neighborhood auxiliary losses yet.
Do NOT modify Task007/Task008 ground truth.
Do NOT tune labels based on model performance.
Do NOT update the production model.

## 1. Primary representations

Compare exactly:

1. CellViT_TOKEN
2. PHIKON_V2_SMALL
3. PHIKON_V2_CONTEXT
4. MIDNIGHT12K_SMALL
5. MIDNIGHT12K_CONTEXT

All five primary conditions must use:
- the same frozen V3_CORE biological labels;
- the same shared detected cells;
- the same matched H&E nucleus centroid;
- the same held-out batch folds;
- the same linear-probe procedure;
- the same secondary MLP-probe procedure;
- the same evaluation metrics.

The only intended difference is the representation.

## 2. Scientific rationale

Task009 showed:
- V3_CORE macro-F1 0.3330 ± 0.0222;
- V3_EXTENDED macro-F1 0.3340 ± 0.0217;
- V3_EXTENDED Neutrophil F1 ~0.1084;
- Neutrophil detector recall ~0.55;
- conditional Neutrophil classification recall ~0.10.

Thus the dominant failure occurs after nucleus detection, during cell-type representation/classification.

The CANVAS design motivates keeping CellViT for nuclear localization while using a pathology foundation model for morphology/context encoding.

## 3. Frozen ground truth

Task007 biological annotation:
`/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`

Task008 accepted Neutrophil eligibility:
`/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`

Primary label condition:
`V3_CORE`

Seven classes:
- Endothelial
- Mesenchymal
- Myeloid
- Neutrophil
- Plasma cell
- T and B
- Tumor

Merge original `Neutrophil` and `Neutrophil_CXCR4` into the broad Neutrophil output class, but preserve subtype metadata.

Do not use V3_EXTENDED in the primary benchmark.

## 4. Data sources

Original H&E:
`/data/lf_data/xenium_data/ID0060276.ome.tif`

Registration:
`/data/lf_data/xenium_data/matrix.csv`

Task009 regenerated V3_CORE dataset:
`/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE`

Task009 result root:
`/data/lf_data/result/task009_v3_retraining`

CellViT backbone:
`/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth`

Historical dataset:
`/data/lf_data/xenium_data/CellViT_dataset`

The historical dataset is FORBIDDEN as an image, label, split, metadata, or embedding source.

## 5. Output root

`/data/lf_data/result/task010_representation_benchmark`

Preserve the previous blocked MUSK-access artefacts under a clearly named provenance/archive subdirectory or leave them untouched.

Create new executable-run outputs under:
- `config/`
- `code/`
- `metrics/`
- `features/`
- `models/`
- `figures/`
- `figure_data/`
- `qc/`
- `logs/`
- `work/`

Do not overwrite Task009.

## 6. Public model provenance and access verification

### 6.1 Phikon-v2 — primary pathology encoder

Official model identifier:
`owkin/phikon-v2`

Expected characteristics to verify from the downloaded model/config:
- pathology-specific ViT-L/16;
- DINOv2-style pretrained backbone;
- official image preprocessing from the model repository;
- standard classification feature = CLS token;
- expected CLS embedding dimension approximately 1024, but record the actual loaded dimension rather than assuming it.

Download only from the official Owkin Hugging Face repository.

Record:
- exact revision/commit if available;
- local checkpoint/cache path;
- SHA256 for downloaded weight file(s);
- model config;
- preprocessing config;
- license text/identifier;
- transformers/torch versions.

Phikon-v2 has a non-commercial license. Record this explicitly in the provenance report. Task010 is a research benchmark and must not make downstream licensing assumptions.

### 6.2 Midnight-12k — independent confirmation encoder

Official model identifier:
`kaiko-ai/midnight`

Use only the publicly available Midnight-12k weights.

Verify from the official model/config:
- DINOv2 pathology foundation model;
- trained on public TCGA data;
- input preprocessing for 224×224 images;
- official normalization mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5);
- official classification embedding = concatenation of CLS token and mean patch-token embedding, unless the loaded official implementation specifies otherwise;
- record actual output dimension rather than assuming it.

Do not use Midnight-92k or Midnight-92k/392 restricted models.

Record:
- exact revision/commit if available;
- local checkpoint/cache path;
- SHA256;
- config;
- preprocessing;
- MIT license;
- environment versions.

### 6.3 Access behavior

Both selected models are expected to be publicly downloadable without gated authorization.

If one model fails because of a transient download/runtime issue:
- retry only with the official repository;
- document the error;
- do not silently substitute another encoder.

If Phikon-v2 works but Midnight-12k fails, Task010 may continue as PARTIAL with CellViT + Phikon-v2, but must not claim the independent confirmation was completed.

If Phikon-v2 itself cannot be loaded, stop before interpreting the representation hypothesis and mark PARTIAL/BLOCKED with the exact reason.

Create:
- `qc/public_model_provenance.md`
- `config/phikon_v2_provenance.json`
- `config/midnight12k_provenance.json`
- `config/foundation_model_environment.txt`

## 7. Physical field-of-view normalization

Audit the H&E physical scale before crop extraction.

Do not trust the inconsistent OME PhysicalSize metadata blindly.

Use the historically validated Xenium–H&E registration geometry and native H&E scale from this project. The previously validated working scale is approximately 0.2125 μm/px; verify it from the source workflow and record the evidence.

Define crop conditions by PHYSICAL FIELD OF VIEW, referenced to a 0.25 μm/px 40× equivalent:

### SMALL
Equivalent to 64×64 px at 0.25 μm/px:
approximately 16×16 μm field of view.

### CONTEXT
Equivalent to 224×224 px at 0.25 μm/px:
approximately 56×56 μm field of view.

If native H&E scale is 0.2125 μm/px, compute the native crop dimensions needed to preserve those physical FOVs, then resize with the encoder's official preprocessing.

Do not define SMALL/CONTEXT by blindly taking 64/224 native pixels.

Create:
- `qc/pixel_scale_audit.md`
- `config/crop_geometry.json`

## 8. Shared-cell cohort — mandatory fairness constraint

Construct:
`SHARED_DETECTED_CORE`

Eligibility:
1. frozen V3_CORE training-eligible biological label;
2. cell belongs to Task009 training batches;
3. successfully matched to a CellViT-detected H&E nucleus;
4. valid CellViT token;
5. valid SMALL crop;
6. valid CONTEXT crop;
7. both public encoders can process the crop;
8. context crop is fully in-bounds;
9. no historical CellViT_dataset dependency.

Use the matched H&E CellViT nucleus centroid as the crop center for all patch-encoder conditions.

All five representations must use EXACTLY the same cell IDs in the primary benchmark.

Save:
- `metrics/shared_cell_manifest.csv.gz`
- `metrics/shared_cell_counts_by_class.csv`
- `metrics/shared_cell_counts_by_batch.csv`
- `qc/shared_cohort_exclusion_summary.md`

Report how many V3_CORE cells are lost at each shared-cohort filtering step.

## 9. CellViT token baseline

Use Task009 regenerated V3_CORE data and the official CellViT-SAM-H checkpoint.

Reuse Task009 detector matches/tokens only when provenance is exact and they originate from the regenerated Task009 V3_CORE dataset.

Otherwise re-extract.

Do not use historical dataset embeddings.

Record per cell:
- cell_id;
- batch;
- class;
- original subtype;
- matched H&E nucleus coordinate;
- detector match distance;
- embedding dimension.

Save:
- `features/cellvit_tokens.pt` or equivalent;
- `features/cellvit_token_manifest.csv.gz`.

## 10. H&E crop generation

Generate all SMALL and CONTEXT crops directly from the original OME-TIFF.

Do not generate them from the old CellViT_dataset.

Do not use the Task009 PNG as the primary crop source if it truncates context or causes double-resampling.

Center on the matched H&E nucleus centroid.

One physical crop can be reused as input to both Phikon-v2 and Midnight-12k; preprocessing after crop extraction must remain encoder-specific.

Generate stratified QC montages showing:
- all seven classes;
- multiple batches;
- conventional Neutrophil;
- Neutrophil_CXCR4;
- SMALL and CONTEXT paired views.

Save:
- `qc/small_crop_montage.pdf`
- `qc/context_crop_montage.pdf`
- `metrics/crop_manifest.csv.gz`.

## 11. Frozen feature extraction

Primary Task010 uses FROZEN encoders only.

Do not fine-tune CellViT, Phikon-v2, or Midnight-12k.

### PHIKON_V2_SMALL / CONTEXT
Use the official Owkin preprocessing and CLS-token feature extraction.

### MIDNIGHT12K_SMALL / CONTEXT
Use official Kaiko preprocessing and the documented classification embedding strategy.

For every feature matrix:
- same shared cell ordering;
- no NaN/Inf;
- record feature dimension;
- record dtype;
- record extraction batch size;
- record GPU used.

Save:
- `features/phikon_v2_small.*`
- `features/phikon_v2_context.*`
- `features/midnight12k_small.*`
- `features/midnight12k_context.*`.

## 12. Exact fold reuse from Task009

Do not generate a new split.

Read Task009 V3_CORE split manifests and recover exact held-out validation batches for folds 0–4.

Every Task010 cell inherits fold from its batch.

For each fold:
- Task010 train batches must equal Task009 V3_CORE train batches;
- Task010 validation batches must equal Task009 V3_CORE validation batches;
- no batch overlap.

Create:
- `metrics/fold_manifest.csv`
- `qc/fold_equivalence_task009.md`.

This is necessary so the five representation conditions are truly paired by fold.

## 13. Primary classifier — identical linear probe

Primary representation benchmark = multinomial linear probe.

Representations:
- CellViT_TOKEN
- PHIKON_V2_SMALL
- PHIKON_V2_CONTEXT
- MIDNIGHT12K_SMALL
- MIDNIGHT12K_CONTEXT

Use identical:
- StandardScaler or equivalent feature standardization;
- multinomial logistic regression;
- class-balanced weights;
- solver;
- stopping criteria;
- common C grid: 0.01, 0.1, 1, 10;
- inner selection procedure.

Prefer grouped inner validation using training batches. If impossible because too few groups, use fixed C=1 for ALL representations rather than using representation-specific tuning.

Save fold-wise probabilities and predictions.

## 14. Secondary classifier — identical MLP probe

Only after linear probing is complete.

For all five representations use an identical probe after necessary input projection:

embedding
→ Linear 256
→ ReLU
→ Dropout 0.5
→ Linear 128
→ ReLU
→ Dropout 0.5
→ Linear 7

Use identical:
- optimizer;
- learning rate;
- weight decay;
- epoch budget;
- early stopping;
- class-imbalance strategy;
- random seed policy.

Do not fine-tune the foundation encoders.

## 15. Seven-class metrics

For linear and MLP probes report:
- accuracy;
- balanced accuracy;
- macro-F1;
- macro-AUPRC;
- macro-AUROC;
- weighted F1;
- MCC;
- lowest-three-class F1.

Per class:
- precision;
- recall;
- F1;
- AUROC;
- AUPRC;
- support.

Report mean ± SD over the same five held-out batch folds.

## 16. Neutrophil-specific endpoints

For each representation/probe report:
- precision;
- recall;
- F1;
- AUROC;
- AUPRC.

Confusion flows:
- Neutrophil → Myeloid;
- Myeloid → Neutrophil;
- Neutrophil → T and B;
- T and B → Neutrophil;
- Neutrophil → Plasma;
- Plasma → Neutrophil.

Also stratify performance metadata for:
- original Neutrophil;
- Neutrophil_CXCR4.

Do not create separate output classes for the subtypes in the main model.

## 17. Binary diagnostic benchmarks

Using the same shared cohort, representations, and held-out batch folds:

### Binary 1
Neutrophil vs Myeloid

### Binary 2
Neutrophil vs T and B

Use the same linear-probe policy for all representations.

Report:
- AUROC;
- AUPRC;
- balanced accuracy;
- F1;
- sensitivity;
- specificity.

These are representation diagnostics, not production classifiers.

## 18. Context-value analysis

For each public encoder compare directly:

- PHIKON_V2_CONTEXT − PHIKON_V2_SMALL
- MIDNIGHT12K_CONTEXT − MIDNIGHT12K_SMALL

for:
- macro-F1;
- macro-AUPRC;
- Neutrophil F1;
- Neutrophil AUPRC;
- binary Neutrophil-vs-Myeloid AUROC/AUPRC;
- binary Neutrophil-vs-T/B AUROC/AUPRC.

This directly answers whether local tissue context adds useful signal beyond the small cell-centered field.

## 19. Cross-encoder agreement

Compare Phikon-v2 and Midnight-12k to determine whether any gain is model-specific.

Compute:
- fold-wise performance correlation;
- per-cell prediction agreement;
- error overlap;
- Neutrophil true-positive overlap;
- shared vs encoder-specific errors.

If both public encoders independently outperform CellViT token in the same direction, treat this as stronger evidence that the CellViT token is the bottleneck.

## 20. Secondary GT-centered morphology upper bound

After the shared-cell benchmark, run a secondary analysis on:
`ALL_V3_CORE_GT_CENTERED`

This may include V3_CORE cells not detected by CellViT.

For BOTH Phikon-v2 and Midnight-12k:
- center the crop using accepted registered H&E target coordinates;
- extract SMALL and CONTEXT embeddings;
- run the same batch-grouped linear probe.

Purpose:
estimate morphology separability if detection were perfect.

Report separately from the shared-cell benchmark.

Do not compare these metrics directly to CellViT token as if the cell universe were identical.

## 21. Representation geometry diagnostics

For SHARED_DETECTED_CORE compute for all five representations:
- class centroid distances;
- within-class cosine distance;
- between-class cosine distance;
- kNN class purity;
- silhouette score;
- batch kNN purity;
- class-vs-batch neighborhood mixing.

UMAP is visualization only.

Determine whether improved class separation is accompanied by stronger batch/domain separation.

## 22. Predefined evidence thresholds

Task010 is exploratory and cannot update production.

### Strong representation gain
At least one public pathology encoder/scale versus CellViT_TOKEN:
- macro-F1 +0.05 absolute or more;
AND
- Neutrophil F1 +0.10 OR Neutrophil AUPRC +0.10 absolute;
AND
- macro-F1 improves in at least 4/5 folds;
AND
- batch separation does not materially worsen.

### Moderate gain
- macro-F1 +0.02 to +0.05;
OR
- Neutrophil F1/AUPRC +0.05 to +0.10;
with reasonably consistent fold direction.

### No meaningful gain
Below those ranges or inconsistent across folds.

### Independent confirmation
If both Phikon-v2 and Midnight-12k show concordant improvement over CellViT token, explicitly report that the result is replicated across two public pathology foundation models.

## 23. Next-step decision tree

If CONTEXT consistently > SMALL and both public encoders > CellViT:
- Task011: multi-scale pathology-FM fusion + neighborhood auxiliary task.

If SMALL > CellViT but CONTEXT adds little:
- Task011: cell/nuclear morphology specialist + limited fine-tuning of the best public encoder.

If immune binary tasks improve strongly but seven-class remains weak:
- Task011: hierarchical immune classifier with Neutrophil/Myeloid/T-B specialist branch.

If only one public encoder improves:
- verify preprocessing/model-specific effects before building a larger architecture.

If neither public encoder improves meaningfully:
- test limited encoder fine-tuning and reassess the intrinsic H&E ceiling.

## 24. Required metrics

Create:
- `metrics/shared_cell_manifest.csv.gz`
- `metrics/shared_cell_counts_by_class.csv`
- `metrics/shared_cell_counts_by_batch.csv`
- `metrics/fold_manifest.csv`
- `metrics/linear_probe_fold_metrics.csv`
- `metrics/linear_probe_summary.csv`
- `metrics/linear_probe_per_class.csv`
- `metrics/mlp_probe_fold_metrics.csv`
- `metrics/mlp_probe_summary.csv`
- `metrics/mlp_probe_per_class.csv`
- `metrics/neutrophil_metrics.csv`
- `metrics/confusion_flows.csv`
- `metrics/binary_neutrophil_vs_myeloid.csv`
- `metrics/binary_neutrophil_vs_tb.csv`
- `metrics/context_value_deltas.csv`
- `metrics/cross_encoder_agreement.csv`
- `metrics/gt_centered_upper_bound.csv`
- `metrics/representation_geometry.csv`
- `metrics/representation_paired_deltas.csv`
- `metrics/decision_summary.json`.

## 25. Required figures

Generate vector PDFs plus source tables:
- Fig1_task010_design.pdf
- Fig2_shared_cohort_composition.pdf
- Fig3_linear_probe_macroF1.pdf
- Fig4_linear_probe_per_class_F1.pdf
- Fig5_neutrophil_metrics.pdf
- Fig6_neutrophil_confusions.pdf
- Fig7_binary_specialist_results.pdf
- Fig8_small_vs_context.pdf
- Fig9_cross_encoder_agreement.pdf
- Fig10_representation_geometry.pdf
- Fig11_fold_paired_deltas.pdf
- Fig12_gt_centered_upper_bound.pdf

Also save SMALL/CONTEXT crop QC montages.

## 26. Required report questions

`TASK010_REPORT.md` must answer:

A. Were Phikon-v2 and Midnight-12k successfully downloaded from their official public repositories?
B. What exact model revisions, SHA256 values, licenses, feature extraction rules, and dimensions were used?
C. What H&E pixel scale and physical crop geometry were used?
D. How many cells entered SHARED_DETECTED_CORE by class and batch?
E. Were the exact Task009 V3_CORE held-out batches reused?
F. What were linear-probe macro-F1/AUPRC for all five representations?
G. What were MLP-probe results?
H. Which representation gave the best Neutrophil precision/recall/F1/AUPRC?
I. Did either public encoder reduce Neutrophil→Myeloid and Neutrophil→T/B confusion?
J. How did each representation perform for Neutrophil vs Myeloid?
K. How did each representation perform for Neutrophil vs T/B?
L. Does CONTEXT outperform SMALL within Phikon-v2?
M. Does CONTEXT outperform SMALL within Midnight-12k?
N. Do Phikon-v2 and Midnight-12k independently support the same conclusion?
O. Did class separation improve without excessive batch separation?
P. What was the public-encoder GT-centered upper-bound performance on all V3_CORE cells?
Q. Is the dominant bottleneck CellViT representation, missing local context, detector limitation, model-specific representation, or intrinsic H&E ambiguity?
R. Which Task011 branch is justified?

Do not update production.

## 27. GitHub synchronization

After execution:
1. update `tasks/task_010.md` to COMPLETED / PARTIAL / BLOCKED;
2. create or replace `reports/task_010_report.md` with the revised-run report while clearly preserving the previous MUSK access block in a provenance note;
3. update `PROJECT_STATUS.md`;
4. commit scripts/configs/small metrics/reports only;
5. do not commit large embeddings, crops, or foundation-model checkpoints;
6. do not commit credentials;
7. push main;
8. no force push.

## 28. Final handoff

Return:
1. model provenance for Phikon-v2 and Midnight-12k;
2. shared cohort size and class counts;
3. exact fold/batch manifest;
4. CellViT_TOKEN linear macro-F1/AUPRC;
5. PHIKON_V2_SMALL linear metrics;
6. PHIKON_V2_CONTEXT linear metrics;
7. MIDNIGHT12K_SMALL linear metrics;
8. MIDNIGHT12K_CONTEXT linear metrics;
9. Neutrophil metrics for all five representations;
10. MLP results;
11. Neutrophil↔Myeloid confusion;
12. Neutrophil↔T/B confusion;
13. binary diagnostic metrics;
14. SMALL-vs-CONTEXT deltas;
15. cross-encoder agreement;
16. representation geometry metrics;
17. GT-centered upper-bound results;
18. best representation;
19. evidence strength and independent-confirmation status;
20. recommended Task011.

Do not modify production and do not modify frozen ground truth.
