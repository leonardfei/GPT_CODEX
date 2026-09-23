# Task 010 — Phikon-first Local Pathology Foundation Model Representation Benchmark

## Status
PENDING — local Phikon-v2 uploaded; run Phase A immediately. Midnight-12k is optional Phase B and must not block Phase A.

## Objective

Test whether the CellViT cell-token representation is the main bottleneck in the current H&E seven-class cell-typing pipeline.

Primary immediate comparison:

1. CellViT_TOKEN
2. PHIKON_V2_SMALL
3. PHIKON_V2_CONTEXT

Midnight-12k will be added later as an independent confirmation encoder after its local upload completes.

Do NOT:
- modify Task007/Task008 ground truth;
- tune labels in response to performance;
- fine-tune CellViT or Phikon-v2 in Phase A;
- build the final multiscale model yet;
- update the production model.

## 1. Frozen ground truth

Use only V3_CORE.

Task007 annotation:
`/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`

Task008 Neutrophil eligibility:
`/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`

Seven output classes:
- Endothelial
- Mesenchymal
- Myeloid
- Neutrophil
- Plasma cell
- T and B
- Tumor

Merge original Neutrophil and Neutrophil_CXCR4 into broad Neutrophil while preserving subtype metadata.

## 2. Data sources

Original H&E:
`/data/lf_data/xenium_data/ID0060276.ome.tif`

Registration:
`/data/lf_data/xenium_data/matrix.csv`

Task009 regenerated V3_CORE dataset:
`/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE`

Task009 split manifest:
`/data/lf_data/result/task009_v3_retraining/metrics/split_manifest.csv`

CellViT backbone:
`/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth`

Forbidden historical dataset:
`/data/lf_data/xenium_data/CellViT_dataset`

No file from the forbidden historical dataset may be used as image, label, split, metadata, or embedding source.

## 3. Local-only Phikon-v2 discovery and provenance

The user has already uploaded Phikon-v2 to the server.

DO NOT attempt internet or Hugging Face download before searching local storage.

Search likely locations under:
- `/data/lf_data/models/`
- `/data/lf_data/`
- user home cache/model directories if needed

Prefer exact directory:
`/data/lf_data/models/phikon-v2`

If not found there, locate directories/files containing:
- `phikon-v2`
- `model.safetensors`
- `config.json`
- `preprocessor_config.json`

The local model is usable only if the directory contains enough files for fully offline loading.

Validate by loading with:
- `local_files_only=True`
- no network access
- no fallback to another model

Record:
- actual local path;
- all model/config files;
- SHA256 of weight file(s);
- architecture/config;
- preprocessing config;
- actual embedding dimension;
- torch/transformers versions;
- GPU and dtype.

Create:
- `qc/phikon_local_provenance.md`
- `config/phikon_v2_provenance.json`
- `config/foundation_model_environment.txt`

If the local upload is incomplete, stop Phase A at `BLOCKED_LOCAL_PHIKON_INCOMPLETE` and report the exact missing files.

## 4. Output root

`/data/lf_data/result/task010_representation_benchmark`

Preserve previous MUSK/Hugging Face blocked-run artefacts as provenance.

New Phase A outputs should be clearly marked as the local-Phikon execution.

## 5. Physical H&E crop geometry

Use the historically validated project scale, not the inconsistent OME PhysicalSize metadata.

Audit and document the validated native H&E scale. The working project value is approximately:
`0.2125 μm/px`

Define by physical field of view:

### SMALL
16 × 16 μm FOV.

At 0.2125 μm/px:
approximately 75 native pixels per side.

Use an odd integer crop size centered exactly on the matched H&E nucleus centroid; preferred:
`75 × 75 native px`

### CONTEXT
56 × 56 μm FOV.

At 0.2125 μm/px:
approximately 263 native pixels per side.

Preferred:
`263 × 263 native px`

After native crop extraction, use Phikon-v2 official preprocessing/resizing.

Create:
- `qc/pixel_scale_audit.md`
- `config/crop_geometry.json`

## 6. Freeze a model-independent shared cohort

Construct and freeze:
`SHARED_DETECTED_CORE`

Eligibility must NOT depend on Midnight-12k or on whether a specific foundation model happens to process a cell.

Required:
1. V3_CORE eligible biological label;
2. Task009 training-batch membership;
3. successfully matched to a CellViT-detected H&E nucleus;
4. valid CellViT token;
5. SMALL crop in-bounds;
6. CONTEXT crop in-bounds;
7. no forbidden historical dataset dependency.

Use the matched CellViT H&E nucleus centroid as the center for BOTH Phikon crop conditions.

Freeze the resulting cell IDs BEFORE extracting Phikon features.

Save:
- `metrics/shared_cell_manifest.csv.gz`
- `metrics/shared_cell_counts_by_class.csv`
- `metrics/shared_cell_counts_by_batch.csv`
- `qc/shared_cohort_exclusion_summary.md`

This exact shared manifest must later be reused for Midnight-12k Phase B.

## 7. CellViT token baseline

Use the same CellViT token extraction semantics as Task009.

Reuse Task009 token/detection outputs only if provenance is exact and they were generated from the regenerated V3_CORE dataset.

Otherwise re-extract.

Record:
- cell_id;
- batch;
- class;
- original subtype;
- matched nucleus coordinates;
- match distance;
- embedding dimension.

Save:
- `features/cellvit_tokens.pt`
- `features/cellvit_token_manifest.csv.gz`

## 8. Phikon-v2 crop generation

Generate SMALL and CONTEXT crops directly from the original OME-TIFF.

Do not crop from the historical dataset.

Do not rely on Task009 patch PNGs if they truncate the requested context or introduce double-resampling.

Create paired QC montages for all seven classes and multiple batches, including:
- conventional Neutrophil;
- Neutrophil_CXCR4.

Save:
- `qc/phikon_small_crop_montage.pdf`
- `qc/phikon_context_crop_montage.pdf`
- `metrics/crop_manifest.csv.gz`

## 9. Frozen Phikon-v2 feature extraction

Use the local Phikon-v2 model fully offline.

Primary feature:
official CLS-token representation from the loaded implementation/config.

Do not assume dimension; record actual runtime dimension.

Conditions:
- PHIKON_V2_SMALL
- PHIKON_V2_CONTEXT

Do not fine-tune the encoder.

Ensure:
- exact same cell ordering as SHARED_DETECTED_CORE;
- no NaN/Inf;
- deterministic eval mode;
- fixed preprocessing.

Save:
- `features/phikon_v2_small.pt`
- `features/phikon_v2_context.pt`
- manifests with cell IDs and extraction metadata.

## 10. Reuse exact Task009 folds

Do not create new folds.

Read Task009 V3_CORE split manifest and reuse the exact held-out validation batch identities for folds 0–4.

Every Task010 cell inherits fold membership by batch.

Create:
- `metrics/fold_manifest.csv`
- `qc/fold_equivalence_task009.md`

No train/validation batch overlap is allowed.

## 11. Phase A primary benchmark — identical linear probe

Representations:
- CellViT_TOKEN
- PHIKON_V2_SMALL
- PHIKON_V2_CONTEXT

Use the exact same cells and folds.

Use identical:
- feature standardization;
- multinomial logistic regression;
- class-balanced weighting;
- solver/stopping criteria;
- common C grid `0.01, 0.1, 1, 10`;
- inner selection rule.

If grouped inner validation is unstable because too few training batches exist, use fixed `C=1` for all representations rather than representation-specific tuning.

Save fold-wise probabilities and predictions.

## 12. Seven-class metrics

For all three representations report:
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

Report mean ± SD over the same five folds.

## 13. Neutrophil-specific analysis

For each representation report:
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

Also stratify results by original subtype metadata:
- Neutrophil;
- Neutrophil_CXCR4.

## 14. Binary diagnostic benchmarks

Run identical linear probes for:

### Neutrophil vs Myeloid
### Neutrophil vs T and B

Report:
- AUROC;
- AUPRC;
- balanced accuracy;
- F1;
- sensitivity;
- specificity.

These are diagnostic representation tests, not production models.

## 15. Context-value analysis

Directly compare:
`PHIKON_V2_CONTEXT - PHIKON_V2_SMALL`

for:
- macro-F1;
- macro-AUPRC;
- Neutrophil F1;
- Neutrophil AUPRC;
- N-vs-Myeloid AUROC/AUPRC;
- N-vs-T/B AUROC/AUPRC.

Interpretation:
- CONTEXT >> SMALL: local tissue context adds useful information;
- SMALL >> CellViT but CONTEXT ≈ SMALL: pathology morphology representation helps, context adds little;
- both ≈ CellViT: frozen Phikon does not solve the problem.

## 16. Representation geometry

For CellViT_TOKEN, PHIKON_V2_SMALL, PHIKON_V2_CONTEXT compute:
- class centroid distances;
- within/between-class cosine distance;
- kNN class purity;
- silhouette score;
- batch kNN purity;
- class-vs-batch mixing.

UMAP is visualization only.

## 17. Phase A stopping point and decision

After the LINEAR benchmark and binary diagnostics complete, write an interim interpretation BEFORE doing anything more complex.

Strong gain:
- macro-F1 +0.05 or more vs CellViT;
AND
- Neutrophil F1 +0.10 OR AUPRC +0.10;
AND
- ≥4/5 folds improve macro-F1;
AND
- no major increase in batch separation.

Moderate gain:
- macro-F1 +0.02 to +0.05;
OR
- N F1/AUPRC +0.05 to +0.10;
with reasonably consistent folds.

No meaningful gain:
below those ranges or inconsistent.

## 18. Secondary MLP probe

Run only after Phase A linear results are saved.

Use the same MLP for all three representations:
embedding → Linear256 → ReLU → Dropout0.5 → Linear128 → ReLU → Dropout0.5 → Linear7

Use identical optimizer, LR, weight decay, epoch budget, early stopping, class-imbalance handling, and seed policy.

Do not fine-tune Phikon.

## 19. GT-centered morphology upper bound

After shared-cell benchmarking, run Phikon SMALL/CONTEXT on:
`ALL_V3_CORE_GT_CENTERED`

This may include V3_CORE cells not detected by CellViT.

Center on accepted registered H&E target coordinates.

Use the same grouped-batch linear probe.

Report separately from the shared-cell comparison.

Purpose:
estimate H&E morphology separability if detection were perfect.

## 20. Midnight-12k Phase B — do not block Phase A

Midnight-12k is still uploading.

DO NOT wait for it.

After Midnight-12k is locally available:
1. verify local provenance and SHA256;
2. use the already frozen `SHARED_DETECTED_CORE` manifest;
3. use the exact same SMALL/CONTEXT physical crops;
4. extract MIDNIGHT12K_SMALL/CONTEXT;
5. run identical linear/MLP probes;
6. add cross-encoder agreement and independent confirmation.

Do not alter the shared cohort after seeing Midnight availability.

## 21. Required Phase A metrics

Create:
- `metrics/shared_cell_manifest.csv.gz`
- `metrics/shared_cell_counts_by_class.csv`
- `metrics/shared_cell_counts_by_batch.csv`
- `metrics/fold_manifest.csv`
- `metrics/linear_probe_fold_metrics.csv`
- `metrics/linear_probe_summary.csv`
- `metrics/linear_probe_per_class.csv`
- `metrics/neutrophil_metrics.csv`
- `metrics/confusion_flows.csv`
- `metrics/binary_neutrophil_vs_myeloid.csv`
- `metrics/binary_neutrophil_vs_tb.csv`
- `metrics/context_value_deltas.csv`
- `metrics/representation_geometry.csv`
- `metrics/representation_paired_deltas.csv`
- `metrics/decision_summary.json`

Then add MLP and GT-centered upper-bound outputs.

## 22. Required Phase A figures

- Fig1_task010A_design.pdf
- Fig2_shared_cohort_composition.pdf
- Fig3_linear_probe_macroF1.pdf
- Fig4_linear_probe_per_class_F1.pdf
- Fig5_neutrophil_metrics.pdf
- Fig6_neutrophil_confusions.pdf
- Fig7_binary_specialist_results.pdf
- Fig8_small_vs_context.pdf
- Fig9_representation_geometry.pdf
- Fig10_fold_paired_deltas.pdf
- Fig11_gt_centered_upper_bound.pdf

Plus crop QC montages.

## 23. Required report questions

The revised report must answer:

A. Where was the locally uploaded Phikon-v2 found?
B. Was it successfully loaded fully offline with `local_files_only=True`?
C. What files, SHA256, config, preprocessing, embedding dimension, dtype, and environment were used?
D. What validated H&E physical scale and native crop sizes were used?
E. How many cells entered SHARED_DETECTED_CORE by class and batch?
F. Were the exact Task009 V3_CORE held-out batches reused?
G. What were linear macro-F1/AUPRC for CellViT_TOKEN, PHIKON_V2_SMALL, PHIKON_V2_CONTEXT?
H. What were Neutrophil precision/recall/F1/AUPRC?
I. Did Phikon reduce N→Myeloid and N→T/B confusion?
J. What were N-vs-Myeloid and N-vs-T/B binary metrics?
K. Did CONTEXT outperform SMALL?
L. Did class separation improve without excessive batch separation?
M. What did the MLP probe show?
N. What was the GT-centered morphology upper bound?
O. Is the main limitation more consistent with CellViT representation, lack of context, detector limitation, or intrinsic H&E ambiguity?
P. Should Task011 be multiscale/contextual, local-morphology focused, hierarchical immune classification, or limited fine-tuning?
Q. Is Midnight-12k still pending or has Phase B been completed?

## 24. GitHub synchronization

After Phase A:
1. update `tasks/task_010.md` status to PARTIAL-PHIKON-COMPLETE or COMPLETED if Midnight Phase B is also done;
2. update `reports/task_010_report.md`;
3. update `PROJECT_STATUS.md`;
4. commit scripts/configs/small metrics/reports only;
5. do not commit large embeddings, crops, checkpoints, or credentials;
6. ordinary push only.

## 25. Final handoff after Phase A

Return:
1. local Phikon path/provenance;
2. shared cohort size and class counts;
3. fold/batch manifest;
4. CellViT_TOKEN linear macro-F1/AUPRC;
5. PHIKON_V2_SMALL linear macro-F1/AUPRC;
6. PHIKON_V2_CONTEXT linear macro-F1/AUPRC;
7. Neutrophil metrics for all three;
8. confusion flows;
9. binary diagnostic metrics;
10. CONTEXT-SMALL deltas;
11. representation geometry;
12. MLP results;
13. GT-centered upper bound;
14. evidence strength;
15. dominant bottleneck interpretation;
16. recommended next branch;
17. Midnight Phase B status.

Do not modify production or frozen ground truth.
