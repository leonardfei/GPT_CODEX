# Task 006 — Xenium Re-annotation, H&E Nuclear-Integrity QC, and High-Quality CellViT Retraining

## Status
PENDING

## Scientific motivation

Tasks 001–005 indicate that classifier-head optimization, strict official training, simple centroid-confidence filtering, and stain normalization do not explain the poor seven-class performance, especially the very low Neutrophil conditional and end-to-end recall.

A new hypothesis must now be tested:

1. the original Xenium cell labels may be insufficiently accurate;
2. low-quality Xenium cells may have been retained;
3. necrotic/no-nucleus regions may contain extracellular or residual RNA and be incorrectly assigned a biological cell type;
4. in particular, neutrophil RNA remnants in necrotic regions may have generated false Neutrophil labels even when no intact nucleus is visible on H&E;
5. these label errors may have contaminated CellViT training and evaluation.

Task 006 therefore rebuilds the Xenium ground truth before any further model-development step.

The task must create a new, derived Xenium v2 annotation and high-quality training set. Do not overwrite any source file.

---

## 1. Input files

All inputs are on the remote server.

### Xenium AnnData

```text
/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad
```

### Original H&E

```text
/data/lf_data/xenium_data/ID0060276.ome.tif
```

### Registration matrix

```text
/data/lf_data/xenium_data/matrix.csv
```

### Previous preprocessing notebook

```text
/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb
```

### Existing CellViT++ code

```text
/data/lf_data/CellViT-plus-plus
```

### Existing production classifier

```text
/data/lf_data/result/model_best.pth
```

### Task output root

```text
/data/lf_data/result/task006_xenium_reannotation
```

Do not alter any of the four source files above.

---

## 2. Overall design

Build a new annotation system in four independent layers:

```text
Xenium transcriptomic QC
        +
coarse biological re-annotation
        +
H&E nuclear-integrity QC
        +
registration/spatial QC
        ↓
Xenium v2 label + confidence
        ↓
high-quality CellViT labels
        ↓
controlled retraining benchmark
```

The CellViT model itself must not be used to decide whether a Xenium cell is high quality.

This is essential to avoid circularity.

Existing CellViT detections/predictions may be used only for descriptive post-hoc comparison after the new Xenium labels have been frozen.

---

## 3. Phase A — Audit the current Xenium object and old labeling pipeline

Before changing anything, inspect the H5AD and notebook.

### A1. AnnData audit

Record:

- shape;
- obs columns;
- var columns;
- layers;
- raw presence;
- obsm keys;
- obsp keys;
- uns keys;
- spatial coordinate columns;
- batch columns;
- current cell-type annotation columns;
- cluster columns;
- count/QC fields;
- cell area / nucleus area fields if present;
- negative-control / blank-probe fields if present;
- segmentation fields if present;
- Harmony representation key;
- whether X contains counts, normalized expression, scaled expression, or another representation.

Do not assume that `.X` is raw expression because the filename contains `harmony`.

For transcript-level marker scoring:
- prefer raw counts or a log-normalized expression layer if present;
- use Harmony only for neighborhood/clustering representation, not for marker-expression values.

Save:

```text
qc/adata_inventory.json
qc/obs_schema.csv
qc/var_schema.csv
```

### A2. Notebook audit

Read the previous notebook and reconstruct exactly:

- how Xenium cells were previously annotated;
- which obs column supplied labels;
- how cells were divided into patches;
- how batch was assigned;
- how train/test were created;
- how spatial coordinates were transformed;
- how `matrix.csv` was applied;
- how H&E coordinates were generated;
- how seven CellViT classes were mapped;
- which cells were previously excluded.

Create:

```text
qc/old_pipeline_audit.md
```

Do not silently reuse an old label column as ground truth.

---

## 4. Phase B — Verify registration before any morphology QC

Use the notebook plus `matrix.csv` to identify the correct transform direction.

Inspect OME-TIFF metadata and record:

- width / height;
- pixel size in µm/px if available;
- pyramid levels;
- channel/color layout.

Transform Xenium cell coordinates into H&E coordinates.

Validation must include:

1. fraction of transformed cells falling within image bounds;
2. coordinate-range sanity checks;
3. representative overlays across the slide;
4. overlays from multiple batches;
5. overlays in tumor, immune-rich, stromal and necrosis-adjacent areas where identifiable.

Save:

```text
qc/registration_summary.csv
figures/Fig1_registration_QC.pdf
qc/registration_overlays/
```

If registration is clearly incorrect, stop morphology-dependent filtering, mark the task PARTIAL and report the issue rather than inventing a transform.

---

## 5. Phase C — Xenium transcriptomic QC

The file name says `remove_necrosis`, but do not assume all residual low-quality or necrosis-associated objects were removed.

### C1. Use available QC fields

Calculate or retrieve, where possible:

- total transcripts / counts;
- detected genes;
- counts per gene;
- negative control counts/fraction;
- blank-probe counts/fraction;
- cell area;
- nucleus area;
- nucleus-to-cell-area ratio;
- local transcript density;
- mitochondrial/ribosomal metrics only if biologically meaningful in the Xenium panel.

### C2. Robust batch-aware outlier detection

Use data-driven robust thresholds, not arbitrary hard-coded global cutoffs.

Primary rule:
- work on log-transformed positive QC variables;
- determine robust center and MAD within batch;
- flag extreme low-count / low-gene cells at approximately median - 3 MAD;
- flag extreme size/area objects at approximately ±4 MAD where the metric supports two-sided QC.

If distributions are clearly multimodal, document this and use a mixture/model-based or valley-based threshold only if justified by the observed distribution.

Do not exclude a cell solely because neutrophils naturally have low RNA content.

Create explicit flags rather than immediately deleting cells:

```text
qc_low_counts
qc_low_genes
qc_extreme_area
qc_high_control_fraction
qc_other
```

Save:

```text
metrics/transcript_qc_by_cell.parquet
metrics/transcript_qc_by_batch.csv
figures/Fig2_transcript_QC.pdf
```

---

## 6. Phase D — Re-annotate biological cell type independently of old labels

The primary training taxonomy remains exactly seven classes:

```text
0 Endothelial
1 Mesenchymal
2 Myeloid
3 Neutrophil
4 Plasma cell
5 T and B
6 Tumor
```

Add non-training states:

```text
Uncertain
Artifact_or_no_nucleus
Mixed_lineage
Low_quality
```

Do not force every Xenium object into one of the seven training classes.

### D1. Candidate marker sets

Use only genes actually present in the Xenium panel.

Candidate marker pools include:

#### Endothelial
PECAM1, VWF, EMCN, KDR, ENG, PLVAP, RAMP2, CA4, ESAM

#### Mesenchymal
COL1A1, COL1A2, COL3A1, COL5A1, COL6A1, DCN, LUM, FAP, THY1, PDGFRA, PDGFRB, ACTA2, TAGLN, RGS5

#### Myeloid
LST1, TYROBP, FCER1G, CTSS, CTSB, CTSD, C1QA, C1QB, C1QC, CD68, CD163, APOE, SPP1, IL1B

#### Neutrophil
FCGR3B, CSF3R, CXCR2, FPR1, SELL, S100A8, S100A9, MNDA, NAMPT

CXCR4 and VEGFA may support a neutrophil subtype but must not be used alone to define Neutrophil identity.

#### Plasma cell
JCHAIN, MZB1, XBP1, SDC1, IGKC, CD79A, DERL3, SSR4

#### T and B
T-cell markers: CD3D, CD3E, TRBC1, TRBC2, LCK, IL7R, CCL5
B-cell markers: MS4A1, CD79A, CD79B, CD19, CD22, CD37, CD74, HLA-DRA, CD83

#### Tumor / epithelial-hepatocytic
EPCAM, KRT8, KRT18, KRT19, KRT7, ALB, APOA1, APOB, GPC3, AFP, KRT17 and other HCC/hepatocyte epithelial genes actually represented in the panel.

The implementation must:
- intersect each marker list with `adata.var_names`;
- record present and absent markers;
- not fabricate genes absent from the panel.

Save:

```text
metrics/marker_availability.csv
```

### D2. Unsupervised structure

Use an appropriate expression representation to construct or reuse:

- PCA / Harmony coordinates;
- neighbors;
- Leiden clusters at several moderate resolutions, e.g. approximately 0.3, 0.5, 0.8, 1.0.

Do not pick a resolution based on CellViT performance.

Assess cluster stability and marker coherence.

### D3. Cluster-level annotation

For each cluster calculate:

- differential-expression markers;
- seven lineage scores;
- fraction of cells expressing lineage-specific markers;
- competing lineage scores;
- original annotation composition for audit only.

Assign a seven-class cluster identity only when evidence is coherent.

Clusters with conflicting or weak evidence must remain `Uncertain` or `Mixed_lineage`.

### D4. Cell-level annotation confidence

For every cell calculate:

- top lineage score;
- second-best lineage score;
- top-minus-second margin;
- cluster-cell agreement;
- number of supportive lineage markers;
- number/strength of conflicting-lineage markers;
- original label agreement/disagreement, audit only.

Create:

```text
new_celltype
annotation_confidence_score
annotation_confidence_rank
cluster_cell_agreement
lineage_score_margin
```

Do not use old labels to force the new label.

---

## 7. Phase E — Neutrophil-specific re-annotation safeguards

This phase is mandatory.

The goal is to prevent RNA debris or necrosis-associated residual transcripts from being treated as intact Neutrophil cells.

A cell may enter the high-confidence Neutrophil training set only if all applicable evidence supports it.

### E1. Transcript evidence

Require:
- Neutrophil is the top lineage identity;
- clear margin over Myeloid and T/B;
- support from multiple available neutrophil markers where panel coverage permits;
- not merely S100A8/S100A9 alone if specific neutrophil markers such as FCGR3B/CSF3R/CXCR2/FPR1 are available.

Report how many putative neutrophils are supported by:
- specific neutrophil markers;
- only generic inflammatory markers;
- mixed Myeloid/Neutrophil signal.

### E2. Spatial / H&E nuclear evidence

A high-confidence Neutrophil must have evidence for a real nucleus near the registered Xenium position.

Do not require a single connected nuclear component because neutrophils can be multilobulated.

Allow a compact group of multiple hematoxylin-positive nuclear lobes.

Reject or flag patterns more consistent with:
- no hematoxylin-positive nucleus;
- only diffuse weak hematoxylin signal;
- many tiny dispersed fragments;
- necrotic nuclear debris;
- large registration displacement.

Create:

```text
neutrophil_transcript_support
neutrophil_nuclear_integrity
neutrophil_debris_suspect
```

---

## 8. Phase F — H&E nuclear-integrity QC for all cells

Use the registered H&E only for cell-quality / nuclear-integrity evidence.

Do not use CellViT classifier predictions.

### F1. H&E patch extraction

Extract a physically consistent local patch around every mapped Xenium centroid.

Determine crop size in microns from actual OME metadata.

Generate at least two scales:
- nucleus/local scale, approximately 15–25 µm field;
- context scale, approximately 40–60 µm field.

Do not assume the remembered pixel size; verify the OME metadata.

### F2. Classical hematoxylin evidence

Use color deconvolution or another transparent non-CellViT image-processing method to quantify:

- hematoxylin-positive fraction;
- total nuclear-component area;
- largest-component area;
- number of nuclear components;
- largest-component / total-component ratio;
- distance from nuclear evidence to expected centroid;
- fragmentation score;
- local texture/entropy if useful.

The purpose is not to classify cell type from H&E.

The purpose is only to identify whether an intact/plausible nucleus is present.

### F3. Conservative morphology flags

Create:

```text
he_nucleus_present
he_low_nuclear_signal
he_fragmented_debris
he_registration_suspect
he_nuclear_integrity_score
```

Use robust batch-aware distributions and obvious geometric failures.

For Neutrophils:
- allow several nearby lobes;
- do not reject solely because more than one nuclear component is present.

Do not over-filter rare or naturally small nuclei.

---

## 9. Phase G — Build final quality/confidence tiers

Do not create a single binary keep/drop decision immediately.

Create at least three derived statuses:

### HQ_CORE

Must satisfy:
- transcript QC pass;
- assigned to one of the seven classes;
- cluster-cell annotation agreement;
- high annotation-confidence margin;
- no strong mixed-lineage conflict;
- valid registration;
- plausible H&E nuclear evidence.

For Neutrophil, additionally require the safeguards in Phase E.

### HQ_EXTENDED

Must satisfy:
- transcript QC pass;
- seven-class label;
- moderate or high annotation confidence;
- valid registration;
- H&E nucleus present or at least no strong evidence of debris.

### EXCLUDE_OR_REVIEW

Includes:
- Low_quality;
- Uncertain;
- Mixed_lineage;
- Artifact_or_no_nucleus;
- debris-suspect neutrophils;
- registration-suspect cells.

Use within-class and within-batch confidence ranking where necessary so one batch does not dominate selection.

Do not allow a high-quality tier to eliminate an entire biological class or batch without explicit documentation.

Save:

```text
metrics/xenium_v2_cell_annotations.parquet
metrics/xenium_v2_cell_annotations.csv.gz
```

The table must include:
- original label;
- new label;
- QC flags;
- annotation scores;
- H&E integrity metrics;
- final quality tier;
- batch;
- spatial coordinates;
- transformed H&E coordinates.

---

## 10. Phase H — Label-change audit

Quantify:

- original label counts;
- new label counts;
- old→new confusion matrix;
- fraction reannotated per original class;
- fraction excluded per original class;
- fraction excluded per batch;
- original Neutrophil → new label;
- original Neutrophil → Low_quality / Artifact / Uncertain;
- cells newly annotated as Neutrophil;
- neutrophil debris-suspect fraction.

Required outputs:

```text
metrics/old_vs_new_label_confusion.csv
metrics/quality_retention_by_class.csv
metrics/quality_retention_by_batch.csv
metrics/neutrophil_reannotation_summary.csv
```

Required figures:

```text
Fig3_old_vs_new_annotation.pdf
Fig4_quality_retention_by_class.pdf
Fig5_neutrophil_reannotation.pdf
```

---

## 11. Phase I — Visual review montages

Generate reviewable H&E montages using registered crops.

For each of the seven classes create representative:
- HQ_CORE;
- HQ_EXTENDED;
- EXCLUDE_OR_REVIEW.

For Neutrophils specifically create larger montages for:
- high-confidence intact neutrophils;
- old-neutrophil labels reclassified to Myeloid;
- old-neutrophil labels excluded for low transcript quality;
- old-neutrophil labels excluded for no visible nucleus;
- debris-suspect/necrosis-associated cases.

Do not use these montages to tune CellViT performance.

Save under:

```text
qc/montages/
figures/Fig6_neutrophil_HE_QC_montage.pdf
```

---

## 12. Phase J — Freeze the Xenium v2 annotation before model training

Before any CellViT retraining:

1. write the complete annotation/QC table;
2. save the derived AnnData:
   ```text
   /data/lf_data/result/task006_xenium_reannotation/adata_xenium_v2_annotated.h5ad
   ```
3. record SHA256;
4. save the final marker definitions and thresholds;
5. save the final quality-tier rules;
6. mark the annotation as frozen.

Do not modify annotation criteria after viewing CellViT CV results.

This avoids model-driven label tuning.

---

## 13. Phase K — Rebuild CellViT training labels

Use the original notebook only as a reference for coordinate transformation, patch construction, batch split and train/test logic.

Create new code; do not modify the source notebook.

Build derivative datasets:

```text
work/CellViT_dataset_v2_CORE
work/CellViT_dataset_v2_EXTENDED
```

Rules:
- reuse H&E image patches if geometrically identical and valid;
- regenerate label CSVs from Xenium v2 annotations;
- preserve seven class IDs;
- exclude non-training states;
- preserve existing batch-based train/test separation;
- preserve the leakage-safe grouped 5-fold definitions for model selection.

Do not use CellViT predictions to choose labels.

Record:
- training cells per class;
- test cells per class;
- cells per batch;
- images with zero labels;
- images with only excluded cells.

---

## 14. Phase L — Controlled retraining benchmark

The first model experiment must isolate the effect of better Xenium labels.

Use the same official CellViT++ SAM-H RAW training stack and fixed recipe used in Tasks 003/005.

Compare:

```text
OLD_LABELS
vs
XENIUM_V2_HQ_EXTENDED
vs
XENIUM_V2_HQ_CORE
```

All must use:
- same SAM-H backbone;
- same fixed classifier recipe;
- same batch-grouped 5-fold structure;
- no test-set tuning.

Primary metrics:
- macro-F1;
- macro-AUPRC;
- balanced accuracy;
- lowest-three-class F1;
- per-class F1/recall/AUPRC;
- Neutrophil precision/recall/F1/AUPRC.

The main scientific question is whether ground-truth reconstruction materially improves classification.

---

## 15. Promotion thresholds

A Xenium-v2 training set becomes eligible for final model training if grouped CV shows:

1. macro-F1 improves by at least +0.03 absolute versus OLD_LABELS;
2. lowest-three-class mean F1 improves by at least +0.03;
3. Neutrophil F1 improves by at least +0.05 OR Neutrophil AUPRC improves by at least +0.05;
4. at least 4/5 folds improve in macro-F1;
5. no previously strong class loses >0.05 F1 without clear balanced benefit;
6. all seven classes remain adequately represented;
7. no batch leakage;
8. annotation rules were frozen before training.

If neither v2 tier satisfies this:
- do not overwrite the production model;
- report whether the remaining limitation is likely H&E representation/detection or annotation uncertainty.

---

## 16. Optional second-stage optimization

Only if a Xenium-v2 label set clearly outperforms OLD_LABELS in the controlled benchmark:

perform a limited classifier-head optimization on the selected v2 dataset.

Use official or previously validated Task 001 settings only.

Do not run a large sweep.

The purpose is to obtain the best practical model only after label quality has been shown to matter.

---

## 17. Final test evaluation

Only if the selected v2 model satisfies the predefined grouped-CV promotion criteria:

- freeze model choices;
- train on all v2 training data;
- run one final evaluation on the frozen v2 test labels.

Report clearly that the ground-truth labels themselves were rebuilt.

Do not compare the new test number directly to old test metrics without noting the changed ground truth.

Report:
- classifier-only paired-cell metrics;
- seven-class macro metrics;
- per-class metrics;
- Neutrophil conditional metrics;
- Neutrophil end-to-end metrics where meaningfully defined.

Do not optimize after the test evaluation.

---

## 18. Required figures

All main analytical figures must be vector PDF with source CSV where applicable.

Create at minimum:

```text
Fig1_registration_QC.pdf
Fig2_transcript_QC.pdf
Fig3_old_vs_new_annotation.pdf
Fig4_quality_retention_by_class.pdf
Fig5_neutrophil_reannotation.pdf
Fig6_neutrophil_HE_QC_montage.pdf
Fig7_marker_scores_by_new_class.pdf
Fig8_embedding_new_annotation.pdf
Fig9_CV_macroF1_old_vs_v2.pdf
Fig10_per_class_F1_old_vs_v2.pdf
Fig11_neutrophil_metrics_old_vs_v2.pdf
```

---

## 19. Required outputs

Create:

```text
/data/lf_data/result/task006_xenium_reannotation/
├── TASK006_REPORT.md
├── adata_xenium_v2_annotated.h5ad
├── config/
├── code/
├── qc/
├── metrics/
├── work/
├── models/
├── figures/
├── figure_data/
└── logs/
```

Required small/summary outputs include:

```text
qc/adata_inventory.json
qc/old_pipeline_audit.md
qc/registration_summary.csv
metrics/marker_availability.csv
metrics/transcript_qc_by_batch.csv
metrics/xenium_v2_cell_annotations.csv.gz
metrics/old_vs_new_label_confusion.csv
metrics/quality_retention_by_class.csv
metrics/quality_retention_by_batch.csv
metrics/neutrophil_reannotation_summary.csv
metrics/cv_summary_old_vs_v2.csv
metrics/cv_per_class_old_vs_v2.csv
```

Large per-cell tables may remain on the server and must not be committed to GitHub.

---

## 20. Required report

Create:

```text
/data/lf_data/result/task006_xenium_reannotation/TASK006_REPORT.md
```

It must explicitly answer:

### A. What was the original Xenium annotation pipeline?

### B. Which original Xenium QC problems were confirmed?

### C. How many cells were reannotated?

### D. How many cells were excluded as low quality / uncertain / no-nucleus / debris?

### E. How many original Neutrophil labels survived as high-confidence intact Neutrophils?

### F. How many old Neutrophil labels were reclassified or excluded, and why?

### G. What markers supported each new seven-class annotation?

### H. How well did the new annotation align with H&E nuclear evidence?

### I. Did Xenium-v2 labels improve grouped-CV CellViT performance?

### J. What happened to Neutrophil F1, recall and AUPRC?

### K. Was a new production model promoted?

### L. What should Task 007 do next?

Do not start Task 007.

---

## 21. GitHub synchronization

After execution:

1. update `tasks/task_006.md` to COMPLETED, PARTIAL or BLOCKED;
2. create `reports/task_006_report.md`;
3. update `PROJECT_STATUS.md`;
4. commit only code, report, configuration and small summary tables/figures;
5. do not commit the OME-TIFF, H5AD, large patch datasets, caches or checkpoints;
6. push to `origin/main`;
7. do not force-push.

---

## 22. Final handoff

Return:

1. original annotation column and workflow;
2. original total cell count;
3. v2 annotated count;
4. HQ_CORE count by class;
5. HQ_EXTENDED count by class;
6. excluded/review count and reasons;
7. old→new label-change rate;
8. original Neutrophil count;
9. high-confidence intact Neutrophil count;
10. neutrophil debris/no-nucleus exclusions;
11. old-label grouped-CV macro-F1;
12. v2 CORE grouped-CV macro-F1;
13. v2 EXTENDED grouped-CV macro-F1;
14. old vs v2 Neutrophil F1/recall/AUPRC;
15. promotion decision;
16. production model path;
17. v2 AnnData path;
18. report path;
19. figure directory;
20. unresolved issues;
21. recommended Task 007 direction.

Do not start Task 007.
