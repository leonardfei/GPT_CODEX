# Task 007 — Xenium 5K Panel-Aware Reannotation and Training-Label QC

## Status
PENDING

## Objective

Rebuild the Xenium annotation using a method appropriate for a targeted ~5,000-gene Xenium panel.

Task 006 must NOT be reused as ground truth because its annotation policy was not panel-aware and caused severe over-filtering:
- old→new label change ~89.7%;
- HQ_CORE retained only ~7.5% of all cells;
- T/B and Neutrophil were disproportionately depleted;
- canonical-marker absence was effectively treated as weak biological evidence even when the gene might not exist in the panel.

Task 007 must correct this methodological problem.

The primary goal is to produce a biologically plausible, panel-aware, auditable Xenium v3 annotation and a high-quality CellViT training-label subset.

DO NOT train CellViT in Task 007.

The annotation must be frozen and reviewed before any model retraining task.

---

## 1. Fixed input files

### Source Xenium AnnData
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

### Historical preprocessing notebook
```text
/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb
```

### Task 006 outputs for audit only
```text
/data/lf_data/result/task006_xenium_reannotation
```

### Task 007 output root
```text
/data/lf_data/result/task007_xenium5k_panelaware
```

Do not modify any source file.

---

## 2. Core methodological principles

### 2.1 Targeted-panel principle

This is a Xenium 5K targeted panel, not whole-transcriptome scRNA-seq.

Therefore:

- absence of a canonical marker that is NOT in the panel is non-informative;
- absence of a marker that IS in the panel is weak negative evidence, not automatic exclusion;
- do not require a fixed number of canonical markers across all cell types;
- do not compare raw mean marker scores from unequal marker lists as if they had the same scale;
- do not force every object into a new class from scratch.

Every gene used for annotation must come from the actual `adata.var_names`.

### 2.2 Original annotation as prior

Use the original biological annotation `obs['cl1']` as a prior, not as unquestioned truth.

The default action should be KEEP unless there is reproducible evidence for:
- a different biological lineage;
- low-quality/non-cell object;
- no-nucleus/debris artifact;
- irreducible ambiguity.

Do not relabel a cell merely because its canonical marker panel is incomplete.

### 2.3 Separate cell quality from cell identity

These are different questions.

#### Cell quality
Is this a real, trainable cell with adequate transcript and nuclear evidence?

#### Cell identity
Which of the seven classes does it belong to?

A cell with weak lineage evidence is not automatically a low-quality cell.

### 2.4 Anti-circularity

Do not use:
- CellViT classifier predictions;
- Task 001–005 classification performance;
- future CellViT CV results

to define Xenium v3 labels.

Task 007 stops before model training.

---

## 3. Seven-class target taxonomy

The eventual CellViT training taxonomy remains:

```text
0 Endothelial
1 Mesenchymal
2 Myeloid
3 Neutrophil
4 Plasma cell
5 T and B
6 Tumor
```

Preserve original finer labels for audit, especially:
- Neutrophil
- Neutrophil_CXCR4

Map them to the seven-class taxonomy only in a separate column.

Non-training/review states are allowed:

```text
Review_uncertain_identity
Review_identity_conflict
Low_quality
Artifact_or_no_nucleus
Debris_necrosis_suspect
```

Do not call a biologically plausible but ambiguous cell `Low_quality`.

---

## 4. Phase A — Actual Xenium 5K panel inventory

Read the exact panel genes from the source AnnData.

Create:

```text
metrics/panel_gene_inventory.csv
metrics/panel_annotation_coverage.csv
```

Record:
- exact number of genes/features;
- all `var_names`;
- gene symbol field if separate;
- negative/control probe features;
- duplicated gene symbols if any;
- which genes are protein-coding vs controls where metadata permits.

For each original class, list:
- panel genes enriched in that class;
- panel genes depleted in that class;
- genes reproducibly informative across batches.

Explicitly verify whether commonly cited markers such as CD3D are in the panel.

Do not use any canonical marker absent from the panel in scoring or filtering.

---

## 5. Phase B — Audit the original annotation and quality fields

Inspect `obs['cl1']` and all relevant QC/segmentation fields.

Report:
- original class counts;
- original class counts by batch;
- original `Low quality` count;
- nucleus_count distribution by class;
- nucleus_area distribution by class;
- cell_area distribution by class;
- total_counts / n_genes distributions by class and batch;
- control-probe fraction where available.

Preserve:
```text
original_cl1
original_cl1_7class
```

Do not modify `cl1`.

Create:
```text
metrics/original_annotation_summary.csv
metrics/original_qc_by_class.csv
metrics/original_qc_by_batch.csv
```

---

## 6. Phase C — Cell-quality QC independent of cell type

Build a quality layer without using biological marker identity.

### C1. Transcript quality

Use batch-aware robust distributions for:
- total transcripts/counts;
- genes detected;
- control-probe fraction;
- cell area;
- nucleus area;
- nucleus count.

Use robust outlier flags rather than forcing a fixed global threshold.

Do NOT exclude a cell solely because it has relatively low RNA if this is typical for its original class.

Therefore calculate thresholds both:
- within batch;
- within original broad class where sample size is adequate.

A cell should be `Low_quality` only for strong technical evidence, such as:
- extreme low transcript/genes relative to its class/batch;
- extreme control-probe contamination;
- implausible segmentation geometry;
- other clearly technical failure.

### C2. Nuclear quality

Use Xenium segmentation fields first:
- nucleus_count;
- nucleus_area;
- nucleus/cell area relationship.

Do not use H&E as the first-line quality gate for all 990k cells.

Cells with `nucleus_count == 0`, implausible nucleus area, or strong segmentation anomaly are H&E-review candidates.

Save per-cell flags:
```text
qc_transcript_pass
qc_segmentation_pass
qc_nucleus_present_xenium
qc_low_quality_reason
```

---

## 7. Phase D — Data-driven panel-aware lineage signatures

Do NOT start from an external canonical marker list.

Derive lineage-discriminating genes from this Xenium 5K dataset itself.

### D1. Candidate anchor cells

Create conservative anchor cells from the original annotation using:
- original class not Low quality;
- transcript QC pass;
- segmentation/nucleus QC pass;
- no obvious registration failure;
- membership in a local/cluster neighborhood substantially enriched for the same original class;
- representation across multiple batches where possible.

Do not require canonical marker presence for anchor definition.

Do not use Task 006 v2 labels.

### D2. Batch-aware differential evidence

For each of the seven classes:

compare anchor cells of that class against the other classes using only genes in the panel.

Prefer batch-aware/reproducible evidence:
- compute class-vs-rest effect sizes within each batch;
- combine across batches using median/robust summary;
- require directionally consistent enrichment across multiple batches.

Record for each gene:
- class;
- median log fold-change/effect size;
- detection fraction difference;
- number/fraction of batches with consistent direction;
- ranking.

Create:
```text
metrics/panel_data_driven_markers.csv
```

These data-driven signatures, not external marker completeness, are the primary transcriptional evidence.

### D3. Signature size

Do not force every class to use the same number of genes if informative genes differ.

However, prevent a class with many selected genes from dominating the score by using standardized/weighted evidence rather than raw mean expression.

---

## 8. Phase E — Cross-fitted class-consistency model

Build a transparent model to evaluate whether each cell is transcriptionally consistent with its original label.

The model is not the final ground truth.

Use leave-one-batch-out or grouped-by-batch cross-fitting.

For each held-out batch:
- build class reference profiles/signatures only from anchor cells in other batches;
- predict class consistency in the held-out batch;
- never train and evaluate the same batch together.

Preferred transparent approaches:
- correlation/cosine similarity to batch-excluded class pseudobulk profiles;
- shrinkage nearest-centroid;
- multinomial regularized logistic regression if needed.

Avoid complex deep-learning label transfer in this task unless clearly justified.

For every cell record:
```text
crossfit_top_class
crossfit_top_score
crossfit_second_score
crossfit_margin
crossfit_original_class_score
crossfit_original_class_rank
```

Calibrate score/margin thresholds from held-out anchor performance rather than arbitrary global quantiles.

Target a high estimated precision for automatic relabeling, preferably >=95% on cross-fitted anchors when feasible.

---

## 9. Phase F — Neighborhood/cluster evidence

Use existing Harmony/neighbor structure or rebuild a panel-appropriate neighborhood graph if needed.

For every cell calculate:
- neighborhood composition of original classes;
- neighborhood composition after excluding Low-quality cells;
- Leiden/cluster original-class composition;
- entropy/purity.

These are supporting evidence only.

Do not require exact agreement with the cluster majority for every cell.

Record:
```text
neighbor_original_label_fraction
neighbor_top_class
neighbor_entropy
cluster_original_label_fraction
cluster_top_class
```

---

## 10. Phase G — Conservative reannotation policy

The new biological label must use a conservative KEEP / RELABEL / REVIEW strategy.

### G1. KEEP original label

KEEP when:
- original label is compatible with cross-fitted transcriptional evidence; OR
- cross-fitted model is uncertain but does not strongly contradict the original label; AND
- local neighborhood does not strongly contradict the original label.

### G2. Automatic RELABEL

Relabel only when all are true:
1. cross-fitted alternative class passes a high-confidence, precision-calibrated threshold;
2. alternative margin is strong;
3. neighborhood/cluster evidence supports the alternative;
4. original class evidence is weak;
5. this rule was calibrated without using CellViT.

Automatic relabeling should be rare and high-specificity.

### G3. REVIEW

Use review states when:
- original and cross-fitted identities disagree but evidence is not decisive;
- two lineages remain plausible;
- neighborhood is heterogeneous;
- the cell lies near a biological transition;
- the panel lacks enough information to distinguish the alternatives.

Do not convert these cells to Low_quality unless technical QC also fails.

---

## 11. Phase H — Neutrophil-specific policy

This is the highest-priority class.

### H1. Preserve original subtype information

Keep separate source fields:
```text
Neutrophil
Neutrophil_CXCR4
```

For seven-class training both map to Neutrophil, but the subtype must remain available for audit.

### H2. Panel-aware neutrophil signature

Derive the neutrophil-vs-Myeloid/T-B signature from actual panel genes and high-quality original anchor cells.

Do not require any marker that is absent from the panel.

Do not require >=2 canonical markers.

Report the actual genes that distinguish:
- Neutrophil vs Myeloid;
- Neutrophil vs T/B;
- Neutrophil_CXCR4 vs other Neutrophil if separable.

### H3. Full-resolution H&E audit for neutrophils

For ALL original Neutrophil / Neutrophil_CXCR4 cells and all newly proposed neutrophils:

extract registered full-resolution H&E crops.

Because OME physical-size metadata conflicts with the historical notebook geometry:
- perform this QC in H&E pixel coordinates;
- do not make micron-scale claims in Task 007;
- use the historical inverse registration as the coordinate mapping only after overlay QC;
- record crop size in pixels.

Use at least two pixel-space crops, e.g. small nuclear-scale and larger local-context windows, selected from the historical 256-pixel patch geometry.

Assess:
- visible hematoxylin-positive nuclear material;
- compact intact nucleus/lobes;
- multilobulated morphology allowed;
- diffuse debris;
- multiple scattered fragments;
- necrotic background;
- registration displacement.

Create:
```text
he_neutrophil_nucleus_status
he_neutrophil_debris_status
he_neutrophil_registration_status
```

Possible statuses:
```text
intact_or_plausible
debris_suspect
no_visible_nucleus
registration_uncertain
review
```

Do not use a level-3 thumbnail as the final neutrophil nuclear-integrity decision.

### H4. Neutrophil training eligibility

A biological Neutrophil label and a CellViT-training eligibility decision are separate.

A cell can remain biologically labelled Neutrophil but be excluded from H&E training because:
- no visible nucleus;
- debris/necrosis;
- registration uncertainty.

Create:
```text
biological_label_v3
trainable_neutrophil
neutrophil_exclusion_reason
```

---

## 12. Phase I — H&E review for other classes

For other six classes:

do not run expensive full-resolution morphology analysis for every cell initially.

Use:
- Xenium segmentation/nucleus QC for primary filtering;
- targeted H&E review for:
  - nucleus_count == 0;
  - extreme nucleus/cell geometry;
  - cells proposed for automatic relabeling;
  - a stratified QC sample of retained training cells from every class and batch.

Generate montages to confirm that retained labels correspond to plausible nuclei.

---

## 13. Phase J — Final v3 label states

Every cell must receive independent fields for biological identity and training eligibility.

### Biological identity
```text
biological_label_v3
label_action = KEEP / RELABEL / REVIEW
label_confidence = high / moderate / low
```

### Quality
```text
cell_quality_status
trainable_cell
training_exclusion_reason
```

Do not encode identity uncertainty as technical low quality.

### Training tiers

Create:

#### TRAIN_CORE
- seven-class biological label;
- KEEP or high-confidence RELABEL;
- technical QC pass;
- segmentation/nuclear QC pass;
- no registration concern;
- for neutrophils, full-resolution H&E nucleus/debris QC pass.

#### TRAIN_EXTENDED
- seven-class biological label;
- KEEP / moderate-confidence compatible label;
- technical QC pass;
- no strong artifact evidence;
- for neutrophils, nucleus status not debris/no-nucleus.

#### REVIEW
- identity conflict;
- technical ambiguity;
- registration uncertainty;
- debris/no-nucleus;
- other non-trainable cases.

Do not impose a fixed target retention fraction.

---

## 14. Safety rails against another over-filtering failure

Before freezing v3 annotations, automatically flag the task for review if ANY occurs:

1. >30% of non-Low-quality original cells are biologically relabeled;
2. >40% of a major original class is sent to identity REVIEW solely because of transcriptional ambiguity;
3. TRAIN_EXTENDED retains <50% of technically valid cells for any major class, except when a documented class-specific artifact explains it;
4. T/B or Neutrophil is depleted >50% by annotation evidence alone rather than technical/no-nucleus QC;
5. any batch loses >50% more training cells than the median batch after adjusting for original class composition.

If triggered:
- do not call the annotation finalized;
- mark Task 007 PARTIAL;
- generate the audit tables/montages;
- do not silently tighten or loosen rules after seeing retention.

---

## 15. Required manual-review material

Generate stratified H&E + Xenium review montages.

### For each of seven classes
At least representative examples of:
- KEEP high confidence;
- RELABEL proposed;
- REVIEW identity conflict;
- technical exclusion.

### Neutrophil
Generate larger dedicated review panels:
- original Neutrophil KEEP;
- original Neutrophil_CXCR4 KEEP;
- old Neutrophil → Myeloid proposed;
- old Neutrophil → T/B proposed;
- no-visible-nucleus;
- debris/necrosis suspect;
- registration uncertain;
- newly proposed Neutrophil.

Each panel should show:
- H&E crop;
- original label;
- proposed v3 label;
- batch;
- Xenium counts/genes;
- nucleus_count/nucleus_area;
- top data-driven class scores;
- relevant panel-aware signature genes actually measured.

Save:
```text
qc/review_montages/
figures/Fig_neutrophil_review_montage.pdf
```

---

## 16. Freeze annotation before any CellViT training

Task 007 must NOT train CellViT.

If the annotation passes the safety rails:

save:
```text
/data/lf_data/result/task007_xenium5k_panelaware/adata_xenium_v3_panelaware.h5ad
```

and:
```text
metrics/xenium_v3_annotations.csv.gz
metrics/training_core_cells.csv.gz
metrics/training_extended_cells.csv.gz
```

Record SHA256 for the derived AnnData and annotation tables.

If safety rails fail, still save the provisional files but name/status them PROVISIONAL and mark the task PARTIAL.

---

## 17. Required quantitative outputs

Create:

```text
metrics/panel_gene_inventory.csv
metrics/panel_annotation_coverage.csv
metrics/original_annotation_summary.csv
metrics/original_qc_by_class.csv
metrics/original_qc_by_batch.csv
metrics/panel_data_driven_markers.csv
metrics/crossfit_class_consistency.csv.gz
metrics/v3_label_transition_matrix.csv
metrics/v3_label_action_summary.csv
metrics/v3_quality_summary.csv
metrics/v3_training_retention_by_class.csv
metrics/v3_training_retention_by_batch.csv
metrics/neutrophil_v3_summary.csv
metrics/neutrophil_panel_signature.csv
metrics/neutrophil_he_qc_summary.csv
```

---

## 18. Required figures

Generate editable/reviewable figures:

```text
Fig1_panel_gene_coverage.pdf
Fig2_original_qc_by_class.pdf
Fig3_data_driven_class_signatures.pdf
Fig4_crossfit_confusion_anchor_cells.pdf
Fig5_original_to_v3_transition.pdf
Fig6_training_retention_by_class.pdf
Fig7_training_retention_by_batch.pdf
Fig8_neutrophil_panel_signature.pdf
Fig9_neutrophil_HE_QC_summary.pdf
Fig10_neutrophil_review_montage.pdf
```

Do not generate CellViT performance figures in Task 007.

---

## 19. Required report

Create:
```text
/data/lf_data/result/task007_xenium5k_panelaware/TASK007_REPORT.md
```

It must explicitly answer:

### A. What genes are actually present in the Xenium 5K panel?

### B. Which genes in the panel are empirically most informative for each of the seven classes?

### C. How well does the original `cl1` annotation agree with cross-fitted panel-aware transcriptional evidence?

### D. What fraction of cells are KEEP, RELABEL, REVIEW, Low_quality, or artifact?

### E. Which classes show the most disagreement?

### F. What are the actual panel-aware Neutrophil-vs-Myeloid distinguishing genes?

### G. How many original Neutrophil / Neutrophil_CXCR4 cells remain biologically consistent with Neutrophil?

### H. How many are excluded from H&E training specifically because of no nucleus / debris / registration concerns?

### I. How many TRAIN_CORE and TRAIN_EXTENDED cells remain for each class and batch?

### J. Did any safety rail trigger?

### K. Is the v3 annotation FINAL or PROVISIONAL?

### L. Is it appropriate to proceed to CellViT retraining?

Do not start retraining.

---

## 20. GitHub workflow

After execution:

1. update `tasks/task_007.md` to COMPLETED or PARTIAL;
2. create `reports/task_007_report.md`;
3. update `PROJECT_STATUS.md`;
4. commit scripts/config/report/small summary outputs only;
5. do not commit H5AD, TIFF, large per-cell tables, patches or image montages if too large;
6. push to `origin/main`;
7. do not force-push.

---

## 21. Final handoff

Return:

1. exact panel gene count;
2. whether CD3D is present;
3. original class counts;
4. KEEP/RELABEL/REVIEW counts;
5. technical Low_quality/artifact counts;
6. TRAIN_CORE counts by class;
7. TRAIN_EXTENDED counts by class;
8. original Neutrophil count;
9. original Neutrophil_CXCR4 count;
10. biologically retained Neutrophil count;
11. neutrophil no-nucleus count;
12. neutrophil debris-suspect count;
13. top panel-aware genes distinguishing Neutrophil from Myeloid;
14. old→v3 biological relabel rate;
15. safety-rail status;
16. FINAL vs PROVISIONAL annotation status;
17. derived AnnData path;
18. report path;
19. review montage path;
20. recommendation on whether to start CellViT retraining.

Do not start CellViT retraining.
