# Task 008 — Neutrophil Nucleus-Centered H&E QC Recalibration

## Status
PARTIAL

## Objective

Resolve the likely over-filtering artifact in Task 007 Neutrophil H&E QC.

Task 007 produced a biologically plausible panel-aware annotation:
- 24,167 original broad Neutrophils;
- 24,107 remained biologically consistent with Neutrophil;
- but only 1,969 were considered trainable because 22,164 were flagged as H&E debris-suspect.

Inspection of the Task 007 implementation showed that debris status was driven by the number and size of hematoxylin-like connected components across the entire 128×128-pixel crop. In immune-rich regions, neighboring nuclei can generate many components even when the target Xenium cell has an intact nucleus.

Task 008 must therefore recalibrate H&E eligibility using TARGET-CENTERED nuclear evidence.

Do not alter the Task 007 biological annotation.
Do not retrain CellViT.
Do not use CellViT predictions to define H&E quality.

---

## 1. Inputs

Source Xenium AnnData:
```text
/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad
```

Source H&E:
```text
/data/lf_data/xenium_data/ID0060276.ome.tif
```

Registration matrix:
```text
/data/lf_data/xenium_data/matrix.csv
```

Task 007 v3 annotation:
```text
/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz
```

Task 007 derived AnnData:
```text
/data/lf_data/result/task007_xenium5k_panelaware/adata_xenium_v3_panelaware.h5ad
```

Task 007 neutrophil crops / QC:
```text
/data/lf_data/result/task007_xenium5k_panelaware/work/neutrophil_crops
/data/lf_data/result/task007_xenium5k_panelaware/metrics/neutrophil_he_qc_summary.csv
```

Task 008 output root:
```text
/data/lf_data/result/task008_neutrophil_he_recalibration
```

Do not modify any source or Task 007 file.

---

## 2. Fixed biological identity

Task 008 is NOT a reannotation task.

Freeze the Task 007 panel-aware biological identity.

For Neutrophils:
- preserve original `Neutrophil`;
- preserve original `Neutrophil_CXCR4`;
- preserve broad seven-class mapping to `Neutrophil`;
- preserve Task 007 KEEP/REVIEW identity decisions.

Task 008 changes only:
```text
H&E nuclear-quality status
H&E training eligibility
```

It must never change:
```text
biological_label_v3
original_cl1
cross-fitted transcriptional identity
```

---

## 3. Reproduce and audit the Task 007 failure mode

Before replacing the rule:

reproduce Task 007 H&E statuses for all 24,167 broad Neutrophil candidates.

Quantify by:
- original subtype;
- batch;
- total counts;
- genes detected;
- Xenium nucleus_count;
- Xenium nucleus_area;
- H&E signal fraction;
- full-patch component count;
- largest component.

Create:
```text
metrics/task007_he_status_audit.csv
```

Test whether the Task 007 debris flag is strongly associated with:
- high local cell density;
- high full-patch component count;
- batch;
- neutrophil subtype;
rather than target-centered nuclear absence.

This is a diagnostic analysis only.

---

## 4. Registration-tolerance calibration using non-Neutrophil reference cells

The expected Xenium centroid may not lie exactly at the H&E nuclear center.

Estimate the empirical registration / centroid-offset tolerance before classifying Neutrophils.

### 4.1 Reference cells

Select a stratified sample of technically valid Task 007 KEEP cells from:
- T and B;
- Myeloid;
- Endothelial;
- Mesenchymal;
- Plasma cell;
- Tumor.

Prefer cells with:
- Xenium nucleus_count >= 1;
- nucleus_area > 0;
- cell_quality_status == Pass;
- in-bounds registration.

Sample across all batches.

Target approximately 1,000–2,000 cells per broad class if available.

### 4.2 H&E target-centered nuclear localization

Use level-0 H&E.

For each reference cell:
- extract 128×128 context crop centered on registered Xenium coordinate;
- perform transparent H&E color deconvolution / hematoxylin optical-density estimation;
- identify candidate nuclear components;
- calculate distance from crop center to the nearest plausible nuclear component;
- calculate whether the component intersects concentric center radii.

Record:
```text
nearest_nuclear_component_distance_px
nearest_component_area_px
nearest_component_mean_H_OD
center_signal_fraction
center_component_count
```

### 4.3 Empirical tolerance

From technically valid reference cells, derive empirical center-to-nucleus tolerance.

Report:
- median;
- 75th;
- 90th;
- 95th;
- 97.5th percentile

of nearest-component distance by class and batch.

Select the Task 008 primary center tolerance from the empirical distribution, preferably approximately the 95th percentile of valid reference cells, with an upper sanity cap documented from pixel-space geometry.

Do not invent a micron conversion.

Save:
```text
metrics/registration_tolerance_reference.csv
metrics/registration_tolerance_summary.csv
figures/Fig1_center_offset_calibration.pdf
```

---

## 5. H&E segmentation method

Use a transparent stain-aware method.

Preferred:
- RGB optical density;
- H&E color deconvolution using a documented hematoxylin vector;
- adaptive/local thresholding on hematoxylin OD where necessary.

Do not use the Task 007 rule:
```text
full 128×128 crop component count >= threshold => debris
```

Do not use CellViT.

For every component calculate:
- area;
- centroid;
- distance to Xenium center;
- perimeter;
- compactness/circularity;
- eccentricity where feasible;
- mean/max hematoxylin OD;
- bounding box;
- component adjacency / inter-component distance.

---

## 6. Multi-scale target-centered regions

For every Neutrophil candidate, use at least:

### Nuclear core
Centered on Xenium coordinate; approximate 32×32 or empirically justified equivalent.

### Local cell window
Approximate 64×64.

### Context window
128×128.

The context window is used only for:
- neighboring-cell density;
- necrotic background;
- tissue context.

The context-window component count must NOT directly define the target cell as debris.

---

## 7. Multilobulated Neutrophil logic

A neutrophil nucleus may contain multiple lobes.

Therefore, target nuclear evidence may consist of:
- one plausible central component; OR
- 2–5 nearby hematoxylin-positive components/lobes that form a compact local group around the expected centroid.

Construct a center-associated component group using:
- empirical center-distance tolerance;
- component-to-component proximity;
- total group area;
- group centroid distance to Xenium centroid.

Do not penalize a cell merely because multiple lobes are present.

Record:
```text
center_associated_n_components
center_group_total_area_px
center_group_centroid_distance_px
center_group_span_px
center_group_H_OD
```

---

## 8. Distinguish four different failure modes

Assign H&E status using separate evidence fields, not one debris flag.

### A. TARGET_NUCLEUS_PRESENT
Plausible center-associated nuclear component/group exists.

### B. NO_TARGET_NUCLEUS
No plausible nuclear component within empirically calibrated centroid tolerance.

### C. FRAGMENTED_TARGET_SUSPECT
There is nuclear signal near the target centroid, but the center-associated group is composed almost entirely of very small dispersed fragments and lacks plausible total nuclear area.

This must be based on CENTER-ASSOCIATED components, not full-crop component count.

### D. REGISTRATION_UNCERTAIN
The crop is out of bounds or target-centered nuclear evidence is inconsistent with the empirically calibrated registration model.

### E. REVIEW
Evidence is intermediate/ambiguous.

Use these exact concepts even if final column names differ.

---

## 9. Integrate Xenium segmentation evidence

Use Xenium nucleus information as an independent supporting layer:
- nucleus_count;
- nucleus_area;
- cell_area.

Important:
- H&E and Xenium segmentation disagreement should produce REVIEW unless one modality gives extremely clear artifact evidence;
- `nucleus_count == 0` does not automatically prove debris;
- H&E center-nucleus present + Xenium nucleus_count == 0 should be reported as segmentation disagreement.

Create:
```text
xenium_he_nucleus_agreement
```

Categories:
```text
agree_present
agree_absent
HE_present_Xenium_absent
HE_absent_Xenium_present
ambiguous
```

---

## 10. Necrosis/debris context score

The user's biological concern is specifically that neutrophil RNA remnants in necrotic tissue may be assigned to objects without intact nuclei.

Therefore compute a separate CONTEXT-level necrosis/debris score from the 128×128 crop.

Possible transparent features:
- low hematoxylin organization;
- eosinophilic homogeneous background;
- many tiny nuclear fragments;
- loss of intact nuclei;
- local fragmentation density.

This context score may SUPPORT exclusion but must not override a clear target-centered intact nucleus.

Create:
```text
context_necrosis_score
context_fragment_density
context_intact_nuclei_density
```

---

## 11. Initial rule must be conservative

Primary Task 008 output must prioritize avoiding false exclusion.

Recommended eligibility logic:

### TRAINABLE_CORE
- biological Neutrophil from frozen Task 007;
- technical transcript QC pass;
- in-bounds registration;
- TARGET_NUCLEUS_PRESENT;
- no strong center-associated fragmentation evidence.

### TRAINABLE_EXTENDED
- biological Neutrophil;
- technical QC pass;
- in-bounds registration;
- TARGET_NUCLEUS_PRESENT OR REVIEW with plausible nuclear signal;
- not NO_TARGET_NUCLEUS;
- not strong FRAGMENTED_TARGET_SUSPECT.

### EXCLUDE_HARD
Only:
- clear NO_TARGET_NUCLEUS with supporting evidence; OR
- clear center-associated fragmented target + high necrosis context; OR
- registration unusable.

### MANUAL_REVIEW
All intermediate conflicts.

Do not turn MANUAL_REVIEW cells into hard exclusions automatically.

---

## 12. Compare Task 007 vs Task 008 statuses

Create a transition table:

```text
Task007 debris_suspect
→ Task008 TARGET_NUCLEUS_PRESENT / NO_TARGET_NUCLEUS / FRAGMENTED / REVIEW
```

Key question:

How many of the 22,164 Task 007 debris-suspect cells actually have a plausible nucleus at the target centroid?

Required:
```text
metrics/task007_to_task008_transition.csv
```

---

## 13. Manual review panels

This is mandatory before declaring Task 008 eligibility final.

Generate stratified full-resolution review montages.

At least 100 examples per important category where available:

1. Task007 debris → Task008 TARGET_NUCLEUS_PRESENT
2. Task007 debris → Task008 NO_TARGET_NUCLEUS
3. Task007 debris → Task008 FRAGMENTED_TARGET_SUSPECT
4. Task007 debris → Task008 REVIEW
5. Task007 intact → Task008 intact
6. Neutrophil_CXCR4 representative cells
7. Conventional Neutrophil representative cells
8. high-necrosis-context cells
9. low-necrosis-context cells

Each tile should display:
- original H&E context crop;
- zoomed central crop;
- Xenium centroid marker;
- center tolerance circle;
- detected center-associated component boundaries if feasible;
- original subtype;
- batch;
- nucleus_count;
- nucleus_area;
- total_counts;
- Task007 status;
- Task008 status.

Save:
```text
qc/review_montages/
figures/Fig7_task008_manual_review.pdf
```

Do not hide borderline examples.

---

## 14. Stability / sensitivity analysis

Evaluate whether status is stable under modest changes to:
- H&E threshold;
- center tolerance radius;
- minimum component area;
- lobe-group distance.

Use a small predefined grid around the empirically calibrated primary parameters.

For every candidate report:
```text
status_stability_fraction
```

Cells whose status changes frequently must be REVIEW rather than hard excluded.

Create:
```text
metrics/parameter_sensitivity.csv
```

---

## 15. Safety rails

Task 008 must be marked PARTIAL / REVIEW_REQUIRED if any occur:

1. >50% of biologically retained Neutrophils are still hard-excluded;
2. >25% of reference non-Neutrophil cells with Xenium nucleus evidence are classified as having no target nucleus;
3. >20 percentage-point difference in hard-exclusion rate between major batches without a documented morphology reason;
4. Neutrophil_CXCR4 hard-exclusion differs from conventional Neutrophil by >20 percentage points without clear evidence;
5. primary conclusions are highly parameter-sensitive.

Do not create an exception simply to bypass a triggered rail.

---

## 16. Outputs

Output root:
```text
/data/lf_data/result/task008_neutrophil_he_recalibration/
```

Required:
```text
TASK008_REPORT.md
metrics/reference_cell_center_qc.csv.gz
metrics/registration_tolerance_reference.csv
metrics/registration_tolerance_summary.csv
metrics/neutrophil_centered_he_qc.csv.gz
metrics/neutrophil_status_summary.csv
metrics/xenium_he_nucleus_agreement.csv
metrics/task007_to_task008_transition.csv
metrics/neutrophil_retention_by_subtype.csv
metrics/neutrophil_retention_by_batch.csv
metrics/parameter_sensitivity.csv
metrics/safety_rail_decision.json
figures/
qc/review_montages/
config/
code/
logs/
```

Do not modify Task 007 AnnData.

If Task 008 passes review, create a separate eligibility table only:
```text
metrics/neutrophil_training_eligibility_task008.csv.gz
```

This table does not alter biological labels.

---

## 17. Required figures

```text
Fig1_center_offset_calibration.pdf
Fig2_task007_failure_mode.pdf
Fig3_target_centered_status_counts.pdf
Fig4_task007_to_task008_transition.pdf
Fig5_retention_by_subtype.pdf
Fig6_retention_by_batch.pdf
Fig7_task008_manual_review.pdf
Fig8_parameter_sensitivity.pdf
```

---

## 18. Report questions

`TASK008_REPORT.md` must answer:

A. Why did the Task 007 crop-level debris rule over-call debris?

B. What is the empirical centroid-to-nucleus offset distribution?

C. What center tolerance was selected and why?

D. Among the 22,164 Task007 debris-suspect cells, how many show a plausible target-centered nucleus?

E. How many are truly no-target-nucleus?

F. How many are center-fragmented suspect?

G. How many remain manual review?

H. What is retention for conventional Neutrophil?

I. What is retention for Neutrophil_CXCR4?

J. Are there batch effects in H&E eligibility?

K. How well do Xenium nucleus fields agree with H&E center evidence?

L. Are results stable to modest parameter changes?

M. Did any safety rail trigger?

N. How many Neutrophils are TRAINABLE_CORE / TRAINABLE_EXTENDED?

O. Is Neutrophil H&E eligibility now reliable enough to proceed to CellViT retraining?

Do not start retraining.

---

## 19. GitHub synchronization

After execution:
1. update `tasks/task_008.md` to COMPLETED/PARTIAL;
2. create `reports/task_008_report.md`;
3. update `PROJECT_STATUS.md`;
4. commit code/report/config/small summary outputs only;
5. do not commit full H&E crops or large per-cell tables;
6. push main;
7. no force push.

---

## 20. Final handoff

Return:
1. reference cells used;
2. empirical median/95th-percentile center offset;
3. selected center tolerance;
4. original broad Neutrophil count;
5. Task007 debris count;
6. Task008 TARGET_NUCLEUS_PRESENT count;
7. NO_TARGET_NUCLEUS count;
8. FRAGMENTED_TARGET_SUSPECT count;
9. REVIEW count;
10. conventional Neutrophil core/extended eligible;
11. CXCR4 Neutrophil core/extended eligible;
12. retention by batch;
13. Xenium/H&E agreement summary;
14. parameter stability;
15. safety-rail status;
16. final eligibility status;
17. eligibility-table path;
18. report path;
19. montage path;
20. recommendation on whether to begin CellViT retraining.

Do not start CellViT retraining.
