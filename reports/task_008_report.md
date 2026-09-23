# Task 008 report — Neutrophil nucleus-centered H&E QC recalibration

Status: PARTIAL / REVIEW_REQUIRED. Task007 biological identity was frozen; only H&E nuclear-quality status and Neutrophil training eligibility were recalibrated. No CellViT predictions were used and no CellViT model was trained.

## Evidence and reproducibility

Inputs were read from the source H5AD, level-0 H&E OME-TIFF, registration matrix, and Task007 annotation listed in `tasks/task_008.md`. Task007 files and source files were not modified. The implementation uses hematoxylin optical-density projection, target-centered components/groups, empirical reference-cell offset calibration, Xenium nucleus agreement, a separate context necrosis score, and a predefined sensitivity grid.

Runtime: Python 3.10.14, NumPy 1.23.5, pandas 1.4.3, h5py 3.9.0, SciPy 1.8.1, pyvips 2.2.3; random seed 42 with deterministic class sampling. Script: [task008_recalibrate.py](../scripts/python/task008_recalibrate.py). Remote command: `python3 /data/lf_data/result/task008_neutrophil_he_recalibration/code/task008_recalibrate.py`.

## Required handoff

1. Reference cells used: **9,000**, 1,500 per class across T and B, Myeloid, Endothelial, Mesenchymal, Plasma cell, and Tumor, stratified over all batches.
2. Empirical center offset: median **8.73 px**; 95th percentile **21.07 px**.
3. Selected center tolerance: **21.07 px**, capped at 48 px and determined from the empirical 95th percentile.
4. Original broad Neutrophil count: **24,167**.
5. Task007 debris-suspect count: **22,164**.
6. Task008 target-centered evidence `TARGET_NUCLEUS_PRESENT`: **22,630**.
7. Final `NO_TARGET_NUCLEUS`: **13**.
8. Final `FRAGMENTED_TARGET_SUSPECT`: **11**.
9. Final `MANUAL_REVIEW`: **1,736**.
10. Conventional Neutrophil: provisional TRAINABLE_CORE **12,903**, TRAINABLE_EXTENDED **13,208** of 14,647.
11. Neutrophil_CXCR4: provisional TRAINABLE_CORE **8,736**, TRAINABLE_EXTENDED **8,899** of 9,520.
12. Batch extended retention: s01A 87.18%; s01B 94.90%; s02A 72.91%; s02B 90.91%; s03A 93.51%; s03B 95.77%; s04B 95.76%; s05A 95.03%; s05B 92.59%; s06A 90.91%; s100 91.66%; s11 93.62%; s22 83.82%; s33 84.82%; s93 93.39%; s98 83.02%.
13. Xenium/H&E agreement: agree_present 22,407; HE_absent_Xenium_present 1,513; HE_present_Xenium_absent 223; agree_absent 24. Conflicts are assigned MANUAL_REVIEW.
14. Parameter stability: mean status stability **97.48%**; 94.77% of cells had stability ≥0.80. The sensitivity grid was one-at-a-time perturbation of H&E threshold, center tolerance, minimum component area, and lobe-group distance.
15. Safety rails: **no computational safety rail triggered**. Hard-exclusion rate 0.099%; reference no-target rate 0%; batch hard-exclusion spread 0.338 percentage points; subtype spread 0.008 percentage points.
16. Final eligibility status: **PARTIAL / REVIEW_REQUIRED**, because manual review is required before declaring the table final.
17. Eligibility table: `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`.
18. Report: `/data/lf_data/result/task008_neutrophil_he_recalibration/TASK008_REPORT.md`.
19. Manual-review montage: `/data/lf_data/result/task008_neutrophil_he_recalibration/figures/Fig7_task008_manual_review.pdf` and `/data/lf_data/result/task008_neutrophil_he_recalibration/qc/review_montages/Fig7_task008_manual_review.pdf`.
20. CellViT retraining: **No**. Wait for manual review and explicit acceptance of Task008 eligibility.

## Interpretation boundary

The result supports the computational finding that Task007's whole-context component rule over-called debris: 21,116 of 22,164 old debris-suspect cells had target-centered nuclear evidence in the final run. This is a QC finding, not proof of biological identity or a substitute for pathology review. The context necrosis score is review prioritization only and does not override clear target-centered evidence.

## Remote outputs

- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/task007_he_status_audit.csv`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/reference_cell_center_qc.csv.gz`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/registration_tolerance_reference.csv`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/registration_tolerance_summary.csv`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_centered_he_qc.csv.gz`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/xenium_he_nucleus_agreement.csv`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/task007_to_task008_transition.csv`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_retention_by_subtype.csv`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_retention_by_batch.csv`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/parameter_sensitivity.csv`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/safety_rail_decision.json`
- `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`
