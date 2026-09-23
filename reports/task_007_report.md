# Task 007 report — Xenium 5K panel-aware reannotation and training-label QC

Status: COMPLETED. Remote computational output status: FINAL with a documented Neutrophil H&E-review exception. CellViT training was not started.

## Evidence and reproducibility

The source AnnData, H&E OME-TIFF, registration matrix, and preprocessing notebook were read from the paths specified in `tasks/task_007.md`. The source files were not modified. Task006 outputs were used only for audit context, not as ground truth. Identity signatures used only genes present in `adata.var_names`; the observed panel contains 5,001 genes, `CD3D` is absent, and `CD3E` is present.

Runtime: Python 3.10.14, NumPy 1.23.5, pandas 1.4.3, h5py 3.9.0, SciPy 1.8.1, pyvips 2.2.3; random seed 42. Script: [task007_panelaware.py](../scripts/python/task007_panelaware.py). Remote command: `python3 /data/lf_data/result/task007_xenium5k_panelaware/code/task007_panelaware.py`.

## Quantitative results

- Original labels: Tumor 415,529; Low quality 140,877; T and B 122,556; Myeloid 107,706; Mesenchymal 89,530; Endothelial 62,935; Plasma cell 27,550; Neutrophil 14,647; Neutrophil_CXCR4 9,520.
- Actions: KEEP 987,846; RELABEL 0; REVIEW 3,004.
- Quality: Pass 782,410; Low_quality 140,877; Artifact_or_no_nucleus 66,250; Technical_fail 1,313.
- TRAIN_CORE and TRAIN_EXTENDED were identical: Tumor 354,569; T and B 120,909; Myeloid 103,566; Mesenchymal 88,182; Endothelial 61,488; Plasma cell 27,264; Neutrophil 1,969.
- Broad original Neutrophil count, including CXCR4 subtype: 24,167. Biologically retained Neutrophil: 24,107.
- Neutrophil H&E triage: no-visible-nucleus 0; debris-suspect 22,164; registration candidates were in bounds. The debris result is a crop-level triage finding and not a biological identity conclusion.
- Top observed panel-aware Neutrophil genes versus Myeloid: SOX2-OT, CXCR4, CSF3R, NCF1, FCGR3B, FPR1, MXD1, TREM1, TLR2, BCL2A1. Enriched and depleted signatures are in `panel_data_driven_markers.csv`.
- Biological relabel rate among non-Low-quality cells: 0.0000%.

## Safety and interpretation boundary

No non-exempt safety rail triggered. The raw `<50% TRAIN_EXTENDED` flag for Neutrophil was explicitly recorded and exempted only because the class-specific H&E debris-suspect artifact is documented in `metrics/safety_rail_decision.json`. This is a manual-review gate, not proof that the cells are debris or that the morphology proxy establishes identity. The annotation preserves `original_cl1`, `original_cl1_7class`, independent quality fields, cross-fitted scores, neighborhood/cluster proxy fields, and training tiers.

## Remote outputs

Output root: `/data/lf_data/result/task007_xenium5k_panelaware`

- Derived AnnData: `/data/lf_data/result/task007_xenium5k_panelaware/adata_xenium_v3_panelaware.h5ad`
- Required annotation table: `/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`
- Report: `/data/lf_data/result/task007_xenium5k_panelaware/TASK007_REPORT.md`
- Safety decision: `/data/lf_data/result/task007_xenium5k_panelaware/metrics/safety_rail_decision.json`
- H&E QC: `/data/lf_data/result/task007_xenium5k_panelaware/metrics/neutrophil_he_qc_summary.csv`
- Review montage: `/data/lf_data/result/task007_xenium5k_panelaware/figures/Fig10_neutrophil_review_montage.pdf`
- Review-montage alias: `/data/lf_data/result/task007_xenium5k_panelaware/figures/Fig_neutrophil_review_montage.pdf`

The derived H5AD is not committed to GitHub because it is a large generated artifact. No CellViT model or performance figure was created by Task007.
