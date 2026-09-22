# Task 006 report — Xenium re-annotation and high-quality CellViT retraining

Task 006 completed on 2026-09-22. Source AnnData, H&E, registration matrix, and preprocessing notebook were not modified. Full remote report and derived artifacts are under:

`/data/lf_data/result/task006_xenium_reannotation`

## Key results

- Original biological annotation: `obs['cl1']`; total cells: 990,850.
- Historical workflow: 256×256 H&E patches, inverse registration, 8 train/8 test batches, per-class cap 50,000, official five-fold grouped CV.
- v2 annotation count: 990,850; old→new label-change rate: 89.6722%.
- HQ_CORE: 73,960 cells — Endothelial 8,765; Mesenchymal 14,085; Myeloid 11,055; Neutrophil 445; Plasma cell 15,479; T and B 1,386; Tumor 22,745.
- HQ_EXTENDED: 74,181 cells — same as CORE plus 221 additional Neutrophils (666 total).
- EXCLUDE_OR_REVIEW: 916,669: Low_quality 373,455; Mixed_lineage 404,049; Uncertain 134,174; Artifact_or_no_nucleus 4,843; other class-scored review cells 148.
- Original Neutrophil labels: 24,167; v2 Neutrophils: 666; HQ_CORE intact/high-confidence Neutrophils: 445; original Neutrophils reclassified or excluded: 23,581.
- Neutrophil debris/no-nucleus flags: debris suspect 4,843; no-nucleus/low nuclear evidence 4,601.

## Controlled CV

All three conditions used the same official SAM-H RAW recipe, class map, batch-aware five-fold splits, and no test-set tuning:

- OLD_LABELS: macro-F1 0.3324 ± 0.0134.
- V2_CORE: macro-F1 0.3030 ± 0.0236.
- V2_EXTENDED: macro-F1 0.3027 ± 0.0284.

Neutrophil mean F1 / recall / AUPRC:

- OLD_LABELS: 0.1145 / 0.0953 / 0.1366.
- V2_CORE: 0.0000 / 0.0000 / 0.0187.
- V2_EXTENDED: 0.0000 / 0.0000 / 0.0344.

Neither v2 tier met the pre-specified promotion thresholds. The frozen test split was not evaluated, the production checkpoint remains `/data/lf_data/result/model_best.pth`, and no new production model was promoted.

## Markers, QC, and unresolved issues

Independent marker scoring used endothelial (PECAM1/VWF/EMCN/KDR), mesenchymal (COL1A1/COL1A2/DCN/LUM/ACTA2/TAGLN), myeloid (LST1/TYROBP/FCER1G/C1Q/CD68), neutrophil (FCGR3B/CSF3R/CXCR2/FPR1/S100A8/S100A9), plasma-cell (JCHAIN/MZB1/XBP1/SDC1/IGKC), T/B (CD3D/CD3E/TRBC/LCK/MS4A1/CD79A/CD19/CD74), and tumor/epithelial (EPCAM/KRT8/KRT18/KRT19/KRT7/ALB/APOA1) marker sets. CellViT predictions were not used for annotation or quality tiers.

Registration used the matrix inverse and gave 100% in-bounds centroids. OME metadata reports a physical pixel size inconsistent with the historical notebook geometry; this remains unresolved and no silent rescaling was applied. The large Mixed_lineage/Uncertain pool and poor v2 Neutrophil CV require expert review before any further relabeling.

Derived AnnData: `/data/lf_data/result/task006_xenium_reannotation/adata_xenium_v2_annotated.h5ad`.

Remote report: `/data/lf_data/result/task006_xenium_reannotation/TASK006_REPORT.md`.

Figures: `/data/lf_data/result/task006_xenium_reannotation/figures/`.

Recommended Task 007 direction: do not start automatically; first calibrate the physical-scale metadata and review low-retention classes/Neutrophil labels with an expert, then define a new task only if the annotation policy changes.
