# Task 009 — CellViT retraining with frozen V3 labels

Status: **COMPLETED; V3 promotion criteria not met.**

## Scope and provenance

Task009 used the frozen Task007 panel-aware biological labels and the manually accepted Task008 target-centered Neutrophil eligibility. All V3 images and labels were regenerated from the original OME-TIFF; **No old CellViT_dataset files were reused.** The historical notebook was used only to reconstruct patch geometry, registration, batch split, and cap behavior. The source H&E, matrix, Task007 annotation, Task008 eligibility, and production checkpoint were not modified.

- H&E: `/data/lf_data/xenium_data/ID0060276.ome.tif`
- Registration: `/data/lf_data/xenium_data/matrix.csv`
- Task007 labels: `/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz`
- Task008 eligibility: `/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz`
- Recipe: official `python3 ./cellvit/train_cell_classifier_head.py --config <CONFIG>`, SAM-H RAW, AdamW, lr 0.001, weight decay 0.0001, hidden 100, dropout 0, 12 epochs, early stopping 5.
- Historical cap: 50,000 cells/class, seed 1234. CV: `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, grouped by batch.

Eligibility rules were fixed before training. For non-Neutrophil classes, V3 required a V3 label in the six non-Neutrophil classes, `cell_quality_status == Pass`, `technical_fail == False`, `qc_nucleus_present_xenium == True`, and `v3_action != REVIEW`. For Neutrophil, the source label was the original Neutrophil/Neutrophil_CXCR4 annotation, with Task008 `TARGET_NUCLEUS_PRESENT` status and Task007 registration `inbounds`; CORE used `trainable_core_task008`, while EXTENDED used the Task008 extended eligibility flag. Low-quality, REVIEW, technical-failure, artifact/no-nucleus, no-target, fragmented, and manual-review cases were excluded according to those frozen fields.

Reproducibility commands were the preparation script, the official CellViT command above for each fold, `python3 code/task009_detection.py`, `python3 code/task009_end_to_end.py`, and `python3 code/task009_analyze.py` under `/data/lf_data/result/task009_v3_retraining/`. The remote software versions and CUDA availability are recorded in `config/environment.txt`; input and production checkpoint checksums are recorded in `config/checksums.json`.

## Results

| condition | macro-F1 | macro-AUPRC | lowest-three F1 |
|---|---:|---:|---:|
| OLD_LABELS | 0.3324 ± 0.0134 | 0.3413 ± 0.0281 | 0.1748 ± 0.0213 |
| V3_CORE | 0.3330 ± 0.0222 | 0.3485 ± 0.0262 | 0.1580 ± 0.0208 |
| V3_EXTENDED | 0.3340 ± 0.0217 | 0.3488 ± 0.0265 | 0.1608 ± 0.0252 |

Uncapped eligible cells were:

| class | CORE | EXTENDED |
|---|---:|---:|
| Endothelial | 61,488 | 61,488 |
| Mesenchymal | 88,182 | 88,182 |
| Myeloid | 103,566 | 103,566 |
| Neutrophil | 21,639 | 22,107 |
| Plasma cell | 27,264 | 27,264 |
| T and B | 120,909 | 120,909 |
| Tumor | 354,569 | 354,569 |

Actual cells used in the regenerated training patches were:

| class | CORE | EXTENDED |
|---|---:|---:|
| Endothelial | 26,268 | 26,268 |
| Mesenchymal | 30,828 | 30,828 |
| Myeloid | 26,302 | 26,302 |
| Neutrophil | 10,486 | 10,721 |
| Plasma cell | 12,986 | 12,986 |
| T and B | 28,452 | 28,452 |
| Tumor | 20,978 | 20,978 |

Neutrophil CV metrics (precision / recall / F1 / AUPRC):

- OLD_LABELS: 0.2004 / 0.0953 / 0.1145 / 0.1366
- V3_CORE: 0.2005 / 0.0992 / 0.1072 / 0.1369
- V3_EXTENDED: 0.2031 / 0.1006 / 0.1084 / 0.1379

Fold-paired macro-F1 deltas versus OLD_LABELS were CORE `[-0.0099, +0.0161, -0.0273, +0.0427, -0.0187]` and EXTENDED `[-0.0103, +0.0193, -0.0212, +0.0419, -0.0216]`; each improved in 2/5 folds.

Myeloid↔Neutrophil confusion changed as follows (rate among true class):

- OLD_LABELS: Myeloid→Neutrophil 3.56%; Neutrophil→Myeloid 19.17%.
- V3_CORE: 2.78%; 19.92%.
- V3_EXTENDED: 2.79%; 15.86%.

CellViT detector recall on the regenerated training patches (CORE / EXTENDED): Endothelial 0.5078 / 0.5077; Mesenchymal 0.5511 / 0.5508; Myeloid 0.6081 / 0.6079; Neutrophil 0.5854 / 0.5777; Plasma cell 0.6576 / 0.6577; T and B 0.7387 / 0.7385; Tumor 0.6687 / 0.6686.

Neutrophil end-to-end evaluation across grouped validation folds:

- V3_CORE: detection recall 0.5588, conditional recall 0.0992, end-to-end recall 0.0549, end-to-end precision 0.0906.
- V3_EXTENDED: detection recall 0.5524, conditional recall 0.1006, end-to-end recall 0.0554, end-to-end precision 0.0912.

## Promotion and next step

Neither V3 condition met the predefined promotion rule: neither reached +0.03 macro-F1, neither reached the Neutrophil +0.05 F1/AUPRC criterion, lowest-three F1 decreased, and only 2/5 folds improved in macro-F1. No secondary optimization, final external test, or production update was performed. The dominant observed bottleneck remains detector/representation performance rather than a demonstrated correction from label eligibility alone; this is a computational finding, not a biological conclusion.

- Selected candidate: none.
- Production checkpoint unchanged: `/data/lf_data/result/model_best.pth`
- Production SHA256: `f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`
- Recommended next task: review the detector/representation bottleneck and design any context-aware or specialist experiment only after Task009 review; do not alter frozen labels in response to these results.

## Remote artefacts

- Report: `/data/lf_data/result/task009_v3_retraining/TASK009_REPORT.md`
- New datasets: `/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE` and `..._EXTENDED`
- Metrics: `/data/lf_data/result/task009_v3_retraining/metrics/`
- Figures: `/data/lf_data/result/task009_v3_retraining/figures/`
- Regeneration audits: `/data/lf_data/result/task009_v3_retraining/qc/dataset_reconstruction_audit.md` and `qc/dataset_regeneration_provenance.md`

No checkpoint, cache, or large patch image is intended for Git.
