# Task 012 — Final Model Export and Publication Figure Package

## Status

COMPLETED on 2026-09-29. Five fold checkpoints, the full-data checkpoint, inference smoke tests, and all 13 figure PDFs passed QC.

## Model artefacts

Final single-model checkpoint: `/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`

Final checkpoint SHA256: `ddb3d21f9492032b5f0509c5dc28de15a29b0074105991c5cd1168055d84f633`

Five fold-specific checkpoints:

| Outer fold | Server path |
|---:|---|
| 0 | `/data/lf_data/result/final_model/ensemble/fold0_midnight_fov12_finalblock_7class.pth` |
| 1 | `/data/lf_data/result/final_model/ensemble/fold1_midnight_fov12_finalblock_7class.pth` |
| 2 | `/data/lf_data/result/final_model/ensemble/fold2_midnight_fov12_finalblock_7class.pth` |
| 3 | `/data/lf_data/result/final_model/ensemble/fold3_midnight_fov12_finalblock_7class.pth` |
| 4 | `/data/lf_data/result/final_model/ensemble/fold4_midnight_fov12_finalblock_7class.pth` |

The ensemble manifest and individual checkpoint hashes are in `config/task012/ensemble_metadata.json`. The final checkpoint hash is also in `config/task012/checkpoint_sha256.txt`. Checkpoints remain on the analysis server and are excluded from Git.

## Cohort and training recipe

The canonical cohort has 96,044 unique cells, seven classes, and eight source batches. The five held-out Task009 CORE folds were reused. No label, eligibility, crop center, split, or biological definition was changed.

Classes in ID order: 0 Endothelial; 1 Mesenchymal; 2 Myeloid; 3 Neutrophil; 4 Plasma cell; 5 T and B; 6 Tumor. The native input is a 57 × 57 RGB H&E crop centered on the matched nucleus (12 μm at 0.2125 μm/px). The crop is converted to float, bicubic-resized to 224 × 224, and normalized with per-channel mean and standard deviation of 0.5. The 3,072-dimensional feature concatenates the CLS token and mean patch-token embedding.

For each outer fold, the corrected Task011 training code selects the epoch using the last lexicographic outer-training batch as grouped inner validation; it then reloads the local Midnight-12k checkpoint and refits on all outer-training cells. All five selected epochs were three. The only trainable encoder portion is the final transformer block. The 3,072 → 7 linear head and final block use AdamW with learning rates 1e-3 and 1e-5 respectively, weight decay 1e-4, and class-balanced weighted cross entropy. The seeded batch-level augmentation consists of 0/90/180/270° rotations, horizontal flips, and mild brightness/contrast. Seed: `20260923`.

After fold export and QC, a fresh Midnight-12k model is trained on all 96,044 development cells for exactly three epochs with this frozen recipe. **No performance estimate is computed on those training cells.** The performance attached to the exported strategy remains the corrected Task011 five-fold grouped cross-validation result until independent slide/patient validation is available.

## Corrected development performance

| Metric | Task011 corrected five-fold mean ± SD |
|---|---:|
| Accuracy | 0.4964 ± 0.0339 |
| Balanced accuracy | 0.4792 ± 0.0301 |
| Macro-F1 | **0.4522 ± 0.0294** |
| Macro-AUPRC | **0.4893 ± 0.0428** |
| Neutrophil F1 | 0.2352 ± 0.1214 |
| Neutrophil one-vs-rest AUPRC | 0.1976 ± 0.1181 |

Corrected independently trained binary probes: Neutrophil versus Myeloid AUROC `0.7202 ± 0.0221`, AUPRC `0.4385 ± 0.1787`; Neutrophil versus T/B AUROC `0.7069 ± 0.0359`, AUPRC `0.3930 ± 0.2118`. Probability-ratio diagnostics are not relabeled as true binary probes.

The Task012 rerun OOF probabilities are used for the confusion matrix and error plots. Their relation to the corrected Task011 fold metrics is recorded as actual per-fold deltas in `qc/task012/exported_fold_reproduction_deltas.csv` and summarized in `qc/task012/task012_export_qc.json`; these are same-recipe reruns, not bitwise-identical historical predictions.
The 96,044-cell OOF file has exactly one held-out prediction per canonical cell, finite normalized probabilities, and a 96,044-count confusion matrix. All seven fold-level metric types differ from corrected Task011 by less than 0.001; the largest absolute difference is 0.000256 for fold 0 lowest-three-class F1. Repeated training is not bitwise deterministic on this GPU stack. An initial overly strict 1e-4 probability-sum assertion on float16 inference output was relaxed to 2e-3 after confirming finite values; the inference utilities normalize probabilities for output, while the raw OOF probabilities were not altered.

## Inference and smoke test

Server scripts: `/data/lf_data/result/final_model/code/inference_single_model.py` and `/data/lf_data/result/final_model/code/inference_ensemble.py`. The single model is recommended for routine inference when runtime and storage matter; the five-model ensemble averages seven-class probabilities and is available when added compute is acceptable. The scripts accept a 57 × 57 crop file, directory, or CSV manifest containing `crop_path`; details are in `/data/lf_data/result/final_model/code/README_inference.md`.

Smoke-test status: **PASS**. Both the single full-data model and five-model ensemble loaded checkpoints, inferred two held-out example crops, produced finite probabilities, and returned rows summing to one. This checks operation, not external predictive validity.

## Publication figure package

All outputs are server-side at `/data/lf_data/result/final_model/figures/` and synchronized to local `figures/task012/`. The 13 one-page vector PDFs are:

1. `Fig1_model_ladder_macroF1.pdf`
2. `Fig2_model_ladder_macroAUPRC.pdf`
3. `Fig3_neutrophil_F1.pdf`
4. `Fig4_neutrophil_AUPRC.pdf`
5. `Fig5_midnight_FOV_sweep.pdf`
6. `Fig6_musk_FOV_sweep.pdf`
7. `Fig7_true_binary_AUROC.pdf`
8. `Fig8_true_binary_AUPRC.pdf`
9. `Fig9_per_class_F1.pdf`
10. `Fig10_confusion_matrix.pdf`
11. `Fig11_fold_pairing.pdf`
12. `Fig12_final_strategy_schematic.pdf`
13. `Fig13_final_recipe_table.pdf`

Figures 1–9 and 11–13 use corrected server metrics. Figure 10 uses Task012 refit OOF predictions and its companion count/row-percentage CSV. The plotted objects and text are editable PDF vectors with embedded fonts; final rendering and image-XObject checks are documented in `qc/task012/`.
The final local `qc/task012/task012_pdf_audit.json` confirms 13 single-page PDFs with extractable text, embedded fonts, and no raster image XObjects. Figures 1 and 10 were also rendered and visually inspected after the final transfer.

## Reproducibility and limitations

Server environment: `/data/lf_data/task010_env/bin/python`; Python 3.10.14, PyTorch 2.7.1+cu128, Transformers 4.45.2, pandas 1.4.3, NumPy 1.23.5, scikit-learn 1.3.0, matplotlib 3.7.1, Pillow 10.3.0, pyvips 2.2.3. GPU: NVIDIA RTX PRO 5000 72GB Blackwell. The non-fatal VIPS ImageMagick module warning seen in previous tasks recurred; crop and model operations continued.

Main code: `scripts/python/task012_export_models.py`, `scripts/python/task012_fold_worker.py`, `scripts/python/task012_collect_parallel.py`, `scripts/python/task012_finalize_figures.py`, and `scripts/python/task012_inference_*.py`. Server commands and run logs are retained under `/data/lf_data/result/final_model/logs/`.

The existing production checkpoint `/data/lf_data/result/model_best.pth` retains SHA256 `f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`. Deployment remains gated by independent slide/patient validation. Detector recall is a separate end-to-end bottleneck and is not measured by this target-centered cell-classifier export.
