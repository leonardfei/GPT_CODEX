# Task 012 — Final Model Export and Publication Figure Package

## Status
IN_PROGRESS — corrected Task011 fold refits and checkpoint export running on the analysis server.

## Goal

Freeze the current best H&E single-cell classification recipe into reproducible deployable checkpoints, export inference utilities, and generate a publication-grade editable vector figure package.

This is a **model export task**, not an independent external-validation task.

Important distinction:
- the exported full-data model is trained on all 96,044 canonical development cells;
- its performance must **not** be estimated on those same training cells;
- the performance attached to the exported recipe remains the corrected Task011 five-fold grouped-CV estimate unless/until an independent slide/patient validation set is evaluated.

## Selected final recipe

Current best development strategy:

- Backbone: Midnight-12k
- Input: target-centered 12 μm × 12 μm H&E crop
- Native crop: 57 × 57 px at 0.2125 μm/px
- Resize/preprocessing: exactly the corrected Task011 preprocessing
- Feature: CLS token + mean patch-token embedding
- Feature dimension: 3072
- Trainable encoder portion: final transformer block only
- Classifier: linear 3072 → 7
- Classes:
  0. Endothelial
  1. Mesenchymal
  2. Myeloid
  3. Neutrophil
  4. Plasma cell
  5. T and B
  6. Tumor
- Epochs: 3
- Backbone LR: 1e-5
- Head LR: 1e-3
- Optimizer: AdamW
- Weight decay: 1e-4
- Loss: class-balanced weighted cross entropy
- Augmentation:
  - 0/90/180/270° rotations
  - horizontal flip
  - mild brightness/contrast only
- Seed: 20260923

Corrected development performance from Task011:
- Macro-F1: 0.4522 ± 0.0294
- Macro-AUPRC: 0.4893 ± 0.0428
- Neutrophil F1: 0.2352 ± 0.1214
- Neutrophil one-vs-rest AUPRC: 0.1976 ± 0.1181
- True N-vs-Myeloid AUROC/AUPRC: 0.7202 ± 0.0221 / 0.4385 ± 0.1787
- True N-vs-T/B AUROC/AUPRC: 0.7069 ± 0.0359 / 0.3930 ± 0.2118

## Canonical inputs

- Canonical cohort:
  `/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz`
- Expected n:
  `96,044`
- Task009 CORE split manifest:
  `/data/lf_data/result/task009_v3_retraining/metrics/split_manifest.csv`
- Original H&E:
  `/data/lf_data/xenium_data/ID0060276.ome.tif`
- Midnight:
  `/data/lf_data/models/midnight-12k`
- Corrected Task011 root:
  `/data/lf_data/result/task011_midnight_local_optimization`
- Environment:
  `/data/lf_data/task010_env/bin/python`

## Output root

`/data/lf_data/result/final_model`

Create:
- ensemble/
- metrics/
- figures/
- config/
- qc/
- logs/
- code/

Do not overwrite any previous Task011 result.

---

## Part A — retrain and export five fold-specific checkpoints

Reproduce the corrected Task011 outer-fold training recipe exactly.

For each of the five Task009 CORE folds:

1. initialize Midnight-12k from the same local checkpoint;
2. freeze all blocks except the final transformer block;
3. initialize the 3072→7 classifier head;
4. use the same outer-training cells as Task011 correction;
5. use the same training-only inner grouped validation rule;
6. select epoch exactly as corrected Task011 did;
7. refit on all outer-training cells for the selected epoch count;
8. evaluate once on untouched outer validation;
9. save the fold-specific checkpoint.

Required checkpoints:

- `ensemble/fold0_midnight_fov12_finalblock_7class.pth`
- `ensemble/fold1_midnight_fov12_finalblock_7class.pth`
- `ensemble/fold2_midnight_fov12_finalblock_7class.pth`
- `ensemble/fold3_midnight_fov12_finalblock_7class.pth`
- `ensemble/fold4_midnight_fov12_finalblock_7class.pth`

Each checkpoint must include:
- encoder_state_dict
- classifier_state_dict
- architecture
- source_model_path
- fov_um
- native_crop_px
- input_size
- feature_dim
- class_names
- fold id
- train/validation batch ids
- selected epoch
- training recipe
- preprocessing config
- seed
- timestamp

Also save:
- `ensemble/ensemble_metadata.json`
- `metrics/oof_predictions.csv.gz`

OOF predictions must contain:
- cell_id
- fold
- true_class_id
- predicted_class_id
- p_class0 ... p_class6

Use the OOF predictions for publication confusion/error plots.

---

## Part B — train and export the single full-data model

After Part A is complete and verified, initialize a fresh Midnight-12k model.

Train on **all 96,044 canonical development cells** for exactly 3 epochs using the frozen final recipe.

Do not use this full-data model to estimate performance on the same cells.

Save:

`/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`

Checkpoint contents:
- encoder_state_dict
- classifier_state_dict
- architecture = Midnight-12k
- fov_um = 12
- native_crop_px = 57
- input_size = 224
- feature_dim = 3072
- class_names
- training_n = 96044
- epochs = 3
- backbone_lr = 1e-5
- head_lr = 1e-3
- optimizer = AdamW
- weight_decay = 1e-4
- loss = class-balanced weighted cross entropy
- augmentation description
- preprocessing config
- source_model_path
- seed
- timestamp
- git commit hash if available

Create:
- `model_metadata.json`
- `preprocessing_config.json`
- `class_mapping.json`
- `checkpoint_sha256.txt`

Compute SHA256 after writing the final checkpoint.

---

## Part C — inference utilities

Create:

- `inference_single_model.py`
- `inference_ensemble.py`
- `README_inference.md`

### Single model script
Must:
1. load the final full-data checkpoint;
2. accept one crop file, a directory of crops, or a manifest;
3. reproduce the exact Task011 preprocessing;
4. output seven class probabilities and predicted class.

### Ensemble script
Must:
1. load the five fold checkpoints;
2. run each model independently;
3. average seven-class probabilities;
4. output averaged probability and final class.

Output CSV columns:
- sample/cell identifier
- predicted_class_id
- predicted_class_name
- p_Endothelial
- p_Mesenchymal
- p_Myeloid
- p_Neutrophil
- p_Plasma
- p_T_and_B
- p_Tumor

Add a smoke test using a small subset of known Task011 crops.

---

## Part D — performance/QC regeneration

Use **corrected Task011 metrics and newly exported OOF predictions**, not training-set performance of the full-data checkpoint.

Regenerate:

### Overall
- accuracy
- balanced accuracy
- Macro-F1
- Macro-AUPRC
- Macro-AUROC
- weighted F1
- lowest-three F1

### Per class
For each of the seven classes:
- precision
- recall
- F1
- AUROC
- AUPRC
- support

### Neutrophil focus
- F1
- precision
- recall
- AUROC
- one-vs-rest AUPRC

### True binary probes
Retain corrected Task011 values:
- N vs Myeloid
- N vs T/B

Do not relabel probability-ratio diagnostics as true binary probes.

### Robustness
- per-fold Macro-F1
- per-fold Neutrophil F1/AUPRC
- batch/spatial kNN purity from corrected Task011

Create:
- `metrics/final_model_summary.csv`
- `metrics/final_per_class.csv`
- `metrics/final_fold_metrics.csv`
- `metrics/final_neutrophil_metrics.csv`
- `metrics/final_true_binary.csv`
- `metrics/final_confusion_matrix.csv`

---

## Part E — publication-grade figure package

Generate all figures as **editable vector PDFs** with embedded text, no rasterized plotting elements unless the source itself is a raster H&E image.

Use a restrained Nature-like palette:
- muted blue
- muted green
- muted red
- muted teal
- grey
- white background
- black axes/text
- thin axis lines
- no chartjunk
- no unnecessary gridlines

Required figures:

1. `Fig1_model_ladder_macroF1.pdf`
   - CellViT aligned
   - Phikon SMALL
   - MUSK FOV8
   - Midnight frozen FOV12
   - Midnight final-block FT

2. `Fig2_model_ladder_macroAUPRC.pdf`

3. `Fig3_neutrophil_F1.pdf`

4. `Fig4_neutrophil_AUPRC.pdf`

5. `Fig5_midnight_FOV_sweep.pdf`

6. `Fig6_musk_FOV_sweep.pdf`

7. `Fig7_true_binary_AUROC.pdf`

8. `Fig8_true_binary_AUPRC.pdf`

9. `Fig9_per_class_F1.pdf`
   - frozen Midnight FOV12 vs final-block FT

10. `Fig10_confusion_matrix.pdf`
    - based on corrected OOF predictions
    - use normalized row percentages plus counts in a companion CSV

11. `Fig11_fold_pairing.pdf`
    - paired per-fold Macro-F1 for frozen FOV12 vs final-block FT
    - connect same fold with lines

12. `Fig12_final_strategy_schematic.pdf`
    - H&E → CellViT nucleus centroid → 12 μm crop → Midnight-12k → final-block FT → 7-class head

13. `Fig13_final_recipe_table.pdf`

Do not hardcode approximate per-class values if corrected server metrics are available; always read them from the corrected Task011 CSVs.

---

## Part F — final report

Create:

`reports/task_012_final_model_export_report.md`

Report must state:

1. exact final checkpoint path
2. exact SHA256
3. fold ensemble checkpoint paths
4. class mapping
5. preprocessing
6. complete training recipe
7. corrected cross-validation performance
8. that full-data model performance is **not** estimated on its own training set
9. single-model inference recommendation
10. ensemble inference recommendation
11. deployment caveat: detector recall remains a separate end-to-end bottleneck
12. external-patient validation is still required before replacing production

---

## Production guardrail

Do NOT overwrite:

`/data/lf_data/result/model_best.pth`

That remains the current production checkpoint until independent slide/patient validation explicitly supports promotion.

Do not modify:
- Task007 labels
- Task008 eligibility
- Task009 canonical data
- Task010/011 historical outputs

Do not commit to GitHub:
- final model checkpoint
- ensemble checkpoints
- large feature tensors
- crop arrays
- raw H&E
- credentials

Commit only:
- task
- scripts
- small metrics
- config
- QC summaries
- report
- figure PDFs if repository size policy permits

---

## Final handoff

Return:

1. final single-model checkpoint path
2. final checkpoint SHA256
3. five ensemble checkpoint paths
4. metadata/config paths
5. inference script paths
6. smoke-test result
7. corrected performance summary
8. figure directory listing
9. final report path
10. confirmation that production model is unchanged
11. recommended next step: independent slide/patient validation
