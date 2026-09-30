# Task 014 — P169.svs End-to-End WSI Inference to QuPath GeoJSON

## Status
PENDING / READY TO RUN

## Goal

Run the validated Task013 whole-slide inference pipeline on:

`/data/lf_data/he_image/P169.svs`

using the same CellViT detector and Task012 Midnight final classifier, and export only the two slide-prefixed QuPath-compatible GeoJSON outputs for P169.

## Input

`/data/lf_data/he_image/P169.svs`

## Output directory

`/data/lf_data/HCC_result`

Required new P169 files:

- `/data/lf_data/HCC_result/P169_cell_detection.geojson`
- `/data/lf_data/HCC_result/P169_cells.geojson`

Existing GeoJSON files from P84 must be preserved.

The shared output directory may contain slide-prefixed GeoJSON pairs for multiple slides, but no CSV/JSON/PNG/PDF/PT/checkpoint/log sidecars.

## Reusable script

Use:

`scripts/python/task013_predict_svs_geojson.py`

The script has been updated so multiple slide-prefixed GeoJSON pairs can coexist in `HCC_result`.

## Models

### Detector

`/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth`

CellViT is used only for nucleus detection/segmentation and contours.

### Final classifier

`/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`

Expected SHA256:

`ddb3d21f9492032b5f0509c5dc28de15a29b0074105991c5cd1168055d84f633`

Classes:

0 Endothelial
1 Mesenchymal
2 Myeloid
3 Neutrophil
4 Plasma cell
5 T and B
6 Tumor

Use the single full-data final model, identical to P84.

## Inference recipe

Replicate Task013 exactly:

1. open P169.svs and verify WSI metadata;
2. require plausible MPP metadata rather than assuming training-slide MPP;
3. run CellViT++ binary WSI detection with standard overlap cleaning;
4. preserve level-0 centroid and contour coordinates;
5. for each retained nucleus, extract a 12 μm × 12 μm physical H&E field centered at its centroid;
6. resample to 57 × 57 RGB;
7. apply Task012 224 × 224 bicubic resize and 0.5/0.5 normalization;
8. run the Midnight final classifier;
9. use argmax seven-class prediction;
10. write CellViT-style QuPath GeoJSON:
    - P169_cells.geojson = contours grouped by final Midnight class
    - P169_cell_detection.geojson = centroids grouped by final Midnight class.

Do not retain CellViT/PanNuke classes as final labels.

## Work directory

Use:

`/data/lf_data/result/task014_p169_wsi_inference`

All detector outputs, logs, QC summaries and temporary files must stay here, not in `HCC_result`.

## Command

Run:

```bash
/data/lf_data/task010_env/bin/python scripts/python/task013_predict_svs_geojson.py \
  --svs /data/lf_data/he_image/P169.svs \
  --output-dir /data/lf_data/HCC_result \
  --work-dir /data/lf_data/result/task014_p169_wsi_inference
```

If the detector is better launched separately for GPU/runtime control, use the same CellViT CLI recipe as P84 first, then let the wrapper reuse the completed `P169_cells.json`.

## Required QC

Before accepting the result verify:

1. P169 MPP and level-0 dimensions are valid;
2. classifier SHA256 matches Task012;
3. number classified == number retained detector nuclei;
4. all seven-class probabilities are finite and normalized;
5. centroids and contours are in level-0 P169 coordinates and within bounds;
6. every polygon ring is closed;
7. only the seven final class names and required RGB colors are present;
8. GeoJSON files parse successfully;
9. class-group counts in polygon and centroid outputs agree exactly;
10. no PanNuke or old CellViT class names remain;
11. P84 outputs are unchanged;
12. no non-GeoJSON sidecars are written to `/data/lf_data/HCC_result`.

## Interpretation

P169 has no reference labels in this task unless separately provided.

Therefore:
- do not calculate independent accuracy/F1/AUROC/AUPRC for P169;
- report predicted class counts/proportions and confidence only as operational QC;
- compare P169 vs P84 class proportions descriptively for domain-shift sanity checking, not biological conclusions.

Given P84 produced 70.9% Tumor and 0.08% Neutrophil predictions, explicitly check whether P169 shows similarly extreme or very different class proportions.

## Internal report

Create:

`reports/task_014_p169_wsi_inference.md`

Include:
- P169 metadata / MPP;
- total detected/classified nuclei;
- per-class count and proportion;
- prediction-confidence summary;
- detector and classification runtime;
- GeoJSON QC;
- exact output paths;
- concise descriptive comparison with P84 predicted proportions;
- statement that quantitative predictive accuracy is not estimable without labels.

## Final handoff

Return:

1. whether P169 inference completed;
2. detected/classified cell count;
3. seven-class counts/proportions;
4. confidence summary;
5. comparison with P84 prediction distribution;
6. GeoJSON QC result;
7. exact paths:
   - `/data/lf_data/HCC_result/P169_cell_detection.geojson`
   - `/data/lf_data/HCC_result/P169_cells.geojson`
8. confirmation that existing P84 GeoJSON outputs remain intact;
9. note that no independent accuracy estimate is available without P169 labels.

