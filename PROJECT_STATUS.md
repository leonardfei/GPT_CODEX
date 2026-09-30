# Project Status

## Current task

Task 013 — P84.svs End-to-End WSI Inference to QuPath GeoJSON — COMPLETED

Completed 2026-09-30. CellViT binary detection and Midnight single-model classification produced 291,029 level-0 nuclei/predictions. Both slide-prefixed GeoJSON files passed built-in and independent geometry/class/count checks. See `reports/task_013_p84_wsi_inference.md`. P84 has no reference labels; no independent accuracy/F1 estimate is available. The 70.9% Tumor and 0.08% Neutrophil prediction proportions need reference-label and QuPath visual review before scientific interpretation.

## Input

`/data/lf_data/he_image/P84.svs`

## Required final output directory

`/data/lf_data/HCC_result`

This directory must contain **only**:

- `P84_cell_detection.geojson`
- `P84_cells.geojson`

## Pipeline

1. CellViT++ SAM-H x40: nucleus detection/segmentation only.
2. Preserve level-0 centroid and contour geometry.
3. Read P84 MPP from SVS metadata.
4. Extract a target-centered 12 μm physical crop for every retained nucleus.
5. Resample crop to 57×57, then apply Task012 Midnight preprocessing.
6. Classify with:
   `/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`
7. Replace CellViT/PanNuke type with the final Midnight seven-class prediction.
8. Export QuPath-compatible CellViT-style with slide-prefixed filenames:
   - P84_cells.geojson = nucleus polygons
   - P84_cell_detection.geojson = nucleus centroids
9. Do not leave any sidecar files in HCC_result.

## Important

P84 has no reference labels in this task, so this run can test operational inference, spatial plausibility, prediction confidence and QuPath compatibility, but cannot yield an independent accuracy/F1 estimate.

## Models

CellViT detector:
`/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth`

Midnight final classifier:
`/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`

Expected classifier SHA256:
`ddb3d21f9492032b5f0509c5dc28de15a29b0074105991c5cd1168055d84f633`

## Task file

`tasks/task_013.md`

## Next step

Await an assigned follow-up. Recommended validation is to open both files in QuPath against P84, inspect nucleus overlays at several tissue regions, and obtain reference labels for quantitative evaluation. Pull origin/main before the next task.

## Production guardrail

Existing production and final checkpoints remain unchanged.

## Last update

2026-09-30


## Output naming rule

For future slides, derive filenames from the SVS basename:
- `<SLIDE>_cell_detection.geojson`
- `<SLIDE>_cells.geojson`

The inference script must derive `<SLIDE>` automatically from the input file stem; do not hardcode P84.
