# Project Status

## Current task

Task 014 — P169.svs End-to-End WSI Inference to QuPath GeoJSON — COMPLETED

Completed 2026-09-30. CellViT binary detection and Task012 Midnight single-model classification produced 153,892 P169 level-0 nucleus predictions. Both P169 GeoJSON outputs passed built-in and independent geometry/class/count checks. Existing P84 outputs and both model checkpoints retained their SHA256 hashes. See `reports/task_014_p169_wsi_inference.md`. P169 has no reference labels; there is no independent accuracy/F1 estimate. Tumor 72.067% and Neutrophil 0.112% are unverified prediction proportions requiring visual/reference-label review.

## Input

`/data/lf_data/he_image/P169.svs`

## Shared output directory

`/data/lf_data/HCC_result`

Required new outputs:

- `P169_cell_detection.geojson`
- `P169_cells.geojson`

Existing P84 outputs must be preserved.

The reusable WSI inference script now supports multiple slide-prefixed GeoJSON pairs in this shared directory and refuses non-GeoJSON sidecars.

## Pipeline

Same validated P84 workflow:

1. CellViT++ SAM-H x40 binary nucleus detection/segmentation
2. preserve level-0 centroid and contour geometry
3. read P169 MPP
4. extract 12 μm physical target-centered crops
5. resample to 57×57
6. Task012 Midnight full-data final classifier
7. export QuPath-compatible slide-prefixed contour and centroid GeoJSON

## Models

CellViT detector:
`/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth`

Midnight classifier:
`/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`

Expected SHA256:
`ddb3d21f9492032b5f0509c5dc28de15a29b0074105991c5cd1168055d84f633`

## P84 reference run

P84 completed successfully:
- 291,029 detected/classified nuclei
- Tumor 70.898%
- Neutrophil 0.080%

These are unverified model outputs, not biological reference proportions. Task014 should compare P169 descriptively as a domain-shift sanity check only.

## Work directory

`/data/lf_data/result/task014_p169_wsi_inference`

## Task file

`tasks/task_014.md`

## Next step

Await an assigned follow-up. Recommended validation: open both P169 GeoJSON files in QuPath on P169.svs, inspect representative tissue regions and nucleus contours, and obtain reference labels to quantify classifier/detector performance. Pull origin/main before the next task.

## Production guardrail

Existing production/final checkpoints and P84 outputs remain unchanged.

## Last update

2026-09-30
