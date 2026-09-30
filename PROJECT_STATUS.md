# Project Status

## Current task

Task 014 — P169.svs End-to-End WSI Inference to QuPath GeoJSON — PENDING / READY TO RUN

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

## Next execution command

```text
Execute task_014.
```

Codex must pull origin/main and follow AGENTS.md plus `tasks/task_014.md`.

## Production guardrail

Existing production/final checkpoints and P84 outputs remain unchanged.

## Last update

2026-09-30
