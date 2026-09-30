# Task 013 — P84.svs End-to-End WSI Inference to QuPath GeoJSON

## Status
PENDING / READY TO RUN

## Goal

Run the current best exported H&E cell-classification pipeline on one new whole-slide H&E image:

`/data/lf_data/he_image/P84.svs`

and write **only two QuPath-compatible GeoJSON files** into:

`/data/lf_data/HCC_result`

Required final files:

1. `/data/lf_data/HCC_result/P84_cell_detection.geojson`
2. `/data/lf_data/HCC_result/P84_cells.geojson`

No CSV, JSON, PNG, PDF, PT, checkpoint, log, or temporary file may remain inside `/data/lf_data/HCC_result`.

### Slide-based output naming rule

Output filenames must be derived automatically from the input slide basename.

For an input:
`/path/to/<SLIDE>.svs`

write:
- `<SLIDE>_cell_detection.geojson`
- `<SLIDE>_cells.geojson`

For this task, `P84.svs` therefore produces:
- `P84_cell_detection.geojson`
- `P84_cells.geojson`

The reusable script must not hardcode `P84`; it must derive `<SLIDE>` from `Path(svs).stem`.

Temporary files, logs, QC, and intermediate detector outputs must be stored outside that directory, under:

`/data/lf_data/result/task013_p84_wsi_inference`

After successful completion, verify that `HCC_result` contains exactly the two required slide-prefixed GeoJSON files.

---

## Important interpretation

P84 currently has no Xenium/manual reference labels in this task.

Therefore do **not** report accuracy, F1, AUROC, AUPRC, or sensitivity for P84 itself.

The Task011 grouped-CV values remain the validated development performance.

For P84, only assess inference sanity internally:
- WSI metadata / MPP validity;
- detected nucleus count;
- seven-class predicted counts/proportions;
- probability confidence distribution;
- obvious technical failures;
- spatial coordinate integrity;
- GeoJSON validity and QuPath compatibility.

Do not write these QC summaries into `/data/lf_data/HCC_result`; store them under the task013 working directory/report.

---

## Models

### Nucleus detector

Use the established CellViT++ SAM-H x40 detector:

`/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth`

CellViT is used **only for nucleus detection/segmentation and contours**.

Do not use the old seven-class CellViT classifier for final cell typing.

### Seven-class classifier

Use the Task012 exported full-data final model:

`/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`

Expected SHA256:

`ddb3d21f9492032b5f0509c5dc28de15a29b0074105991c5cd1168055d84f633`

Seven classes:

0. Endothelial
1. Mesenchymal
2. Myeloid
3. Neutrophil
4. Plasma cell
5. T and B
6. Tumor

Use the single final model for this first P84 test.

Do **not** run the five-model ensemble unless the single-model run fails technically or a later task explicitly requests ensemble comparison.

---

## Step 1 — Input validation

Before inference:

1. verify `P84.svs` exists and can be opened;
2. record:
   - level-0 width/height;
   - number of pyramid levels;
   - MPP-X / MPP-Y;
   - objective power if present;
   - scanner vendor if present;
3. verify the final Midnight checkpoint SHA256;
4. verify local Midnight source model path from checkpoint metadata;
5. verify CellViT detector checkpoint exists.

### MPP requirement

Do not assume P84 has the same pixel size as the training slide.

The classifier requires a **12 μm physical field of view**.

For each detected centroid:
- convert 12 μm to native level-0 pixels using P84 MPP;
- crop that physical field centered on the level-0 centroid;
- bicubic-resample to **57 × 57 px**;
- then apply Task012 preprocessing:
  - float RGB;
  - resize to 224 × 224;
  - normalize with mean/std 0.5.

If MPP metadata are missing or implausible, stop rather than silently assuming 0.2125 μm/px.

---

## Step 2 — CellViT nucleus detection / segmentation

Use the local CellViT++ WSI pipeline and preserve the detector's native WSI coordinate geometry.

The final detector result must supply for every retained nucleus:

- level-0 centroid [x, y];
- polygon contour in level-0 pixel coordinates;
- unique internal cell id;
- any detector confidence if available.

Use CellViT's standard overlap cleaning / edge-cell de-duplication.

Do not use CellViT's original PanNuke class as the final classification.

### Resolution handling

Inspect P84 MPP before selecting CellViT inference resolution.

Use the supported CellViT++ resolution/WSI pipeline appropriate to the slide, preserving correct level-0 re-alignment.

Do not apply an arbitrary coordinate scale.

---

## Step 3 — Midnight classification on detected nuclei

Perform crop extraction on-the-fly if practical; do not save millions of crop PNGs.

For each CellViT nucleus centroid:

1. take a 12 μm × 12 μm level-0 physical crop from P84;
2. resample to 57 × 57 px;
3. run the Task012 Midnight final model;
4. obtain seven-class probabilities;
5. assign argmax class.

Keep internally:
- p_Endothelial
- p_Mesenchymal
- p_Myeloid
- p_Neutrophil
- p_Plasma
- p_T_and_B
- p_Tumor

But final QuPath GeoJSON only needs classification labels/colors/geometry unless adding probabilities can be done without breaking the CellViT/QuPath structure.

Do not alter or filter predictions using the old CellViT class.

---

## Step 4 — QuPath-compatible GeoJSON format

Match the CellViT/CellViT++ QuPath GeoJSON convention.

Official CellViT behavior groups objects by predicted cell type:
- segmentation output uses polygon/MultiPolygon geometry from cell contours;
- detection output uses point/MultiPoint geometry from centroids;
- `properties.classification.name` stores the class label;
- `properties.classification.color` stores the RGB classification color.

The official CellViT GeoJSON exporter uses its template helpers and creates one grouped GeoJSON object per detected class.

Reference behavior:
- `P84_cells.geojson`: segmentation polygons/contours
- `P84_cell_detection.geojson`: detection centroids

Use the **Midnight seven-class prediction** to populate classification, while retaining CellViT centroid and contour geometry. CellViT supports GeoJSON output for QuPath, and its converter groups contours or centroids by class and writes classification name/color in the feature properties. citeturn881068search0turn479133view0

### Required classification names

Exactly use:

- `Endothelial`
- `Mesenchymal`
- `Myeloid`
- `Neutrophil`
- `Plasma cell`
- `T and B`
- `Tumor`

### Required stable colors

Use the following RGB colors in both slide-prefixed GeoJSON files:

- Endothelial: [76, 120, 168]
- Mesenchymal: [242, 207, 91]
- Myeloid: [178, 121, 162]
- Neutrophil: [114, 183, 178]
- Plasma cell: [255, 157, 166]
- T and B: [84, 162, 75]
- Tumor: [228, 87, 86]

Coordinates must remain in P84 level-0 pixel coordinates so the GeoJSON overlays correctly in QuPath.

### Geometry

`P84_cells.geojson`
- use CellViT nucleus contours;
- close every polygon ring;
- preserve valid coordinate order [x, y];
- grouped by predicted class following CellViT's GeoJSON convention.

`P84_cell_detection.geojson`
- use CellViT nucleus centroids;
- grouped by predicted class following CellViT's GeoJSON convention.

Do not substitute synthetic circles/bounding boxes for CellViT contours.

---

## Step 5 — Required QC before accepting outputs

Internally verify:

1. number of classified cells == number of retained CellViT nuclei;
2. every nucleus has exactly one seven-class prediction;
3. all class probabilities are finite and normalized;
4. all centroids lie inside level-0 slide bounds;
5. all contours lie within valid bounds after CellViT coordinate realignment;
6. polygon rings are closed;
7. predicted classes are limited to the seven allowed labels;
8. both slide-prefixed GeoJSON files parse successfully;
9. both files contain all predicted classes that are present;
10. GeoJSON object counts / coordinate totals match the classified-cell counts;
11. no old PanNuke classification names remain;
12. no old Task001 seven-class CellViT classifier output is used.

If possible, perform a programmatic QuPath-format sanity check using the same CellViT template structure.

CellViT/CellViT++ officially exports `cells.geojson` and `cell_detection.geojson` for QuPath loading. citeturn881068search0turn881068search3

---

## Step 6 — Output hygiene

Before final copy:

`mkdir -p /data/lf_data/HCC_result`

Remove stale P84 outputs from that directory if they are from this test.

After all QC passes, atomically write/copy only:

`/data/lf_data/HCC_result/P84_cell_detection.geojson`

`/data/lf_data/HCC_result/P84_cells.geojson`

Final assertion:

`find /data/lf_data/HCC_result -maxdepth 1 -type f`

must return exactly:
- `P84_cell_detection.geojson`
- `P84_cells.geojson`

For future slides, apply the same `<SLIDE>_...` naming rule.

No sidecar files in `HCC_result`.

---

## Step 7 — Internal task report

Create in the repository/work area, not HCC_result:

`reports/task_013_p84_wsi_inference.md`

Record:

- input SVS path;
- P84 MPP and dimensions;
- detector checkpoint;
- classifier checkpoint and SHA256;
- total detected/classified cells;
- predicted count/proportion per seven classes;
- probability-confidence summary;
- runtime;
- any warnings;
- GeoJSON validation;
- exact final output paths.

Explicitly state that P84 has no reference labels in this task and therefore the run does not provide an independent accuracy/F1 estimate.

---

## Required code

Create a reusable script:

`scripts/python/task013_predict_svs_geojson.py`

Design it so a later slide can be run as:

```bash
/data/lf_data/task010_env/bin/python scripts/python/task013_predict_svs_geojson.py \
  --svs /path/to/slide.svs \
  --output-dir /path/to/qupath_geojson_output
```

For this task use:

```bash
/data/lf_data/task010_env/bin/python scripts/python/task013_predict_svs_geojson.py \
  --svs /data/lf_data/he_image/P84.svs \
  --output-dir /data/lf_data/HCC_result
```

The reusable script must derive output filenames from the input SVS stem and preserve the rule that the specified output directory contains only the two slide-prefixed GeoJSON files for that run.

---

## Production guardrails

Do not modify:
- `/data/lf_data/result/model_best.pth`
- `/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`
- Task007/Task008 labels
- Task009 canonical objects
- Task010–Task012 historical results

Do not commit:
- P84.svs
- GeoJSON files if extremely large
- detector intermediate arrays
- crops
- model checkpoints

Commit only:
- task specification
- reusable script
- small QC/report summaries

---

## Final handoff

Return:

1. whether P84 inference completed successfully;
2. number of detected/classified nuclei;
3. brief seven-class count summary;
4. whether GeoJSON QC passed;
5. exact final paths:
   - `/data/lf_data/HCC_result/P84_cell_detection.geojson`
   - `/data/lf_data/HCC_result/P84_cells.geojson`
6. confirmation that `HCC_result` contains only those two files;
7. explicit note that prediction accuracy cannot be measured without reference labels.

