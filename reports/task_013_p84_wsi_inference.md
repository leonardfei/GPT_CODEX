# Task 013 — P84.svs WSI inference to QuPath GeoJSON

## Status and scope

COMPLETED on 2026-09-30. This task performs operational nucleus detection and seven-class classification on an unlabeled WSI. P84 has no Xenium/manual reference labels in this task; therefore **no independent P84 accuracy, F1, AUROC, AUPRC, or sensitivity can be calculated**. The corrected Task011 grouped-CV estimates remain development-only reference values, not P84 performance.

## Inputs and design

- Slide: `/data/lf_data/he_image/P84.svs`; Aperio SVS, 45,696 × 39,424 level-0 pixels, 7 pyramid levels, objective 20×. OpenSlide metadata report MPP-X = MPP-Y = 0.276812 μm/px. No training-slide MPP was assumed.
- CellViT++ SAM-H x40 detector: `/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth`; SHA256 `356418f19d9d478f164c7a31f85274584fefaa02355815c09f52346c658c8ec4`.
- Detector used the official in-memory WSI CLI, binary segmentation (`--binary`), nominal 0.25 MPP network resolution. The implementation accepted this 0.276812-MPP slide natively and set target patch MPP to 0.276812. The standard 1024-pixel patch / 64-pixel overlap and built-in overlap cleaning were used. No PanNuke or older CellViT seven-class typing is retained.
- Midnight classifier: `/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`; verified SHA256 `ddb3d21f9492032b5f0509c5dc28de15a29b0074105991c5cd1168055d84f633`. The Task012 single final model is used, not the five-model ensemble.
- Each retained level-0 centroid is the center of a physical 12 μm × 12 μm H&E field. The field is bicubic-resampled to a 57 × 57 RGB crop, then processed with the Task012 224 × 224 bicubic resize and 0.5/0.5 channel normalization. The source-model CLS token and mean patch embedding form the 3,072-dimensional input to the seven-class head. No probabilities were thresholded or reweighted.

## Reproducible execution

Reusable script: `scripts/python/task013_predict_svs_geojson.py`. Server copy: `/data/lf_data/result/task013_p84_wsi_inference/task013_predict_svs_geojson.py`.

Official detector command (executed first; output reused by the wrapper):

```bash
/data/lf_data/task010_env/bin/python /data/lf_data/CellViT-plus-plus/cellvit/detect_cells.py \
  --model /data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --binary --gpu 0 --resolution 0.25 --batch_size 8 \
  --outdir /data/lf_data/result/task013_p84_wsi_inference/detector \
  process_wsi --wsi_path /data/lf_data/he_image/P84.svs
```

Classification and export command:

```bash
/data/lf_data/task010_env/bin/python /data/lf_data/result/task013_p84_wsi_inference/task013_predict_svs_geojson.py \
  --svs /data/lf_data/he_image/P84.svs \
  --output-dir /data/lf_data/HCC_result
```

The script derives output names from `Path(svs).stem`; it does not hardcode P84. It keeps intermediates, logs, and QC under `/data/lf_data/result/task013_p84_wsi_inference` and refuses to delete unrelated files in the requested output directory. This is deterministic inference; no random seed is applicable.

Server environment: Python 3.10.14, PyTorch 2.7.1+cu128, Transformers 4.45.2, NumPy 1.23.5, OpenSlide Python 1.3.1, OpenCV 4.5.4-dev, Ray 2.9.3. Hardware: NVIDIA RTX PRO 5000 72GB Blackwell. The CellViT source directory is not itself a Git repository, so no CellViT source commit is available. Detailed execution logs are retained under the task013 server working directory.

## Observed results and QC

CellViT detected **291,029** nuclei after standard overlap cleaning; Midnight classified exactly **291,029**. The source detector JSON has only `Background`/`Cell` type labels, and its available detection `type_prob` mean was 0.9376 (10th percentile 0.8032). The detector centroid range was x = 1,892–41,824 and y = 1,911–37,398 level-0 pixels.

| Midnight prediction | Count | Proportion |
|---|---:|---:|
| Endothelial | 45,067 | 15.485% |
| Mesenchymal | 7,886 | 2.710% |
| Myeloid | 26,635 | 9.152% |
| Neutrophil | 234 | 0.080% |
| Plasma cell | 2,061 | 0.708% |
| T and B | 2,812 | 0.966% |
| Tumor | 206,334 | 70.898% |

Maximum seven-class prediction probability: mean 0.5511, 10th percentile 0.2647, median 0.5053, 90th percentile 0.8952. All probability vectors were finite and normalized before export. No threshold was used to suppress low-confidence predictions.

The exporter validated 291,029 polygon objects and 291,029 centroid objects. Independent post-export parsing in `scripts/python/task013_verify_outputs.py` confirmed exactly seven present class groups in each file, exact per-class count agreement, valid CellViT-style `Feature`/`MultiPolygon` and `Feature`/`MultiPoint` structure, required names and RGB colors, closed polygon rings, level-0 bounds, and each centroid inside its paired contour bounding box. The independent QC JSON is `qc/task013/task013_independent_geojson_qc.json`. A seeded 20,000-point overlay on the SVS thumbnail (`figures/task013/P84_spatial_qc_overlay.png`) was visually reviewed; detections followed tissue rather than the blank slide background. The overlay checks broad spatial alignment, not microscopic contour accuracy or actual QuPath UI loading.

The very high Tumor prediction proportion (70.9%) and very low Neutrophil proportion (0.08%) are **observed classifier outputs**, not verified biological cell proportions. Their disparity from the development cohort warrants targeted visual and reference-label review for possible slide composition, stain/domain shift, detector selection, or classifier error. No such cause can be established without P84 labels.

Official CellViT detection took approximately 5 min 43 s; the subsequent checkpoint/hash checks, Midnight classification, export and built-in QC took 1,380.6 s (23 min 00.6 s). Logs: `/data/lf_data/result/task013_p84_wsi_inference/logs/detector.log` and `pipeline.log`. The independent format and spatial QC ran afterward. Post-run SHA256 checks confirmed that the Task012 classifier retained `ddb3d21f9492032b5f0509c5dc28de15a29b0074105991c5cd1168055d84f633` and the existing production checkpoint `/data/lf_data/result/model_best.pth` retained `f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`.

## Output and limitations

Final server output directory `/data/lf_data/HCC_result` contains **only**:

- `/data/lf_data/HCC_result/P84_cell_detection.geojson` (5,160,072 bytes)
- `/data/lf_data/HCC_result/P84_cells.geojson` (158,946,662 bytes)

The approximately 159 MB GeoJSON and 384 MB source SVS remain on the analysis server and are not committed to Git. The script, small QC summaries, spatial overview and reports are committed. Both GeoJSON files parsed successfully and passed the programmatic CellViT/QuPath structure check. Actual loading and close-up visual inspection in QuPath have not been performed; that remains a downstream validation step.

The GeoJSON files represent algorithmic nucleus segmentation and classifier predictions, not verified cell identities. Plausibility and format checks cannot establish biological validity or independent accuracy. Independent slide/patient labeling and QuPath visual review remain necessary before clinical or downstream scientific interpretation.
