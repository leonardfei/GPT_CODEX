# Task 015 — GHIST classification-only HCC cell typing

## Status

**BLOCKED_NEEDS_LOCAL_GHIST_UPLOAD** (2026-10-01). No GHIST model was trained, and no GHIST-versus-Midnight performance comparison exists. The task specification explicitly requires stopping if server networking prevents acquisition of the official GHIST source; it forbids substituting an unofficial architecture.

## Official source and blocker

The required official source is [SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST), GPL-3.0, expected at `/data/lf_data/models/GHIST`. The three designated server locations were absent. A full `git clone` reached an incomplete transfer and failed with `curl 56 Recv failure: Connection timed out`, `early EOF`, and `invalid index-pack output`. A retry with `--depth 1 --filter=blob:none` failed with `GnuTLS recv error (-110): The TLS connection was non-properly terminated`. The failed attempts left no usable `/data/lf_data/models/GHIST` checkout; no source commit SHA can honestly be recorded. Details are in `config/task015/ghist_source_provenance.json`, also copied to `/data/lf_data/result/task015_ghist_celltyping/config/ghist_source_provenance.json`. No upstream files were edited or alternative model implemented.

## Input observations and work safely completed

The canonical cohort at `/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz` contains 96,044 unique cell IDs, seven frozen classes, eight batches, no missing essential labels/coordinates, and native H&E positions within the 50,000 × 23,451 image. Task010's fold manifest assigns each canonical cell to exactly one of the unchanged Task009 grouped outer validation folds (fold sizes 19,232 / 16,484 / 24,883 / 6,714 / 28,731).

The H&E TIFF's generic OpenSlide MPP value is implausible (~352.783 μm/px). The project-validated 0.2125 μm/px from Task010 was used explicitly for the already-started CellViT binary detection; no full-slide resize or coordinate rescaling was introduced. This produced 620,205 retained binary nucleus instances after standard overlap cleaning. They are stored, **unmapped**, at `/data/lf_data/result/task015_ghist_celltyping/work/detector/ID0060276_cells.json` (309 MB) with detection companion JSON and log outside Git. Existing Task010 matched CellViT *token centroids* have no contour instance IDs, so valid-mask coverage for canonical cells is still unknown and a new audited one-to-one mapping is required before training.

Command used for binary detection:

```bash
/data/lf_data/task010_env/bin/python /data/lf_data/CellViT-plus-plus/cellvit/detect_cells.py \
  --model /data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --binary --gpu 0 --resolution 0.25 --batch_size 8 \
  --outdir /data/lf_data/result/task015_ghist_celltyping/work/detector \
  process_wsi --wsi_path /data/lf_data/xenium_data/ID0060276.ome.tif \
  --wsi_properties '{"slide_mpp":0.2125,"magnification":40}'
```

The log is `/data/lf_data/result/task015_ghist_celltyping/logs/cellvit_binary_detection.log`. Input QC used Python 3.10.14, PyTorch 2.7.1+cu128, pandas 1.4.3, SciPy 1.8.1, OpenSlide Python 1.3.1 and Ray 2.9.3 in `/data/lf_data/task010_env/bin/python`. Binary detection is inference; no stochastic training seed applied. See `qc/task015/input_audit.md` and `qc/task015/canonical_instance_alignment.md`.

## Required benchmark status

The GHIST architecture adapter, patch/instance manifest, fold-0 five-epoch pilot, five-fold GHIST_CT_OFFICIAL and GHIST_CT_BALANCED training, true binary probes, morphology-head QC, paired Midnight comparison, optional neighborhood stage, decision gate, figures and full-data model export were **not run**. No performance claim or model-promotion decision is possible. The pilot audit is `qc/task015/pilot_audit.md` and explicitly records NOT RUN.

## Resumption condition

Provide an unmodified official GHIST checkout at `/data/lf_data/models/GHIST` (or explicitly authorize local download and transfer), including its `.git` history or an immutable commit SHA plus source archive. Once available, verify the official commit and GPL-3.0 license, audit the canonical-to-CellViT one-to-one contour mapping and fold-safe masks, and only then implement and run the pilot. Do not train on P84/P169 or touch the Task012 Midnight/production models.
