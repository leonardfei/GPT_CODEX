# Task 015 pilot audit

Status: **PILOT_GATE_PASS** (2026-10-01). Official GHIST commit `917456be305fc82e92293ea272812e79675e821c` and all 96,044 canonical CellViT contours are verified. Fold-0 training used outer-training batches `s01A,s01B,s04B,s11,s22`; `s93` was the training-only inner-validation batch; outer validation `s02A,s06A` was not used for checkpoint selection.

One-batch split-safe loss smoke test passed with batch size 8, 194 supervised cells, finite logits, pixel CE `2.1089`, cell CE `1.9537`, and successful backward/optimizer step. The 30-pixel-overlap geometry audit passed for all 96,044 cells; the fold-0 inner-validation `s93` set has 9,607 cells on 617 selected overlap patches, and every selected instance was present in its rasterized patch mask. These tests are not the five-epoch pilot.

The five-epoch GHIST_CT_OFFICIAL pilot completed successfully on server GPU 0 with command:

```bash
PYTHONPATH=/data/lf_data/result/task015_ghist_celltyping/code CUDA_VISIBLE_DEVICES=0 \
  /data/lf_data/task010_env/bin/python \
  /data/lf_data/result/task015_ghist_celltyping/code/task015_train_ghist_celltype.py \
  --fold 0 --pilot --epochs 5
```

Log: `/data/lf_data/result/task015_ghist_celltyping/logs/pilot_fold0_official.log`. Per-epoch metrics and five checkpoints are server-side under `qc/pilot_epoch_metrics.json` and `models/pilot_fold0_official/`. The mean training loss declined each epoch: `1.8726, 1.7273, 1.6449, 1.5995, 1.5575`. Inner-validation Macro-F1 was `0.0695, 0.1864, 0.2207, 0.2455, 0.2366`; Neutrophil AUPRC was `0.1584, 0.3812, 0.4444, 0.4183, 0.4737`. These are pilot inner-validation diagnostics, **not** outer-fold estimates.

Each epoch produced exactly **9,607** unique inner-validation cell IDs after 30-pixel-overlap largest-area deduplication, with finite probabilities. Peak allocated GPU memory remained `13.6959–13.6960 GiB` across epochs. Training/inner/outer cell IDs and biological batches were disjoint by construction and assertion. Independent raster-mask QC verified all 96,044 canonical instances in both non-overlap training and selected overlap inference masks. No outer-validation label contributed to a training loss or pilot checkpoint choice. HED augmentation was disabled because official `stainlib` is absent; no substitute stain method was introduced. Mixed precision was disabled for the stability pilot. No full-CV or production model conclusion follows from this gate alone.
