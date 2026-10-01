# Task 015 pilot audit

Status: **RUNNING; gate not yet evaluated** (2026-10-01). Official GHIST commit `917456be305fc82e92293ea272812e79675e821c` and all 96,044 canonical CellViT contours are verified. Fold-0 training uses outer-training batches `s01A,s01B,s04B,s11,s22`; `s93` is the untouched training-only inner-validation batch; outer validation `s02A,s06A` remains unused for checkpoint selection.

One-batch split-safe loss smoke test passed with batch size 8, 194 supervised cells, finite logits, pixel CE `2.1089`, cell CE `1.9537`, and successful backward/optimizer step. The 30-pixel-overlap geometry audit passed for all 96,044 cells; the fold-0 inner-validation `s93` set has 9,607 cells on 617 selected overlap patches, and every selected instance was present in its rasterized patch mask. These tests are not the five-epoch pilot.

The five-epoch GHIST_CT_OFFICIAL pilot was launched on server GPU 0 with command:

```bash
PYTHONPATH=/data/lf_data/result/task015_ghist_celltyping/code CUDA_VISIBLE_DEVICES=0 \
  /data/lf_data/task010_env/bin/python \
  /data/lf_data/result/task015_ghist_celltyping/code/task015_train_ghist_celltype.py \
  --fold 0 --pilot --epochs 5
```

Log: `/data/lf_data/result/task015_ghist_celltyping/logs/pilot_fold0_official.log`. Per-epoch metrics/checkpoints will be server-side. Final loss trend, validation count, unique IDs, finite probabilities, GPU-memory stability and pass/fail decision are **pending**, not assumed. HED augmentation is disabled because official `stainlib` is absent; no substitute stain method was introduced. Mixed precision is disabled for the initial stability pilot.
