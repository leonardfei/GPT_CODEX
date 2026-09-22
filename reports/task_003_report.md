# Task 003 — Strict Official CellViT++ Retraining Report

## Outcome

Task 003 completed as a strict official CellViT++ control experiment. The five grouped folds and the final all-training run used the official entry point:

```text
python3 ./cellvit/train_cell_classifier_head.py --config <YAML>
```

The native test evaluation used:

```text
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir <FINAL_OFFICIAL_LOGDIR> \
  --dataset_path /data/lf_data/xenium_data/CellViT_dataset \
  --cellvit_path /data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --input_shape 256 256
```

No custom PyTorch training loop, manual optimizer loop, or manual checkpoint writer was used. WandB was not used because the remote environment had no configured API key; official CLI fallback YAMLs were used.

## Data and split controls

- Train images: 5,002; test images: 6,107.
- Seven classes with labels 0–6.
- Five leakage-safe grouped folds from Task 1 were retained, grouped by batch with no train/validation batch overlap.
- Exact train/test image duplicates: 0.
- Source CellViT++ code hashes and environment records are in `/data/lf_data/result/task003_official`.
- Task 1 production checkpoint was protected and remained unchanged.

## Grouped-CV results

| Metric | Mean | SD |
|---|---:|---:|
| Macro F1 | 0.3324 | 0.0134 |
| Balanced accuracy | 0.3455 | 0.0195 |
| Macro AUROC | 0.7314 | 0.0167 |
| Macro AUPRC | 0.3413 | 0.0281 |
| Lowest-three-class F1 | 0.1748 | 0.0213 |

Official validation model selection used AUROC. The five best epochs were 10, 7, 4, 7, and 4; the final all-training run used the median of 7 fixed epochs.

## Native test result

The official native JSON reported:

- classifier-global F1/accuracy: 0.3986;
- classifier-global AUROC: 0.7439;
- classifier-global AP: 0.3495;
- CellViT detection F1: 0.3563;
- TIA binary detection F1: 0.3472.

Native per-class Ocelot scores are preserved separately and are marked with their scope. They are not treated as directly comparable to Task 1 classifier-only per-class metrics.

## Task 1 comparison and decision

Task 1 selected grouped-CV macro F1 was `0.3417±0.0218`; Task 3 official grouped-CV macro F1 was `0.3324±0.0134`, a delta of `-0.0092`. Lowest-three-class grouped-CV F1 was `0.2267` for Task 1 and `0.1748` for Task 3, a delta of `-0.0519`.

The required promotion thresholds were not met. The strict official candidate is therefore not promoted. Production remains:

```text
/data/lf_data/result/model_best.pth
```

The Task 3 candidate is retained for audit and reproducibility at:

```text
/data/lf_data/result/task003_official/model_official_best.pth
```

Complete remote artifacts (metrics, figures, native evaluation outputs, environment, source hashes, commands, manifest, and report) are under:

```text
/data/lf_data/result/task003_official
```
