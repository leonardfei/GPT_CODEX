# Task 001 Report — CellViT++ classifier-head optimization

## Status

COMPLETED on the authorized remote server. Scientific source data and the pretrained backbone were preserved.

## Input and execution

- Task specification: `/Users/longfei/Downloads/task_001.md`
- CellViT++ path: `/data/lf_data/CellViT-plus-plus`
- Dataset: `/data/lf_data/xenium_data/CellViT_dataset`
- Derived output root: `/data/lf_data/result`
- Pretrained checkpoint was read only; its SHA256 is recorded in the remote `run_manifest.json`.
- The CellViT++ directory is not a Git repository, so no source commit hash was available.

## QC and leakage control

- 5,002 training images and 6,107 test images were inspected; train/test exact image-hash overlap was zero.
- The supplied source folds mixed batches between train and validation. They were not used for model selection.
- Derived splits use `StratifiedGroupKFold(n_splits=5, random_state=42)` grouped by batch, with no train/validation batch overlap.
- The test set was not used for hyperparameter search, cross-validation, or final training.
- Original dataset files under `/data/lf_data/xenium_data` were not modified. Derived symlinks/cache and all outputs are under `/data/lf_data/result`.

## Model selection and final training

- 20 head-only candidate configurations were searched on the first grouped fold.
- The top three candidates were evaluated across all five grouped folds.
- Selection rule: mean macro F1, then balanced accuracy, macro AUROC/AUPRC, and lower variability.
- Selected head: AdamW, learning rate `0.0003`, weight decay `0.0001`, hidden dimension `128`, dropout `0.2`, inverse-square-root frequency weighting.
- Final head training used all 5,002 training images for 12 epochs, guided by the median best epoch from grouped CV.
- Final checkpoint: `/data/lf_data/result/model_best.pth`.

## Frozen-model test results

These classifier metrics cover 89,189 detected cells paired to ground truth by the native CellViT++ evaluation pathway; they are not whole-image metrics.

| Metric | Value |
|---|---:|
| Accuracy | 0.3883 |
| Balanced accuracy | 0.3514 |
| Macro F1 | 0.3440 |
| Weighted F1 | 0.3794 |
| Macro AUROC | 0.7400 |
| Macro AUPRC | 0.3441 |
| MCC | 0.2762 |
| Native detection F1 | 0.3563 |

Grouped test-batch bootstrap 95% intervals are in `/data/lf_data/result/metrics/test_bootstrap_ci.csv`.

## Remote artifacts

- Final report: `/data/lf_data/result/FINAL_REPORT.md`
- Run manifest: `/data/lf_data/result/run_manifest.json`
- Metrics: `/data/lf_data/result/metrics/`
- Figures and vector figure data: `/data/lf_data/result/figures/` and `/data/lf_data/result/figure_data/`
- Error analysis: `/data/lf_data/result/error_analysis/`
- Environment snapshots: `/data/lf_data/result/config/`
- Reproducible task scripts: `/data/lf_data/result/code/`

## Deviations and limitations

- The task's supplied fold files were replaced only in the derived result directory because their batch assignments leaked across train and validation; source data and source fold files were preserved.
- The remote project lacks Git metadata; the checkpoint hash, environment snapshot, configs, scripts, logs, and manifest provide the execution audit trail.
- Native CellViT++ inference reports classifier quality separately from detector quality; the final report preserves that distinction.

## Credentials and Git safety

The supplied server password was used only interactively and was not stored in any project file, remote result, commit, or report. No force-push was used.

## GitHub synchronization

- Remote: `https://github.com/leonardfei/GPT_CODEX.git`
- Branch: `main`
- Local completion commit: `091c48c`
- Push status: ordinary push attempted twice, but the current environment could not connect to `github.com:443`; the local commit is ready and the remote-tracking branch remains at `1c9e8b8` until connectivity is restored.
