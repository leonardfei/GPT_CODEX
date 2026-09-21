# Task 001 — CellViT++ classifier-head optimization

## Source instruction document

The attached task specification was supplied at `/Users/longfei/Downloads/task_001.md`. Its instructions define the scientific scope, remote paths, CellViT++ training/evaluation requirements, leakage controls, required metrics, figures, and output structure.

## User request

Execute the attached task through this project workflow using the authorized remote server session. The server credentials were used only for authentication and were not written to project files, logs, reports, or Git history.

## Scope constraints carried into execution

- Do not modify `/data/lf_data/xenium_data` or overwrite the pretrained CellViT++ backbone.
- Tune and train the classifier head only.
- Keep the test set untouched during search, grouped cross-validation, model selection, and final training.
- Use `/data/lf_data/result` for derived outputs, logs, metrics, figures, reports, and checkpoints.
- Do not force-push or commit credentials.

## Status

COMPLETED — see `reports/task_001_report.md` for the workflow audit and remote artifact paths.
