# Task 015 CV temporary-disk failure and restart audit

Date: 2026-10-01 (Asia/Shanghai). Status: **RECOVERED; new CV run in progress**.

The first two-GPU `GHIST_CT_OFFICIAL` / `GHIST_CT_BALANCED` fold-0 selection runs stopped advancing after epoch 25/50. Both logs showed Python/PyTorch multiprocessing failing to create `pymp-*` directories in `/tmp` with `OSError: [Errno 28] No space left on device`. At diagnosis, the 50 GB root filesystem was 100% used with only 66 MB available; `/data` still had about 253 GB available and `/dev/shm` about 93 GB available. Neither fold had produced an outer-validation result.

The original selection checkpoint payloads contain model weights and epoch metadata but **not** AdamW optimizer moments or DataLoader/random-number-generator state. Loading epoch 25 into a newly initialized optimizer would change the planned training trajectory. Therefore the two stalled Task015 processes were stopped and both fold-0 selection runs were restarted from epoch 1 with their original fixed seeds and unchanged scientific configuration. No scientific thresholds, labels, folds, augmentation, class weights, model architecture, or checkpoint-selection rules were changed.

Preserved, not deleted:

- `/data/lf_data/result/task015_ghist_celltyping/work/failed_root_full_20261001_GHIST_CT_OFFICIAL_fold0/` — 25 model checkpoints and selection history;
- `/data/lf_data/result/task015_ghist_celltyping/work/failed_root_full_20261001_GHIST_CT_BALANCED_fold0/` — 25 model checkpoints and selection history;
- `/data/lf_data/result/task015_ghist_celltyping/logs/cv_GHIST_CT_OFFICIAL_fold0_failed_root_full_20261001.log`;
- `/data/lf_data/result/task015_ghist_celltyping/logs/cv_GHIST_CT_BALANCED_fold0_failed_root_full_20261001.log`;
- `/data/lf_data/result/task015_ghist_celltyping/logs/cv_launcher_failed_root_full_20261001.log`.

The launcher now creates `/data/lf_data/result/task015_ghist_celltyping/work/tmp` and exports `TMPDIR`, `TMP`, and `TEMP` to that path before starting Python. A preflight asserts that Python's `tempfile.gettempdir()` resolves there and that the data volume has more than 50 GiB free. An independent four-worker PyTorch DataLoader smoke test completed 16 batches using this directory. The restarted official and balanced fold-0 processes were observed alive with the new `TMPDIR` in their process environments, and each advanced past `train_epoch=1 step=100/399` with finite mean loss (`2.00420` and `2.07678`, respectively). `/data` had about 253 GB free at that check; root `/tmp` still had only about 62 MB. New top-level log: `/data/lf_data/result/task015_ghist_celltyping/logs/cv_launcher.log`.

The task remains `IN_PROGRESS`. Pilot QC remains valid; no GHIST outer-fold performance or model-promotion conclusion is available yet.
