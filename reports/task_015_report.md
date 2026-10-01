# Task 015 report

Status: **IN_PROGRESS** (2026-10-01). Official GHIST commit `917456be305fc82e92293ea272812e79675e821c` was downloaded locally and uploaded to the server with matching checksums. Reproducing the Task010 patch-level CellViT pipeline mapped valid contours one-to-one to all 96,044 canonical cells. Exact folds and 30-pixel-overlap largest-area geometry passed QC; the classification-only adapter passed a split-safe loss smoke test. The five-epoch fold-0 pilot is running. No accepted pilot or five-fold metrics exist yet, and no model-promotion decision has been made.

See [the detailed task report](task_015_ghist_celltyping.md), [input audit](../qc/task015/input_audit.md), and [source provenance](../config/task015/ghist_source_provenance.json).
