# Task 015 report

Status: **IN_PROGRESS** (2026-10-08). Official GHIST commit `917456be305fc82e92293ea272812e79675e821c` was downloaded locally and uploaded with matching checksums. Canonical contours, exact folds, split-safe masks, overlap geometry and the pilot passed QC. The initial fold-0 disk failure and restart are documented. All ten two-variant five-fold CV runs are complete and 96,044 unique OOF cells per variant passed coverage/probability checks. Mean Macro-F1 is 0.4345 (OFFICIAL), 0.4225 (BALANCED), versus 0.4522 for corrected Midnight; Neutrophil AUPRC/F1 gains meet the prespecified moderate gate, but not the strong gate. Full-data OFFICIAL training and pixel-head QC remain in progress. No production model has been replaced.

See [the detailed task report](task_015_ghist_celltyping.md), [CV restart audit](../qc/task015/cv_restart_audit.md), [input audit](../qc/task015/input_audit.md), and [source provenance](../config/task015/ghist_source_provenance.json).
