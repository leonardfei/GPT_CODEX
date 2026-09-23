# CellViT alignment correction

The old CellViT token tensor and manifest each contained 96,044 rows. The old extraction order was joined to the canonical cohort using `image|local_x|local_y|class_id`; all 96,044 rows (100%) changed positional index. Exact key uniqueness, one-to-one coverage, no missing/extra cells, and element-wise canonical cell_id equality were asserted.

Corrected output: `features/cellvit_tokens_aligned.pt` with 1,280-dimensional features. The pre-fix 0.1102 macro-F1 comparison is superseded; corrected CellViT macro-F1 is 0.3364 ± 0.0263.
