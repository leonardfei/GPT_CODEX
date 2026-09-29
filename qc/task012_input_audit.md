# Task012 input and design audit

Date: 2026-09-29. Server: `/data/lf_data`.

## Confirmed inputs

- Canonical cohort: `/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz` exists with 96,044 rows and 19 columns.
- `cell_id` is unique for all 96,044 rows.
- Required `cell_id`, `class_id`, `batch`, `he_x`, and `he_y` fields have zero missing values.
- Class IDs are exactly 0–6 with counts: Endothelial 13,339; Mesenchymal 16,988; Myeloid 15,994; Neutrophil 6,139; Plasma cell 8,539; T and B 21,018; Tumor 14,027.
- Eight source batches are present.
- Task009 CORE split manifest exists at `/data/lf_data/result/task009_v3_retraining/metrics/split_manifest.csv`; five rows define folds 0–4 with disjoint training and validation batches.
- Reusable corrected Task011 FOV12 crop cache exists at `/data/lf_data/result/task011_midnight_local_optimization/work/midnight_fov12_crops_uint8.npy` with approximately 893 MB on disk. Export code asserts exact shape `(96044, 57, 57, 3)` and `uint8` before training.
- Midnight-12k source checkpoint exists at `/data/lf_data/models/midnight-12k/model.safetensors` (approximately 4.3 GB). Its Task010-verified SHA256 is `52c14f20386ca17c2af8a7bf32c31c352668a8fbf6aefc88d86be6eaa0c72ca1`.
- H&E coordinates are native pixels; the validated scale is 0.2125 μm/px, so FOV12 uses an odd 57 × 57 native crop centered on the matched nucleus.

## Prespecified design retained

Training uses the corrected Task011 module without changing labels, split batches, crop centers, augmentation, class weights, optimizer, or the epoch-selection rule. Each outer validation fold is evaluated once. The full-data final model is trained separately and is not evaluated on its 96,044 training cells.

Output root: `/data/lf_data/result/final_model`. Existing production remains `/data/lf_data/result/model_best.pth`; its expected SHA256 is `f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164` and is checked before and after final export.
