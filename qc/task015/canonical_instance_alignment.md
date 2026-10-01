# Task 015 canonical-to-instance alignment QC

Status: **PASS for reproducible Task010-equivalent patch-level mapping**, with an important ambiguity caveat. No GHIST model has been trained yet.

The independently executed whole-WSI binary CellViT run returned 620,205 nuclei. A naive nearest-center 15-pixel match covered only 93,126 of 96,044 canonical cells, with only 88,179 distinct nearest nuclei; this was rejected for supervised training. Its tiling differed from the Task010 patch-level detection that defined the frozen cohort.

We reran the exact Task010 `DetectionOnlyExperiment` CellViT-SAM-H-x40-AMP inference on its 5,005 original 256×256 training patches, with the same 15-native-pixel `pair_coordinates` rule, seed `20260923`, and no stain normalization. The run produced 252,921 patch-local binary detections. It matched **96,044/96,044 unique canonical cell IDs (100%)** to **96,044 unique `(image, instance_id)` pairs**; all mapped contours have at least three vertices. Patch-level contour and cell mappings are stored server-side under `/data/lf_data/result/task015_ghist_celltyping/data_manifest/patch_contours.jsonl.gz` and `/data/lf_data/result/task015_ghist_celltyping/metrics/canonical_instance_manifest.csv.gz`.

The matched nucleus centroids were compared against the frozen Task010 `cellvit_tokens_aligned_manifest.csv.gz`: all 96,044 match within `1e-6` native pixels; maximum numerical difference was `6.36e-14` pixels. Thus the contour assignment reproduces the existing detection-defined cohort rather than introducing a different matching scheme.

Matching-distance quantiles (pixels; 0/25/50/75/90/95/99/100%): `0.0393 / 4.4974 / 7.1845 / 9.9301 / 12.4839 / 13.6622 / 14.7222 / 14.9997`. **33,988/96,044 (35.39%)** matched cells had more than one detected candidate within 15 pixels. The one-to-one Hungarian pairing resolves these identically to Task010, but this does not prove biological assignment is unambiguous. This limitation must accompany model results.

The canonical cells occur on 4,814 of the 5,005 patches. No canonical-bearing patch contains multiple biological batches. All five exact Task010 outer folds have unique cell IDs, disjoint training/validation batches, and expected validation sizes `19,232 / 16,484 / 24,883 / 6,714 / 28,731`. The corrected training-only inner-validation batches are `s93` for folds 0–3 and `s22` for fold 4. Full machine-readable audit: `/data/lf_data/result/task015_ghist_celltyping/qc/patch_manifest_audit.json`.

No labels, thresholds, fold memberships, Task009–014 historical outputs, or production checkpoints were changed.
