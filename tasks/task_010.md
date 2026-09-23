# Task 010 — Corrected CellViT Alignment + Local Midnight-12k Integration

## Status
PENDING-CORRECTION-AND-MIDNIGHT — correct the CellViT token alignment issue, then run a five-representation local-only benchmark including the user-uploaded Midnight-12k.

## Why this correction is required

The previous Phase A produced a valid Phikon-v2 signal but the Task010 CellViT_TOKEN baseline is not trusted.

The previous implementation saved CellViT token features in extraction/DataLoader order and later merged the cohort metadata without explicitly reordering the token tensor to the post-merge cohort order. Length equality was checked, but exact feature-to-cell identity was not asserted. This can cause feature/label misalignment and can explain the near-random CellViT Task010 baseline.

Therefore:
- archive the pre-fix Task010 metrics/report as provenance;
- do not use the pre-fix CellViT_TOKEN metrics for any scientific conclusion;
- preserve the existing valid Phikon feature tensors, but verify their exact cell IDs before reuse;
- regenerate all canonical Task010 comparison metrics after strict per-cell alignment;
- integrate the locally uploaded Midnight-12k in the same corrected benchmark.

Do NOT modify frozen biological labels, Task008 eligibility, Task009 outputs, or the production model.

---

## 1. Canonical frozen cohort

The canonical shared cohort remains the already frozen:

`SHARED_DETECTED_CORE`

Expected size from Phase A:
`96,044` cells.

Canonical manifest:
`/data/lf_data/result/task010_representation_benchmark/metrics/shared_cell_manifest.csv.gz`

This cohort MUST NOT be redefined because Midnight-12k is now available.

Required canonical ordering:
- sort/order exactly as the saved shared manifest;
- use `cell_id` as the primary identity;
- retain a secondary composite key:
  `image|local_x|local_y|class_id`
  only for joining feature manifests that do not contain cell_id.

Create:
- `metrics/canonical_cell_order.csv.gz`
- `qc/canonical_alignment_audit.md`

The canonical order file must contain:
- canonical_index;
- cell_id;
- image;
- local_x;
- local_y;
- class_id;
- class_name;
- batch.

---

## 2. Archive the flawed pre-fix benchmark

Before changing canonical metrics, create:

`/data/lf_data/result/task010_representation_benchmark/archive/pre_alignment_fix/`

Copy or move SMALL metric/report artefacts from the pre-fix run into this archive, including at minimum:
- linear probe metrics;
- MLP metrics;
- binary diagnostic metrics;
- confusion flows;
- paired deltas;
- representation geometry;
- decision summary;
- any report snapshot necessary to reconstruct the previous interpretation.

Do NOT archive/remove the large valid Phikon feature tensors unless needed.

Create:
`qc/pre_alignment_fix_archive_manifest.md`

Explicitly state:
- Phikon SMALL/CONTEXT feature extraction itself was not identified as misaligned;
- the corrected run nevertheless re-validates Phikon IDs against the canonical cohort;
- all old comparison metrics are superseded by corrected canonical metrics.

---

## 3. Correct CellViT token alignment — mandatory

Existing files:
- `features/cellvit_tokens.pt`
- `features/cellvit_token_manifest.csv.gz`

The old token tensor may be reused only after reconstructing an exact identity map.

### 3.1 Build exact token identity

Join the CellViT token manifest to the canonical cohort using the strongest available identity:

Preferred:
`cell_id`

If the token manifest lacks cell_id, join on:
`image + local_x + local_y + class_id`

Then recover the canonical `cell_id`.

Required assertions:
1. token-manifest key uniqueness;
2. canonical cohort key uniqueness;
3. exactly one token per canonical cell;
4. exactly 96,044 canonical cells represented;
5. no extra/duplicate token identity after the join;
6. no missing canonical cell;
7. token tensor row count equals token manifest row count BEFORE reindexing;
8. after reindexing, the stored cell_id vector equals canonical_cell_order.cell_id element-for-element.

Do NOT accept a mere length equality check.

### 3.2 Save corrected CellViT tensor

Reindex CellViT token features to canonical cell order.

Save:
- `features/cellvit_tokens_aligned.pt`
- `features/cellvit_tokens_aligned_manifest.csv.gz`

The corrected tensor payload must include:
- features;
- ordered cell_ids;
- ordered composite keys;
- source checkpoint;
- alignment method;
- SHA256/checksum of the original token tensor if feasible.

Create:
`qc/cellvit_alignment_correction.md`

This audit must state how many rows changed positional index relative to the old tensor order.

### 3.3 Sanity check versus Task009

The corrected Task010 CellViT probe uses a different shared-cell universe/probe recipe from Task009, so exact equality is NOT required.

However:
- compare corrected CellViT macro-F1/AUPRC and Neutrophil metrics against Task009 V3_CORE;
- if corrected CellViT remains near random, investigate further before interpreting pathology-FM gains;
- explicitly determine whether the previous ~0.110 macro-F1 was primarily caused by alignment error.

---

## 4. Re-validate Phikon tensors against canonical cell IDs

Existing:
- `features/phikon_v2_small.pt`
- `features/phikon_v2_context.pt`

Both were saved with ordered `cell_ids`.

Before reuse assert:
- len(cell_ids) == 96,044;
- exact equality to canonical_cell_order.cell_id;
- no duplicate/missing IDs;
- feature row count matches IDs;
- finite features.

If exact equality fails:
- reorder using stored cell_ids;
- save corrected aligned copies.

Canonical filenames:
- `features/phikon_v2_small_aligned.pt`
- `features/phikon_v2_context_aligned.pt`

If already aligned exactly, symbolic/copy reuse is acceptable but record the audit.

Create:
`qc/phikon_alignment_audit.md`

Do NOT re-extract Phikon features unless the saved IDs are invalid.

---

## 5. Local Midnight-12k discovery — local only

The user has uploaded Midnight-12k to the server.

DO NOT attempt any network/Hugging Face download before local discovery.

Search likely paths:
- `/data/lf_data/models/midnight-12k`
- `/data/lf_data/models/midnight`
- `/data/lf_data/`
- user home/model/cache directories if needed.

Also search for uploaded archives containing names such as:
- midnight
- midnight-12k
- model.safetensors
- config.json

If an archive is found:
- validate archive integrity;
- extract into a dedicated directory under `/data/lf_data/models/`;
- do not overwrite Phikon or CellViT environments/files.

Use only the PUBLIC Midnight-12k model.
Do NOT use restricted Midnight-92k variants.

Load fully offline.
No network fallback.
No substitute encoder.

If Midnight is incomplete, mark:
`PARTIAL-CORRECTION-COMPLETE-MIDNIGHT-INCOMPLETE`
and report exact missing files, but still complete the CellViT correction and corrected Phikon benchmark.

---

## 6. Midnight provenance and official feature rule

Verify the locally uploaded model/config/README and record the actual runtime architecture.

Expected public Midnight-12k behavior from the official project:
- pathology DINOv2-family encoder;
- input 224 × 224;
- normalization mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), unless local official config states otherwise;
- classification embedding is concatenation of:
  - CLS token;
  - mean patch-token embedding.

Do NOT hard-code expected dimension.
Record runtime dimension.

Create:
- `config/midnight12k_provenance.json`
- `qc/midnight12k_local_provenance.md`

Record:
- local path;
- file list/sizes;
- SHA256 of weight file(s);
- architecture/model type;
- patch size;
- input preprocessing;
- feature extraction rule;
- embedding dimension;
- dtype;
- environment versions;
- GPU;
- offline loading method.

---

## 7. Reuse exactly the Phase A physical crops

Do not alter crop geometry after seeing Phikon results.

Validated project scale:
`0.2125 μm/px`

SMALL:
- 16 μm FOV;
- 75 × 75 native px.

CONTEXT:
- 56 μm FOV;
- 263 × 263 native px.

Both are centered on the same matched H&E nucleus centroid.

Use the existing canonical crop manifest:
`metrics/crop_manifest.csv.gz`

For Midnight:
- extract the exact same native physical fields from the original OME-TIFF;
- apply Midnight-specific official preprocessing after native crop extraction.

Do NOT use a different cell center or FOV for Midnight.

---

## 8. Extract Midnight features in canonical order

Generate:
- `features/midnight12k_small.pt`
- `features/midnight12k_context.pt`

Each payload MUST include:
- features;
- ordered cell_ids;
- condition;
- embedding dimension;
- preprocessing metadata;
- feature rule;
- seed.

After extraction assert:
`midnight_cell_ids == canonical_cell_ids`
element-for-element.

If extraction runs in a different order, explicitly reorder before saving canonical aligned tensors:
- `features/midnight12k_small_aligned.pt`
- `features/midnight12k_context_aligned.pt`

No NaN/Inf.

---

## 9. Corrected five-representation primary benchmark

Canonical representations:

1. CELLVIT_TOKEN_ALIGNED
2. PHIKON_V2_SMALL
3. PHIKON_V2_CONTEXT
4. MIDNIGHT12K_SMALL
5. MIDNIGHT12K_CONTEXT

All must use:
- exact same 96,044 canonical cells;
- exact same labels;
- exact same Task009 V3_CORE held-out batch folds;
- exact same standardization policy;
- exact same linear probe;
- exact same class weighting;
- exact same C/regularization policy;
- exact same seeds.

No encoder fine-tuning.

### Linear probe

Use multinomial softmax/logistic regression.

Keep the previous common fixed `C=1` if that is required for exact comparability, OR switch all five representations to the same nested training-only C-selection procedure. Do not use a different selection rule by representation.

If changing the probe recipe from Phase A, rerun all five conditions.

---

## 10. TRUE binary specialist probes — replace old diagnostics

The previous Task010 "binary" analysis derived pairwise probabilities from the seven-class model. That is not a true binary specialist classifier.

Archive those old metrics as pre-fix diagnostics.

Now train separate binary linear probes from scratch using only the relevant training cells.

### Binary A
`Neutrophil vs Myeloid`

### Binary B
`Neutrophil vs T and B`

For every outer fold and every representation:
- restrict training cells to the two target classes;
- fit StandardScaler on outer-training cells only;
- fit a class-balanced binary logistic/softmax linear probe;
- evaluate only on the corresponding two classes in the untouched outer validation batches.

Report:
- AUROC;
- AUPRC;
- balanced accuracy;
- F1;
- sensitivity;
- specificity;
- precision.

Save:
- `metrics/true_binary_neutrophil_vs_myeloid.csv`
- `metrics/true_binary_neutrophil_vs_tb.csv`
- `metrics/true_binary_summary.csv`

Do not label the old probability-ratio diagnostics as binary specialist models in the corrected report.

---

## 11. Fix MLP validation leakage

The old MLP used outer validation performance for early stopping/model selection. This must not be used as canonical MLP performance.

Archive old MLP metrics.

For each outer fold and each representation:

1. outer validation batches remain untouched until final evaluation;
2. use ONLY outer-training batches for model selection;
3. create grouped inner validation using training batches only;
4. select epoch/early stopping based only on inner validation;
5. then reinitialize the same architecture;
6. refit on ALL outer-training cells for the selected epoch count;
7. evaluate ONCE on outer validation.

If grouped inner validation has too few batches:
- use a deterministic prespecified epoch count chosen without outer validation;
- apply the exact same rule to all five representations.

Architecture remains:
embedding → Linear256 → ReLU → Dropout0.5 → Linear128 → ReLU → Dropout0.5 → Linear7

Save:
- `metrics/mlp_probe_fold_metrics_corrected.csv`
- `metrics/mlp_probe_summary_corrected.csv`
- `qc/mlp_nested_validation_audit.md`

---

## 12. Corrected metrics and comparisons

For the five canonical representations report:

Overall:
- accuracy;
- balanced accuracy;
- macro-F1;
- macro-AUPRC;
- macro-AUROC;
- weighted F1;
- MCC;
- lowest-three-class F1.

Per class:
- precision;
- recall;
- F1;
- AUROC;
- AUPRC;
- support.

Neutrophil:
- precision;
- recall;
- F1;
- AUROC;
- AUPRC.

Confusion flows:
- Neutrophil → Myeloid;
- Myeloid → Neutrophil;
- Neutrophil → T and B;
- T and B → Neutrophil;
- Neutrophil → Plasma;
- Plasma → Neutrophil.

Paired fold deltas:
- Phikon SMALL − corrected CellViT;
- Phikon CONTEXT − corrected CellViT;
- Midnight SMALL − corrected CellViT;
- Midnight CONTEXT − corrected CellViT;
- each CONTEXT − its corresponding SMALL;
- Midnight SMALL − Phikon SMALL;
- Midnight CONTEXT − Phikon CONTEXT.

---

## 13. Representation geometry — corrected five-way comparison

Run the same geometry diagnostics for all five canonical representations:
- within-class cosine distance;
- between-class cosine distance;
- class-separation ratio;
- within-batch cosine distance;
- between-batch cosine distance;
- batch-separation ratio;
- kNN class purity;
- kNN batch purity;
- silhouette score.

Use the same deterministic 10k-cell subset across ALL five representations.

Save the selected 10k canonical cell IDs:
`metrics/geometry_subset_cell_ids.csv`

This removes representation-specific sampling differences.

Interpret "batch" cautiously as spatial/batch structure because these batches derive from regions of the paired slide.

---

## 14. Cross-encoder agreement

Compare Phikon and Midnight at matched scales.

For SMALL and CONTEXT separately compute:
- per-cell predicted-class agreement;
- Cohen's kappa;
- correct/correct overlap;
- error/error overlap;
- Neutrophil true-positive overlap;
- one-correct/one-wrong fractions;
- fold-wise metric correlation.

Save:
`metrics/cross_encoder_agreement.csv`

Purpose:
determine whether any gain is pathology-FM general or specific to Phikon.

---

## 15. GT-centered upper bound

Existing Phikon GT-centered features/results may be reused after provenance verification.

Add Midnight GT-centered SMALL/CONTEXT on the same ALL_V3_CORE_GT_CENTERED universe if computationally feasible.

If Midnight GT-centered extraction is too expensive for the first corrected pass:
- complete the corrected shared-cohort five-way benchmark first;
- mark GT-centered Midnight as secondary pending;
- do not delay the main corrected conclusion.

---

## 16. Required canonical output names

Create corrected canonical outputs:

- `metrics/linear_probe_fold_metrics_corrected.csv`
- `metrics/linear_probe_summary_corrected.csv`
- `metrics/linear_probe_per_class_corrected.csv`
- `metrics/neutrophil_metrics_corrected.csv`
- `metrics/confusion_flows_corrected.csv`
- `metrics/true_binary_neutrophil_vs_myeloid.csv`
- `metrics/true_binary_neutrophil_vs_tb.csv`
- `metrics/true_binary_summary.csv`
- `metrics/representation_paired_deltas_corrected.csv`
- `metrics/representation_geometry_corrected.csv`
- `metrics/cross_encoder_agreement.csv`
- `metrics/mlp_probe_fold_metrics_corrected.csv`
- `metrics/mlp_probe_summary_corrected.csv`
- `metrics/decision_summary_corrected.json`

Do not overwrite the old files without archiving them first.

---

## 17. Required corrected figures

Generate corrected vector PDFs:

- Fig1_task010_corrected_design.pdf
- Fig2_corrected_macroF1_fiveway.pdf
- Fig3_corrected_macroAUPRC_fiveway.pdf
- Fig4_corrected_per_class_F1.pdf
- Fig5_corrected_neutrophil_metrics.pdf
- Fig6_corrected_true_binary_results.pdf
- Fig7_corrected_small_vs_context.pdf
- Fig8_corrected_representation_geometry.pdf
- Fig9_corrected_cross_encoder_agreement.pdf
- Fig10_corrected_fold_paired_deltas.pdf
- Fig11_corrected_MLP.pdf

---

## 18. Corrected interpretation rules

Do NOT retain the previous "Phikon +0.226 macro-F1 vs CellViT" claim unless it survives the corrected alignment.

The corrected report must explicitly answer:

A. Was the old CellViT Task010 baseline misaligned?
B. What proportion/count of CellViT token rows changed position after canonical reindexing?
C. What is the corrected CellViT macro-F1/AUPRC and Neutrophil F1/AUPRC?
D. How does corrected CellViT compare with Task009 V3_CORE?
E. Does Phikon SMALL still outperform corrected CellViT?
F. Does Phikon CONTEXT still underperform Phikon SMALL?
G. What are Midnight SMALL/CONTEXT metrics?
H. Does Midnight independently reproduce the SMALL > CONTEXT pattern?
I. Does Midnight independently outperform corrected CellViT?
J. Which encoder is strongest at SMALL?
K. Which encoder is strongest for Neutrophil?
L. What do the TRUE binary specialist probes show?
M. Does the corrected nested MLP change the ranking?
N. How large is spatial/batch separation for each representation?
O. Are Phikon and Midnight gains concordant at the per-cell level?
P. Is the dominant bottleneck truly CellViT representation, local morphology extraction, domain/spatial batch structure, or intrinsic H&E ambiguity?
Q. What Task011 experiment is justified?

---

## 19. Production and ground-truth guardrails

Do NOT:
- change `/data/lf_data/result/model_best.pth`;
- change Task007 labels;
- change Task008 eligibility;
- modify Task009 datasets/metrics;
- relabel cells based on model performance;
- fine-tune encoders in Task010.

Task010 remains diagnostic.

---

## 20. GitHub synchronization

After the corrected run:

1. set `tasks/task_010.md` to COMPLETED if Midnight shared-cohort Phase B is complete, otherwise PARTIAL with exact remaining item;
2. replace `reports/task_010_report.md` with a corrected report that clearly marks the previous CellViT baseline as superseded;
3. update `PROJECT_STATUS.md`;
4. commit:
   - correction scripts;
   - Midnight integration scripts;
   - small configs;
   - corrected metrics;
   - reports;
5. do not commit large feature tensors/crops/checkpoints;
6. preserve pre-fix artefacts under archive/provenance;
7. push main normally.

## 21. Final handoff

Return:
1. CellViT alignment audit;
2. number/percent rows reordered;
3. corrected five-way linear metrics;
4. corrected Neutrophil metrics;
5. true binary specialist metrics;
6. corrected MLP metrics;
7. Phikon SMALL vs CONTEXT delta;
8. Midnight SMALL vs CONTEXT delta;
9. corrected CellViT vs Phikon/Midnight deltas;
10. representation geometry;
11. cross-encoder agreement;
12. GT-centered status/results;
13. whether the prior Phase A conclusion changed;
14. strongest encoder/scale;
15. dominant bottleneck interpretation;
16. recommended Task011.

Do not modify production or frozen biological ground truth.
