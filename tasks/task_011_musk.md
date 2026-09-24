# Task 011-MUSK — Gated MUSK benchmark on the canonical H&E cell-typing cohort

## Status
PENDING / READY-LOCAL-MODEL — user reports MUSK code and weights have been uploaded to the server and extracted. Execute using local files only; do not contact Hugging Face unless local discovery fails and explicit access is required.

## Goal

Benchmark the official MUSK pathology foundation model against the current corrected Midnight-12k pipeline on exactly the same H&E cell-typing cohort, labels, crop centers, and held-out folds.

This is a diagnostic extension of Task011. It does NOT replace Task012 independent validation.

Primary question:

> Does frozen MUSK provide better target-cell morphology representations than the current Midnight-12k FOV12 baseline and F1 final-block candidate?

## Security / credential guardrail

A Hugging Face access token must NEVER be:
- committed to GitHub;
- written into task files, scripts, config JSON, shell history, reports, logs, notebooks, or environment snapshots;
- printed by scripts;
- passed as a literal command-line argument.

Only read it from a process environment variable such as:
`HF_TOKEN`

Do not persist the token to disk.

If an authenticated model download succeeds, unset/remove the token from the process environment after download.

## Official MUSK sources

Use ONLY:
- code: `lilab-stanford/MUSK`
- weights: `xiangjx/musk`

Model:
`musk_large_patch16_384`

Official model card / README behavior to reproduce:
- input size 384 × 384;
- resize to 384, center crop 384;
- ImageNet inception normalization;
- vision feature extraction:
  - `with_head=False`
  - `out_norm=False`
  - `return_global=True`
- for linear-probe / MIL feature extraction, official MUSK recommends `ms_aug=True`;
- `ms_aug=False` will be retained as a sensitivity analysis at the best FOV.

Record the actual output dimension at runtime. Do not assume it.

License:
CC-BY-NC-ND-4.0 / academic non-commercial use only.
Do not commit, redistribute, or package the weights.
Do not commit derived large embeddings or fine-tuned MUSK checkpoints.

## 1. Access and local model discovery

Preferred local directories:
- `/data/lf_data/models/musk`
- `/data/lf_data/models/MUSK`
- `/data/lf_data/models/xiangjx_musk`
- `/data/lf_data/models/MUSK-code`

First search local storage for:
- official MUSK source package;
- `model.safetensors`;
- any Hugging Face snapshot for `xiangjx/musk`.

If both official code and weights are already local, use them and do NOT contact Hugging Face.

If weights are not local:
- authenticated download is permitted only if `HF_TOKEN` exists in the process environment;
- never print the token;
- never save credentials in the repository;
- download to a local model directory such as `/data/lf_data/models/musk`.

If server networking prevents Hugging Face access:
- stop the model-download part with status `BLOCKED_NEEDS_LOCAL_MUSK_UPLOAD`;
- do NOT modify benchmark labels or use an unofficial mirror;
- report the exact model/code files that need to be downloaded on the user's local computer and uploaded.

Required provenance:
- `config/musk_provenance.json`
- `qc/musk_access_audit.md`

Record:
- official repository/model identifiers;
- local source-code path;
- local weight path;
- SHA256 of weight file;
- MUSK source commit/release if recoverable;
- Python / torch / timm / torchvision / fairscale versions;
- exact preprocessing;
- feature extraction flags;
- output dimension;
- GPU/dtype;
- license;
- whether model was loaded entirely offline after download.

## 2. Canonical cohort and folds

Reuse Task010/Task011 canonical cohort EXACTLY:

`/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz`

Expected:
`96,044` cells.

Reuse exact Task009 V3_CORE outer folds:
`/data/lf_data/result/task009_v3_retraining/metrics/split_manifest.csv`

Do not:
- create new random folds;
- drop cells because of MUSK processing unless a technical failure is unavoidable;
- alter labels;
- alter Neutrophil eligibility.

All feature matrices must be ordered element-for-element by canonical `cell_id`.

## 3. H&E geometry

Original H&E:
`/data/lf_data/xenium_data/ID0060276.ome.tif`

Validated native scale:
`0.2125 μm/px`

Use the same matched H&E nucleus center as Task010/011.

Primary MUSK FOV sweep:

### MUSK_FOV8
8 μm per side.
Nearest odd native crop:
37 × 37 px.

### MUSK_FOV12
12 μm per side.
57 × 57 px.
This matches the current Midnight BEST_FOV.

### MUSK_FOV16
16 μm per side.
75 × 75 px.
This matches the Task010 Midnight SMALL comparison.

### MUSK_FOV24
24 μm per side.
113 × 113 px.

### MUSK_FOV56
56 μm per side.
263 × 263 px.
Retained as a large-context/CANVAS-scale reference, not assumed to be optimal for cell typing.

All native crops are subsequently passed through the official MUSK 384 × 384 preprocessing.

Create:
- `config/musk_crop_geometry.json`
- `metrics/musk_crop_manifest.csv.gz`
- paired QC montages for FOV8/FOV12/FOV16/FOV24/FOV56.

## 4. Primary MUSK feature extraction

For each FOV, frozen MUSK only.

Primary official feature:
```
model(
    image=x,
    with_head=False,
    out_norm=False,
    ms_aug=True,
    return_global=True
)[0]
```

Save canonical-aligned server-side features:
- `features/musk_fov8_ms.pt`
- `features/musk_fov12_ms.pt`
- `features/musk_fov16_ms.pt`
- `features/musk_fov24_ms.pt`
- `features/musk_fov56_ms.pt`

Each tensor must include ordered `cell_ids` and runtime dimension metadata.

Do not commit these large tensors.

Assertions:
- exactly 96,044 rows;
- exact cell-id equality to canonical order;
- no NaN/Inf;
- deterministic eval mode;
- no label-dependent extraction.

## 5. Primary linear-probe benchmark

Use exactly the same corrected Task010/011 linear-probe policy and outer folds.

For every MUSK FOV report:
- accuracy;
- balanced accuracy;
- macro-F1;
- macro-AUPRC;
- macro-AUROC;
- weighted F1;
- MCC if available;
- lowest-three F1;
- per-class precision / recall / F1 / AUROC / AUPRC.

Neutrophil-specific:
- precision;
- recall;
- F1;
- AUROC;
- one-vs-rest AUPRC.

TRUE binary probes, independently trained on outer-training cells:
- Neutrophil vs Myeloid;
- Neutrophil vs T/B.

Binary metrics:
- AUROC;
- AUPRC;
- F1;
- sensitivity;
- specificity;
- precision.

Use five-fold mean ± SD.

## 6. Direct comparison baselines

Include these existing corrected baselines in the same comparison table:

### Frozen baselines
- CELLVIT_TOKEN_ALIGNED: macro-F1 0.3364
- PHIKON_V2_SMALL: macro-F1 0.3366
- MIDNIGHT12K_FOV12 frozen: macro-F1 0.4204

### Current best development candidate
- MIDNIGHT F1_FINAL_BLOCK FOV12 corrected:
  - macro-F1 0.4522
  - macro-AUPRC 0.4893
  - Neutrophil F1 0.2352
  - Neutrophil one-vs-rest AUPRC 0.1976

Important:
A frozen MUSK linear probe is directly comparable to frozen Midnight/Phikon for representation quality.
Do NOT claim that frozen MUSK is inferior/superior to the fine-tuned Midnight model without explicitly noting that the latter was partially fine-tuned.

## 7. MUSK multiscale sensitivity

After selecting BEST_MUSK_FOV from the `ms_aug=True` primary sweep, extract the SAME FOV with:
`ms_aug=False`

Condition:
`MUSK_BEST_FOV_SINGLE_SCALE`

Compare official multiscale vs single-scale:
- macro-F1;
- macro-AUPRC;
- Neutrophil F1/AUPRC;
- true N-vs-Myeloid AUROC/AUPRC;
- true N-vs-T/B AUROC/AUPRC;
- batch/spatial representation purity.

This determines whether MUSK's official multiscale inference is actually beneficial for nucleus-centered cell typing.

## 8. Representation geometry and spatial/batch robustness

Use the same deterministic 10k canonical cell subset from Task010.

For each MUSK condition:
- class kNN purity;
- batch/spatial kNN purity;
- class separation;
- batch separation;
- silhouette if computationally practical.

Compare with:
- Midnight frozen FOV12;
- Midnight F1 final-block if feature geometry is available.

Do not interpret batch identity as purely technical batch because the data derive from one paired specimen and spatial regions.

## 9. Predefined MUSK decision rules

Select BEST_MUSK_FOV by:
1. highest mean macro-F1;
2. if within 0.01, higher Neutrophil one-vs-rest AUPRC;
3. if still tied, lower batch/spatial purity.

### Strong frozen MUSK gain over frozen Midnight
- macro-F1 >= +0.03
AND
- Neutrophil F1 or AUPRC >= +0.03
AND
- macro-F1 improves in >=4/5 folds
AND
- no batch/spatial purity increase >0.05.

### Moderate frozen MUSK gain
- macro-F1 +0.01 to +0.03
OR
- Neutrophil F1/AUPRC +0.02 to +0.03,
with consistent folds.

### No meaningful frozen gain
Below those thresholds or inconsistent.

Do NOT fine-tune MUSK in this task.
First establish frozen representation value and licensing implications.

## 10. Required outputs

Output root:
`/data/lf_data/result/task011_musk_benchmark`

Required metrics:
- `metrics/musk_fov_fold_metrics.csv`
- `metrics/musk_fov_summary.csv`
- `metrics/musk_per_class.csv`
- `metrics/musk_neutrophil_metrics.csv`
- `metrics/musk_true_binary.csv`
- `metrics/musk_true_binary_summary.csv`
- `metrics/musk_multiscale_sensitivity.csv`
- `metrics/musk_representation_geometry.csv`
- `metrics/musk_vs_existing_baselines.csv`
- `metrics/musk_paired_fold_deltas.csv`
- `metrics/decision_summary.json`

Required QC/config:
- `config/musk_provenance.json`
- `config/musk_crop_geometry.json`
- `qc/musk_access_audit.md`
- `qc/musk_alignment_audit.md`
- crop montages.

Required report:
`reports/task_011_musk_report.md`

## 11. Report questions

The report must answer:

A. Was official MUSK successfully accessed and loaded?
B. What exact source/weight revision and SHA256 were used?
C. What feature dimension was produced with ms_aug=True and False?
D. Which physical FOV is optimal?
E. Does MUSK FOV12 outperform frozen Midnight FOV12?
F. Does any MUSK FOV outperform Midnight F1 final-block despite being frozen?
G. What are MUSK Neutrophil F1 and one-vs-rest AUPRC?
H. What are true N-vs-Myeloid and N-vs-T/B results?
I. Does official multiscale inference help or hurt cell typing?
J. Does MUSK encode stronger spatial/batch identity than Midnight?
K. Is MUSK a better candidate than Midnight for Task012?
L. If MUSK is promising, what is the next permitted experiment under the model license?

## 12. Production and licensing guardrails

Do not modify:
`/data/lf_data/result/model_best.pth`

Do not change Task007/008 ground truth.

Do not commit:
- Hugging Face token;
- MUSK weights;
- large MUSK embeddings;
- MUSK fine-tuned checkpoints.

No MUSK fine-tuning in this task.

Task012 independent validation remains separate.

## 13. Git synchronization

After completion:
1. update this task status;
2. create `reports/task_011_musk_report.md`;
3. update `PROJECT_STATUS.md`;
4. commit scripts/config/small metrics/report only;
5. ordinary push;
6. never commit or log credentials.

## 14. Final handoff

Return:
1. model access/download status;
2. local MUSK paths and provenance;
3. best MUSK FOV;
4. frozen MUSK five-fold macro-F1/AUPRC;
5. per-class F1/AUPRC;
6. Neutrophil F1/AUPRC;
7. true binary metrics;
8. multiscale vs single-scale delta;
9. geometry/spatial robustness;
10. paired comparison with Midnight frozen;
11. comparison with Midnight F1 final-block;
12. whether MUSK should replace Midnight as the primary Task012 candidate;
13. license implications for any next-stage fine-tuning.

Production remains unchanged.
