# Project Status

## Current task

Task 010 — Corrected CellViT alignment + local Midnight-12k integration — PARTIAL-CORRECTED-FIVEWAY-COMPLETE; Midnight GT-centered secondary pending

## Last completed task

Task 009 — CellViT retraining with frozen panel-aware Xenium labels and recalibrated Neutrophil eligibility — COMPLETED; no production promotion

## Repository status

Tasks 001–009 are complete.

Task007 established the frozen Xenium 5K panel-aware biological annotation.

Task008 established and manually validated target-centered Neutrophil H&E eligibility.

Task009 regenerated all V3 datasets from the original OME-TIFF and showed that corrected ground truth alone did not materially improve the official SAM-H classifier:
- V3_CORE macro-F1 0.3330 ± 0.0222
- V3_EXTENDED macro-F1 0.3340 ± 0.0217
- V3_EXTENDED Neutrophil F1 0.1084
- Neutrophil end-to-end recall ~0.055

## Task010 corrected benchmark status

The locally uploaded Phikon-v2 Phase A completed on the frozen 96,044-cell SHARED_DETECTED_CORE. The subsequent correction reindexed all CellViT tokens by canonical composite identity and integrated local Midnight-12k.

Observed pre-fix Phikon results:
- PHIKON_V2_SMALL macro-F1 0.3366 ± 0.0243
- PHIKON_V2_CONTEXT macro-F1 0.2726 ± 0.0253
- PHIKON_V2_SMALL Neutrophil F1 0.1885
- PHIKON_V2_CONTEXT Neutrophil F1 0.1414

Review of the Phase A code identified a CellViT token-to-cell ordering error:
- CellViT tokens were saved in extraction/DataLoader order;
- cohort metadata was subsequently merged/reordered;
- the token tensor was not explicitly reindexed to the post-merge cohort order;
- only length equality was checked.

Therefore the pre-fix Task010 CellViT baseline (macro-F1 0.1102) is archived and superseded. All 96,044 rows changed positional index during canonical reindexing.

Corrected five-way linear macro-F1:
- CELLVIT_TOKEN_ALIGNED 0.3364 ± 0.0263
- PHIKON_V2_SMALL 0.3366 ± 0.0243
- PHIKON_V2_CONTEXT 0.2726 ± 0.0253
- MIDNIGHT12K_SMALL 0.4116 ± 0.0312
- MIDNIGHT12K_CONTEXT 0.2806 ± 0.0221

Midnight-12k was loaded fully offline from `/data/lf_data/models/midnight-12k`; weight SHA256 is `52c14f20386ca17c2af8a7bf32c31c352668a8fbf6aefc88d86be6eaa0c72ca1`. Production and frozen ground truth remain unchanged.

The existing Phikon feature tensors were revalidated against the canonical cell IDs before reuse; both retained tensors match the canonical order exactly.

## Current corrected Task010 objective

Correct the CellViT alignment bug and run a canonical five-representation benchmark:

1. CELLVIT_TOKEN_ALIGNED
2. PHIKON_V2_SMALL
3. PHIKON_V2_CONTEXT
4. MIDNIGHT12K_SMALL
5. MIDNIGHT12K_CONTEXT

The user has uploaded Midnight-12k locally. Codex must search local storage first and must not attempt a network download before local discovery.

Preferred local paths:
- /data/lf_data/models/midnight-12k
- /data/lf_data/models/midnight

If an archive is found, validate it and extract under /data/lf_data/models/.

Use only the public Midnight-12k model, never Midnight-92k restricted variants.

## Canonical shared cohort

Reuse and freeze the existing:
`/data/lf_data/result/task010_representation_benchmark/metrics/shared_cell_manifest.csv.gz`

Expected n:
`96,044`

Do not redefine the cohort after Midnight availability.

Canonical identity:
- primary: cell_id
- secondary join key when needed: image + local_x + local_y + class_id

Every feature tensor must be reordered/asserted against the same canonical cell order.

## Mandatory correction items

1. Archive pre-alignment-fix metrics/report.
2. Create canonical cell order.
3. Reindex CellViT tokens to exact canonical cell IDs.
4. Assert exact element-wise ID alignment for CellViT, Phikon, and Midnight.
5. Re-run corrected five-way linear probes.
6. Replace prior probability-ratio "binary" diagnostics with TRUE independently trained binary probes:
   - Neutrophil vs Myeloid
   - Neutrophil vs T and B
7. Fix MLP outer-validation leakage by using training-only inner grouped validation or a prespecified epoch rule.
8. Run corrected five-way geometry on the same deterministic 10k-cell subset.
9. Compute Phikon-vs-Midnight cross-encoder agreement.
10. Preserve production and frozen ground truth.

## Physical crop conditions

Validated H&E scale:
`0.2125 μm/px`

SMALL:
- ~16 μm FOV
- 75×75 native px

CONTEXT:
- ~56 μm FOV
- 263×263 native px

Midnight must use the exact same native crop centers/FOVs as Phikon, with only encoder-specific preprocessing differing.

## Model provenance

Phikon-v2:
- local path: /data/lf_data/models/phikon-v2
- SHA256: 261ae680fa699b3b951597fd57aa19c02ef735805acb104b93af69b36d928569
- frozen CLS-token embedding

Midnight-12k:
- locally uploaded by user
- path/SHA256/runtime architecture to be discovered and recorded during corrected Task010
- expected public classification embedding: concat(CLS token, mean patch tokens), subject to verification from local official files/config

## Current production model

`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Production remains unchanged.

## Task010 output root

`/data/lf_data/result/task010_representation_benchmark`

Pre-fix results must be preserved under:
`archive/pre_alignment_fix/`

Corrected canonical metrics use filenames ending in:
`_corrected`

## Pending tasks

- Complete the secondary Midnight GT-centered upper-bound analysis if required.
- Do not start Task011 until corrected five-way results are reviewed.
- Do not modify production.
- Do not modify Task007/Task008 ground truth.
- Do not modify Task009 outputs.

## Latest workflow files

- tasks/task_010.md
- reports/task_010_report.md
- PROJECT_STATUS.md
- scripts/python/task010_phikon_phase_a.py
- scripts/python/task010_corrected_midnight.py

## Next execution command

```text
Execute task_010.
```

Codex must pull origin/main before execution and follow AGENTS.md plus the corrected tasks/task_010.md.

## Last update

2026-09-23
