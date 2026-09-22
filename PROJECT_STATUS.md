# Project Status

## Current task

Task 005 — multi-backbone, stain-domain, and Neutrophil detection benchmark — COMPLETED

## Last completed/closed task

Task 005 — multi-backbone, stain-domain, and Neutrophil detection benchmark — COMPLETED; no condition promoted and production unchanged

## Repository status

Tasks 001–003 are complete. Task 004 primary three-tier experiment is complete and has been reviewed. The remaining secondary threshold-sensitivity CV matrix is not required before proceeding because the primary result already showed that stricter centroid-only filtering did not materially improve seven-class or weak-class performance.

Task 005 has been completed under `/data/lf_data/result/task005_backbone_domain`. Only the installed official-compatible SAM-H checkpoint was available; RAW and official stain-normalized conditions were evaluated with the fixed recipe and grouped batch 5-fold splits. No condition met the promotion guardrails, so the production model remains unchanged and Task 006 has not started.

Scientific source data must remain unchanged.

Task 005 remote outputs must be written under:

`/data/lf_data/result/task005_backbone_domain`

## GitHub synchronization

- Repository: `leonardfei/GPT_CODEX`
- Branch: `main`
- Tasks 001–004 workflow reports/specifications are synchronized.
- Task 005 specification added at `tasks/task_005.md`.
- No force-push should be used.
- Credentials and gated-model tokens must not be committed.

## Current production model

`/data/lf_data/result/model_best.pth`

SHA256:

`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Task 001 remains the production model.

## Evidence motivating Task 005

### Task 001
- grouped-CV macro-F1: approximately 0.342
- test macro-F1: 0.3440

### Task 002
- GT match rate: 0.594
- ambiguous GT assignments: 36,775
- frozen SAM-H class silhouette: approximately -0.02
- nearest-neighbor class purity: approximately 0.25
- nearest-neighbor batch purity substantially higher than class purity

### Task 003
- strict official grouped-CV macro-F1: 0.3324 ± 0.0134
- strict official training did not outperform Task 001

### Task 004
- ALL_MATCHED macro-F1: 0.3324 ± 0.0134
- HIGH_CONFIDENCE macro-F1: 0.3401 ± 0.0164
- ULTRA_HIGH_CONFIDENCE macro-F1: 0.3226 ± 0.0199
- HIGH_CONFIDENCE improved macro-F1 only +0.0077
- lowest-three-class F1 decreased under stricter filtering
- Neutrophil F1 decreased by more than 0.03 in HIGH_CONFIDENCE
- stricter filtering increased nearest-neighbor batch purity more than class purity

These results suggest that the dominant remaining limitations are likely:
- backbone representation quality;
- histology/stain/domain shift;
- CellViT detection recall;
- insufficient Neutrophil-specific morphology representation.

## Task 005 primary questions

1. Which already-available official CellViT++ backbone best supports the seven-class HCC taxonomy?
2. Does official stain normalization reduce batch/domain structure without hurting class performance?
3. What is per-class detection recall?
4. What is Neutrophil detection recall?
5. What is Neutrophil conditional classifier recall?
6. What is Neutrophil end-to-end recall?
7. Which backbone/condition best balances overall macro-F1 and Neutrophil F1/AUPRC?

## Task 005 design

Primary candidate backbones, only if already installed and compatible:
- SAM-H
- UNI
- Virchow2
- optional Virchow
- optional ViT256 reference

Primary stain conditions:
- RAW
- official STAIN_NORMALIZED

All comparisons must use:
- the same seven classes;
- the same leakage-safe grouped 5-fold split definitions;
- the official CellViT++ classifier training stack;
- one fixed primary classifier recipe.

## Task 005 result

- SAM-H RAW: macro-F1 `0.3324 ± 0.0134`, macro-AUPRC `0.3413 ± 0.0281`, lowest-three F1 `0.1748 ± 0.0213`.
- SAM-H STAIN_NORMALIZED: macro-F1 `0.3295 ± 0.0121`, macro-AUPRC `0.3467 ± 0.0310`, lowest-three F1 `0.1603 ± 0.0185`.
- Best overall and best Neutrophil-F1 condition: SAM-H RAW.
- Neutrophil detection recall / conditional recall / end-to-end recall: RAW `0.5467 / 0.0824 / 0.0450`; normalized `0.5870 / 0.0678 / 0.0398`.
- Promotion: `not_promoted`; no final candidate or test evaluation; production model unchanged.
- Remote report: `/data/lf_data/result/task005_backbone_domain/TASK005_REPORT.md`.
- Local report: `reports/task_005_report.md`.

## Pending tasks

- Task 006 — not started; do not start until Task 005 is reviewed by Web GPT.

## Latest workflow files

- `tasks/task_001.md`
- `reports/task_001_report.md`
- `tasks/task_002.md`
- `reports/task_002_report.md`
- `tasks/task_003.md`
- `reports/task_003_report.md`
- `tasks/task_004.md`
- `reports/task_004_report.md`
- `tasks/task_005.md`
- `reports/task_005_report.md`
- `PROJECT_STATUS.md`

## Next execution command

```text
Execute task_005.
```

Codex must pull `origin/main` before execution and follow `AGENTS.md` plus `tasks/task_005.md`.

## Last update

2026-09-22
