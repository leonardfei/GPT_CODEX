# Project Status

## Current task

Task 011-MUSK — gated MUSK benchmark on the canonical H&E cell-typing cohort — COMPLETED

Current execution: official local MUSK code/weights were loaded fully offline and the five-FOV frozen benchmark, true binary probes, geometry QC, and single-scale sensitivity were completed on the canonical 96,044-cell cohort. The final report and small reviewable outputs are synchronized locally and are included in the completion commit. No Hugging Face token or network model download was used.

## Previous completed task

Task 011 correction — fine-tuning promotion gate and true binary probes — COMPLETED

Corrected Task011 preferred development candidate:
- Midnight F1 final-block at FOV12
- macro-F1 0.4522 ± 0.0294
- macro-AUPRC 0.4893 ± 0.0428
- Neutrophil F1 0.2352 ± 0.1214
- Neutrophil one-vs-rest AUPRC 0.1976 ± 0.1181
- strict promotion gate PASS
- production unchanged

## New MUSK benchmark

The user has downloaded the official MUSK code and weights locally, uploaded them to the server, and extracted them. Task011-MUSK should now use local discovery first and run offline.

Official resources:
- code: lilab-stanford/MUSK
- weights: xiangjx/musk
- model: musk_large_patch16_384
- input: 384×384
- official image feature extraction for linear probe/MIL:
  - with_head=False
  - out_norm=False
  - ms_aug=True
  - return_global=True
- license: CC-BY-NC-ND-4.0, academic non-commercial use only

Task file:
`tasks/task_011_musk.md`

## Security requirement

Never commit, print, log, or persist Hugging Face credentials.

The access token must be supplied only through a process environment variable such as:
`HF_TOKEN`

Do not store the token in:
- GitHub
- scripts
- config files
- notebooks
- shell command arguments
- reports/logs

## Server/model access behavior

First search for an already-local official MUSK source and checkpoint.

Preferred local paths:
- /data/lf_data/models/musk
- /data/lf_data/models/MUSK
- /data/lf_data/models/xiangjx_musk
- /data/lf_data/models/MUSK-code

If weights/code are already local, benchmark fully offline.

If not local:
- use authenticated Hugging Face access only if HF_TOKEN exists;
- do not print or persist the token.

If server networking blocks Hugging Face:
- stop download stage with BLOCKED_NEEDS_LOCAL_MUSK_UPLOAD;
- do not use unofficial mirrors;
- preserve the benchmark task for execution after local-machine download/upload.

## Benchmark design

Canonical cohort:
`96,044` cells

Exact Task009 V3_CORE held-out folds reused.

Primary MUSK frozen FOV sweep:
- 8 μm
- 12 μm
- 16 μm
- 24 μm
- 56 μm

Validated scale:
`0.2125 μm/px`

Primary feature extraction:
official MUSK `ms_aug=True`.

After selecting BEST_MUSK_FOV, also test:
`ms_aug=False`
at the same FOV.

Metrics:
- seven-class macro-F1/AUPRC
- per-class metrics
- Neutrophil F1/AUPRC
- true N-vs-Myeloid binary probe
- true N-vs-T/B binary probe
- batch/spatial geometry
- paired comparison with Midnight frozen FOV12
- comparison with corrected Midnight F1 final-block

No MUSK fine-tuning in this task.

Completed result:
- Best frozen MUSK candidate: `MUSK_FOV8` (8 μm; 37 native pixels).
- Macro-F1: `0.3377 ± 0.0383`; macro-AUPRC: `0.3488 ± 0.0371`.
- Relative to frozen Midnight FOV12: macro-F1 `−0.0827`, Neutrophil F1 `−0.0227`, Neutrophil AUPRC `−0.0205`, with `0/5` outer folds improved.
- Official `ms_aug=True` outperformed the same-FOV single-scale sensitivity by `+0.0097` macro-F1 and `+0.0106` macro-AUPRC.
- Task012 primary replacement is **not recommended**; retain corrected Midnight `F1_FINAL_BLOCK` as the primary validation candidate.
- Production model, ground-truth labels, eligibility rules, and Task009 outputs were unchanged.

## Existing comparison baselines

Frozen:
- CellViT aligned macro-F1 0.3364
- Phikon SMALL macro-F1 0.3366
- Midnight FOV12 frozen macro-F1 0.4204

Current best development candidate:
- Midnight F1 final-block FOV12 macro-F1 0.4522

## Production guardrail

Current production:
`/data/lf_data/result/model_best.pth`

SHA256:
`f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164`

Production remains unchanged.

## Completion handoff

```text
Wait for the next assigned task.
```

Local-first requirement:
- discover the uploaded MUSK code and weight directories under `/data/lf_data/models/`;
- verify source/weights/provenance and SHA256;
- load fully offline;
- do not contact Hugging Face unless local discovery fails.

Codex must pull origin/main and follow AGENTS.md plus `tasks/task_011_musk.md`.

## Last update

2026-09-25
