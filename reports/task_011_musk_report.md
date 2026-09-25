# Task 011-MUSK — Frozen Official MUSK Benchmark Report

## Status and scope

Task 011-MUSK is **COMPLETED**. This was a frozen representation benchmark on the canonical H&E cell-typing cohort. It did not fine-tune MUSK, change the production model, change ground-truth labels, change eligibility rules, or alter Task009 outputs.

The benchmark used the exact canonical cohort of 96,044 cells and the exact Task009 V3_CORE five outer folds. The official MUSK source and checkpoint were already available on the server and were loaded entirely offline; no Hugging Face token or network model download was used.

## Executive result

The best frozen MUSK candidate was `MUSK_FOV8` (8 μm, 37 native pixels) with official multiscale augmentation:

| Metric | Five-fold mean ± SD |
|---|---:|
| Accuracy | 0.3766 ± 0.0345 |
| Balanced accuracy | 0.3637 ± 0.0328 |
| Macro-F1 | **0.3377 ± 0.0383** |
| Macro-AUPRC | **0.3488 ± 0.0371** |
| Macro-AUROC | 0.7401 ± 0.0242 |
| Weighted F1 | 0.3786 ± 0.0275 |
| MCC | 0.2641 ± 0.0407 |
| Lowest-three F1 | 0.2070 ± 0.0608 |

Relative to the frozen Midnight FOV12 representation, MUSK FOV8 had macro-F1 delta **−0.0827**, Neutrophil F1 delta **−0.0227**, and Neutrophil one-vs-rest AUPRC delta **−0.0205**. Macro-F1 improved in **0/5** paired outer folds. Under the prespecified decision gate, this is `NO_MEANINGFUL_GAIN`.

The Task012 recommendation is unchanged: retain corrected Midnight `F1_FINAL_BLOCK` as the primary validation candidate. MUSK remains a secondary frozen representation candidate and should not replace the primary candidate based on this benchmark.

## A–C. Official access and provenance

| Item | Observed value |
|---|---|
| Official source | `lilab-stanford/MUSK` |
| Local source on server | `/data/lf_data/models/MUSK-code` |
| Source commit | `714b666969c1911e5efe70d991140a21030f4ef3` |
| Official model | `xiangjx/musk` / `musk_large_patch16_384` |
| Weight path | `/data/lf_data/models/musk/model.safetensors` |
| Weight SHA256 | `83d9f2429ec3f04fc3bdc4bc6d8e6c78035ea196fdce9fe05d76ea0e8cad0375` |
| Runtime | Python 3.10.14; PyTorch 2.7.1+cu128; CUDA |
| GPU and dtype | NVIDIA RTX PRO 5000 72GB Blackwell; `torch.float16` |
| Runtime dimensions | 2,048 with `ms_aug=True`; 1,024 with `ms_aug=False` |
| Feature flags | `with_head=False`, `out_norm=False`, `return_global=True` |
| Preprocessing | resize 384, center crop 384×384, mean/std 0.5/0.5/0.5 |
| License | CC-BY-NC-ND-4.0; academic non-commercial use only |

The complete provenance and access audit are in `config/task011_musk/musk_provenance.json` and `qc/task011_musk/musk_access_audit.md`.

## D–F. Cohort, folds, and crop geometry

The canonical cell order and labels were reused without modification. Exact held-out folds were reused from Task009 V3_CORE. Native scale was 0.2125 μm/px. The five evaluated crops were 8, 12, 16, 24, and 56 μm, corresponding to native crop sides of 37, 57, 75, 113, and 263 pixels. All crops were passed through official MUSK 384×384 preprocessing.

The crop manifest and review montages are stored under `metrics/task011_musk/` and `figures/task011_musk/`. Fold equivalence is documented in `qc/task011_musk/fold_equivalence_task009.md`.

## G–H. Frozen MUSK FOV sweep

| Candidate | Native side | Macro-F1 | Macro-AUPRC | Macro-AUROC |
|---|---:|---:|---:|---:|
| MUSK_FOV8 | 37 px | **0.3377 ± 0.0383** | **0.3488 ± 0.0371** | **0.7401 ± 0.0242** |
| MUSK_FOV12 | 57 px | 0.3283 ± 0.0294 | 0.3379 ± 0.0322 | 0.7279 ± 0.0244 |
| MUSK_FOV16 | 75 px | 0.3165 ± 0.0250 | 0.3237 ± 0.0284 | 0.7181 ± 0.0238 |
| MUSK_FOV24 | 113 px | 0.2948 ± 0.0230 | 0.3033 ± 0.0247 | 0.7037 ± 0.0256 |
| MUSK_FOV56 | 263 px | 0.2523 ± 0.0258 | 0.2685 ± 0.0250 | 0.6749 ± 0.0320 |

The declared selection rule therefore chooses FOV8. The observed pattern is that larger context progressively reduced the frozen linear-probe scores in this cell-typing benchmark; this is a representation finding, not a biological conclusion.

## I. Best-FOV per-class metrics

Values below are five-fold means ± sample SD for MUSK FOV8.

| Class | F1 | AUPRC |
|---|---:|---:|
| Endothelial | 0.3892 ± 0.0559 | 0.3814 ± 0.0446 |
| Mesenchymal | 0.2793 ± 0.0679 | 0.3046 ± 0.0844 |
| Myeloid | 0.2038 ± 0.0423 | 0.2235 ± 0.0314 |
| Neutrophil | 0.1911 ± 0.1174 | 0.1465 ± 0.1021 |
| Plasma cell | 0.2649 ± 0.1217 | 0.2245 ± 0.1360 |
| T and B | 0.4415 ± 0.0816 | 0.5019 ± 0.1012 |
| Tumor | 0.5939 ± 0.0737 | 0.6595 ± 0.1062 |

The Neutrophil result is F1 `0.1911 ± 0.1174`; its independent one-vs-rest AUPRC is `0.1465 ± 0.1021`.

## J. Independent true binary probes

Each probe was trained independently within each outer training fold and evaluated on the corresponding held-out cells.

| Comparison | AUROC | AUPRC | F1 | Sensitivity | Specificity | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Neutrophil vs Myeloid | 0.6727 ± 0.0314 | 0.3968 ± 0.1573 | 0.4168 ± 0.1472 | 0.5127 ± 0.1246 | 0.7309 ± 0.1048 | 0.3708 ± 0.1686 |
| Neutrophil vs T/B | 0.6734 ± 0.0385 | 0.3513 ± 0.1911 | 0.3760 ± 0.1824 | 0.5303 ± 0.0973 | 0.7137 ± 0.0838 | 0.3181 ± 0.1918 |

These binary results show measurable separability, but they do not overcome the prespecified seven-class frozen-representation comparison against Midnight FOV12.

## K. Multiscale sensitivity

At the selected FOV8, official multiscale extraction (`ms_aug=True`) was compared with single-scale extraction (`ms_aug=False`):

| Metric | Multiscale | Single-scale | Single − multi |
|---|---:|---:|---:|
| Macro-F1 | 0.3377 | 0.3279 | −0.0097 |
| Macro-AUPRC | 0.3488 | 0.3382 | −0.0106 |
| Neutrophil F1 | 0.1911 | 0.1817 | −0.0093 |
| Neutrophil AUPRC | 0.1465 | 0.1341 | −0.0124 |
| N vs Myeloid AUROC | 0.6727 | 0.6632 | −0.0094 |
| N vs T/B AUROC | 0.6734 | 0.6398 | −0.0336 |

The primary `ms_aug=True` setting is therefore retained for the frozen MUSK result.

## L. Geometry, baseline comparison, and decision

For the deterministic 10,000-cell geometry subset at MUSK FOV8:

- class separation ratio: `1.0376`;
- batch separation ratio: `1.0951`;
- spatial separation ratio: `1.5749`;
- class kNN purity: `0.2794`;
- batch kNN purity: `0.5152`;
- spatial kNN purity: `0.0251`.

Relative to frozen Midnight FOV12, the batch-purity delta was `−0.1030` and the spatial-purity delta was `−0.00993`. These QC values do not alter the primary decision because the macro-F1 and Neutrophil gates were not met.

| Representation | Type | Macro-F1 | Macro-AUPRC | Neutrophil F1 | Neutrophil AUPRC |
|---|---|---:|---:|---:|---:|
| CellViT aligned | Frozen | 0.3364 | 0.3418 | 0.1885 | 0.1327 |
| Phikon SMALL | Frozen | 0.3366 | 0.3434 | 0.1885 | 0.1349 |
| Midnight FOV12 | Frozen | 0.4116 | 0.4327 | 0.2175 | 0.1620 |
| Midnight F1 final-block | Partially fine-tuned | **0.4522** | **0.4893** | **0.2352** | **0.1976** |
| MUSK FOV8 | Frozen | 0.3377 | 0.3488 | 0.1911 | 0.1465 |

The partially fine-tuned Midnight candidate is not directly comparable to frozen representations, but remains the strongest observed development candidate. No MUSK fine-tuning was performed. Any future MUSK fine-tuning requires an explicit academic non-commercial license review under CC-BY-NC-ND-4.0 and must not redistribute weights.

## Reproducibility, QC, and artefacts

- Script: `scripts/python/task011_musk.py`.
- Primary remote execution root: `/data/lf_data/result/task011_musk_benchmark`.
- Primary command: `/data/lf_data/task010_env/bin/python scripts/python/task011_musk.py --stage all`.
- Finalization command: `/data/lf_data/task010_env/bin/python scripts/python/task011_musk.py --stage finalize`.
- Seed: `20260924`.
- Canonical inputs: Task010 canonical cell order and Task009 V3_CORE split manifest, with matched H&E centers from the existing workflow.
- Reviewable local outputs: `metrics/task011_musk/`, `config/task011_musk/`, `qc/task011_musk/`, and `figures/task011_musk/`.
- Large MUSK feature tensors remain on the server and were not committed to Git.
- The previously observed `vips-magick.so` warning was non-fatal; all required extraction, probe, QC, and finalization stages completed.
- No credentials, tokens, raw source data, or model weights were committed.

The machine-readable final decision is `metrics/task011_musk/decision_summary.json`.
