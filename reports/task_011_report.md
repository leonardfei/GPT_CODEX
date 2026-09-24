# Task 011 Corrected Report — Fine-tuning Promotion Gate and Binary Probes

## Status and scope

Task 011 correction is complete. The previous Task 011 report's promotion statement is superseded because it used N-vs-Myeloid AUPRC as a substitute for the prespecified Neutrophil one-vs-rest AUPRC, called seven-class probability ratios “true binary” probes, and reported a fold-0 binary summary instead of a five-fold mean. The corrected rerun fixes all three issues without changing biological ground truth, eligibility rules, Task009 outputs, or the production checkpoint.

The analysis used the canonical 96,044-cell cohort, exact Task009 V3_CORE five-fold held-out batch splits, FOV12 = 57 native pixels, seed `20260923`, and the final transformer block with the original class-balanced fine-tuning rules. Old outputs were preserved remotely under `/data/lf_data/result/task011_midnight_local_optimization/archive/pre_task011_correction/` and recorded in `qc/task011_pre_correction_archive_manifest.md`.

## Corrected seven-class result

The corrected F1 final-block candidate achieved:

| Metric | Corrected five-fold mean ± SD |
|---|---:|
| Macro-F1 | 0.4522 ± 0.0294 |
| Macro-AUPRC | 0.4893 ± 0.0428 |
| Accuracy | 0.4964 ± 0.0339 |
| Balanced accuracy | 0.4792 ± 0.0301 |
| Lowest-three F1 | 0.2866 ± 0.0268 |
| Neutrophil F1 | 0.2352 ± 0.1214 |
| Neutrophil one-vs-rest AUPRC | 0.1976 ± 0.1181 |

Relative to frozen FOV12 (macro-F1 0.4204, Neutrophil F1 0.2138, Neutrophil one-vs-rest AUPRC 0.1671):

- macro-F1 gain: **+0.0318**;
- Neutrophil F1 gain: **+0.0214**;
- Neutrophil one-vs-rest AUPRC gain: **+0.0306**;
- macro-F1 improved in **5/5** outer folds;
- batch kNN purity delta: **+0.0483**;
- spatial kNN purity delta: **−0.00094**.

The strict Task 011 promotion gate therefore **passes**: the macro-F1 threshold, the Neutrophil one-vs-rest AUPRC alternative threshold, the fold-improvement requirement, and both purity guardrails are satisfied. This is a corrected conclusion; it is not carried forward from the previous, methodologically incorrect evidence.

## Corrected true binary probes

For each outer fold, an independent class-balanced binary linear probe was trained on embeddings from that fold's fine-tuned encoder. StandardScaler was fit only on outer-training embeddings, and evaluation was performed once on the corresponding outer validation set. Values below are five-fold mean ± sample SD.

| Comparison | AUROC | AUPRC | F1 | Sensitivity | Specificity | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Neutrophil vs Myeloid | 0.7202 ± 0.0221 | 0.4385 ± 0.1787 | 0.4561 ± 0.1453 | 0.5443 ± 0.1140 | 0.7627 ± 0.0805 | 0.4116 ± 0.1696 |
| Neutrophil vs T/B | 0.7069 ± 0.0359 | 0.3930 ± 0.2118 | 0.4020 ± 0.1803 | 0.5179 ± 0.1154 | 0.7641 ± 0.0738 | 0.3538 ± 0.1970 |

The independently trained probes are distinct from the following probability-ratio diagnostics, which are retained only for comparison and are not called true binary models:

| Probability-ratio diagnostic | AUROC | AUPRC | F1 |
|---|---:|---:|---:|
| Neutrophil vs Myeloid | 0.7236 ± 0.0278 | 0.4461 ± 0.1595 | 0.4537 ± 0.1445 |
| Neutrophil vs T/B | 0.7027 ± 0.0295 | 0.3779 ± 0.1982 | 0.3969 ± 0.1942 |

## Development decision and Task012 handoff

`F1_FINAL_BLOCK` remains both the highest observed development candidate and the strict gate-preferred candidate. The corrected result does not change the Task012 candidate, but it does replace the previous evidence with the corrected audit trail. Task012 should independently validate `F1_FINAL_BLOCK` on untouched slide/patient-level data; no production replacement is authorized by this development result alone.

No production model, ground-truth label, eligibility rule, or Task009 output was modified. Large embeddings, crops, and checkpoints remain remote and were not committed.

## Reproducibility and QC

- Correction scripts: `scripts/python/task011_finetune_correction.py` and `scripts/python/task011_finalize_correction.py`.
- Remote environment: `/data/lf_data/task010_env/bin/python` (Python 3.10.14, PyTorch 2.7.1+cu128, pandas 1.4.3, scikit-learn 1.3.0).
- Main correction command: `/data/lf_data/task010_env/bin/python code/task011_finetune_correction.py`.
- Finalization command: `/data/lf_data/task010_env/bin/python code/task011_finalize_correction.py --root /data/lf_data/result/task011_midnight_local_optimization`.
- Inner validation selected epoch 3 for all five folds; no outer validation labels were used for epoch selection.
- The only runtime warning was the previously observed non-fatal VIPS ImageMagick module warning; all five outer folds completed successfully.
- Corrected metrics and audit files are under `metrics/task011_correction/`, `config/task011_correction/`, and `qc/task011_correction/`.
