# Task 011 Correction — Fine-tuning Promotion Gate and True Binary Probes

## Status
COMPLETED

Completed 2026-09-24. The corrected five-fold rerun, independent binary probes, strict gate audit, archived provenance, report replacement, and Git synchronization were completed. See `reports/task_011_report.md` and the corrected metrics/config/QC directories.

## Purpose

Correct three methodological/reporting errors in the completed Task011 fine-tuning stage without changing the frozen biological ground truth or production model.

The errors to correct are:

1. The Task011 promotion gate incorrectly used N-vs-Myeloid AUPRC in place of Neutrophil one-vs-rest AUPRC.
2. The reported fine-tuned "true binary" endpoints were actually probability-ratio diagnostics from the seven-class model.
3. The final report summarized those pairwise diagnostics using the first/fold-0 row rather than the five-fold mean.

The underlying seven-class F1 fine-tuning result remains potentially valid and must be rerun/re-audited, not discarded.

## Inputs

Task011 output root:
`/data/lf_data/result/task011_midnight_local_optimization`

Canonical cohort:
`/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz`

Task009 folds:
`/data/lf_data/result/task009_v3_retraining/metrics/split_manifest.csv`

Midnight:
`/data/lf_data/models/midnight-12k`

Corrected scripts in repository:
- `scripts/python/task011_finetune_correction.py`
- `scripts/python/task011_finalize_correction.py`

Environment:
`/data/lf_data/task010_env/bin/python`

## 1. Preserve old outputs

Before rerunning, preserve the current Task011 fine-tuning/promotional artefacts as provenance.

Create:
`/data/lf_data/result/task011_midnight_local_optimization/archive/pre_task011_correction/`

Copy at minimum:
- metrics/fine_tune_fold_metrics.csv
- metrics/fine_tune_summary.csv
- metrics/fine_tune_per_class.csv
- metrics/fine_tune_true_binary.csv
- metrics/fine_tune_geometry.csv
- metrics/model_ladder.csv
- metrics/decision_summary.json
- config/fine_tuning_gate.json
- qc/fine_tune_inner_validation.csv

Do not delete the originals.

Create:
`qc/task011_pre_correction_archive_manifest.md`

## 2. Rerun fine-tuning correction

Run:
`/data/lf_data/task010_env/bin/python scripts/python/task011_finetune_correction.py`

The corrected run must preserve:
- canonical n=96,044;
- exact Task009 CORE outer folds;
- FOV12 = 57 native px;
- same inner grouped epoch-selection rule;
- final transformer block only;
- same class-balanced loss;
- same augmentation policy;
- same seed 20260923.

The corrected script must write only corrected-suffix outputs and must not overwrite the original fine-tuning metrics.

## 3. True binary probe requirement

For every outer fold after the fold-specific Midnight encoder has been fine-tuned on that fold's outer-training data:

### N vs Myeloid
- extract embeddings from the SAME fold-specific fine-tuned encoder for relevant outer-training cells;
- train a class-balanced binary linear probe using only outer-training embeddings/labels;
- fit StandardScaler only on outer-training embeddings;
- evaluate once on relevant outer-validation cells.

### N vs T/B
Same rule.

This is the canonical true-binary result.

Do not train a single binary probe on heterogeneous cross-fold OOF embeddings.

Do not use:
`p_N/(p_N+p_other)`
as the true binary classifier.

Probability-ratio results may be retained only as:
`pairwise_probability_diagnostic`.

## 4. Corrected promotion gate

Run:
`/data/lf_data/task010_env/bin/python scripts/python/task011_finalize_correction.py --root /data/lf_data/result/task011_midnight_local_optimization`

The promotion criteria MUST use:

- Macro-F1 gain relative to frozen FOV12;
- Neutrophil F1 gain relative to frozen FOV12;
- Neutrophil ONE-VS-REST AUPRC gain relative to frozen FOV12;
- number of outer folds with macro-F1 improvement;
- batch/spatial purity deltas.

Strict criterion:

`macro-F1 gain >= 0.03`
AND
`(Neutrophil F1 gain >= 0.03 OR Neutrophil one-vs-rest AUPRC gain >= 0.03)`
AND
`macro-F1 improves in >=4/5 folds`
AND
`batch purity increase <=0.05`
AND
`spatial purity increase <=0.05`.

Do not substitute pairwise AUPRC for one-vs-rest Neutrophil AUPRC.

## 5. Corrected binary summary

Canonical true binary results must report mean ± SD across all five outer folds for:
- AUROC;
- AUPRC;
- F1;
- sensitivity;
- specificity;
- precision.

At minimum:
- N vs Myeloid
- N vs T/B

Do not report fold 0 as the final summary.

## 6. Corrected canonical outputs

Required:

- metrics/fine_tune_fold_metrics_corrected.csv
- metrics/fine_tune_summary_corrected.csv
- metrics/fine_tune_per_class_corrected.csv
- metrics/fine_tune_pairwise_probability_diagnostics_corrected.csv
- metrics/fine_tune_true_binary_corrected.csv
- metrics/fine_tune_true_binary_summary_corrected.csv
- metrics/fine_tune_geometry_corrected.csv
- metrics/model_ladder_corrected.csv
- metrics/decision_summary_corrected.json
- config/fine_tuning_gate_corrected.json
- config/fine_tuning_correction_run.json
- qc/fine_tune_inner_validation_corrected.csv
- qc/task011_correction_audit.json

## 7. Interpretation rules

The corrected report must distinguish:

### Strict gate status
Whether F1_FINAL_BLOCK passes the predeclared Task011 promotion gate.

### Best observed development candidate
The model with the highest observed macro-F1 in Task011.

These are not necessarily the same.

If F1_FINAL_BLOCK has the highest observed macro-F1 but fails the strict promotion gate narrowly:
- do not call it "promoted";
- call it the "best observed development candidate";
- retain FOV12 as the strict gate-preferred frozen model;
- recommend independent validation of BOTH FOV12 and F1_FINAL_BLOCK.

## 8. Correct Task011 report

Replace:
`reports/task_011_report.md`

with a corrected report that explicitly states:
- the previous promotion statement is superseded;
- why it was wrong;
- corrected one-vs-rest Neutrophil AUPRC gain;
- corrected strict gate result;
- corrected true binary five-fold means;
- pairwise probability diagnostics separately;
- whether seven-class macro-F1 remains improved;
- whether the recommended Task012 candidate changes.

Do not hide the earlier error.

## 9. Update project status

Update:
`PROJECT_STATUS.md`

Only after corrected results exist.

Do not prepare Task012 before this correction is complete.

## 10. Git synchronization

Commit:
- the two correction scripts;
- corrected small metrics;
- corrected QC/config files;
- corrected Task011 report;
- task/status updates.

Do NOT commit:
- checkpoints;
- crop arrays;
- large embeddings;
- credentials.

Push main normally.

## 11. Final handoff

Return:
1. corrected seven-class macro-F1/AUPRC;
2. corrected Neutrophil F1/AUPRC;
3. macro-F1 gain vs frozen FOV12;
4. Neutrophil F1 gain vs frozen FOV12;
5. Neutrophil one-vs-rest AUPRC gain vs frozen FOV12;
6. strict promotion gate pass/fail;
7. outer-fold improvement count;
8. batch/spatial purity deltas;
9. true N-vs-Myeloid five-fold mean ± SD metrics;
10. true N-vs-T/B five-fold mean ± SD metrics;
11. pairwise probability diagnostics five-fold means separately;
12. best observed development candidate;
13. strict gate-preferred model;
14. Task012 validation recommendation.

Production and frozen biological ground truth must remain unchanged.
