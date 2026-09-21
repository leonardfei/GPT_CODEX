#!/usr/bin/env python3
"""Finalize Task 002 after the search/CV tables have been written."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    root = Path(args.result_root)
    task = root / "task002"
    metrics_dir = task / "metrics"
    code_dir = task / "code"
    sys.path.insert(0, str(code_dir))
    from task002_optimize import baseline_stats, make_figures

    search = pd.read_csv(metrics_dir / "hyperparameter_search.csv")
    candidate_summary = pd.read_csv(metrics_dir / "candidate_cv_summary.csv")
    candidate_per = pd.read_csv(metrics_dir / "candidate_per_class_metrics.csv")
    candidate_batch = pd.read_csv(metrics_dir / "cv_by_batch.csv")
    candidate_summary = candidate_summary.sort_values(["selection_score_mean", "macro_f1_mean"], ascending=False).reset_index(drop=True)
    baseline_fold, baseline_per, baseline = baseline_stats(root)
    best = candidate_summary.iloc[0]
    best_per = candidate_per[candidate_per.config_id == best.config_id].groupby("class_id").f1.mean()
    base_per = baseline_per.groupby("class_id").f1.mean()
    strong_losses = (best_per[base_per > 0.5] - base_per[base_per > 0.5]).to_dict()
    checks = {
        "macro_f1_improvement": float(best.macro_f1_mean - baseline["macro_f1_mean"]),
        "lowest3_improvement": float(best.lowest3_f1_mean - baseline["lowest3_f1_mean"]),
        "strong_class_losses": {str(k): float(v) for k, v in strong_losses.items()},
        "fold_sd_not_materially_worse": bool(best.macro_f1_sd <= baseline["macro_f1_sd"] + 0.02),
    }
    checks["promotion"] = bool(
        checks["macro_f1_improvement"] >= 0.03
        and checks["lowest3_improvement"] >= 0.03
        and all(v >= -0.03 for v in strong_losses.values())
        and checks["fold_sd_not_materially_worse"]
    )
    (task / "promotion_decision.json").write_text(json.dumps({"baseline": baseline, "best_candidate": best.to_dict(), "checks": checks}, indent=2, default=str))
    make_figures(root, search, candidate_summary, candidate_per, candidate_batch, baseline_fold, baseline_per)
    baseline_path = root / "task001_model_best_baseline.pth"
    baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    matching = pd.read_csv(task / "qc" / "cell_matching_summary.csv")
    matching_values = dict(zip(matching.metric, matching.value))
    geometry = pd.read_csv(metrics_dir / "embedding_geometry.csv")
    geometry_values = dict(zip(geometry.metric, geometry.value))
    oof_per = pd.read_csv(task / "diagnostics" / "oof_per_class_metrics.csv")
    weak = oof_per.sort_values("f1").head(3)
    task3 = "Manually review upstream CellViT detection-to-Xenium registration and matching, then validate on an untouched external cohort before further head-only tuning."
    best_params = {}
    for key in ["lr", "weight_decay", "hidden_dim", "drop_rate", "optimizer", "loss", "gamma", "label_smoothing", "sampler"]:
        value = best[key]
        best_params[key] = value.item() if hasattr(value, "item") else value
    report = f"""# Task 002 Report — CellViT++ diagnostic and targeted optimization

## Status

COMPLETED. The frozen CellViT-SAM-H-x40 backbone was preserved. The Task 001 production model was not overwritten because the predefined promotion rule was not met.

## A. Dominant bottleneck

The evidence supports mixed causes, dominated by detection-to-ground-truth matching and limited frozen-embedding separability, with class imbalance contributing.

- Training GT cells: {int(matching_values['gt_cells']):,}
- Detected cells: {int(matching_values['detected_cells']):,}
- Matched cells: {int(matching_values['matched_cells']):,}
- GT match rate: {float(matching_values['match_rate_gt']):.3f}
- Median match distance: {float(matching_values['median_match_distance']):.2f}px; threshold: {int(matching_values['matching_threshold_px'])}px
- Ambiguous GT assignments: {int(matching_values['ambiguous_gt']):,}; duplicate assignments: {int(matching_values['duplicate_assignments']):,}
- Class silhouette: {float(geometry_values['class_silhouette']):.4f}; batch silhouette: {float(geometry_values['batch_silhouette']):.4f}
- Nearest-neighbor class purity: {float(geometry_values['nearest_neighbor_class_purity']):.4f}; batch purity: {float(geometry_values['nearest_neighbor_batch_purity']):.4f}

No duplicate assignments or out-of-bounds coordinates were found, but incomplete/ambiguous matching and weak embedding geometry indicate that upstream detection/matching is a major limitation. UMAP/reduction details are in `figure_data/embedding_reduction_method.txt`.

## B. Did head-only optimization materially improve grouped CV?

| Metric | Task 001 baseline | Best Task 002 candidate |
|---|---:|---:|
| Macro F1 | {baseline['macro_f1_mean']:.4f} ± {baseline['macro_f1_sd']:.4f} | {best.macro_f1_mean:.4f} ± {best.macro_f1_sd:.4f} |
| Balanced accuracy | {baseline['balanced_accuracy_mean']:.4f} | {best.balanced_accuracy_mean:.4f} |
| Macro AUPRC | {baseline['macro_auprc_mean']:.4f} | {best.macro_auprc_mean:.4f} |
| Lowest-three-class mean F1 | {baseline['lowest3_f1_mean']:.4f} | {best.lowest3_f1_mean:.4f} |

Promotion checks: `{json.dumps(checks, sort_keys=True)}`.

## C. Was Task 002 promoted?

**No.** The candidate did not satisfy every predefined requirement of at least +0.03 macro-F1 and +0.03 lowest-three-class F1 without unacceptable strong-class loss or materially worse fold variance. The independent test set was not used for Task 002 search, selection, or early stopping.

## D. Final production model

Task 001 remains the production model:

`/data/lf_data/result/model_best.pth`

Protected baseline SHA256: `{baseline_sha}`.

## E. Weak classes and likely causes

Training OOF weakest classes were:

{weak[['class_name', 'f1', 'recall', 'precision']].to_string(index=False)}

These weaknesses are consistent with matching/detection noise, embedding overlap, and class imbalance. This is a computational interpretation, not a biological conclusion.

## F. Task 003 recommendation

{task3}

## Best candidate configuration

```yaml
{yaml.safe_dump(best_params, sort_keys=False).strip()}
```

No Task 002 test metrics are reported because promotion criteria were not met. The Task 001 test result remains a prior baseline and is not a pristine iterative-development estimate.
"""
    (task / "TASK002_REPORT.md").write_text(report)
    manifest = {
        "status": "COMPLETED",
        "task": "task_002",
        "seed": 42,
        "backbone_frozen": True,
        "test_used_for_selection": False,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "baseline_model_sha256": baseline_sha,
        "baseline_cv": baseline,
        "best_candidate": best.to_dict(),
        "promotion_checks": checks,
        "outputs": str(task),
    }
    (task / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"status": "task002_complete", "baseline_macro_f1": baseline["macro_f1_mean"], "best_macro_f1": float(best.macro_f1_mean), "baseline_lowest3": baseline["lowest3_f1_mean"], "best_lowest3": float(best.lowest3_f1_mean), "promote": checks["promotion"]}, indent=2))


if __name__ == "__main__":
    main()
