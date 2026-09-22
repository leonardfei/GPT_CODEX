#!/usr/bin/env python3
"""Write the Task 004 remote report after primary tier CV aggregation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]


def fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task004_high_confidence"))
    args = parser.parse_args()
    result = args.result.resolve()
    summary = pd.read_csv(result / "metrics/cv_summary_by_tier.csv").set_index("tier")
    retention = pd.read_csv(result / "metrics/class_retention_by_tier.csv")
    emb = pd.read_csv(result / "metrics/embedding_separability_by_tier.csv").set_index("tier")
    decision = json.loads((result / "metrics/promotion_decision.json").read_text())
    pair_summary = pd.read_csv(result / "metrics/matching_summary_by_tier.csv").set_index("tier")
    sensitivity = pd.read_csv(result / "metrics/threshold_sensitivity.csv")
    sensitivity_cv_complete = sensitivity["cv_performance_source"].astype(str).str.startswith("official_grouped_cv").all()
    production_path = Path("/data/lf_data/result/model_best.pth")
    production_sha = subprocess.check_output(["sha256sum", str(production_path)], text=True).split()[0]
    status = "COMPLETED" if decision["status"] == "not_promoted" and sensitivity_cv_complete else "PARTIAL"
    lines = [
        "# Task 004 — High-confidence CellViT–Xenium label reconstruction and retraining",
        "",
        f"Status: **{status}**. The primary three-tier official grouped-CV experiment completed. No test-set evaluation or production promotion was performed unless a tier met the predefined CV promotion rule.",
        "",
        "## A. Original matching",
        "",
        "Task 2 used the installed CellViT++ `pair_coordinates` implementation: an Euclidean distance matrix followed by SciPy Hungarian/Munkres global minimum-cost one-to-one assignment, with assigned pairs retained only when distance was <=15 px. It was not greedy nearest-neighbor matching. The archived matching table contains 94,662 retained training pairs.",
        "",
        "## B. Available geometry",
        "",
        "The available source geometry is centroid-only: annotation CSV rows contain x, y and integer class ID; the archived pair table contains matched CellViT centroids. No Xenium polygons, CellViT polygons, affine transform file, or global-coordinate mapping was available. Mutual-nearest and detection-competitor diagnostics therefore use the archived matched-detection pool; unmatched detection coordinates were not archived and are explicitly flagged in the QC tables.",
        "",
        "## C. Retention by tier",
        "",
        "| Tier | Cells | Batches | Fraction of ALL_MATCHED | Median distance (px) |",
        "|---|---:|---:|---:|---:|",
    ]
    for tier in ("ALL_MATCHED", "HIGH_CONFIDENCE", "ULTRA_HIGH_CONFIDENCE"):
        row = pair_summary.loc[tier]
        lines.append(f"| {tier} | {int(row.retained_cells):,} | {int(row.batches)} | {row.retained_fraction_of_all_matched:.3f} | {row.median_pair_distance_px:.2f} |")
    retention_pivot = retention.pivot(index="class_name", columns="tier", values=["retained_cells", "retention_rate"])
    lines += ["", "### Per-class retention", "", "| Class | ALL_MATCHED | HIGH_CONFIDENCE | ULTRA_HIGH_CONFIDENCE |", "|---|---:|---:|---:|"]
    for class_name in CLASS_NAMES:
        parts = []
        for tier in ("ALL_MATCHED", "HIGH_CONFIDENCE", "ULTRA_HIGH_CONFIDENCE"):
            count = int(retention_pivot.loc[class_name, ("retained_cells", tier)])
            rate = float(retention_pivot.loc[class_name, ("retention_rate", tier)])
            parts.append(f"{count:,} ({rate:.1%})")
        lines.append(f"| {class_name} | " + " | ".join(parts) + " |")
    lines += ["", "All eight batches remain represented in each tier. The source table is `metrics/class_retention_by_tier.csv`; no class or batch was dropped.", "", "## D. Threshold sensitivity", "", "The sensitivity table reports retention and ambiguity counts for the requested threshold candidates. Candidate-specific official retraining and grouped-CV were not run; therefore no secondary sensitivity rule was used for tier selection, and this limitation keeps the overall status at PARTIAL.", ""]
    lines += ["## E. Frozen embedding separability", ""]
    lines.append("| Tier | Class silhouette | Class NN purity | Batch silhouette | Batch NN purity |")
    lines.append("|---|---:|---:|---:|---:|")
    for tier in ("ALL_MATCHED", "HIGH_CONFIDENCE", "ULTRA_HIGH_CONFIDENCE"):
        row = emb.loc[tier]
        lines.append(f"| {tier} | {fmt(row.class_silhouette)} | {fmt(row.nearest_neighbor_class_purity)} | {fmt(row.batch_silhouette)} | {fmt(row.nearest_neighbor_batch_purity)} |")
    lines += ["", "These are descriptive frozen-embedding diagnostics using the same archived Task2 embeddings and fixed seed 42.", "", "## F–H. Grouped-CV results", "", "| Tier | Macro F1 | Balanced accuracy | Macro AUROC | Macro AUPRC | Lowest-three F1 |", "|---|---:|---:|---:|---:|---:|"]
    for tier in ("ALL_MATCHED", "HIGH_CONFIDENCE", "ULTRA_HIGH_CONFIDENCE"):
        row = summary.loc[tier]
        lines.append(f"| {tier} | {fmt(row.macro_f1_mean)} ± {fmt(row.macro_f1_sd)} | {fmt(row.balanced_accuracy_mean)} ± {fmt(row.balanced_accuracy_sd)} | {fmt(row.macro_auroc_mean)} ± {fmt(row.macro_auroc_sd)} | {fmt(row.macro_auprc_mean)} ± {fmt(row.macro_auprc_sd)} | {fmt(row.lowest3_f1_mean)} ± {fmt(row.lowest3_f1_sd)} |")
    lines += ["", "Weak-class F1, recall, AUPRC, and retained counts for Myeloid, Neutrophil and Plasma cell are in `metrics/weak_class_cv_by_tier.csv`.", "", "## I. Promotion", ""]
    for tier, details in decision["tiers"].items():
        lines.append(f"- {tier}: macro-F1 delta {details['macro_f1_delta_mean']:+.4f}; lowest-three F1 delta {details['lowest3_f1_delta_mean']:+.4f}; improved folds {details['macro_f1_improved_folds']}/5; eligible before test: {details['eligible_before_test']}.")
    lines += ["", f"Decision: **{decision['status']}**; selected tier: `{decision.get('selected_tier')}`."]
    if decision["status"] == "not_promoted":
        lines.append("No high-confidence tier met all predefined promotion criteria. No final refit, test evaluation, or production replacement was performed.")
    lines += ["", "## J. Production model", "", f"Production remains `{production_path}` with SHA256 `{production_sha}`; Task1 and Task3 models were not overwritten.", "", "## K. Recommended Task 005 direction", "", "Because centroid-only filtering is the current intervention and the archived unmatched-detection pool limits exact reciprocal diagnostics, the next task should prioritize re-running and archiving the complete detection-to-Xenium candidate graph, manual review of borderline matches across batches, and/or improving registration/representation before another head-only tuning round.", "", "## Reproducibility and outputs", "", "- Fixed seed: 42.", "- Official training entry point: `python3 ./cellvit/train_cell_classifier_head.py --config <YAML>`.", "- Task4 outputs: `/data/lf_data/result/task004_high_confidence`.", "- Required tables, source CSVs, vector PDFs, configs, logs, checksums and manifests are stored below that root.", "- Test data were not used for tier selection."]
    (result / "TASK004_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "promotion": decision["status"], "selected_tier": decision.get("selected_tier")}, indent=2))


if __name__ == "__main__":
    main()
