#!/usr/bin/env python3
"""Write the Task 005 benchmark report after aggregation and figure generation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
CONDITIONS = ["SAM-H_RAW", "SAM-H_STAIN_NORMALIZED"]


def fmt(value: object) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def condition_label(condition: str) -> str:
    return "RAW" if condition == "SAM-H_RAW" else "STAIN_NORMALIZED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task005_backbone_domain"))
    args = parser.parse_args()
    result = args.result.resolve()
    inventory = pd.read_csv(result / "metrics/backbone_inventory.csv")
    summary = pd.read_csv(result / "metrics/cv_summary.csv").set_index("condition")
    classes = pd.read_csv(result / "metrics/cv_per_class_metrics.csv")
    detect = pd.read_csv(result / "metrics/per_class_detection_recall.csv")
    neutro = pd.read_csv(result / "metrics/neutrophil_end_to_end.csv").set_index("condition")
    emb = pd.read_csv(result / "metrics/embedding_domain_diagnostics.csv").set_index("condition")
    ranking = pd.read_csv(result / "metrics/model_ranking.csv").set_index("condition")
    decision = json.loads((result / "metrics/promotion_decision.json").read_text())
    production = Path("/data/lf_data/result/model_best.pth")
    production_sha = subprocess.check_output(["sha256sum", str(production)], text=True).split()[0]

    tested = inventory[inventory.official_load_success].backbone_name.unique().tolist()
    lines = [
        "# Task 005 — Multi-Backbone, Stain-Domain, and Neutrophil Detection Benchmark",
        "",
        "Status: **COMPLETED**. The available official-compatible backbone and both supported stain conditions were evaluated with the fixed official classifier recipe and the shared grouped five-fold splits. No test evaluation or production promotion was performed because the promotion criteria were not met.",
        "",
        "## A. Available and tested backbones",
        "",
        f"The checkpoint inventory found {len(inventory)} checkpoint(s). Officially loadable and tested: **{', '.join(tested)}**. UNI, Virchow, Virchow2 and ViT256 were not present in the checkpoint directory and were not downloaded. The available checkpoint is recorded in `metrics/backbone_inventory.csv` with SHA256, file size, architecture and embedding dimension.",
        "",
        "## B. Stain/domain conditions",
        "",
        "Both official DetectionDataset Macenko conditions were tested: RAW and STAIN_NORMALIZED. No custom stain algorithm was introduced. Both conditions used the same seven classes, the same batch-grouped folds, seed 42, and the same AdamW/head recipe.",
        "",
        "## C. Grouped-CV classification",
        "",
        "| Condition | Macro-F1 | Macro-AUPRC | Lowest-three F1 |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summary.loc[condition]
        lines.append(f"| {condition_label(condition)} | {fmt(row.macro_f1_mean)} ± {fmt(row.macro_f1_sd)} | {fmt(row.macro_auprc_mean)} ± {fmt(row.macro_auprc_sd)} | {fmt(row.lowest3_f1_mean)} ± {fmt(row.lowest3_f1_sd)} |")
    lines += ["", "Per-class precision, recall, F1, AUROC, AUPRC and support are in `metrics/cv_per_class_metrics.csv`.", "", "## D. Neutrophil classification and recognition", "", "| Condition | Precision | CV recall | F1 | AUPRC | Detection recall | Conditional classifier recall | End-to-end recall |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        cls = classes[(classes.condition == condition) & (classes.class_id == 3)]
        row = neutro.loc[condition]
        lines.append(f"| {condition_label(condition)} | {fmt(cls.precision.mean())} | {fmt(cls.recall.mean())} | {fmt(cls.f1.mean())} | {fmt(cls.auprc.mean())} | {fmt(row.detection_recall)} | {fmt(row.conditional_classifier_recall)} | {fmt(row.end_to_end_recall)} |")
    lines += ["", "The three quantities are kept separate: detection recall is any valid detector-to-GT pair; conditional recall is prediction of Neutrophil among paired Neutrophils; end-to-end recall uses all GT Neutrophils as denominator. End-to-end precision is not reported as a scalar because the archived detection audit does not provide the complete unmatched-detection false-positive denominator.", "", "## E. Per-class detection recall", "", "| Class | RAW recall | STAIN_NORMALIZED recall |", "|---|---:|---:|"]
    for class_id, class_name in enumerate(CLASS_NAMES):
        vals = [detect[(detect.condition == condition) & (detect.class_id == class_id)].detection_recall.iloc[0] for condition in CONDITIONS]
        lines.append(f"| {class_name} | {fmt(vals[0])} | {fmt(vals[1])} |")
    lines += ["", "Detection metrics and pairing-distance summaries are in `metrics/per_class_detection_recall.csv`; the underlying pair tables are in `qc/`.", "", "## F. Embedding/domain diagnostics", "", "| Condition | Class silhouette | Batch silhouette | Class NN purity | Batch NN purity |", "|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        row = emb.loc[condition]
        lines.append(f"| {condition_label(condition)} | {fmt(row.class_silhouette)} | {fmt(row.batch_silhouette)} | {fmt(row.nearest_neighbor_class_purity)} | {fmt(row.nearest_neighbor_batch_purity)} |")
    lines += ["", "Stain normalization changed the domain diagnostics as reported in `metrics/stain_effect_summary.csv`; it was not preferred solely for lowering batch structure.", "", "## G. Best conditions and promotion", "", f"Best overall primary-score condition: **{decision['best_overall_condition']}**. Best Neutrophil-F1 condition: **{decision['best_neutrophil_condition']}**.", "", "| Condition | Primary score | Macro-F1 | Neutrophil F1 | Eligible before test |", "|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        row = ranking.loc[condition]
        lines.append(f"| {condition_label(condition)} | {fmt(row.primary_score)} | {fmt(row.macro_f1_mean)} | {fmt(row.neutrophil_f1_mean)} | {bool(row.eligible_before_test)} |")
    lines += ["", f"Promotion decision: **{decision['status']}**. No condition met all predefined macro-F1, Neutrophil, lowest-three-class, class-loss and leakage guardrails. No final candidate was trained, no test evaluation was run, and no new candidate path exists.", "", "## H. Production model", "", f"Production remains `{production}` with SHA256 `{production_sha}`. Tasks 001–004 outputs and the production model were not overwritten.", "", "## I. Recommended Task 006 direction", "", "The evidence supports prioritizing a detection-model adaptation or context-aware representation with an explicit Neutrophil specialist objective, while preserving grouped batch validation. A larger backbone sweep should wait until additional compatible weights are actually available; stain normalization alone should not be treated as the primary solution.", "", "## Reproducibility and outputs", "", "- Official command: `python3 ./cellvit/train_cell_classifier_head.py --config <YAML>`.", "- Fixed seed: 42; five batch-grouped folds; seven classes unchanged.", "- Task5 output root: `/data/lf_data/result/task005_backbone_domain`.", "- Required metrics: `metrics/`.", "- Required vector figures and source data: `figures/` and `figure_data/`.", "- Environment, checkpoint checksums, configs and manifest: `config/`.", "- Task 006 was not started."]
    (result / "TASK005_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "COMPLETED", "best_overall": decision["best_overall_condition"], "best_neutrophil": decision["best_neutrophil_condition"], "promotion": decision["status"]}, indent=2))


if __name__ == "__main__":
    main()
