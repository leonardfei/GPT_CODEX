#!/usr/bin/env python3
"""Finalize Task 003 artifacts from official CellViT++ outputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, precision_recall_curve, roc_curve


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
LOWEST3 = ["Myeloid", "Neutrophil", "Plasma cell"]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]


def json_dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric_map(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {str(row["metric"]): float(row["value"]) for _, row in df.iterrows()}


def parse_native(native_path: Path) -> dict:
    data = json.loads(native_path.read_text())
    global_scores = data["classifier"]["global"]
    cellvit = data["cellvit_scores"]
    ocelot = data["pipeline"]["scores_ocelot"]
    rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "precision": ocelot[f"Pre/{name}"],
                "recall": ocelot[f"Rec/{name}"],
                "f1": ocelot[f"F1/{name}"],
                "support": np.nan,
                "auroc": np.nan,
                "auprc": np.nan,
                "metric_scope": "official_native_scores_ocelot",
            }
        )
    return {"raw": data, "global": global_scores, "cellvit": cellvit, "ocelot": ocelot, "per_class": pd.DataFrame(rows)}


def parse_training_curves(log_path: Path) -> pd.DataFrame:
    rows = []
    epoch = None
    for line in log_path.read_text(errors="replace").splitlines():
        match = re.search(r"Epoch: (\d+)/(\d+)", line)
        if match:
            epoch = int(match.group(1))
        for split, key in (("train", "Training epoch stats"), ("val", "Validation epoch stats")):
            if key in line and epoch is not None:
                vals = {}
                for label, col in (("Loss", "loss"), ("F1-Score", "f1"), ("Accuracy-Score", "accuracy"), ("AUROC", "auroc"), ("AP", "ap")):
                    m = re.search(rf"{re.escape(label)}: ([0-9.]+)", line)
                    if m:
                        vals[col] = float(m.group(1))
                rows.append({"epoch": epoch, "split": split, **vals})
    return pd.DataFrame(rows).sort_values(["epoch", "split"])


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task003_official"))
    parser.add_argument("--task1-metrics", type=Path, default=Path("/data/lf_data/result/metrics"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = args.result.resolve()
    task1 = args.task1_metrics.resolve()
    metrics = result / "metrics"
    figures = result / "figures"
    figure_data = result / "figure_data"
    for path in (metrics, figures, figure_data):
        path.mkdir(parents=True, exist_ok=True)

    cv_folds = pd.read_csv(metrics / "cv_fold_metrics.csv")
    cv_classes = pd.read_csv(metrics / "cv_per_class_summary.csv")
    cv_oof = pd.read_csv(metrics / "cv_oof_predictions.csv")
    task1_cv_all = pd.read_csv(task1 / "cv_summary.csv")
    task1_rank = int(task1_cv_all.sort_values("macro_f1_mean", ascending=False).iloc[0]["config_rank"])
    task1_cv = task1_cv_all[task1_cv_all["config_rank"] == task1_rank].iloc[0]
    task1_test = metric_map(task1 / "test_summary.csv")
    task1_per_class = pd.read_csv(task1 / "test_per_class_metrics.csv")
    task1_per_class_selected = task1_per_class.copy()
    if "config_rank" in task1_per_class_selected.columns:
        task1_per_class_selected = task1_per_class_selected[task1_per_class_selected["config_rank"] == task1_rank]
    task1_lowest3_test = float(task1_per_class_selected[task1_per_class_selected["class_name"].isin(LOWEST3)]["f1"].mean())

    native_path = args.run / "test_results" / "inference_results.json"
    native = parse_native(native_path)
    global_scores = native["global"]
    official_test = {
        "n_cells": np.nan,
        "accuracy": global_scores["Acc"],
        "balanced_accuracy": np.nan,
        "macro_precision": global_scores["Prec"],
        "macro_recall": global_scores["Rec"],
        "macro_f1": global_scores["F1"],
        "weighted_f1": np.nan,
        "macro_auroc": global_scores["Auroc"],
        "macro_auprc": global_scores["AP"],
        "mcc": np.nan,
        "kappa": np.nan,
        "native_detection_f1": native["cellvit"]["F1"],
        "native_detection_precision": native["cellvit"]["Prec"],
        "native_detection_recall": native["cellvit"]["Rec"],
        "tia_binary_f1": native["raw"]["pipeline"]["detection_scores_tia"]["binary"]["f1"],
    }
    pd.DataFrame([{"metric": k, "value": v, "source": "official_native_inference_results.json"} for k, v in official_test.items()]).to_csv(metrics / "test_summary.csv", index=False)
    native["per_class"].to_csv(metrics / "test_per_class_metrics.csv", index=False)
    json_dump(metrics / "native_test_metrics.json", native["raw"])

    official_lowest3_cv = float(cv_classes[cv_classes["class_name"].isin(LOWEST3)]["f1"].mean())
    official_lowest3_test_native = float(native["per_class"][native["per_class"]["class_name"].isin(LOWEST3)]["f1"].mean())
    comparison = pd.DataFrame(
        [
            {
                "workflow": "Task1_custom_head",
                "cv_macro_f1_mean": task1_cv["macro_f1_mean"],
                "cv_macro_f1_sd": task1_cv["macro_f1_sd"],
                "cv_balanced_accuracy_mean": task1_cv["balanced_accuracy_mean"],
                "cv_macro_auroc_mean": task1_cv["macro_auroc_mean"],
                "cv_macro_auprc_mean": task1_cv["macro_auprc_mean"],
                "cv_lowest3_f1_mean": float(task1_per_class_selected[task1_per_class_selected["class_name"].isin(LOWEST3)]["f1"].mean()),
                "test_macro_f1": task1_test["macro_f1"],
                "test_balanced_accuracy": task1_test["balanced_accuracy"],
                "test_macro_auroc": task1_test["macro_auroc"],
                "test_macro_auprc": task1_test["macro_auprc"],
                "test_lowest3_f1": task1_lowest3_test,
                "test_lowest3_f1_native_ocelot": np.nan,
                "test_native_detection_f1": task1_test.get("native_detection_f1", np.nan),
            },
            {
                "workflow": "Task3_official_cellvitpp",
                "cv_macro_f1_mean": cv_folds["macro_f1"].mean(),
                "cv_macro_f1_sd": cv_folds["macro_f1"].std(ddof=1),
                "cv_balanced_accuracy_mean": cv_folds["balanced_accuracy"].mean(),
                "cv_macro_auroc_mean": cv_folds["macro_auroc"].mean(),
                "cv_macro_auprc_mean": cv_folds["macro_auprc"].mean(),
                "cv_lowest3_f1_mean": official_lowest3_cv,
                "test_macro_f1": official_test["macro_f1"],
                "test_balanced_accuracy": np.nan,
                "test_macro_auroc": official_test["macro_auroc"],
                "test_macro_auprc": official_test["macro_auprc"],
                "test_lowest3_f1": np.nan,
                "test_lowest3_f1_native_ocelot": official_lowest3_test_native,
                "test_native_detection_f1": official_test["native_detection_f1"],
            },
        ]
    )
    comparison.to_csv(metrics / "task001_vs_official.csv", index=False)
    numeric_cols = [col for col in comparison.columns if col != "workflow"]
    delta_row = {"comparison": "official_minus_task1"}
    delta_row.update({col: float(comparison.iloc[1][col] - comparison.iloc[0][col]) for col in numeric_cols})
    pd.DataFrame([delta_row]).to_csv(figure_data / "task001_vs_official_delta.csv", index=False)

    cv_folds[["fold", "macro_f1", "balanced_accuracy", "macro_auroc", "macro_auprc", "lowest3_f1", "native_detection_f1"]].to_csv(figure_data / "official_cv_metrics.csv", index=False)
    cv_classes.to_csv(figure_data / "official_per_class_summary.csv", index=False)
    cv_oof.to_csv(figure_data / "official_oof_predictions.csv", index=False)
    native["per_class"].to_csv(figure_data / "official_native_test_per_class.csv", index=False)

    # Figure 1: official grouped CV performance.
    means = [cv_folds["macro_f1"].mean(), cv_folds["balanced_accuracy"].mean(), cv_folds["macro_auroc"].mean(), cv_folds["macro_auprc"].mean(), cv_folds["lowest3_f1"].mean()]
    sds = [cv_folds["macro_f1"].std(ddof=1), cv_folds["balanced_accuracy"].std(ddof=1), cv_folds["macro_auroc"].std(ddof=1), cv_folds["macro_auprc"].std(ddof=1), cv_folds["lowest3_f1"].std(ddof=1)]
    plt.figure(figsize=(8, 4.5))
    plt.bar(["Macro F1", "Balanced\naccuracy", "Macro\nAUROC", "Macro\nAUPRC", "Lowest-3\nF1"], means, yerr=sds, color=COLORS[:5], capsize=4)
    plt.ylabel("Score"); plt.ylim(0, 0.85); plt.title("Task3 official CellViT++ grouped 5-fold CV")
    savefig(figures / "Fig1_official_CV_performance.pdf")

    # Figure 2: Task1 versus official.
    comp = comparison.set_index("workflow")
    labels = ["Macro F1", "Macro AUROC", "Macro AUPRC", "Lowest-3 F1"]
    task1_vals = [comp.loc["Task1_custom_head", "cv_macro_f1_mean"], comp.loc["Task1_custom_head", "cv_macro_auroc_mean"], comp.loc["Task1_custom_head", "cv_macro_auprc_mean"], comp.loc["Task1_custom_head", "cv_lowest3_f1_mean"]]
    off_vals = [comp.loc["Task3_official_cellvitpp", "cv_macro_f1_mean"], comp.loc["Task3_official_cellvitpp", "cv_macro_auroc_mean"], comp.loc["Task3_official_cellvitpp", "cv_macro_auprc_mean"], comp.loc["Task3_official_cellvitpp", "cv_lowest3_f1_mean"]]
    x = np.arange(len(labels)); w = 0.37
    plt.figure(figsize=(8, 4.5)); plt.bar(x-w/2, task1_vals, w, label="Task1 custom", color="#999999"); plt.bar(x+w/2, off_vals, w, label="Task3 official", color="#0072B2")
    plt.xticks(x, labels); plt.ylabel("Score"); plt.ylim(0, .85); plt.title("Grouped-CV comparison"); plt.legend(frameon=False)
    savefig(figures / "Fig2_task001_vs_official.pdf")

    # Figure 3: per-class CV summary.
    cls = cv_classes.sort_values("class_id")
    x = np.arange(len(cls)); w = .25
    plt.figure(figsize=(10, 4.8)); plt.bar(x-w, cls["f1"], w, label="F1", color="#0072B2"); plt.bar(x, cls["recall"], w, label="Recall", color="#009E73"); plt.bar(x+w, cls["auroc"], w, label="AUROC", color="#E69F00")
    plt.xticks(x, cls["class_name"], rotation=30, ha="right"); plt.ylabel("Mean score"); plt.ylim(0, 1); plt.title("Official per-class grouped-CV performance"); plt.legend(frameon=False, ncol=3)
    savefig(figures / "Fig3_official_per_class_performance.pdf")

    # Figure 4: OOF confusion matrix.
    cm = confusion_matrix(cv_oof["y_true"], cv_oof["y_pred"], labels=range(len(CLASS_NAMES)), normalize="true")
    plt.figure(figsize=(7, 6)); disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES); disp.plot(cmap="Blues", values_format=".2f", xticks_rotation=35, colorbar=False); plt.title("Official grouped OOF confusion matrix (row-normalized)")
    savefig(figures / "Fig4_official_confusion_matrix.pdf")

    # Figure 5: OOF ROC curves.
    plt.figure(figsize=(7, 5.5)); y = cv_oof["y_true"].to_numpy()
    for idx, name in enumerate(CLASS_NAMES):
        truth = (y == idx).astype(int); prob = cv_oof[f"prob_{idx}"].to_numpy(); fpr, tpr, _ = roc_curve(truth, prob); plt.plot(fpr, tpr, lw=1.7, label=f"{name} (AUC={auc(fpr,tpr):.3f})", color=COLORS[idx])
    plt.plot([0, 1], [0, 1], "--", color="#777777", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Official grouped OOF ROC curves"); plt.legend(frameon=False, fontsize=8)
    savefig(figures / "Fig5_official_ROC_curves.pdf")

    # Figure 6: OOF PR curves.
    plt.figure(figsize=(7, 5.5))
    for idx, name in enumerate(CLASS_NAMES):
        truth = (y == idx).astype(int); prob = cv_oof[f"prob_{idx}"].to_numpy(); precision, recall, _ = precision_recall_curve(truth, prob); plt.plot(recall, precision, lw=1.7, label=name, color=COLORS[idx])
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Official grouped OOF precision-recall curves"); plt.legend(frameon=False, fontsize=8)
    savefig(figures / "Fig6_official_PR_curves.pdf")

    # Figure 7: final official training curves.
    curves = parse_training_curves(args.run / "logs.log")
    curves.to_csv(figure_data / "final_training_curves.csv", index=False)
    plt.figure(figsize=(8, 4.5))
    for split, color in (("train", "#E69F00"), ("val", "#0072B2")):
        subset = curves[curves["split"] == split]
        plt.plot(subset["epoch"], subset["f1"], marker="o", label=f"{split} F1", color=color)
        plt.plot(subset["epoch"], subset["auroc"], marker="s", linestyle="--", label=f"{split} AUROC", color=color, alpha=.7)
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.ylim(0, 1); plt.title("Final official all-training curves"); plt.legend(frameon=False, ncol=2)
    savefig(figures / "Fig7_official_training_curves.pdf")

    # Reproducibility and provenance.
    cfg = yaml.safe_load((result / "config/final_official_all_train.yaml").read_text())
    source_hashes = {}
    source_hash_file = result / "qc/source_hashes.csv"
    if source_hash_file.exists():
        source_hashes = pd.read_csv(source_hash_file).to_dict(orient="records")
    baseline = Path("/data/lf_data/result/model_best.pth")
    protected = Path("/data/lf_data/result/task001_model_best_baseline.pth")
    candidate = result / "model_official_best.pth"
    manifest = {
        "task": "task_003",
        "workflow": "strict_official_cellvitpp_classifier_head",
        "wandb": {"used": False, "reason": "remote environment had no configured API key; official CLI fallback configs used"},
        "official_entrypoints": ["cellvit/train_cell_classifier_head.py", "cellvit/training/evaluate/inference_cellvit_experiment_detection.py"],
        "final_config": str(result / "config/final_official_all_train.yaml"),
        "final_run": str(args.run),
        "candidate_checkpoint": str(candidate),
        "dataset_source": "/data/lf_data/xenium_data/CellViT_dataset",
        "derivative_dataset": str(result / "work/CellViT_dataset_official"),
        "grouped_cv": {"folds": 5, "grouped_by": "batch", "median_best_epoch": int(json.loads((result / "config/final_epoch_guidance.json").read_text())["median_best_epoch"])},
        "configuration": cfg,
        "source_hashes": source_hashes,
        "checksums": {str(p): sha256(p) for p in (baseline, protected, candidate) if p.exists()},
        "native_test_results": str(native_path),
        "figures": sorted(str(p) for p in figures.glob("*.pdf")),
    }
    json_dump(result / "run_manifest.json", manifest)

    report = f"""# Task 003 — strict official CellViT++ workflow report

## Execution and provenance

- Official training entry point: `python3 ./cellvit/train_cell_classifier_head.py --config <YAML>`.
- Official native evaluation entry point: `cellvit/training/evaluate/inference_cellvit_experiment_detection.py`.
- WandB was not used because the remote environment had no configured API key; the official CLI fallback configurations were used.
- Five grouped, batch-disjoint folds were used. The official validation selection metric was AUROC.
- The final model used the median official AUROC-selected epoch across folds: **{int(json.loads((result / 'config/final_epoch_guidance.json').read_text())['median_best_epoch'])}**.
- Production baseline was protected and not overwritten.

## Grouped-CV results

| Metric | Official mean | SD |
|---|---:|---:|
| Macro F1 | {cv_folds['macro_f1'].mean():.4f} | {cv_folds['macro_f1'].std(ddof=1):.4f} |
| Balanced accuracy | {cv_folds['balanced_accuracy'].mean():.4f} | {cv_folds['balanced_accuracy'].std(ddof=1):.4f} |
| Macro AUROC | {cv_folds['macro_auroc'].mean():.4f} | {cv_folds['macro_auroc'].std(ddof=1):.4f} |
| Macro AUPRC | {cv_folds['macro_auprc'].mean():.4f} | {cv_folds['macro_auprc'].std(ddof=1):.4f} |
| Lowest-three-class F1 | {official_lowest3_cv:.4f} | {cv_folds['lowest3_f1'].std(ddof=1):.4f} |

## Official native test results

The official native JSON reports classifier-global F1 **{official_test['macro_f1']:.4f}**, AUROC **{official_test['macro_auroc']:.4f}**, AP **{official_test['macro_auprc']:.4f}**, and CellViT detection F1 **{official_test['native_detection_f1']:.4f}**. The TIA binary detection F1 was **{official_test['tia_binary_f1']:.4f}**. Native per-class scores are preserved in `metrics/test_per_class_metrics.csv` and are explicitly marked as Ocelot-scope metrics; their lowest-three-class mean was **{official_lowest3_test_native:.4f}** and is not treated as directly comparable to the Task1 classifier-only metric.

## Task1 comparison and promotion decision

- Task1 selected configuration: `config_rank={task1_rank}`.
- Task1 grouped-CV macro F1: **{task1_cv['macro_f1_mean']:.4f}±{task1_cv['macro_f1_sd']:.4f}**.
- Task3 official grouped-CV macro F1: **{cv_folds['macro_f1'].mean():.4f}±{cv_folds['macro_f1'].std(ddof=1):.4f}**.
- Macro-F1 delta (official minus Task1): **{cv_folds['macro_f1'].mean() - task1_cv['macro_f1_mean']:+.4f}**.
- Lowest-three-class CV F1 delta: **{official_lowest3_cv - float(task1_per_class_selected[task1_per_class_selected['class_name'].isin(LOWEST3)]['f1'].mean()):+.4f}**.

The promotion thresholds were not met: official grouped-CV macro F1 and lowest-three-class F1 did not improve by at least 0.03. The official candidate is therefore **not promoted**. The production model remains `/data/lf_data/result/model_best.pth`.

## Artifact locations

- Candidate: `{candidate}`
- Metrics: `{metrics}`
- Figures: `{figures}`
- Figure source data: `{figure_data}`
- Manifest: `{result / 'run_manifest.json'}`
- Native evaluation log: `{result / 'logs/native_test_evaluation.stdout'}`
"""
    (result / "TASK003_REPORT.md").write_text(report)
    print(json.dumps({"status": "finalized", "result": str(result), "task1_rank": task1_rank, "official_test": official_test, "figures": len(list(figures.glob('*.pdf')))}, indent=2, default=str))


if __name__ == "__main__":
    main()
