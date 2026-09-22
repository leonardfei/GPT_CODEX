#!/usr/bin/env python3
"""Generate editable vector PDF figures for Task 005."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
CONDITIONS = ["SAM-H_RAW", "SAM-H_STAIN_NORMALIZED"]
PALETTE = {"SAM-H_RAW": "#999999", "SAM-H_STAIN_NORMALIZED": "#0072B2"}


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def label(condition: str) -> str:
    return "RAW" if condition == "SAM-H_RAW" else "STAIN_NORMALIZED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task005_backbone_domain"))
    args = parser.parse_args()
    result = args.result.resolve()
    figures, data = result / "figures", result / "figure_data"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    folds = pd.read_csv(result / "metrics/cv_fold_metrics.csv")
    summary = pd.read_csv(result / "metrics/cv_summary.csv")
    classes = pd.read_csv(result / "metrics/cv_per_class_metrics.csv")
    neut = pd.read_csv(result / "metrics/neutrophil_metrics.csv")
    detect = pd.read_csv(result / "metrics/per_class_detection_recall.csv")
    e2e = pd.read_csv(result / "metrics/neutrophil_end_to_end.csv")
    emb = pd.read_csv(result / "metrics/embedding_domain_diagnostics.csv")
    effect = pd.read_csv(result / "metrics/stain_effect_summary.csv")
    ranking = pd.read_csv(result / "metrics/model_ranking.csv")
    cm = pd.read_csv(result / "metrics/confusion_matrices.csv")

    # Figure 1: macro-F1 with individual fold points.
    folds.to_csv(data / "cv_fold_metrics.csv", index=False)
    folds[["condition", "fold", "macro_f1"]].to_csv(data / "Fig1_macro_F1.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for x, condition in enumerate(CONDITIONS):
        values = folds[folds.condition == condition].macro_f1.to_numpy()
        ax.scatter(np.full(len(values), x), values, color=PALETTE[condition], zorder=3)
        row = summary[summary.condition == condition].iloc[0]
        ax.errorbar(x, row.macro_f1_mean, yerr=row.macro_f1_sd, color=PALETTE[condition], capsize=4, fmt="o", markersize=8)
    ax.set_xticks(range(2), [label(c) for c in CONDITIONS]); ax.set_ylabel("Grouped-CV macro-F1"); ax.set_title("Task 005 backbone/stain benchmark")
    save(fig, figures / "Fig1_backbone_macroF1.pdf")

    # Figure 2: macro-AUPRC.
    folds[["condition", "fold", "macro_auprc"]].to_csv(data / "Fig2_macro_AUPRC.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for x, condition in enumerate(CONDITIONS):
        row = summary[summary.condition == condition].iloc[0]
        ax.errorbar(x, row.macro_auprc_mean, yerr=row.macro_auprc_sd, color=PALETTE[condition], capsize=4, fmt="o", markersize=8)
        ax.scatter(np.full(len(folds[folds.condition == condition]), x), folds[folds.condition == condition].macro_auprc, color=PALETTE[condition], alpha=.8)
    ax.set_xticks(range(2), [label(c) for c in CONDITIONS]); ax.set_ylabel("Grouped-CV macro-AUPRC"); ax.set_title("Macro-AUPRC by condition")
    save(fig, figures / "Fig2_backbone_macroAUPRC.pdf")

    # Figure 3: per-class F1.
    classes.to_csv(data / "cv_per_class_metrics.csv", index=False)
    classes[["condition", "fold", "class_id", "class_name", "f1"]].to_csv(data / "Fig3_per_class_F1.csv", index=False)
    means = classes.groupby(["condition", "class_id", "class_name"], as_index=False).f1.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.2)); x = np.arange(len(CLASS_NAMES)); width = .36
    for offset, condition in zip([-width / 2, width / 2], CONDITIONS):
        sub = means[means.condition == condition].sort_values("class_id")
        ax.bar(x + offset, sub["mean"], width, yerr=sub["std"], capsize=3, label=label(condition), color=PALETTE[condition])
    ax.set_xticks(x, CLASS_NAMES, rotation=30, ha="right"); ax.set_ylabel("F1"); ax.set_ylim(0, .8); ax.set_title("Per-class grouped-CV F1"); ax.legend(frameon=False)
    save(fig, figures / "Fig3_backbone_per_class_F1.pdf")

    # Figure 4: Neutrophil conditional metrics.
    neut.to_csv(data / "Fig4_neutrophil_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8)); x = np.arange(2); width = .22
    for offset, col, name in [(-width, "neutrophil_precision", "Precision"), (0, "conditional_classifier_recall", "Recall"), (width, "neutrophil_f1", "F1")]:
        vals = [neut[neut.condition == condition][col].iloc[0] for condition in CONDITIONS]
        ax.bar(x + offset, vals, width, label=name)
    ax.set_xticks(x, [label(c) for c in CONDITIONS]); ax.set_ylim(0, 1); ax.set_ylabel("Metric"); ax.set_title("Neutrophil conditional classifier metrics"); ax.legend(frameon=False)
    save(fig, figures / "Fig4_neutrophil_metrics.pdf")

    # Figure 5: per-class detection recall.
    detect.to_csv(data / "per_class_detection_recall.csv", index=False)
    detect.to_csv(data / "Fig5_per_class_detection_recall.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5.2)); x = np.arange(len(CLASS_NAMES))
    for offset, condition in zip([-width / 2, width / 2], CONDITIONS):
        sub = detect[detect.condition == condition].sort_values("class_id")
        ax.bar(x + offset, sub.detection_recall, width, label=label(condition), color=PALETTE[condition])
    ax.set_xticks(x, CLASS_NAMES, rotation=30, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("Detection recall"); ax.set_title("Per-class CellViT detection recall"); ax.legend(frameon=False)
    save(fig, figures / "Fig5_per_class_detection_recall.pdf")

    # Figure 6: Neutrophil detection/conditional/end-to-end recall.
    e2e.to_csv(data / "Fig6_neutrophil_end_to_end.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8)); x = np.arange(2); width = .24
    for offset, col, name in [(-width, "detection_recall", "Detection"), (0, "conditional_classifier_recall", "Conditional classifier"), (width, "end_to_end_recall", "End-to-end")]:
        ax.bar(x + offset, [e2e[e2e.condition == condition][col].iloc[0] for condition in CONDITIONS], width, label=name)
    ax.set_xticks(x, [label(c) for c in CONDITIONS]); ax.set_ylim(0, 1); ax.set_ylabel("Neutrophil recall"); ax.set_title("Neutrophil recognition decomposition"); ax.legend(frameon=False)
    save(fig, figures / "Fig6_neutrophil_end_to_end.pdf")

    # Figure 7: class vs batch embedding diagnostics.
    emb.to_csv(data / "embedding_domain_diagnostics.csv", index=False)
    emb.to_csv(data / "Fig7_embedding_domain_diagnostics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    x = np.arange(2)
    axes[0].bar(x - width / 2, [emb[emb.condition == c].nearest_neighbor_class_purity.iloc[0] for c in CONDITIONS], width, label="Class purity")
    axes[0].bar(x + width / 2, [emb[emb.condition == c].nearest_neighbor_batch_purity.iloc[0] for c in CONDITIONS], width, label="Batch purity")
    axes[0].set_xticks(x, [label(c) for c in CONDITIONS]); axes[0].set_ylim(0, 1); axes[0].set_title("Nearest-neighbor purity"); axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, [emb[emb.condition == c].class_silhouette.iloc[0] for c in CONDITIONS], width, label="Class silhouette")
    axes[1].bar(x + width / 2, [emb[emb.condition == c].batch_silhouette.iloc[0] for c in CONDITIONS], width, label="Batch silhouette")
    axes[1].set_xticks(x, [label(c) for c in CONDITIONS]); axes[1].set_title("Silhouette"); axes[1].legend(frameon=False)
    save(fig, figures / "Fig7_embedding_class_vs_batch.pdf")

    # Figure 8: RAW versus stain-normalized effect.
    effect.to_csv(data / "stain_effect_summary.csv", index=False)
    effect.to_csv(data / "Fig8_stain_normalization_effect.csv", index=False)
    cols = [("macro_f1_mean", "Macro F1"), ("macro_auprc_mean", "Macro AUPRC"), ("neutrophil_f1", "Neutrophil F1"), ("conditional_classifier_recall", "Neutrophil recall"), ("neutrophil_auprc", "Neutrophil AUPRC"), ("lowest3_f1_mean", "Lowest-3 F1")]
    raw = effect[effect.condition == CONDITIONS[0]].iloc[0]; normalized = effect[effect.condition == CONDITIONS[1]].iloc[0]
    deltas = [normalized[col] - raw[col] for col, _ in cols]
    fig, ax = plt.subplots(figsize=(9, 4.8)); ax.bar(np.arange(len(cols)), deltas, color="#009E73"); ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(np.arange(len(cols)), [name for _, name in cols], rotation=30, ha="right"); ax.set_ylabel("STAIN_NORMALIZED − RAW"); ax.set_title("Stain-normalization effect")
    save(fig, figures / "Fig8_stain_normalization_effect.pdf")

    # Figure 9: confusion matrices for both conditions.
    cm.to_csv(data / "confusion_matrices.csv", index=False)
    cm.to_csv(data / "Fig9_confusion_matrices.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, condition in zip(axes, CONDITIONS):
        matrix = cm[cm.condition == condition].pivot_table(index="true_id", columns="pred_id", values="count", aggfunc="sum").reindex(index=range(7), columns=range(7), fill_value=0).to_numpy()
        row_norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        sns.heatmap(row_norm, ax=ax, cmap="Blues", vmin=0, vmax=1, cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot=True, fmt=".2f")
        ax.set_title(label(condition)); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    save(fig, figures / "Fig9_confusion_best_conditions.pdf")

    # Figure 10: primary score ranking.
    ranking.to_csv(data / "model_ranking.csv", index=False)
    ranking.to_csv(data / "Fig10_primary_score_ranking.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.8)); order = ranking.sort_values("primary_score"); ax.barh([label(c) for c in order.condition], order.primary_score, color=[PALETTE[c] for c in order.condition]); ax.set_xlabel("Primary score"); ax.set_title("Task 005 primary score ranking")
    save(fig, figures / "Fig10_primary_score_ranking.pdf")
    print(f"generated {len(list(figures.glob('*.pdf')))} PDF figures")


if __name__ == "__main__":
    main()
