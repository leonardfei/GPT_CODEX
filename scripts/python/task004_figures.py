#!/usr/bin/env python3
"""Generate Task 004 QC and comparison figures from server-side summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]
TIERS = ["ALL_MATCHED", "HIGH_CONFIDENCE", "ULTRA_HIGH_CONFIDENCE"]


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()


def tier_masks(features: pd.DataFrame) -> dict[str, pd.Series]:
    high = features.mutual_nearest_neighbor_matched_pool & (features.pair_distance_px <= 8) & ((features.distance_margin_px >= 4) | (features.distance_ratio <= .7)) & (features.n_gt_candidates_within_15px == 1)
    ultra = features.mutual_nearest_neighbor_matched_pool & (features.pair_distance_px <= 5) & ((features.distance_margin_px >= 5) | (features.distance_ratio <= .6)) & (features.n_gt_candidates_within_10px == 1) & (features.n_matched_detection_candidates_within_10px == 1)
    return {"ALL_MATCHED": pd.Series(True, index=features.index), "HIGH_CONFIDENCE": high, "ULTRA_HIGH_CONFIDENCE": ultra}


def load_tensor(path: Path) -> np.ndarray:
    return torch.load(path, map_location="cpu", weights_only=False).detach().cpu().numpy()


def run_dir(result: Path, tier: str, fold: int) -> Path:
    return sorted((result / "runs" / tier / f"fold_{fold}").glob("*/val_results/scores.json"))[-1].parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task004_high_confidence"))
    parser.add_argument("--source", type=Path, default=Path("/data/lf_data/xenium_data/CellViT_dataset"))
    args = parser.parse_args()
    result = args.result.resolve(); source = args.source.resolve()
    figures = result / "figures"; data_dir = result / "figure_data"; qc = result / "qc" / "overlay_examples"
    figures.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True); qc.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(result / "metrics/matching_pair_features.csv")
    masks = tier_masks(features)
    retention = pd.read_csv(result / "metrics/class_retention_by_tier.csv")
    cv = pd.read_csv(result / "metrics/cv_fold_metrics_by_tier.csv")
    summary = pd.read_csv(result / "metrics/cv_summary_by_tier.csv")
    classes = pd.read_csv(result / "metrics/cv_per_class_metrics_by_tier.csv")
    embedding = pd.read_csv(result / "metrics/embedding_separability_by_tier.csv")
    for name, frame in {"matching_pair_features": features, "class_retention": retention, "cv_fold_metrics": cv, "cv_summary": summary, "cv_per_class_metrics": classes, "embedding_separability": embedding}.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False)

    # Figure 1: representative matching overlays. One retained/excluded example per class and batch where possible.
    selected = []
    high_mask = masks["HIGH_CONFIDENCE"]
    ultra_mask = masks["ULTRA_HIGH_CONFIDENCE"]
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        for tier, mask in (("EXCLUDED_BY_HIGH_CONFIDENCE", ~high_mask), ("HIGH_CONFIDENCE", high_mask), ("ULTRA_HIGH_CONFIDENCE", ultra_mask)):
            sub = features[(features.class_id == cls_id) & mask].sort_values(["batch", "pair_distance_px"])
            batches = list(sub.batch.drop_duplicates())[:2]
            for batch in batches:
                rows = sub[sub.batch == batch]
                if len(rows):
                    selected.append((tier, rows.iloc[0]))
    with matplotlib.backends.backend_pdf.PdfPages(figures / "Fig1_matching_confidence_examples.pdf") as pdf:
        for idx, (tier, row) in enumerate(selected):
            image_path = source / "train" / "images" / str(row.image)
            if not image_path.exists(): image_path = image_path.with_suffix(".png")
            if not image_path.exists(): continue
            image = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(4.8, 4.8)); ax.imshow(image)
            ax.scatter([row.gt_x], [row.gt_y], marker="x", s=75, c="yellow", linewidths=2, label="Xenium/GT centroid")
            ax.scatter([row.det_x], [row.det_y], marker="o", s=60, facecolors="none", edgecolors="cyan", linewidths=2, label="CellViT centroid")
            ax.set_xlim(max(0, row.det_x - 50), min(256, row.det_x + 50)); ax.set_ylim(min(256, row.det_y + 50), max(0, row.det_y - 50)); ax.axis("off")
            ax.set_title(f"{tier}\n{row.class_name}; {row.batch}; d={row.pair_distance_px:.1f}px")
            ax.legend(fontsize=7, loc="upper right"); fig.tight_layout(); pdf.savefig(fig); fig.savefig(qc / f"overlay_{idx:03d}_{tier}.png", dpi=180); plt.close(fig)
    pd.DataFrame([{"tier": t, "image": r.image, "batch": r.batch, "class_id": r.class_id, "class_name": r.class_name, "pair_distance_px": r.pair_distance_px} for t, r in selected]).to_csv(data_dir / "matching_overlay_examples.csv", index=False)

    # Figure 2: distance distributions.
    plt.figure(figsize=(8, 4.8))
    for tier, color in zip(TIERS, COLORS[:3]):
        values = features.loc[masks[tier], "pair_distance_px"]
        plt.hist(values, bins=np.linspace(0, 15, 31), histtype="step", linewidth=2, color=color, label=f"{tier} (n={len(values):,})", density=True)
    plt.xlabel("Current pair distance (px)"); plt.ylabel("Density"); plt.title("CellViT–Xenium matching distances"); plt.legend(frameon=False)
    savefig(figures / "Fig2_matching_distance_distribution.pdf")

    # Figure 3: class retention.
    wide = retention.pivot(index="class_name", columns="tier", values="retention_rate").reindex(CLASS_NAMES)
    x = np.arange(len(CLASS_NAMES)); w = .25
    plt.figure(figsize=(10, 4.8))
    for offset, tier, color in [(-w, "ALL_MATCHED", "#999999"), (0, "HIGH_CONFIDENCE", "#0072B2"), (w, "ULTRA_HIGH_CONFIDENCE", "#D55E00")]: plt.bar(x + offset, wide[tier], w, label=tier, color=color)
    plt.xticks(x, CLASS_NAMES, rotation=30, ha="right"); plt.ylabel("Retention among ALL_MATCHED"); plt.ylim(0, 1.05); plt.title("Class retention by confidence tier"); plt.legend(frameon=False)
    savefig(figures / "Fig3_class_retention_by_tier.pdf")

    # Figure 4: embedding separability.
    emb_long = embedding.melt(id_vars="tier", value_vars=["class_silhouette", "batch_silhouette", "nearest_neighbor_class_purity", "nearest_neighbor_batch_purity"], var_name="metric", value_name="value")
    emb_long.to_csv(data_dir / "embedding_separability_long.csv", index=False)
    plt.figure(figsize=(9, 4.8)); x = np.arange(4); w = .25
    metrics = ["class_silhouette", "batch_silhouette", "nearest_neighbor_class_purity", "nearest_neighbor_batch_purity"]
    for offset, tier, color in [(-w, "ALL_MATCHED", "#999999"), (0, "HIGH_CONFIDENCE", "#0072B2"), (w, "ULTRA_HIGH_CONFIDENCE", "#D55E00")]: plt.bar(x + offset, [embedding.loc[embedding.tier == tier, m].iloc[0] for m in metrics], w, label=tier, color=color)
    plt.xticks(x, ["Class\nsilhouette", "Batch\nsilhouette", "Class NN\npurity", "Batch NN\npurity"]); plt.ylabel("Value"); plt.title("Frozen embedding separability"); plt.legend(frameon=False)
    savefig(figures / "Fig4_embedding_separability_by_tier.pdf")

    # Figure 5: fold-level macro F1.
    plt.figure(figsize=(8, 4.8)); x = np.arange(5)
    for tier, color in zip(TIERS, ["#999999", "#0072B2", "#D55E00"]):
        sub = cv[cv.tier == tier].sort_values("fold"); plt.plot(x, sub.macro_f1, marker="o", linewidth=2, label=tier, color=color)
    plt.xticks(x, [f"Fold {i}" for i in range(5)]); plt.ylabel("Macro F1"); plt.ylim(0, .6); plt.title("Grouped 5-fold macro F1 by confidence tier"); plt.legend(frameon=False)
    savefig(figures / "Fig5_CV_macroF1_by_tier.pdf")

    # Figure 6: per-class F1 means with fold points.
    mean_cls = classes.groupby(["tier", "class_id", "class_name"], as_index=False).f1.mean().rename(columns={"f1": "f1_mean"})
    mean_cls.to_csv(data_dir / "per_class_f1_by_tier.csv", index=False)
    plt.figure(figsize=(10, 5)); x = np.arange(len(CLASS_NAMES)); w = .25
    for offset, tier, color in [(-w, "ALL_MATCHED", "#999999"), (0, "HIGH_CONFIDENCE", "#0072B2"), (w, "ULTRA_HIGH_CONFIDENCE", "#D55E00")]:
        sub = mean_cls[mean_cls.tier == tier].sort_values("class_id"); plt.bar(x + offset, sub.f1_mean, w, label=tier, color=color)
    plt.xticks(x, CLASS_NAMES, rotation=30, ha="right"); plt.ylabel("Mean F1"); plt.ylim(0, .8); plt.title("Per-class grouped-CV F1"); plt.legend(frameon=False)
    savefig(figures / "Fig6_per_class_F1_by_tier.pdf")

    # Figure 7: weak-class recall.
    weak = classes[classes.class_id.isin([2, 3, 4])].groupby(["tier", "class_id", "class_name"], as_index=False).recall.mean()
    weak.to_csv(data_dir / "weak_class_recall_by_tier.csv", index=False)
    plt.figure(figsize=(8, 4.8)); x = np.arange(3); w = .25
    for offset, tier, color in [(-w, "ALL_MATCHED", "#999999"), (0, "HIGH_CONFIDENCE", "#0072B2"), (w, "ULTRA_HIGH_CONFIDENCE", "#D55E00")]:
        sub = weak[weak.tier == tier].sort_values("class_id"); plt.bar(x + offset, sub.recall, w, label=tier, color=color)
    plt.xticks(x, ["Myeloid", "Neutrophil", "Plasma cell"]); plt.ylabel("Mean recall"); plt.ylim(0, 1); plt.title("Weak-class grouped-CV recall"); plt.legend(frameon=False)
    savefig(figures / "Fig7_weak_class_recall_by_tier.pdf")

    # Figures 8–10: pooled validation confusion matrices.
    confusion_figure_numbers = {"ALL_MATCHED": 8, "HIGH_CONFIDENCE": 9, "ULTRA_HIGH_CONFIDENCE": 10}
    for tier in TIERS:
        ys, preds = [], []
        for fold in range(5):
            run = run_dir(result, tier, fold); ys.append(load_tensor(run / "val_results/gt.pt").astype(int).reshape(-1)); preds.append(load_tensor(run / "val_results/predictions.pt").astype(int).reshape(-1))
        cm = confusion_matrix(np.concatenate(ys), np.concatenate(preds), labels=range(len(CLASS_NAMES)), normalize="true")
        plt.figure(figsize=(7, 6)); disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES); disp.plot(cmap="Blues", values_format=".2f", xticks_rotation=35, colorbar=False); plt.title(f"{tier} pooled validation confusion matrix")
        savefig(figures / f"Fig{confusion_figure_numbers[tier]}_confusion_matrix_{tier}.pdf")
    print(f"generated {len(list(figures.glob('*.pdf')))} PDF figures")


if __name__ == "__main__":
    main()
