#!/usr/bin/env python3
"""Aggregate Task-006 CV outputs and create the required comparison figures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


ROOT = Path("/data/lf_data/result/task006_xenium_reannotation")
TASK5 = Path("/data/lf_data/result/task005_backbone_domain")
SOURCE_H5AD = Path("/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad")
CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
N_CLASSES = len(CLASSES)


def load_tensor(path: Path) -> np.ndarray:
    return torch.load(path, map_location="cpu", weights_only=False).detach().cpu().numpy()


def best_epoch(log_path: Path) -> int:
    current = 0
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(r"Epoch: (\d+)/", line)
        if m:
            current = int(m.group(1))
        m = re.search(r"Validation epoch stats:.*F1-Score: ([0-9.]+).*AUROC: ([0-9.]+)", line)
        if m:
            rows.append((current, float(m.group(2))))
    return max(rows, key=lambda x: x[1])[0] if rows else 0


def task6_run(tier: str, fold: int) -> Path:
    candidates = sorted((ROOT / "runs" / f"XENIUM_V2_HQ_{tier}" / f"fold_{fold}").glob("*/val_results"))
    if not candidates:
        raise FileNotFoundError(f"missing completed Task6 run: {tier} fold {fold}")
    return candidates[-1].parent


def run_metrics(run: Path, condition: str, fold: int) -> tuple[dict, list[dict]]:
    val = run / "val_results"
    y = load_tensor(val / "gt.pt").astype(int).reshape(-1)
    pred = load_tensor(val / "predictions.pt").astype(int).reshape(-1)
    prob = load_tensor(val / "probabilities.pt").astype(float)
    labels = np.arange(N_CLASSES)
    y_bin = label_binarize(y, classes=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    auroc, auprc = [], []
    for i in labels:
        try:
            auroc.append(float(roc_auc_score(y_bin[:, i], prob[:, i])))
            auprc.append(float(average_precision_score(y_bin[:, i], prob[:, i])))
        except ValueError:
            auroc.append(float("nan"))
            auprc.append(float("nan"))
    row = {
        "condition": condition,
        "backbone": "SAM-H",
        "stain_condition": "RAW",
        "fold": fold,
        "run_dir": str(run),
        "best_epoch": best_epoch(run / "logs.log"),
        "n_cells": len(y),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "macro_auroc": float(np.nanmean(auroc)),
        "macro_auprc": float(np.nanmean(auprc)),
        "mcc": matthews_corrcoef(y, pred),
        "lowest3_f1": float(np.mean(f1[[2, 3, 4]])),
    }
    classes = []
    for i, name in enumerate(CLASSES):
        classes.append({
            "condition": condition,
            "backbone": "SAM-H",
            "stain_condition": "RAW",
            "fold": fold,
            "class_id": i,
            "class_name": name,
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "auroc": auroc[i],
            "auprc": auprc[i],
            "support": int(support[i]),
        })
    return row, classes


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, g in df.groupby("condition", sort=False):
        row = {
            "condition": condition,
            "backbone": "SAM-H",
            "stain_condition": "RAW",
            "n_folds": len(g),
        }
        for col in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc", "mcc", "lowest3_f1", "best_epoch"]:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_sd"] = g[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_folds = pd.read_csv(TASK5 / "metrics/cv_fold_metrics.csv")
    old_folds = old_folds[old_folds["condition"] == "SAM-H_RAW"].copy()
    old_folds["condition"] = "OLD_LABELS"
    old_folds["stain_condition"] = "RAW"
    old_per = pd.read_csv(TASK5 / "metrics/cv_per_class_metrics.csv")
    old_per = old_per[old_per["condition"] == "SAM-H_RAW"].copy()
    old_per["condition"] = "OLD_LABELS"
    old_per["stain_condition"] = "RAW"
    fold_rows = [old_folds]
    per_rows = [old_per]
    for tier in ("CORE", "EXTENDED"):
        fr, pr = [], []
        for fold in range(5):
            row, classes = run_metrics(task6_run(tier, fold), f"V2_{tier}", fold)
            fr.append(row)
            pr.extend(classes)
        fold_rows.append(pd.DataFrame(fr))
        per_rows.append(pd.DataFrame(pr))
    fold_df = pd.concat(fold_rows, ignore_index=True)
    per_df = pd.concat(per_rows, ignore_index=True)
    summary = summarize(fold_df)
    ROOT.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(ROOT / "metrics/cv_fold_metrics_old_vs_v2.csv", index=False)
    per_df.to_csv(ROOT / "metrics/cv_per_class_old_vs_v2.csv", index=False)
    summary.to_csv(ROOT / "metrics/cv_summary_old_vs_v2.csv", index=False)
    summary.to_csv(ROOT / "metrics/cv_summary.csv", index=False)
    per_df.to_csv(ROOT / "metrics/cv_per_class_metrics.csv", index=False)
    return fold_df, per_df, summary


def promotion(folds: pd.DataFrame, per: pd.DataFrame, summary: pd.DataFrame) -> dict:
    old = summary.loc[summary.condition == "OLD_LABELS"].iloc[0]
    old_fold = folds[folds.condition == "OLD_LABELS"].set_index("fold")
    rows = []
    for tier in ("CORE", "EXTENDED"):
        name = f"V2_{tier}"
        s = summary.loc[summary.condition == name].iloc[0]
        p = per[per.condition == name]
        n = p[p.class_name == "Neutrophil"]
        vf = folds[folds.condition == name].set_index("fold")
        common = old_fold.index.intersection(vf.index)
        improve = int((vf.loc[common, "macro_f1"] > old_fold.loc[common, "macro_f1"]).sum())
        max_loss = float((p.groupby("class_name").f1.mean() - per[per.condition == "OLD_LABELS"].groupby("class_name").f1.mean()).min())
        delta_macro = float(s.macro_f1_mean - old.macro_f1_mean)
        delta_lowest3 = float(s.lowest3_f1_mean - old.lowest3_f1_mean)
        delta_nf1 = float(n.f1.mean() - per[(per.condition == "OLD_LABELS") & (per.class_name == "Neutrophil")].f1.mean())
        delta_nauprc = float(n.auprc.mean() - per[(per.condition == "OLD_LABELS") & (per.class_name == "Neutrophil")].auprc.mean())
        meets = bool(
            delta_macro >= 0.03
            and delta_lowest3 >= 0.03
            and (delta_nf1 >= 0.05 or delta_nauprc >= 0.05)
            and improve >= 4
            and max_loss >= -0.05
            and (p.groupby(["fold", "class_name"]).support.min() > 0).all()
        )
        rows.append({
            "tier": tier,
            "condition": name,
            "eligible": meets,
            "macro_f1_delta": delta_macro,
            "lowest3_f1_delta": delta_lowest3,
            "neutrophil_f1_delta": delta_nf1,
            "neutrophil_auprc_delta": delta_nauprc,
            "folds_macro_f1_improved": improve,
            "minimum_class_f1_delta": max_loss,
            "test_run_allowed": False,
        })
    eligible = [r for r in rows if r["eligible"]]
    selected = max(eligible, key=lambda r: r["macro_f1_delta"]) if eligible else None
    decision = {
        "promotion_rule": {
            "macro_f1_delta_min": 0.03,
            "lowest3_f1_delta_min": 0.03,
            "neutrophil_f1_or_auprc_delta_min": 0.05,
            "folds_macro_f1_improved_min": 4,
            "minimum_class_f1_delta_min": -0.05,
        },
        "candidates": rows,
        "selected_tier": selected["tier"] if selected else None,
        "promotion_approved": bool(selected),
        "test_run_performed": False,
        "production_overwritten": False,
        "reason": "No v2 tier met all pre-specified promotion criteria; test was not run and production was not overwritten." if not selected else "A v2 tier met all pre-specified criteria; final test still requires the explicit one-time qualification step.",
    }
    (ROOT / "metrics/promotion_decision.json").write_text(json.dumps(decision, indent=2))
    pd.DataFrame(rows).to_csv(ROOT / "metrics/promotion_candidates.csv", index=False)
    return decision


def save_pdf(path: Path, fig: plt.Figure) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure6() -> None:
    label_dir = ROOT / "work/CellViT_dataset_v2_EXTENDED/test/labels"
    image_dir = ROOT / "work/CellViT_dataset_v2_EXTENDED/test/images"
    examples = []
    for label in sorted(label_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(label, header=None)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty or not (frame.iloc[:, 2].astype(int) == 3).any():
            continue
        img = image_dir / f"{label.stem}.png"
        if img.exists():
            examples.append((img, frame))
        if len(examples) >= 20:
            break
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (img_path, frame) in zip(axes, examples):
        ax.imshow(Image.open(img_path).convert("RGB"))
        neut = frame[frame.iloc[:, 2].astype(int) == 3]
        ax.scatter(neut.iloc[:, 0], neut.iloc[:, 1], s=12, facecolors="none", edgecolors="cyan", linewidths=0.8)
        ax.set_title(img_path.stem, fontsize=6)
        ax.axis("off")
    save_pdf(ROOT / "figures/Fig6_neutrophil_HE_QC_montage.pdf", fig)


def figure7() -> None:
    path = ROOT / "metrics/marker_scores_by_cell.csv.gz"
    df = pd.read_csv(path)
    class_col = next((c for c in ["new_celltype", "annotation", "class_name"] if c in df.columns), None)
    score_cols = [c for c in df.columns if c.startswith("score_") and c.replace("score_", "") in CLASSES]
    # marker_scores_by_cell is written in source AnnData row order.  Attach
    # the frozen class labels only for the exact row-count match; this keeps
    # the figure independent of any model prediction.
    if class_col is None:
        ann_path = ROOT / "metrics/xenium_v2_cell_annotations.csv.gz"
        ann = pd.read_csv(ann_path, usecols=["new_celltype"])
        if len(ann) == len(df):
            df["new_celltype"] = ann["new_celltype"].to_numpy()
            class_col = "new_celltype"
    if class_col is None or not score_cols:
        return
    df = df[df[class_col].isin(CLASSES)].copy()
    if len(df) > 12000:
        df = df.sample(12000, random_state=42)
    means = df.groupby(class_col)[score_cols].mean().reindex(CLASSES)
    means.columns = [c.replace("score_", "") for c in means.columns]
    means.to_csv(ROOT / "figure_data/Fig7_marker_scores_by_new_class.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(means.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(means.columns)), means.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(means.index)), means.index)
    ax.set_title("Mean independent marker scores by frozen v2 class")
    fig.colorbar(im, ax=ax, label="score")
    save_pdf(ROOT / "figures/Fig7_marker_scores_by_new_class.pdf", fig)


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values])


def figure8() -> None:
    ann = pd.read_csv(ROOT / "metrics/xenium_v2_cell_annotations.csv.gz", usecols=["cell_id", "new_celltype"])
    labels = ann.set_index("cell_id")["new_celltype"]
    with h5py.File(SOURCE_H5AD, "r") as f:
        xy = np.asarray(f["obsm"]["X_umap"][:], dtype=float)
        ids = decode(f["obs"]["cell_id"][:])
    lab = pd.Series(ids).map(labels).fillna("Other").to_numpy()
    rng = np.random.default_rng(42)
    take = rng.choice(len(xy), size=min(30000, len(xy)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in CLASSES + ["Other"]:
        sel = take[lab[take] == name]
        if len(sel):
            ax.scatter(xy[sel, 0], xy[sel, 1], s=1.2, alpha=0.45, label=name)
    ax.set_title("Existing Harmony/UMAP embedding colored by frozen v2 annotation")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(markerscale=5, fontsize=7, loc="best")
    save_pdf(ROOT / "figures/Fig8_embedding_new_annotation.pdf", fig)


def figures(folds: pd.DataFrame, per: pd.DataFrame, summary: pd.DataFrame) -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "figure_data").mkdir(exist_ok=True)
    figure6()
    figure7()
    figure8()
    cmp = summary[summary.condition.isin(["OLD_LABELS", "V2_CORE", "V2_EXTENDED"])]
    cmp.to_csv(ROOT / "figure_data/Fig9_macroF1_old_vs_v2.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cmp.condition, cmp.macro_f1_mean, yerr=cmp.macro_f1_sd, capsize=4)
    ax.set_ylabel("Macro-F1 (mean ± SD)")
    ax.set_title("SAM-H RAW CV: old labels vs frozen v2 tiers")
    save_pdf(ROOT / "figures/Fig9_CV_macroF1_old_vs_v2.pdf", fig)

    p = per[per.condition.isin(["OLD_LABELS", "V2_CORE", "V2_EXTENDED"])]
    pivot = p.groupby(["condition", "class_name"], as_index=False).f1.mean().pivot(index="class_name", columns="condition", values="f1").reindex(CLASSES)
    pivot.to_csv(ROOT / "figure_data/Fig10_per_class_F1_old_vs_v2.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1: old labels vs frozen v2 tiers")
    ax.legend(title="condition", fontsize=8)
    save_pdf(ROOT / "figures/Fig10_per_class_F1_old_vs_v2.pdf", fig)

    n = p[p.class_name == "Neutrophil"].groupby("condition", as_index=False)[["f1", "recall", "auprc"]].mean().set_index("condition").reindex(["OLD_LABELS", "V2_CORE", "V2_EXTENDED"])
    n.to_csv(ROOT / "figure_data/Fig11_neutrophil_metrics_old_vs_v2.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    n.plot.bar(ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Neutrophil F1, recall and AUPRC")
    ax.legend(fontsize=8)
    save_pdf(ROOT / "figures/Fig11_neutrophil_metrics_old_vs_v2.pdf", fig)


def main() -> None:
    folds, per, summary = aggregate()
    decision = promotion(folds, per, summary)
    figures(folds, per, summary)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
