#!/usr/bin/env python3
"""Create audit tables, figures, manifest, and final report for task_001."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import socket
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def savefig(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def scalar_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, n_classes: int) -> dict:
    per_auc, per_ap = [], []
    for cls in range(n_classes):
        target = (y == cls).astype(int)
        try:
            per_auc.append(float(roc_auc_score(target, prob[:, cls])))
        except ValueError:
            per_auc.append(float("nan"))
        try:
            per_ap.append(float(average_precision_score(target, prob[:, cls])))
        except ValueError:
            per_ap.append(float("nan"))
    return {
        "n_cells": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0, labels=list(range(n_classes)))),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0, labels=list(range(n_classes)))),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0, labels=list(range(n_classes)))),
        "macro_auroc": float(np.nanmean(per_auc)),
        "macro_auprc": float(np.nanmean(per_ap)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred)),
    }


def per_class_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    labels = list(range(len(names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0
    )
    rows = []
    for cls in labels:
        target = (y == cls).astype(int)
        try:
            auroc = float(roc_auc_score(target, prob[:, cls]))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(target, prob[:, cls]))
        except ValueError:
            auprc = float("nan")
        rows.append(
            {
                "class_id": cls,
                "class_name": names[cls],
                "precision": float(precision[cls]),
                "recall": float(recall[cls]),
                "f1": float(f1[cls]),
                "support": int(support[cls]),
                "auroc": auroc,
                "auprc": auprc,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cellvit-path", required=True)
    args = parser.parse_args()

    root = Path(args.result_root)
    metrics_dir, figure_data, figures, errors = [root / x for x in ("metrics", "figure_data", "figures", "error_analysis")]
    for path in (metrics_dir, figure_data, figures, errors):
        path.mkdir(parents=True, exist_ok=True)

    label_map_raw = yaml.safe_load((root / "config" / "label_map.yaml").read_text())
    if all(isinstance(v, (int, float)) for v in label_map_raw.values()):
        names = {int(v): str(k) for k, v in label_map_raw.items()}
    else:
        names = {int(k): str(v) for k, v in label_map_raw.items()}
    n_classes = len(names)
    y = np.load(metrics_dir / "test_ground_truth.npy")
    pred = np.load(metrics_dir / "test_predictions.npy")
    prob = np.load(metrics_dir / "test_probabilities.npy")
    metadata = json.loads((metrics_dir / "test_metadata.json").read_text())
    per_class = per_class_metrics(y, pred, prob, names)
    summary = scalar_metrics(y, pred, prob, n_classes)
    extraction = read_json(metrics_dir / "test_extraction_summary.json", {})
    summary.update(
        {
            "native_detection_f1": extraction.get("detection_f1_mean"),
            "native_detection_precision": extraction.get("detection_precision_mean"),
            "native_detection_recall": extraction.get("detection_recall_mean"),
        }
    )
    pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()]).to_csv(
        metrics_dir / "test_summary.csv", index=False
    )
    per_class.to_csv(metrics_dir / "test_per_class_metrics.csv", index=False)

    patch_meta_path = root / "data" / "patch_metadata.csv"
    patch_meta = pd.read_csv(patch_meta_path)
    batch_by_image = {
        Path(str(row.image)).stem: str(row.batch)
        for row in patch_meta.itertuples()
    }
    meta_rows = []
    for i, item in enumerate(metadata):
        image = str(item[2])
        meta_rows.append(
            {
                "index": i,
                "row": float(item[0]),
                "col": float(item[1]),
                "image": image,
                "batch": batch_by_image.get(Path(image).stem, "UNKNOWN"),
                "true_id": int(y[i]),
                "pred_id": int(pred[i]),
                "true_name": names[int(y[i])],
                "pred_name": names[int(pred[i])],
                "confidence": float(prob[i].max()),
                "correct": bool(y[i] == pred[i]),
            }
        )
    meta_df = pd.DataFrame(meta_rows)

    # Grouped bootstrap by test batch, preserving the natural batch/slide unit.
    rng = np.random.default_rng(42)
    groups = [idx.to_numpy() for _, idx in meta_df.groupby("batch").groups.items()]
    boot_rows = []
    for rep in range(200):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in sampled])
        vals = scalar_metrics(y[idx], pred[idx], prob[idx], n_classes)
        vals["replicate"] = rep
        boot_rows.append(vals)
    boot = pd.DataFrame(boot_rows)
    ci_rows = []
    for metric in ["macro_f1", "balanced_accuracy", "macro_auroc", "macro_auprc"]:
        values = boot[metric].dropna().to_numpy()
        ci_rows.append(
            {
                "metric": metric,
                "estimate": summary[metric],
                "bootstrap_mean": float(np.mean(values)),
                "ci_lower_2.5pct": float(np.percentile(values, 2.5)),
                "ci_upper_97.5pct": float(np.percentile(values, 97.5)),
                "n_replicates": int(len(values)),
                "group_unit": "test batch",
            }
        )
    pd.DataFrame(ci_rows).to_csv(metrics_dir / "test_bootstrap_ci.csv", index=False)

    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_rows = []
    for i in range(n_classes):
        for j in range(n_classes):
            cm_rows.append({"true_id": i, "true_name": names[i], "pred_id": j, "pred_name": names[j], "count": int(cm[i, j]), "row_normalized": float(cm_norm[i, j])})
    pd.DataFrame(cm_rows).to_csv(figure_data / "fig_confusion_matrix.csv", index=False)
    meta_df.loc[~meta_df.correct].sort_values("confidence", ascending=False).head(1000).to_csv(errors / "false_predictions.csv", index=False)
    pd.DataFrame([
        {"true_name": names[i], "pred_name": names[j], "count": int(cm[i, j])}
        for i in range(n_classes) for j in range(n_classes) if i != j and cm[i, j] > 0
    ]).sort_values("count", ascending=False).to_csv(errors / "confusion_pairs.csv", index=False)
    (errors / "summary.md").write_text(
        "# Error analysis\n\n"
        f"The frozen model produced {int((~meta_df.correct).sum()):,} incorrect predictions out of {len(meta_df):,} paired detected test cells. "
        "The false-prediction table is limited to the 1,000 highest-confidence errors for review.\n",
        encoding="utf-8",
    )

    # Curves and calibration data.
    roc_rows, pr_rows = [], []
    for cls in range(n_classes):
        target = (y == cls).astype(int)
        fpr, tpr, _ = roc_curve(target, prob[:, cls])
        auc = per_class.loc[per_class.class_id == cls, "auroc"].iloc[0]
        roc_rows.extend({"class_id": cls, "class_name": names[cls], "fpr": float(a), "tpr": float(b), "auroc": float(auc)} for a, b in zip(fpr, tpr))
        prec, rec, _ = precision_recall_curve(target, prob[:, cls])
        ap = per_class.loc[per_class.class_id == cls, "auprc"].iloc[0]
        pr_rows.extend({"class_id": cls, "class_name": names[cls], "recall": float(a), "precision": float(b), "auprc": float(ap)} for a, b in zip(rec, prec))
    roc_df, pr_df = pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)
    roc_df.to_csv(figure_data / "fig_roc_curves.csv", index=False)
    pr_df.to_csv(figure_data / "fig_pr_curves.csv", index=False)
    bins = np.linspace(0, 1, 11)
    calib_rows = []
    confidence = prob.max(axis=1)
    correctness = (pred == y).astype(float)
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidence >= left) & ((confidence < right) if right < 1 else (confidence <= right))
        calib_rows.append({"bin_left": left, "bin_right": right, "count": int(mask.sum()), "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan, "accuracy": float(correctness[mask].mean()) if mask.any() else np.nan})
    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(figure_data / "fig_calibration.csv", index=False)

    # Copy source tables used by the figures into the reproducible figure_data directory.
    source_table_names = {
        "class_distribution.csv": "fig_class_distribution.csv",
        "cv_fold_metrics.csv": "fig_cv_metrics.csv",
        "hyperparameter_trials.csv": "fig_hyperparameter_trials.csv",
        "final_training_history.csv": "fig_final_training_history.csv",
    }
    for src_name, dst_name in source_table_names.items():
        src = root / "metrics" / src_name if src_name != "class_distribution.csv" else root / "data" / src_name
        if src.exists():
            pd.read_csv(src).to_csv(figure_data / dst_name, index=False)
    per_class.to_csv(figure_data / "fig_test_per_class_metrics.csv", index=False)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    # Figure 1: class distribution.
    dist = pd.read_csv(root / "data" / "class_distribution.csv")
    name_col = next((c for c in ("class_name", "class") if c in dist.columns), "class_name")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = dist[name_col].astype(str)
    if "training_count" in dist.columns and "test_count" in dist.columns:
        xx = np.arange(len(dist))
        ax.bar(xx - 0.2, dist.training_count.astype(float), 0.4, label="train")
        ax.bar(xx + 0.2, dist.test_count.astype(float), 0.4, label="test")
        ax.set_xticks(xx, labels, rotation=35, ha="right")
        ax.legend()
    else:
        count_col = next((c for c in ("cell_count", "count", "cells") if c in dist.columns), None)
        if count_col is None:
            dist = per_class.rename(columns={"class_name": "class_name", "support": "cell_count"})
            count_col, labels = "cell_count", per_class.class_name.astype(str)
        ax.bar(labels, dist[count_col].astype(float), color=colors)
        ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Cells"); ax.set_title("Training and test cell-class distribution")
    savefig(fig, figures / "figure_1_class_distribution")

    # Figure 2: grouped CV summary.
    cv = pd.read_csv(root / "metrics" / "cv_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cv)); labels = [f"config {int(v)}" for v in cv.config_rank]
    ax.bar(x - 0.18, cv.macro_f1_mean, 0.36, yerr=cv.macro_f1_sd, label="Macro F1")
    ax.bar(x + 0.18, cv.balanced_accuracy_mean, 0.36, yerr=cv.balanced_accuracy_sd, label="Balanced accuracy")
    ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Grouped five-fold CV model selection")
    savefig(fig, figures / "figure_2_cv_metrics")

    # Figure 3: test per-class metrics.
    fig, ax = plt.subplots(figsize=(9, 5))
    xx = np.arange(n_classes); width = 0.25
    for offset, col in zip((-width, 0, width), ("precision", "recall", "f1")):
        ax.bar(xx + offset, per_class[col], width, label=col)
    ax.set_xticks(xx, [names[i] for i in range(n_classes)], rotation=35, ha="right"); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Frozen-model test per-class metrics")
    savefig(fig, figures / "figure_3_test_per_class_metrics")

    # Figure 4: normalized confusion matrix.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, values, title, fmt in zip(axes, (cm, cm_norm), ("Counts", "Row-normalized"), ("d", ".2f")):
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if values is cm_norm else None)
        ax.set_xticks(range(n_classes), [names[i] for i in range(n_classes)], rotation=45, ha="right")
        ax.set_yticks(range(n_classes), [names[i] for i in range(n_classes)]); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(n_classes):
            for j in range(n_classes): ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, figures / "figure_4_confusion_matrix")

    # Figure 5: ROC.
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in range(n_classes):
        d = roc_df[roc_df.class_id == cls]; ax.plot(d.fpr, d.tpr, label=f"{names[cls]} (AUC={per_class.loc[per_class.class_id == cls, 'auroc'].iloc[0]:.3f})", color=colors[cls])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC curves"); ax.legend(fontsize=8)
    savefig(fig, figures / "figure_5_roc_curves")

    # Figure 6: PR.
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in range(n_classes):
        d = pr_df[pr_df.class_id == cls]; ax.plot(d.recall, d.precision, label=f"{names[cls]} (AP={per_class.loc[per_class.class_id == cls, 'auprc'].iloc[0]:.3f})", color=colors[cls])
    ax.set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision-recall curves"); ax.legend(fontsize=8)
    savefig(fig, figures / "figure_6_pr_curves")

    # Figure 7: hyperparameter trials.
    trials = pd.read_csv(root / "metrics" / "hyperparameter_trials.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(trials.trial_id, trials.macro_f1, "o-"); ax.set(xlabel="Trial", ylabel="Fold-0 macro F1", title="Head-only hyperparameter search"); ax.grid(alpha=0.25)
    savefig(fig, figures / "figure_7_hyperparameter_trials")

    # Figure 8: final training history.
    hist = pd.read_csv(root / "metrics" / "final_training_history.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in ("train_loss", "loss"):
        if col in hist.columns: ax.plot(hist.epoch, hist[col], marker="o", label=col)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Final head training history"); ax.legend(); ax.grid(alpha=0.25)
    savefig(fig, figures / "figure_8_final_training_history")

    # Figure 9: calibration.
    fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.plot(calib_df.mean_confidence, calib_df.accuracy, "o-", label="Frozen model"); ax.set(xlabel="Mean confidence", ylabel="Accuracy", title="Confidence calibration"); ax.legend(); ax.grid(alpha=0.25)
    savefig(fig, figures / "figure_9_calibration")

    ckpt = Path(args.cellvit_path)
    final_ckpt = root / "model_best.pth"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    official = read_json(root / "logs" / "final_model" / "test_results" / "inference_results.json", {})
    manifest = {
        "task": "task_001",
        "status": "COMPLETED",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "repository_path": "/data/lf_data/CellViT-plus-plus",
        "repository_commit": "unavailable: path is not a Git repository",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()},
        "dataset": {"source": "/data/lf_data/xenium_data/CellViT_dataset", "train_images": 5002, "test_images": 6107, "test_exact_image_overlap": 0, "test_cells_after_detection_pairing": int(len(y))},
        "split_strategy": "StratifiedGroupKFold(n_splits=5, random_state=42), group=batch, dominant patch class for stratification",
        "seed": 42,
        "pretrained_checkpoint_sha256": sha(ckpt),
        "final_model_sha256": sha(final_ckpt),
        "selected_hyperparameters": yaml.safe_load((root / "config" / "best_hyperparameters.yaml").read_text()).get("selected"),
        "test_metrics": summary,
        "official_native_inference": official,
        "source_data_modified": False,
        "test_used_for_tuning": False,
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    ci = pd.read_csv(metrics_dir / "test_bootstrap_ci.csv")
    report = f"""# task_001 final report

## Status

COMPLETED. The frozen classifier head was trained and evaluated through the native CellViT++ detection-plus-classification inference path.

## Data and leakage controls

- Source dataset: `/data/lf_data/xenium_data/CellViT_dataset`; source images and labels were read only.
- Training images: 5,002. Test images: 6,107. Exact train/test image-hash overlap: 0.
- The supplied fold files mixed batches between train and validation. They were not used for selection. Derived five-fold splits use `StratifiedGroupKFold` grouped by batch and have no train/validation batch overlap.
- The test set was not used for hyperparameter tuning, cross-validation, or final training.

## Selected model

- Optimizer: AdamW; learning rate 0.0003; weight decay 0.0001; hidden dimension 128; dropout 0.2.
- Loss weighting: inverse square-root class frequency.
- Final training used all 5,002 training images and 12 epochs, the median robust epoch guidance from grouped CV.
- Final checkpoint: `/data/lf_data/result/model_best.pth`.

## Test results

The metrics below are classifier metrics on the 89,189 detected cells that were paired to ground-truth cells by the native CellViT++ evaluation pathway; they are not whole-image metrics.

| Metric | Value |
|---|---:|
| Accuracy | {summary['accuracy']:.4f} |
| Balanced accuracy | {summary['balanced_accuracy']:.4f} |
| Macro F1 | {summary['macro_f1']:.4f} |
| Weighted F1 | {summary['weighted_f1']:.4f} |
| Macro AUROC | {summary['macro_auroc']:.4f} |
| Macro AUPRC | {summary['macro_auprc']:.4f} |
| MCC | {summary['mcc']:.4f} |
| Native detection F1 | {summary['native_detection_f1']:.4f} |

Grouped bootstrap 95% intervals by test batch are in `metrics/test_bootstrap_ci.csv`.

## Reproducibility and limitations

- The remote CellViT++ directory is not a Git repository, so no repository commit hash was available; the pretrained checkpoint SHA256 and environment manifest are recorded in `run_manifest.json`.
- The report preserves the native detection pipeline's paired-cell evaluation definition. Detection quality and classifier quality are reported separately.
- The main generated artifacts are under `/data/lf_data/result`: metrics, figure data, figures, configs, logs, error analysis, checkpoint, and this report.
"""
    (root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "macro_f1": summary["macro_f1"], "macro_auroc": summary["macro_auroc"], "n_cells": int(len(y))}, indent=2))


if __name__ == "__main__":
    main()
