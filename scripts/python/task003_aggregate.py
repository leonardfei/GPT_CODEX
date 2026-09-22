#!/usr/bin/env python3
"""Aggregate official CellViT++ validation artifacts without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
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


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
LOWEST3 = [2, 3, 4]


def load_tensor(path: Path) -> np.ndarray:
    return torch.load(path, map_location="cpu", weights_only=False).detach().cpu().numpy()


def find_run(result: Path, name: str) -> Path:
    candidates = sorted((result / "runs" / name).glob("*/val_results/scores.json"))
    if not candidates:
        raise FileNotFoundError(f"No completed official run found for {name}")
    return candidates[-1].parent.parent


def parse_best_epoch(log_path: Path) -> tuple[int, float, float]:
    epoch = None
    rows: list[tuple[int, float, float]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = re.search(r"Epoch: (\d+)/(\d+)", line)
        if match:
            epoch = int(match.group(1))
        match = re.search(r"Validation epoch stats:.*F1-Score: ([0-9.]+).*AUROC: ([0-9.]+)", line)
        if match and epoch is not None:
            rows.append((epoch, float(match.group(1)), float(match.group(2))))
    if not rows:
        return 0, float("nan"), float("nan")
    best = max(rows, key=lambda row: row[2])
    return best


def native_detection_f1(log_path: Path) -> float:
    values = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = re.search(r"Extraction detection metrics - F1: ([0-9.]+)", line)
        if match:
            values.append(float(match.group(1)))
    return values[1] if len(values) > 1 else (values[0] if values else float("nan"))


def metric_row(run: Path, fold: int, name: str) -> tuple[dict, list[dict], pd.DataFrame]:
    val = run / "val_results"
    y = load_tensor(val / "gt.pt").astype(int).reshape(-1)
    pred = load_tensor(val / "predictions.pt").astype(int).reshape(-1)
    prob = load_tensor(val / "probabilities.pt").astype(float)
    if prob.ndim == 1:
        prob = np.column_stack([1 - prob, prob])
    y_bin = label_binarize(y, classes=np.arange(len(CLASS_NAMES)))
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0)
    try:
        macro_auroc = roc_auc_score(y_bin, prob, average="macro", multi_class="ovr")
    except ValueError:
        macro_auroc = float("nan")
    try:
        macro_auprc = average_precision_score(y_bin, prob, average="macro")
    except ValueError:
        macro_auprc = float("nan")
    class_rows = []
    for idx, name_class in enumerate(CLASS_NAMES):
        try:
            class_auc = roc_auc_score(y_bin[:, idx], prob[:, idx])
        except ValueError:
            class_auc = float("nan")
        try:
            class_ap = average_precision_score(y_bin[:, idx], prob[:, idx])
        except ValueError:
            class_ap = float("nan")
        class_rows.append({"fold": fold, "class_id": idx, "class_name": name_class, "precision": p[idx], "recall": r[idx], "f1": f[idx], "auroc": class_auc, "auprc": class_ap, "support": int(support[idx])})
    epoch, official_f1, official_auroc = parse_best_epoch(run / "logs.log")
    scores = json.loads((val / "scores.json").read_text())
    row = {
        "fold": fold,
        "run_name": name,
        "run_dir": str(run),
        "best_epoch": epoch,
        "official_validation_f1": official_f1,
        "official_validation_auroc": official_auroc,
        "scores_json_auroc": scores.get("AUROC/Validation"),
        "scores_json_f1": scores.get("F1-Score/Validation"),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "mcc": matthews_corrcoef(y, pred),
        "lowest3_f1": float(np.mean(f[LOWEST3])),
        "native_detection_f1": native_detection_f1(run / "logs.log"),
        "n_cells": len(y),
    }
    pred_df = pd.DataFrame({"fold": fold, "y_true": y, "y_pred": pred})
    for idx in range(prob.shape[1]):
        pred_df[f"prob_{idx}"] = prob[:, idx]
    return row, class_rows, pred_df


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task003_official"))
    args = parser.parse_args()
    result = args.result.resolve()
    runs = [(0, "fold_0_official_baseline"), *[(fold, f"fold_{fold}_official_winner") for fold in range(1, 5)]]
    fold_rows: list[dict] = []
    class_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    for fold, name in runs:
        run = find_run(result, name)
        row, classes, pred_df = metric_row(run, fold, name)
        fold_rows.append(row)
        class_rows.extend(classes)
        pred_frames.append(pred_df)
    folds = pd.DataFrame(fold_rows)
    classes = pd.DataFrame(class_rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    result.joinpath("metrics").mkdir(exist_ok=True)
    folds.to_csv(result / "metrics/cv_fold_metrics.csv", index=False)
    classes.to_csv(result / "metrics/cv_per_class_metrics.csv", index=False)
    preds.to_csv(result / "metrics/cv_oof_predictions.csv", index=False)

    summary = {}
    for col in ["official_validation_auroc", "scores_json_auroc", "official_validation_f1", "scores_json_f1", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc", "mcc", "lowest3_f1", "native_detection_f1"]:
        summary[f"{col}_mean"] = float(folds[col].mean())
        summary[f"{col}_sd"] = float(folds[col].std(ddof=1))
    summary["n_folds"] = 5
    pd.DataFrame([summary]).to_csv(result / "metrics/cv_summary.csv", index=False)

    class_summary = classes.groupby(["class_id", "class_name"], as_index=False).agg({"precision": "mean", "recall": "mean", "f1": "mean", "auroc": "mean", "auprc": "mean", "support": "sum"})
    class_summary.to_csv(result / "metrics/cv_per_class_summary.csv", index=False)
    median_epoch = int(round(float(folds["best_epoch"].median())))
    (result / "config/final_epoch_guidance.json").write_text(json.dumps({"best_epochs": folds["best_epoch"].tolist(), "median_best_epoch": median_epoch, "rule": "median official AUROC-selected epoch across five grouped folds"}, indent=2))

    final_cfg_path = result / "config/final_official_all_train.yaml"
    final_cfg = yaml.safe_load(final_cfg_path.read_text())
    final_cfg["training"]["epochs"] = max(1, median_epoch)
    final_cfg["training"]["early_stopping_patience"] = None
    final_cfg_path.write_text(yaml.safe_dump(final_cfg, sort_keys=False))
    print(json.dumps({"status": "aggregated", "folds": fold_rows, "summary": summary, "median_best_epoch": median_epoch}, indent=2, default=float))


if __name__ == "__main__":
    main()
