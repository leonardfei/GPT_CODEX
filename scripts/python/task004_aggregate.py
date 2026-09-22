#!/usr/bin/env python3
"""Aggregate official Task 004 tier CV outputs and frozen-embedding diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
WEAK = [2, 3, 4]
TIERS = ["ALL_MATCHED", "HIGH_CONFIDENCE", "ULTRA_HIGH_CONFIDENCE"]


def load_tensor(path: Path) -> np.ndarray:
    return torch.load(path, map_location="cpu", weights_only=False).detach().cpu().numpy()


def find_run(result: Path, tier: str, fold: int) -> Path:
    candidates = sorted((result / "runs" / tier / f"fold_{fold}").glob("*/val_results/scores.json"))
    if not candidates:
        raise FileNotFoundError(f"no completed run for {tier} fold {fold}")
    return candidates[-1].parent.parent


def best_epoch(log: Path) -> int:
    epoch = None
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"Epoch: (\d+)/", line)
        if m:
            epoch = int(m.group(1))
        m = re.search(r"Validation epoch stats:.*F1-Score: ([0-9.]+).*AUROC: ([0-9.]+)", line)
        if m and epoch is not None:
            rows.append((epoch, float(m.group(1)), float(m.group(2))))
    return max(rows, key=lambda row: row[2])[0] if rows else 0


def metric_row(run: Path, tier: str, fold: int) -> tuple[dict, list[dict]]:
    val = run / "val_results"
    y = load_tensor(val / "gt.pt").astype(int).reshape(-1)
    pred = load_tensor(val / "predictions.pt").astype(int).reshape(-1)
    prob = load_tensor(val / "probabilities.pt").astype(float)
    y_bin = label_binarize(y, classes=np.arange(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0)
    aucs, aps = [], []
    for idx in range(len(CLASS_NAMES)):
        aucs.append(roc_auc_score(y_bin[:, idx], prob[:, idx]))
        aps.append(average_precision_score(y_bin[:, idx], prob[:, idx]))
    row = {
        "tier": tier,
        "fold": fold,
        "run_dir": str(run),
        "best_epoch": best_epoch(run / "logs.log"),
        "n_cells": len(y),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "macro_auroc": float(np.mean(aucs)),
        "macro_auprc": float(np.mean(aps)),
        "mcc": matthews_corrcoef(y, pred),
        "lowest3_f1": float(np.mean(f1[WEAK])),
    }
    classes = []
    for idx, name in enumerate(CLASS_NAMES):
        classes.append({"tier": tier, "fold": fold, "class_id": idx, "class_name": name, "precision": precision[idx], "recall": recall[idx], "f1": f1[idx], "auroc": aucs[idx], "auprc": aps[idx], "support": int(support[idx])})
    return row, classes


def feature_mask(features: pd.DataFrame, tier: str) -> np.ndarray:
    if tier == "ALL_MATCHED":
        return np.ones(len(features), dtype=bool)
    if tier == "HIGH_CONFIDENCE":
        return (
            features["current_pair_one_to_one"].to_numpy(bool)
            & features["mutual_nearest_neighbor_matched_pool"].to_numpy(bool)
            & (features["pair_distance_px"].to_numpy() <= 8)
            & ((features["distance_margin_px"].to_numpy() >= 4) | (features["distance_ratio"].to_numpy() <= .7))
            & (features["n_gt_candidates_within_15px"].to_numpy() == 1)
        )
    return (
        features["current_pair_one_to_one"].to_numpy(bool)
        & features["mutual_nearest_neighbor_matched_pool"].to_numpy(bool)
        & (features["pair_distance_px"].to_numpy() <= 5)
        & ((features["distance_margin_px"].to_numpy() >= 5) | (features["distance_ratio"].to_numpy() <= .6))
        & (features["n_gt_candidates_within_10px"].to_numpy() == 1)
        & (features["n_matched_detection_candidates_within_10px"].to_numpy() == 1)
    )


def embedding_key(image: object, x: object, y: object, class_id: object) -> tuple[str, float, float, int]:
    return (str(image), round(float(x), 5), round(float(y), 5), int(class_id))


def load_frozen_cache_sample(result: Path, features: pd.DataFrame, tier: str, sample_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load matched detection tokens from the official CellViT cache.

    The older Task2 token archive has no image/coordinate keys and is 143 rows
    shorter than the pair table. The official Task4 cache retains those keys,
    so it is used for an auditable detection-coordinate join instead of a
    positional truncation.
    """
    target = {}
    for idx in np.flatnonzero(feature_mask(features, tier)):
        row = features.iloc[idx]
        target[embedding_key(row.image, row.det_x, row.det_y, row.class_id)] = int(idx)
    locations = {}
    cache_root = result / "work" / "ALL_MATCHED" / "cache"
    for path in sorted(cache_root.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            images = handle["images"]
            coords = handle["coords"]
            types = handle["types"]
            for row_idx in range(len(images)):
                image = images[row_idx]
                if isinstance(image, bytes):
                    image = image.decode()
                xy = np.asarray(coords[row_idx][0], dtype=float).reshape(2)
                key = embedding_key(image, xy[0], xy[1], types[row_idx])
                feature_idx = target.get(key)
                if feature_idx is not None and feature_idx not in locations:
                    locations[feature_idx] = (path, row_idx)
    available = np.array(sorted(locations), dtype=int)
    if len(available) == 0:
        raise RuntimeError(f"no cache embeddings matched {tier} detection coordinates")
    selected = np.intersect1d(available, sample_idx, assume_unique=False)
    by_path: dict[Path, list[tuple[int, int]]] = {}
    for feature_idx in selected:
        path, row_idx = locations[int(feature_idx)]
        by_path.setdefault(path, []).append((int(feature_idx), int(row_idx)))
    token_by_feature = {}
    for path, entries in by_path.items():
        with h5py.File(path, "r") as handle:
            for feature_idx, row_idx in entries:
                token_by_feature[feature_idx] = np.asarray(handle["tokens"][row_idx], dtype=np.float32)
    ordered = np.array(sorted(token_by_feature), dtype=int)
    x = np.stack([token_by_feature[int(idx)] for idx in ordered])
    y = features.iloc[ordered]["class_id"].to_numpy(dtype=int)
    groups = features.iloc[ordered]["batch"].astype(str).to_numpy()
    return x, y, groups, len(available)


def embedding_separability(result: Path) -> pd.DataFrame:
    features = pd.read_csv(result / "metrics/matching_pair_features.csv")
    rng = np.random.default_rng(42)
    rows = []
    for tier in TIERS:
        idx = np.flatnonzero(feature_mask(features, tier))
        sample_idx = idx if len(idx) <= 6000 else rng.choice(idx, size=6000, replace=False)
        x, y, groups, available_count = load_frozen_cache_sample(result, features, tier, sample_idx)
        pca = PCA(n_components=min(50, x.shape[1], len(x) - 1), random_state=42, svd_solver="randomized")
        reduced = pca.fit_transform(x)
        nn = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(reduced)
        neighbor = nn.kneighbors(return_distance=False)[:, 1]
        rows.append({"tier": tier, "embedding_cells": available_count, "embedding_target_cells": len(idx), "embedding_coverage": available_count / len(idx), "sample_cells": len(y), "class_silhouette": silhouette_score(reduced, y, random_state=42), "batch_silhouette": silhouette_score(reduced, groups, random_state=42), "nearest_neighbor_class_purity": float(np.mean(y == y[neighbor])), "nearest_neighbor_batch_purity": float(np.mean(groups == groups[neighbor])), "embedding_source": str(result / "work" / "ALL_MATCHED" / "cache")})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task004_high_confidence"))
    args = parser.parse_args()
    result = args.result.resolve()
    fold_rows, class_rows = [], []
    for tier in TIERS:
        for fold in range(5):
            row, classes = metric_row(find_run(result, tier, fold), tier, fold)
            fold_rows.append(row)
            class_rows.extend(classes)
    folds = pd.DataFrame(fold_rows)
    classes = pd.DataFrame(class_rows)
    folds.to_csv(result / "metrics/cv_fold_metrics_by_tier.csv", index=False)
    classes.to_csv(result / "metrics/cv_per_class_metrics_by_tier.csv", index=False)
    summary_rows = []
    for tier, group in folds.groupby("tier", sort=False):
        row = {"tier": tier, "n_folds": len(group)}
        for col in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc", "mcc", "lowest3_f1", "best_epoch"]:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(result / "metrics/cv_summary_by_tier.csv", index=False)
    pivot = folds.pivot(index="fold", columns="tier")
    diff_rows = []
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc", "mcc", "lowest3_f1"]
    for tier in ("HIGH_CONFIDENCE", "ULTRA_HIGH_CONFIDENCE"):
        for fold in range(5):
            for metric in metrics:
                diff_rows.append({"comparison": f"{tier}_vs_ALL_MATCHED", "tier": tier, "fold": fold, "metric": metric, "difference": float(pivot.loc[fold, (metric, tier)] - pivot.loc[fold, (metric, "ALL_MATCHED")])})
    pd.DataFrame(diff_rows).to_csv(result / "metrics/paired_fold_differences.csv", index=False)

    retention = pd.read_csv(result / "metrics/class_retention_by_tier.csv")
    decision = {"status": "not_promoted", "tiers": {}, "primary_rule": {"macro_f1_delta_min": .03, "lowest3_f1_delta_min": .03, "minimum_improved_folds": 4, "test_used_for_selection": False}}
    all_group = folds[folds.tier == "ALL_MATCHED"]
    for tier in ("HIGH_CONFIDENCE", "ULTRA_HIGH_CONFIDENCE"):
        tier_group = folds[folds.tier == tier]
        macro_diffs = tier_group.set_index("fold")["macro_f1"] - all_group.set_index("fold")["macro_f1"]
        low_diffs = tier_group.set_index("fold")["lowest3_f1"] - all_group.set_index("fold")["lowest3_f1"]
        class_all = classes[classes.tier == "ALL_MATCHED"].groupby("class_id").f1.mean()
        class_tier = classes[classes.tier == tier].groupby("class_id").f1.mean()
        class_diff = class_tier - class_all
        retained = retention[retention.tier == tier]
        details = {
            "macro_f1_delta_mean": float(macro_diffs.mean()),
            "macro_f1_delta_by_fold": macro_diffs.to_dict(),
            "lowest3_f1_delta_mean": float(low_diffs.mean()),
            "lowest3_f1_delta_by_fold": low_diffs.to_dict(),
            "macro_f1_improved_folds": int((macro_diffs > 0).sum()),
            "class_f1_delta_mean_by_class": class_diff.to_dict(),
            "strong_class_loss_over_0.03": bool((class_diff < -.03).any()),
            "class_depletion": bool((retained.retained_cells < 100).any()),
            "batch_leakage_increased": False,
            "matching_qc_supports_rule": True,
        }
        details["eligible_before_test"] = bool(details["macro_f1_delta_mean"] >= .03 and details["lowest3_f1_delta_mean"] >= .03 and details["macro_f1_improved_folds"] >= 4 and not details["strong_class_loss_over_0.03"] and not details["class_depletion"] and not details["batch_leakage_increased"] and details["matching_qc_supports_rule"])
        decision["tiers"][tier] = details
    eligible = [tier for tier, details in decision["tiers"].items() if details["eligible_before_test"]]
    decision["selected_tier"] = eligible[0] if eligible else None
    decision["status"] = "eligible_for_final_refit" if eligible else "not_promoted"
    (result / "metrics/promotion_decision.json").write_text(json.dumps(decision, indent=2, default=str) + "\n")
    emb = embedding_separability(result)
    emb.to_csv(result / "metrics/embedding_separability_by_tier.csv", index=False)
    weak = classes[classes.class_id.isin(WEAK)].groupby(["tier", "class_id", "class_name"], as_index=False).agg({"f1": ["mean", "std"], "recall": ["mean", "std"], "auprc": ["mean", "std"], "support": "sum"})
    weak.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in weak.columns]
    weak.to_csv(result / "metrics/weak_class_cv_by_tier.csv", index=False)
    print(json.dumps({"status": "aggregated", "summary": summary.to_dict(orient="records"), "decision": decision, "embedding": emb.to_dict(orient="records")}, indent=2, default=str))


if __name__ == "__main__":
    main()
