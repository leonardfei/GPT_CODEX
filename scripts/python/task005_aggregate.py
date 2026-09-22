#!/usr/bin/env python3
"""Aggregate Task 005 official CV, detection, end-to-end and domain metrics."""

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
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
N_CLASSES = len(CLASS_NAMES)
NEUTROPHIL = 3
WEAK = [2, 3, 4]
TIERS = ["SAM-H_RAW", "SAM-H_STAIN_NORMALIZED"]


def load_tensor(path: Path) -> np.ndarray:
    return torch.load(path, map_location="cpu", weights_only=False).detach().cpu().numpy()


def find_run(result: Path, condition: str, fold: int) -> Path:
    candidates = sorted((result / "runs" / condition / f"fold_{fold}").glob("*/val_results/scores.json"))
    if not candidates:
        raise FileNotFoundError(f"no completed run for {condition} fold {fold}")
    return candidates[-1].parent.parent


def best_epoch(log: Path) -> int:
    current, rows = None, []
    for line in log.read_text(errors="replace").splitlines():
        match = re.search(r"Epoch: (\d+)/", line)
        if match:
            current = int(match.group(1))
        match = re.search(r"Validation epoch stats:.*F1-Score: ([0-9.]+).*AUROC: ([0-9.]+)", line)
        if match and current is not None:
            rows.append((current, float(match.group(1)), float(match.group(2))))
    return max(rows, key=lambda row: row[2])[0] if rows else 0


def metrics_for_run(run: Path, condition: str, fold: int) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    val = run / "val_results"
    y = load_tensor(val / "gt.pt").astype(int).reshape(-1)
    pred = load_tensor(val / "predictions.pt").astype(int).reshape(-1)
    prob = load_tensor(val / "probabilities.pt").astype(float)
    y_bin = label_binarize(y, classes=np.arange(N_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(N_CLASSES), zero_division=0)
    aurocs, auprcs = [], []
    for idx in range(N_CLASSES):
        aurocs.append(float(roc_auc_score(y_bin[:, idx], prob[:, idx])))
        auprcs.append(float(average_precision_score(y_bin[:, idx], prob[:, idx])))
    row = {
        "condition": condition,
        "backbone": "SAM-H",
        "stain_condition": "STAIN_NORMALIZED" if "STAIN_NORMALIZED" in condition else "RAW",
        "fold": fold,
        "run_dir": str(run),
        "best_epoch": best_epoch(run / "logs.log"),
        "n_cells": len(y),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "macro_auroc": float(np.mean(aurocs)),
        "macro_auprc": float(np.mean(auprcs)),
        "mcc": matthews_corrcoef(y, pred),
        "lowest3_f1": float(np.mean(f1[WEAK])),
    }
    classes = []
    for idx, name in enumerate(CLASS_NAMES):
        classes.append({
            "condition": condition,
            "backbone": "SAM-H",
            "stain_condition": row["stain_condition"],
            "fold": fold,
            "class_id": idx,
            "class_name": name,
            "precision": precision[idx],
            "recall": recall[idx],
            "f1": f1[idx],
            "auroc": aurocs[idx],
            "auprc": auprcs[idx],
            "support": int(support[idx]),
        })
    return row, classes, y, pred


def summarize_cv(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in folds.groupby("condition", sort=False):
        row = {"condition": condition, "backbone": "SAM-H", "stain_condition": group.stain_condition.iloc[0], "n_folds": len(group)}
        for col in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc", "mcc", "lowest3_f1", "best_epoch"]:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def cache_key(image: object, x: object, y: object, class_id: object) -> tuple[str, float, float, int]:
    return (str(image), round(float(x), 5), round(float(y), 5), int(class_id))


def read_cache_rows(cache_root: Path, sample_keys: set[tuple] | None = None):
    """Yield cache rows, retaining only first occurrence of each detection key."""
    seen = set()
    for path in sorted(cache_root.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            images, coords, types = handle["images"], handle["coords"], handle["types"]
            for idx in range(len(images)):
                image = images[idx]
                if isinstance(image, bytes):
                    image = image.decode()
                xy = np.asarray(coords[idx][0], dtype=float).reshape(2)
                key = cache_key(image, xy[0], xy[1], types[idx])
                if key in seen or (sample_keys is not None and key not in sample_keys):
                    continue
                seen.add(key)
                yield key, path, idx, int(types[idx])


def embedding_domain(result: Path) -> pd.DataFrame:
    metadata = pd.read_csv("/data/lf_data/result/data/patch_metadata.csv")
    batch_map = {Path(str(row.image)).stem: str(row.batch) for row in metadata.itertuples()}
    rows = []
    rng = np.random.default_rng(42)
    for condition in TIERS:
        cache_root = result / "work" / condition / "dataset" / "cache"
        locations = {}
        for key, path, idx, class_id in read_cache_rows(cache_root):
            locations[key] = (path, idx, class_id, batch_map.get(Path(key[0]).stem, "UNKNOWN"))
        keys = sorted(locations)
        sample_keys = keys if len(keys) <= 6000 else [keys[i] for i in rng.choice(len(keys), size=6000, replace=False)]
        selected = []
        for key in sample_keys:
            path, idx, class_id, batch = locations[key]
            with h5py.File(path, "r") as handle:
                token = np.asarray(handle["tokens"][idx], dtype=np.float32)
            selected.append((token, class_id, batch))
        x = np.stack([row[0] for row in selected])
        y = np.asarray([row[1] for row in selected], dtype=int)
        groups = np.asarray([row[2] for row in selected], dtype=str)
        reduced = PCA(n_components=min(50, x.shape[1], len(x) - 1), random_state=42, svd_solver="randomized").fit_transform(x)
        neighbor = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(reduced).kneighbors(return_distance=False)[:, 1]
        rows.append({
            "condition": condition,
            "backbone": "SAM-H",
            "stain_condition": "STAIN_NORMALIZED" if "STAIN_NORMALIZED" in condition else "RAW",
            "embedding_cells": len(keys),
            "sample_cells": len(y),
            "class_silhouette": silhouette_score(reduced, y, random_state=42),
            "batch_silhouette": silhouette_score(reduced, groups, random_state=42),
            "nearest_neighbor_class_purity": float(np.mean(y == y[neighbor])),
            "nearest_neighbor_batch_purity": float(np.mean(groups == groups[neighbor])),
            "embedding_source": str(cache_root),
        })
    return pd.DataFrame(rows)


def detection_metrics(result: Path, source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_gt = {class_id: 0 for class_id in range(N_CLASSES)}
    for label_file in sorted((source / "train" / "labels").glob("*.csv")):
        frame = pd.read_csv(label_file, header=None)
        for class_id, count in frame.iloc[:, 2].astype(int).value_counts().items():
            if int(class_id) in all_gt:
                all_gt[int(class_id)] += int(count)
    rows, pair_frames = [], []
    for condition in TIERS:
        pairs = pd.read_csv(result / "qc" / f"detection_pairs_{condition}.csv")
        pair_frames.append(pairs)
        for class_id, name in enumerate(CLASS_NAMES):
            subset = pairs[pairs.class_id == class_id]
            matched = len(subset)
            rows.append({
                "condition": condition,
                "backbone": "SAM-H",
                "stain_condition": "STAIN_NORMALIZED" if "STAIN_NORMALIZED" in condition else "RAW",
                "class_id": class_id,
                "class_name": name,
                "gt_count": all_gt[class_id],
                "matched_detection_count": matched,
                "detection_recall": matched / max(all_gt[class_id], 1),
                "unmatched_gt_count": all_gt[class_id] - matched,
                "median_pair_distance_px": subset.distance_px.median() if matched else np.nan,
                "p95_pair_distance_px": subset.distance_px.quantile(0.95) if matched else np.nan,
            })
    return pd.DataFrame(rows), pd.concat(pair_frames, ignore_index=True)


def cache_for_val(cache_root: Path, val_images: list[str], expected: int) -> list[tuple[str, Path, int]]:
    wanted = set(val_images)
    matches = []
    for path in sorted(cache_root.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            images = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["images"]]
            if len(images) == expected and set(images).issubset(wanted):
                matches.append((path, images))
    if len(matches) != 1:
        raise RuntimeError(f"expected one validation cache shard for {expected} cells, found {[str(path) for path, _ in matches]}")
    path, images = matches[0]
    return [(image, path, idx) for idx, image in enumerate(images)]


def oof_predictions(result: Path, condition: str) -> pd.DataFrame:
    cache_root = result / "work" / condition / "dataset" / "cache"
    rows = []
    for fold in range(5):
        run = find_run(result, condition, fold)
        y = load_tensor(run / "val_results/gt.pt").astype(int).reshape(-1)
        pred = load_tensor(run / "val_results/predictions.pt").astype(int).reshape(-1)
        prob = load_tensor(run / "val_results/probabilities.pt").astype(float)
        split = pd.read_csv(f"/data/lf_data/result/splits/fold_{fold}/val.csv")
        val_col = "image" if "image" in split.columns else split.columns[0]
        val_list = split[val_col].astype(str).tolist()
        cache_rows = cache_for_val(cache_root, val_list, len(y))
        handles = {}
        cache_records = {}
        try:
            for _, path, idx in cache_rows:
                handles.setdefault(path, h5py.File(path, "r"))
            for _, path, idx in cache_rows:
                handle = handles[path]
                image = handle["images"][idx]
                image = image.decode() if isinstance(image, bytes) else str(image)
                xy = np.asarray(handle["coords"][idx][0], dtype=float).reshape(2)
                cache_records[(path, idx)] = (image, xy, int(handle["types"][idx]))
        finally:
            for handle in handles.values():
                handle.close()
        cache_types = np.asarray([cache_records[(path, idx)][2] for _, path, idx in cache_rows], dtype=int)
        if not np.array_equal(cache_types, y):
            raise RuntimeError(f"cache/validation label order mismatch for {condition} fold {fold}")
        for idx, (_, path, cache_idx) in enumerate(cache_rows):
            image, xy, _ = cache_records[(path, cache_idx)]
            row = {"condition": condition, "fold": fold, "image": image, "det_x": xy[0], "det_y": xy[1], "class_id": int(y[idx]), "pred_id": int(pred[idx])}
            row.update({f"prob_{class_id}": float(prob[idx, class_id]) for class_id in range(N_CLASSES)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["join_key"] = list(zip(frame.image, frame.det_x.round(5), frame.det_y.round(5), frame.class_id))
    return frame


def end_to_end_metrics(result: Path, detection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, flows = [], []
    all_gt_neutrophils = int(detection[(detection.condition == TIERS[0]) & (detection.class_id == NEUTROPHIL)].shape[0])
    # GT total is read from detection recall table, not from the paired subset.
    det_recall = pd.read_csv(result / "metrics/per_class_detection_recall.csv")
    for condition in TIERS:
        pred = oof_predictions(result, condition)
        pred_lookup = pred.set_index("join_key")
        det = detection[detection.condition == condition].copy()
        det["join_key"] = list(zip(det.image, det.det_x.round(5), det.det_y.round(5), det.class_id))
        det = det.join(pred_lookup[["pred_id"]], on="join_key", how="left", rsuffix="_classifier")
        neut = det[det.class_id == NEUTROPHIL]
        matched_neut = len(neut)
        classified_neut = int((neut.pred_id == NEUTROPHIL).sum())
        gt_neut = int(det_recall[(det_recall.condition == condition) & (det_recall.class_id == NEUTROPHIL)].gt_count.iloc[0])
        predicted_neut = det[det.pred_id == NEUTROPHIL]
        tp = int(((predicted_neut.class_id == NEUTROPHIL)).sum())
        rows.append({
            "condition": condition,
            "backbone": "SAM-H",
            "stain_condition": "STAIN_NORMALIZED" if "STAIN_NORMALIZED" in condition else "RAW",
            "gt_neutrophil_count": gt_neut,
            "matched_neutrophil_count": matched_neut,
            "detection_recall": matched_neut / max(gt_neut, 1),
            "conditional_classifier_recall": classified_neut / max(matched_neut, 1),
            "end_to_end_recall": classified_neut / max(gt_neut, 1),
            "conditional_classifier_precision": tp / max(len(predicted_neut), 1),
            "end_to_end_precision": np.nan,
            "classifier_join_coverage": float(det.pred_id.notna().mean()),
        })
        for true_id, pred_id in [(3, 2), (2, 3), (3, 5), (5, 3), (3, 4), (4, 3)]:
            flows.append({"condition": condition, "true_class": CLASS_NAMES[true_id], "predicted_class": CLASS_NAMES[pred_id], "count": int(((det.class_id == true_id) & (det.pred_id == pred_id)).sum())})
    return pd.DataFrame(rows), pd.DataFrame(flows)


def baseline_values() -> dict:
    task1_summary = pd.read_csv("/data/lf_data/result/metrics/cv_summary.csv")
    selected = task1_summary[task1_summary.config_rank == 2].iloc[0]
    task1_class = pd.read_csv("/data/lf_data/result/metrics/cv_per_class_metrics.csv")
    task1_class = task1_class[task1_class.config_rank == 2]
    task1_neutro_frame = task1_class[task1_class.class_index == NEUTROPHIL]
    task1_neutro = float(task1_neutro_frame.f1.mean())
    task1_neutro_auprc = float(task1_neutro_frame.auprc.mean()) if "auprc" in task1_neutro_frame.columns else np.nan
    task1_lowest3 = float(task1_class[task1_class.class_index.isin(WEAK)].groupby("fold").f1.mean().mean())
    task4_class = pd.read_csv("/data/lf_data/result/task004_high_confidence/metrics/cv_per_class_metrics_by_tier.csv")
    task4_all = task4_class[(task4_class.tier == "ALL_MATCHED") & (task4_class.class_id == NEUTROPHIL)]
    task4_neutro = float(task4_all.f1.mean())
    task4_neutro_auprc = float(task4_all.auprc.mean())
    auprc_reference = max(value for value in [task1_neutro_auprc, task4_neutro_auprc] if np.isfinite(value))
    return {"task1_macro_f1": float(selected.macro_f1_mean), "task1_neutrophil_f1": task1_neutro, "task1_neutrophil_auprc": task1_neutro_auprc, "task1_lowest3_f1": task1_lowest3, "task4_all_neutrophil_f1": task4_neutro, "task4_all_neutrophil_auprc": task4_neutro_auprc, "neutrophil_reference_f1": max(task1_neutro, task4_neutro), "neutrophil_reference_auprc": auprc_reference}


def ranking(summary: pd.DataFrame, classes: pd.DataFrame, neutro: pd.DataFrame, embedding: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_values()
    rows = []
    raw_classes = classes[classes.condition == "SAM-H_RAW"].groupby("class_id").f1.mean()
    raw_detection_recall = float(neutro[neutro.condition == "SAM-H_RAW"].detection_recall.iloc[0])
    for row in summary.itertuples():
        n = neutro[neutro.condition == row.condition].iloc[0]
        cls = classes[classes.condition == row.condition].groupby("class_id").f1.mean()
        strong_loss = bool(((cls - raw_classes) < -.05).any())
        neutrophil_auprc = classes[(classes.condition == row.condition) & (classes.class_id == NEUTROPHIL)].auprc.mean()
        score = .35 * row.macro_f1_mean + .20 * row.macro_auprc_mean + .20 * cls.loc[NEUTROPHIL] + .15 * neutrophil_auprc + .10 * row.lowest3_f1_mean
        emb_row = embedding[embedding.condition == row.condition].iloc[0]
        eligible = (
            row.macro_f1_mean - baseline["task1_macro_f1"] >= .03
            and cls.loc[NEUTROPHIL] - baseline["neutrophil_reference_f1"] >= .05
            and (n.detection_recall - raw_detection_recall >= .05 or neutrophil_auprc - baseline["neutrophil_reference_auprc"] >= .05)
            and row.lowest3_f1_mean >= baseline["task1_lowest3_f1"]
            and not strong_loss
        )
        rows.append({
            "condition": row.condition,
            "backbone": "SAM-H",
            "stain_condition": row.stain_condition,
            "primary_score": score,
            "macro_f1_mean": row.macro_f1_mean,
            "macro_auprc_mean": row.macro_auprc_mean,
            "neutrophil_f1_mean": cls.loc[NEUTROPHIL],
            "neutrophil_auprc_mean": neutrophil_auprc,
            "neutrophil_detection_recall": n.detection_recall,
            "neutrophil_detection_recall_delta_vs_raw": n.detection_recall - raw_detection_recall,
            "lowest3_f1_mean": row.lowest3_f1_mean,
            "batch_silhouette": emb_row.batch_silhouette,
            "batch_nn_purity": emb_row.nearest_neighbor_batch_purity,
            "strong_class_loss_over_0.05_vs_raw": strong_loss,
            "eligible_before_test": bool(eligible),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task005_backbone_domain"))
    parser.add_argument("--source", type=Path, default=Path("/data/lf_data/xenium_data/CellViT_dataset"))
    args = parser.parse_args()
    result = args.result.resolve()
    fold_rows, class_rows, flow_rows, full_cm_rows = [], [], [], []
    for condition in TIERS:
        for fold in range(5):
            row, classes, y, pred = metrics_for_run(find_run(result, condition, fold), condition, fold)
            fold_rows.append(row)
            class_rows.extend(classes)
            for true_id in range(N_CLASSES):
                for pred_id in range(N_CLASSES):
                    full_cm_rows.append({"condition": condition, "fold": fold, "true_id": true_id, "pred_id": pred_id, "count": int(((y == true_id) & (pred == pred_id)).sum())})
            for true_id, pred_id in [(3, 2), (2, 3), (3, 5), (5, 3), (3, 4), (4, 3)]:
                flow_rows.append({"condition": condition, "fold": fold, "true_class": CLASS_NAMES[true_id], "predicted_class": CLASS_NAMES[pred_id], "count": int(((y == true_id) & (pred == pred_id)).sum())})
    folds, classes = pd.DataFrame(fold_rows), pd.DataFrame(class_rows)
    folds.to_csv(result / "metrics/cv_fold_metrics.csv", index=False)
    classes.to_csv(result / "metrics/cv_per_class_metrics.csv", index=False)
    summary = summarize_cv(folds)
    summary.to_csv(result / "metrics/cv_summary.csv", index=False)
    pd.DataFrame(flow_rows).to_csv(result / "metrics/confusion_flows.csv", index=False)
    pd.DataFrame(full_cm_rows).to_csv(result / "metrics/confusion_matrices.csv", index=False)
    detection, detection_pairs = detection_metrics(result, args.source)
    detection.to_csv(result / "metrics/per_class_detection_recall.csv", index=False)
    end_to_end, end_flows = end_to_end_metrics(result, detection_pairs)
    end_to_end.to_csv(result / "metrics/neutrophil_end_to_end.csv", index=False)
    pd.concat([pd.DataFrame(flow_rows), end_flows.assign(fold="OOF")], ignore_index=True).to_csv(result / "metrics/neutrophil_confusion_flows.csv", index=False)
    embedding = embedding_domain(result)
    embedding.to_csv(result / "metrics/embedding_domain_diagnostics.csv", index=False)
    nmetrics = end_to_end.copy()
    nmetrics["neutrophil_f1"] = classes[classes.class_id == NEUTROPHIL].groupby("condition").f1.mean().reindex(nmetrics.condition).to_numpy()
    nmetrics["neutrophil_precision"] = classes[classes.class_id == NEUTROPHIL].groupby("condition").precision.mean().reindex(nmetrics.condition).to_numpy()
    nmetrics["neutrophil_recall"] = classes[classes.class_id == NEUTROPHIL].groupby("condition").recall.mean().reindex(nmetrics.condition).to_numpy()
    nmetrics["neutrophil_auroc"] = classes[classes.class_id == NEUTROPHIL].groupby("condition").auroc.mean().reindex(nmetrics.condition).to_numpy()
    nmetrics["neutrophil_auprc"] = classes[classes.class_id == NEUTROPHIL].groupby("condition").auprc.mean().reindex(nmetrics.condition).to_numpy()
    nmetrics.to_csv(result / "metrics/neutrophil_metrics.csv", index=False)
    rank = ranking(summary, classes, end_to_end, embedding)
    rank.to_csv(result / "metrics/model_ranking.csv", index=False)
    merged = summary.merge(nmetrics[["condition", "neutrophil_f1", "conditional_classifier_recall", "detection_recall", "neutrophil_auprc"]], on="condition").merge(embedding[["condition", "batch_silhouette", "nearest_neighbor_batch_purity"]], on="condition")
    merged.to_csv(result / "metrics/stain_effect_summary.csv", index=False)
    decision = {"status": "not_promoted", "best_overall_condition": rank.sort_values("primary_score", ascending=False).iloc[0].condition, "best_neutrophil_condition": rank.sort_values("neutrophil_f1_mean", ascending=False).iloc[0].condition, "eligible_before_test": rank[rank.eligible_before_test].condition.tolist(), "test_used_for_selection": False, "primary_score_formula": "0.35*macro_F1 + 0.20*macro_AUPRC + 0.20*Neutrophil_F1 + 0.15*Neutrophil_AUPRC + 0.10*lowest3_F1", "production_model_unchanged": True, "production_path": "/data/lf_data/result/model_best.pth"}
    (result / "metrics/promotion_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({"summary": summary.to_dict("records"), "detection": detection[detection.class_id == NEUTROPHIL].to_dict("records"), "neutrophil_end_to_end": end_to_end.to_dict("records"), "embedding": embedding.to_dict("records"), "ranking": rank.to_dict("records"), "decision": decision}, indent=2, default=str))


if __name__ == "__main__":
    main()
