#!/usr/bin/env python3
"""Prepare Task 004 confidence tiers from the existing training matches.

This script does not touch the source dataset. It uses the archived Task 002
matched-pair table and source GT annotations to create derivative datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_labels(path: Path) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            try:
                rows.append((int(round(float(row[0]))), int(round(float(row[1]))), int(row[2])))
            except ValueError:
                continue
    return rows


def image_batch_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    image_col = "image" if "image" in df.columns else df.columns[0]
    batch_col = "batch" if "batch" in df.columns else df.columns[1]
    return {Path(str(row[image_col])).stem: str(row[batch_col]) for _, row in df.iterrows()}


def make_features(pairs: pd.DataFrame, annotations: dict[str, list[tuple[int, int, int]]]) -> pd.DataFrame:
    records: list[dict] = []
    for image, group in pairs.groupby("image", sort=False):
        group = group.reset_index(drop=True)
        gt = np.asarray([(x, y) for x, y, _ in annotations.get(Path(str(image)).stem, [])], dtype=float)
        if gt.size == 0:
            gt = np.empty((0, 2), dtype=float)
        det = group[["det_x", "det_y"]].to_numpy(dtype=float)
        gt_labels = annotations.get(Path(str(image)).stem, [])
        used_gt: set[int] = set()
        for det_idx, row in group.iterrows():
            current_gt = np.asarray([float(row.gt_x), float(row.gt_y)])
            if len(gt):
                distances = np.linalg.norm(gt - det[det_idx], axis=1)
                exact = [i for i, (x, y, c) in enumerate(gt_labels) if i not in used_gt and abs(x - row.gt_x) < 1e-4 and abs(y - row.gt_y) < 1e-4 and c == int(row.true_id)]
                gt_idx = exact[0] if exact else int(np.argmin(np.where(np.isin(np.arange(len(gt)), list(used_gt)), np.inf, distances)))
                used_gt.add(gt_idx)
                other = np.delete(distances, gt_idx)
                d2 = float(other.min()) if len(other) else np.nan
                n_gt_15 = int(np.sum(distances <= 15.0))
                n_gt_10 = int(np.sum(distances <= 10.0))
                nearest_gt_idx = int(np.argmin(distances))
            else:
                distances = np.empty(0)
                gt_idx = -1
                d2 = np.nan
                n_gt_15 = 0
                n_gt_10 = 0
                nearest_gt_idx = -1
            if len(det):
                det_distances = np.linalg.norm(det - current_gt, axis=1)
                n_det_15 = int(np.sum(det_distances <= 15.0))
                n_det_10 = int(np.sum(det_distances <= 10.0))
                nearest_det_idx = int(np.argmin(det_distances))
            else:
                n_det_15 = 0
                n_det_10 = 0
                nearest_det_idx = -1
            d1 = float(row.distance)
            margin = float(d2 - d1) if np.isfinite(d2) else np.nan
            ratio = float(d1 / d2) if np.isfinite(d2) and d2 > 0 else (0.0 if d1 == 0 and np.isfinite(d2) else np.nan)
            mutual = bool(nearest_gt_idx == gt_idx and nearest_det_idx == det_idx)
            records.append(
                {
                    "image": row.image,
                    "batch": row.batch,
                    "gt_x": float(row.gt_x),
                    "gt_y": float(row.gt_y),
                    "det_x": float(row.det_x),
                    "det_y": float(row.det_y),
                    "class_id": int(row.true_id),
                    "class_name": row.true_name,
                    "pair_distance_px": d1,
                    "d1_nearest_gt_px": float(distances.min()) if len(distances) else np.nan,
                    "d2_competing_gt_px": d2,
                    "distance_margin_px": margin,
                    "distance_ratio": ratio,
                    "mutual_nearest_neighbor_matched_pool": mutual,
                    "current_pair_one_to_one": True,
                    "n_gt_candidates_within_15px": n_gt_15,
                    "n_gt_candidates_within_10px": n_gt_10,
                    "n_matched_detection_candidates_within_15px": n_det_15,
                    "n_matched_detection_candidates_within_10px": n_det_10,
                    "competing_unmatched_detections_available": False,
                    "current_baseline_pred_id": int(row.pred_id),
                    "current_baseline_confidence": float(row.confidence),
                }
            )
    return pd.DataFrame(records)


def tier_masks(features: pd.DataFrame) -> dict[str, pd.Series]:
    all_mask = pd.Series(True, index=features.index)
    high = (
        features["current_pair_one_to_one"]
        & features["mutual_nearest_neighbor_matched_pool"]
        & (features["pair_distance_px"] <= 8)
        & ((features["distance_margin_px"] >= 4) | (features["distance_ratio"] <= 0.70))
        & (features["n_gt_candidates_within_15px"] == 1)
    )
    ultra = (
        features["current_pair_one_to_one"]
        & features["mutual_nearest_neighbor_matched_pool"]
        & (features["pair_distance_px"] <= 5)
        & ((features["distance_margin_px"] >= 5) | (features["distance_ratio"] <= 0.60))
        & (features["n_gt_candidates_within_10px"] == 1)
        & (features["n_matched_detection_candidates_within_10px"] == 1)
    )
    return {"ALL_MATCHED": all_mask, "HIGH_CONFIDENCE": high, "ULTRA_HIGH_CONFIDENCE": ultra}


def write_derivative_dataset(root: Path, source: Path, split_root: Path, tier: str, annotations: dict[str, list[tuple[int, int, int]]], kept: set[tuple[str, int, int, int]]) -> None:
    dst = root / "work" / tier
    (dst / "train").mkdir(parents=True, exist_ok=True)
    (dst / "test").mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        image_link = dst / split / "images"
        label_dir = dst / split / "labels"
        label_dir.mkdir(exist_ok=True)
        if not image_link.exists():
            image_link.symlink_to(source / split / "images", target_is_directory=True)
        source_labels = source / split / "labels"
        for label_file in sorted(source_labels.glob("*.csv")):
            stem = label_file.stem
            rows = read_labels(label_file)
            if split == "train" and tier != "ALL_MATCHED":
                rows = [row for row in rows if (stem, row[0], row[1], row[2]) in kept]
            with (label_dir / label_file.name).open("w", newline="") as handle:
                csv.writer(handle).writerows(rows)
    shutil.copy2(source / "label_map.yaml", dst / "label_map.yaml")
    split_dst = dst / "splits"
    if split_dst.exists():
        shutil.rmtree(split_dst)
    shutil.copytree(split_root, split_dst)
    (dst / "train_configs").mkdir(exist_ok=True)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("/data/lf_data/result/task002/diagnostics/baseline_train_matched_predictions.csv"))
    parser.add_argument("--source", type=Path, default=Path("/data/lf_data/xenium_data/CellViT_dataset"))
    parser.add_argument("--split-root", type=Path, default=Path("/data/lf_data/result/splits"))
    parser.add_argument("--metadata", type=Path, default=Path("/data/lf_data/result/data/patch_metadata.csv"))
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task004_high_confidence"))
    args = parser.parse_args()
    result = args.result.resolve()
    for sub in ("config", "code", "work", "qc", "metrics", "figures", "figure_data", "logs", "models"):
        (result / sub).mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs)
    annotations = {p.stem: read_labels(p) for p in sorted((args.source / "train" / "labels").glob("*.csv"))}
    features = make_features(pairs, annotations)
    features.to_csv(result / "metrics/matching_pair_features.csv", index=False)
    masks = tier_masks(features)
    batch_map = image_batch_map(args.metadata)
    features["batch"] = features["image"].map(lambda x: batch_map.get(Path(str(x)).stem, "UNKNOWN"))
    tiers = []
    class_rows = []
    batch_rows = []
    summaries = []
    for tier, mask in masks.items():
        subset = features[mask].copy()
        tiers.append(subset.assign(tier=tier))
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            total = int((features["class_id"] == cls_id).sum())
            retained = int((subset["class_id"] == cls_id).sum())
            class_rows.append({"tier": tier, "class_id": cls_id, "class_name": cls_name, "all_matched_cells": total, "retained_cells": retained, "retention_rate": retained / total if total else np.nan})
        for batch, group in features.groupby("batch", dropna=False):
            total = len(group)
            retained = int((subset["batch"] == batch).sum())
            batch_rows.append({"tier": tier, "batch": batch, "all_matched_cells": total, "retained_cells": retained, "retention_rate": retained / total if total else np.nan})
        summaries.append({
            "tier": tier,
            "retained_cells": len(subset),
            "retained_fraction_of_all_matched": len(subset) / len(features),
            "images": subset.image.nunique(),
            "batches": subset.batch.nunique(),
            "median_pair_distance_px": subset.pair_distance_px.median(),
            "p95_pair_distance_px": subset.pair_distance_px.quantile(.95),
            "mutual_nn_rate_matched_pool": subset.mutual_nearest_neighbor_matched_pool.mean(),
            "ambiguous_gt_rate_15px": (subset.n_gt_candidates_within_15px > 1).mean(),
            "ambiguous_gt_rate_10px": (subset.n_gt_candidates_within_10px > 1).mean(),
            "unmatched_detection_competitor_data_available": False,
        })
        kept = {(str(row.image), int(round(row.gt_x)), int(round(row.gt_y)), int(row.class_id)) for row in subset.itertuples()}
        write_derivative_dataset(result, args.source, args.split_root, tier, annotations, kept)
    tier_features = pd.concat(tiers, ignore_index=True)
    tier_features.to_csv(result / "figure_data/matching_pair_features_by_tier.csv", index=False)
    pd.DataFrame(summaries).to_csv(result / "metrics/matching_summary_by_tier.csv", index=False)
    pd.DataFrame(class_rows).to_csv(result / "metrics/class_retention_by_tier.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(result / "metrics/batch_retention_by_tier.csv", index=False)

    sensitivity_rows = []
    for threshold in (5, 6, 8, 10, 12, 15):
        conditions = {
            "mutual_nn_only": features.mutual_nearest_neighbor_matched_pool,
            "margin_ge_3": features.mutual_nearest_neighbor_matched_pool & (features.distance_margin_px >= 3),
            "margin_ge_5": features.mutual_nearest_neighbor_matched_pool & (features.distance_margin_px >= 5),
            "ratio_le_0.6": features.mutual_nearest_neighbor_matched_pool & (features.distance_ratio <= .6),
            "ratio_le_0.7": features.mutual_nearest_neighbor_matched_pool & (features.distance_ratio <= .7),
            "ratio_le_0.8": features.mutual_nearest_neighbor_matched_pool & (features.distance_ratio <= .8),
        }
        for condition, condition_mask in conditions.items():
            mask = condition_mask & (features.pair_distance_px <= threshold) & (features.n_gt_candidates_within_15px == 1)
            sensitivity_rows.append({"distance_threshold_px": threshold, "ambiguity_condition": condition, "retained_cells": int(mask.sum()), "retained_fraction": float(mask.mean()), "cv_performance_source": "to_be_filled_after_primary_official_CV"})
    pd.DataFrame(sensitivity_rows).to_csv(result / "metrics/threshold_sensitivity.csv", index=False)

    geometry_rows = [
        {"object": "train_images", "available": True, "count": len(list((args.source / "train" / "images").glob("*"))), "details": "RGB patch images"},
        {"object": "test_images", "available": True, "count": len(list((args.source / "test" / "images").glob("*"))), "details": "RGB patch images"},
        {"object": "xenium_centroids", "available": True, "count": sum(len(v) for v in annotations.values()), "details": "CSV x,y,class rows"},
        {"object": "xenium_nucleus_polygons", "available": False, "count": 0, "details": "not present in source dataset"},
        {"object": "xenium_cell_polygons", "available": False, "count": 0, "details": "not present in source dataset"},
        {"object": "cellvit_centroids", "available": True, "count": len(pairs), "details": "archived current matched detection centroids only"},
        {"object": "cellvit_nucleus_polygons", "available": False, "count": 0, "details": "not archived in Task2 pair table"},
        {"object": "affine_alignment_transform", "available": False, "count": 0, "details": "no separate transform file found"},
        {"object": "patch_to_global_mapping", "available": False, "count": 0, "details": "analysis uses patch-local coordinates"},
        {"object": "batch_identifiers", "available": True, "count": len(batch_map), "details": "data/patch_metadata.csv"},
    ]
    pd.DataFrame(geometry_rows).to_csv(result / "qc/geometry_inventory.csv", index=False)
    (result / "qc/current_matching_definition.md").write_text(
        """# Current matching definition

The archived Task 2 training audit uses the installed CellViT++ `pair_coordinates` implementation. For each image, GT/Xenium centroid coordinates are set A and CellViT detection centroids are set B. The implementation constructs an Euclidean distance matrix, applies SciPy `linear_sum_assignment` (Hungarian/Munkres) for globally optimal unique assignment, and then retains assigned pairs whose assignment distance is <=15 px. Pairs outside the radius are removed after assignment. Thus the current rule is a global one-to-one Hungarian assignment followed by a 15 px radius filter, not greedy nearest-neighbor matching. A GT and a detection can have multiple candidates within 15 px before assignment; the retained pairing itself is unique.

Available source geometry is centroid-only: annotation CSVs contain x, y and integer class IDs. No Xenium polygons, CellViT polygons, affine transform file, or global coordinate mapping was found in the supplied derivative inputs. Task 4 therefore uses centroid-only confidence rules. The archived Task 2 pair table contains matched detections but not the full unmatched-detection coordinate pool; mutual-nearest and detection-competitor fields are consequently computed against the archived matched-detection pool and explicitly marked as such.

The exact implementation was verified in `cellvit/training/utils/tools.py` on the remote CellViT++ installation; the Task 2 reconstruction code is preserved under the Task 4 `code/` directory.
"""
    )
    baseline = Path("/data/lf_data/result/model_best.pth")
    protected = Path("/data/lf_data/result/task001_model_best_baseline.pth")
    t3 = Path("/data/lf_data/result/task003_official/model_official_best.pth")
    manifest = {
        "task": "task_004",
        "seed": 42,
        "source_dataset": str(args.source),
        "pair_table": str(args.pairs),
        "pair_table_sha256": sha256(args.pairs),
        "tiers": {k: int(v.sum()) for k, v in masks.items()},
        "protected_checksums": {str(p): sha256(p) for p in (baseline, protected, t3) if p.exists()},
        "geometry": "centroid_only; no polygons or affine transform available",
        "unmatched_detection_pool_archived": False,
    }
    (result / "config/preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
