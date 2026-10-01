#!/usr/bin/env python3
"""Reproduce Task010 patch-wise CellViT matching and retain instance contours.

This is an audit/preparation stage only; it does not train GHIST or alter labels.
The 15-pixel pairing radius and CellViT invocation are identical to Task010.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

ROOT = Path("/data/lf_data/result/task015_ghist_celltyping")
CANON = Path("/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz")
DATASET = Path("/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE")
CELLVIT = Path("/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth")
SEED = 20260923
RADIUS = 15
PATCH_META = DATASET / "patch_metadata.csv"
FOLDS = Path("/data/lf_data/result/task010_representation_benchmark/metrics/fold_manifest.csv")

sys.path.insert(0, "/data/lf_data/CellViT-plus-plus")
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.training.evaluate.inference_cellvit_experiment_detection import CellViTInfExpDetection
from cellvit.training.utils.tools import pair_coordinates


class DetectionOnlyExperiment(CellViTInfExpDetection):
    def __init__(self, *args, split="train", **kwargs):
        self._split = split
        super().__init__(*args, **kwargs)

    def _load_model(self, checkpoint_path):
        return None, {
            "data": {"num_classes": 7},
            "training": {"mixed_precision": True},
            "transformations": {"normalize": {"mean": [0.5] * 3, "std": [0.5] * 3}},
        }

    def _load_dataset(self, transforms, normalize_stains):
        dataset = DetectionDataset(
            dataset_path=self.dataset_path, split=self._split,
            normalize_stains=normalize_stains, transforms=transforms,
        )
        dataset.cache_dataset()
        return dataset


def name(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def build_patch_manifest():
    canonical = pd.read_csv(CANON)
    instances = pd.read_csv(ROOT / "metrics/canonical_instance_manifest.csv.gz")
    patches = pd.read_csv(PATCH_META)
    patches = patches[(patches.condition == "CORE") & (patches.split == "train")].copy()
    assert len(canonical) == 96044
    assert len(instances) == len(canonical)
    assert set(canonical.cell_id) == set(instances.cell_id)
    assert not instances.cell_id.duplicated().any()
    assert instances.contour_valid.all()
    assert patches.image.is_unique
    patch_batches = canonical.groupby("image").batch.nunique()
    assert int(patch_batches.max()) == 1
    counts = canonical.groupby("image").size().rename("n_canonical").reset_index()
    out = patches[["patch_id", "image", "source_image", "batch", "x", "y"]].merge(
        counts, on="image", how="left", validate="one_to_one")
    out["n_canonical"] = out.n_canonical.fillna(0).astype(int)
    out["image_path"] = out.image.map(lambda s: str(DATASET / "train" / "images" / (s + ".png")))
    assert out.image_path.map(lambda s: Path(s).is_file()).all()
    assert int(out.n_canonical.sum()) == len(canonical)
    folds = pd.read_csv(FOLDS)
    assert set(folds.role) == {"train", "val"}
    fold_audit = []
    for fold, group in folds.groupby("fold"):
        assert len(group) == len(canonical)
        assert group.cell_id.is_unique
        assert set(group.cell_id) == set(canonical.cell_id)
        assert not group.groupby("batch").role.nunique().gt(1).any()
        val_batches = sorted(group.loc[group.role == "val", "batch"].unique())
        train_batches = sorted(group.loc[group.role == "train", "batch"].unique())
        assert not set(val_batches) & set(train_batches)
        fold_audit.append({"fold": int(fold), "train_cells": int((group.role == "train").sum()),
                           "val_cells": int((group.role == "val").sum()),
                           "train_batches": train_batches, "val_batches": val_batches,
                           "inner_val_batch": train_batches[-1]})
    assert len(fold_audit) == 5
    out.to_csv(ROOT / "metrics/patch_manifest.csv.gz", index=False, compression="gzip")
    audit = {"canonical_cells": len(canonical), "matched_instances": len(instances),
             "training_patches": len(out), "patches_with_canonical": int((out.n_canonical > 0).sum()),
             "multi_batch_canonical_patches": int((patch_batches > 1).sum()), "folds": fold_audit}
    (ROOT / "qc/patch_manifest_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="Only for schema smoke tests")
    parser.add_argument("--build-patch-manifest", action="store_true")
    args = parser.parse_args()
    if args.build_patch_manifest:
        build_patch_manifest()
        return

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    canonical = pd.read_csv(CANON)
    required = ["cell_id", "image", "local_x", "local_y", "class_id", "batch", "patch_id", "he_x", "he_y"]
    assert len(canonical) == 96044
    assert not canonical[required].isna().any().any()
    assert not canonical.cell_id.duplicated().any()
    canonical["key"] = list(zip(canonical.image.astype(str), canonical.local_x.astype(int), canonical.local_y.astype(int), canonical.class_id.astype(int)))
    assert not canonical.key.duplicated().any()
    key_to_row = {key: row for key, row in zip(canonical.key, canonical.itertuples(index=False))}

    exp = DetectionOnlyExperiment(
        logdir=ROOT / "logs/cellvit_patch_matching", cellvit_path=CELLVIT,
        dataset_path=DATASET, input_shape=[256, 256], normalize_stains=False,
        gpu=args.gpu, comment="task015_reproduce_task010_contours", split="train",
    )
    loader = DataLoader(exp.inference_dataset, batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False,
                        collate_fn=exp.inference_dataset.collate_batch)
    postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
    output = ROOT / "data_manifest"
    output.mkdir(parents=True, exist_ok=True)
    contour_path = output / ("patch_contours_smoke.jsonl.gz" if args.max_batches else "patch_contours.jsonl.gz")
    rows = []
    seen = set()
    n_pred = n_patches = 0
    t0 = time.time()
    with gzip.open(contour_path, "wt") as contours, torch.no_grad():
        for bi, (images, gt_batch, types_batch, image_names) in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            _, _, pred_dicts, _, _, _ = exp._get_cellvit_result(
                images=images, cell_gt_batch=gt_batch, types_batch=types_batch,
                image_names=image_names, postprocessor=postprocessor,
            )
            for true_centroids, true_types, raw_name in zip(gt_batch, types_batch, image_names):
                nm = name(raw_name)
                pred_dict = pred_dicts[raw_name] if raw_name in pred_dicts else pred_dicts[nm]
                pred_values = list(pred_dict.values())
                pred_xy = np.asarray([item["centroid"] for item in pred_values], dtype=float).reshape(-1, 2)
                true_xy = np.asarray(true_centroids, dtype=float).reshape(-1, 2)
                true_types = np.asarray(true_types, dtype=int).reshape(-1)
                pairs = np.empty((0, 2), dtype=int)
                candidate_counts = np.zeros(len(true_xy), dtype=int)
                second_distances = np.full(len(true_xy), np.nan)
                if len(pred_xy) and len(true_xy):
                    pairs, _, _ = pair_coordinates(true_xy, pred_xy, RADIUS)
                    tree = cKDTree(pred_xy)
                    candidate_counts = np.asarray([len(c) for c in tree.query_ball_point(true_xy, RADIUS)], dtype=int)
                    if len(pred_xy) > 1:
                        second_distances = tree.query(true_xy, k=2)[0][:, 1]
                pred_to_canonical = {}
                for true_idx, pred_idx in pairs:
                    tx, ty = np.rint(true_xy[int(true_idx)]).astype(int)
                    key = (nm, int(tx), int(ty), int(true_types[int(true_idx)]))
                    if key not in key_to_row:
                        continue
                    if key in seen:
                        raise RuntimeError(f"Duplicate canonical matching: {key}")
                    seen.add(key)
                    row = key_to_row[key]
                    item = pred_values[int(pred_idx)]
                    local_id = int(list(pred_dict.keys())[int(pred_idx)])
                    distance = float(np.linalg.norm(pred_xy[int(pred_idx)] - true_xy[int(true_idx)]))
                    pred_to_canonical[local_id] = row.cell_id
                    rows.append({"cell_id": row.cell_id, "batch": row.batch,
                                 "patch_id": int(row.patch_id), "image": nm,
                                 "class_id": int(row.class_id), "instance_id": local_id,
                                 "pred_x": float(item["centroid"][0]),
                                 "pred_y": float(item["centroid"][1]),
                                 "match_distance": distance,
                                 "candidate_count_within_15px": int(candidate_counts[int(true_idx)]),
                                 "second_nearest_distance": float(second_distances[int(true_idx)]),
                                 "contour_n_vertices": len(item["contour"]),
                                 "contour_valid": len(item["contour"]) >= 3})
                n_pred += len(pred_values)
                n_patches += 1
                contours.write(json.dumps({"image": nm, "instances": [
                    {"instance_id": int(cid), "centroid": np.asarray(item["centroid"]).tolist(),
                     "contour": np.asarray(item["contour"]).tolist(),
                     "cell_id": pred_to_canonical.get(int(cid))}
                    for cid, item in pred_dict.items()
                ]}, separators=(",", ":")) + "\n")
            if bi % 25 == 0:
                print(f"batches={bi} patches={n_patches} canonical={len(seen)} elapsed_min={(time.time()-t0)/60:.1f}", flush=True)

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        assert not manifest.cell_id.duplicated().any()
        assert not manifest.duplicated(["image", "instance_id"]).any()
    suffix = "_smoke" if args.max_batches else ""
    manifest_path = ROOT / "metrics" / f"canonical_instance_manifest{suffix}.csv.gz"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, compression="gzip")
    summary = {
        "source": "Task010 DetectionOnlyExperiment patch-wise CellViT-SAM-H-x40-AMP",
        "seed": SEED, "matching_radius_native_pixels": RADIUS,
        "canonical_total": len(canonical), "matched_canonical": len(seen),
        "matched_fraction": len(seen) / len(canonical),
        "valid_contour_count": int(manifest.contour_valid.sum()) if not manifest.empty else 0,
        "ambiguous_candidate_count": int((manifest.candidate_count_within_15px > 1).sum()) if not manifest.empty else 0,
        "ambiguous_candidate_fraction": float((manifest.candidate_count_within_15px > 1).mean()) if not manifest.empty else 0,
        "patches_processed": n_patches, "detected_instances": n_pred,
        "max_batches": args.max_batches, "elapsed_minutes": (time.time()-t0)/60,
        "match_distance_quantiles": {str(q): float(manifest.match_distance.quantile(q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]} if not manifest.empty else {},
    }
    summary_path = ROOT / "qc" / f"patch_matching_summary{suffix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
