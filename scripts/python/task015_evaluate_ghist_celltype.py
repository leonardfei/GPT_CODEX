#!/usr/bin/env python3
"""Audit 30-pixel-overlap GHIST validation coverage and area deduplication.

This stage prepares an exact, label-independent patch choice for each frozen
canonical CellViT contour. It does not yet produce model predictions.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyvips
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, average_precision_score, log_loss, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/data/lf_data/result/task015_ghist_celltyping")
PATCH_SIZE = 256
OVERLAP = 30
STEP = PATCH_SIZE - OVERLAP
WIDTH, HEIGHT = 50000, 23451


def starts(length):
    values = list(range(0, length - PATCH_SIZE, STEP))
    values.append(length - PATCH_SIZE)
    return np.asarray(sorted(set(values)), dtype=np.int32)


def clipped_area(polygon, x, y):
    xmin = max(int(polygon[:, 0].min()), x)
    xmax = min(int(polygon[:, 0].max()) + 1, x + PATCH_SIZE)
    ymin = max(int(polygon[:, 1].min()), y)
    ymax = min(int(polygon[:, 1].max()) + 1, y + PATCH_SIZE)
    if xmin >= xmax or ymin >= ymax:
        return 0
    mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)
    shifted = polygon - np.asarray([xmin, ymin], dtype=np.int32)
    cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2)], 1)
    return int(mask.sum())


def candidates_for_contour(polygon, xstarts, ystarts):
    xmin, xmax = int(polygon[:, 0].min()), int(polygon[:, 0].max()) + 1
    ymin, ymax = int(polygon[:, 1].min()), int(polygon[:, 1].max()) + 1
    xs = xstarts[(xstarts < xmax) & (xstarts + PATCH_SIZE > xmin)]
    ys = ystarts[(ystarts < ymax) & (ystarts + PATCH_SIZE > ymin)]
    for y in ys:
        for x in xs:
            area = clipped_area(polygon, int(x), int(y))
            if area > 0:
                yield int(x), int(y), area


def overlap_qc():
    patches = pd.read_csv(ROOT / "metrics/patch_manifest.csv.gz")
    canonical = pd.read_csv("/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz")
    assert len(canonical) == 96044 and canonical.cell_id.is_unique
    patch_origins = {r.image: (int(r.x), int(r.y)) for r in patches.itertuples(index=False)}
    xstarts, ystarts = starts(WIDTH), starts(HEIGHT)
    rows = []
    seen = set()
    with gzip.open(ROOT / "data_manifest/patch_contours.jsonl.gz", "rt") as source:
        for line in source:
            record = json.loads(line)
            ox, oy = patch_origins[record["image"]]
            for inst in record["instances"]:
                cell_id = inst["cell_id"]
                if cell_id is None:
                    continue
                if cell_id in seen:
                    raise RuntimeError(f"Repeated canonical contour: {cell_id}")
                seen.add(cell_id)
                polygon = np.asarray(inst["contour"], dtype=np.int32) + np.asarray([ox, oy], dtype=np.int32)
                for x, y, area in candidates_for_contour(polygon, xstarts, ystarts):
                    rows.append((cell_id, x, y, area))
    if seen != set(canonical.cell_id):
        raise RuntimeError(f"Overlapping inference source missing {len(set(canonical.cell_id)-seen)} canonical contours")
    candidates = pd.DataFrame(rows, columns=["cell_id", "patch_x", "patch_y", "visible_area"])
    if candidates.empty:
        raise RuntimeError("No overlapping inference candidates")
    candidates.sort_values(["cell_id", "visible_area", "patch_y", "patch_x"],
                           ascending=[True, False, True, True], inplace=True, kind="stable")
    candidates["selected_largest_area"] = ~candidates.duplicated("cell_id")
    selected = candidates[candidates.selected_largest_area].copy()
    if len(selected) != 96044 or selected.cell_id.duplicated().any():
        raise RuntimeError("Overlap deduplication did not yield exactly 96,044 cells")
    candidates.to_csv(ROOT / "metrics/overlap_patch_candidates.csv.gz", index=False, compression="gzip")
    selected.to_csv(ROOT / "metrics/overlap_largest_area_manifest.csv.gz", index=False, compression="gzip")
    folds = pd.read_csv("/data/lf_data/result/task010_representation_benchmark/metrics/fold_manifest.csv")
    per_fold = []
    for fold, group in folds.groupby("fold"):
        val = group[group.role == "val"]
        matched = val.merge(selected[["cell_id", "patch_x", "patch_y", "visible_area"]],
                            on="cell_id", how="left", validate="one_to_one")
        if len(matched) != len(val) or matched.visible_area.isna().any():
            raise RuntimeError(f"Fold {fold} has missing or duplicate validation predictions")
        per_fold.append({"fold": int(fold), "validation_cells": len(val),
                         "unique_selected_cells": matched.cell_id.nunique(),
                         "inference_patches": matched[["patch_x", "patch_y"]].drop_duplicates().shape[0]})
    counts = candidates.groupby("cell_id").size()
    audit = {"status": "OVERLAP_GEOMETRY_PASS", "patch_size": PATCH_SIZE,
             "overlap_pixels": OVERLAP, "step_pixels": STEP,
             "wsi_width": WIDTH, "wsi_height": HEIGHT,
             "canonical_cells": len(seen), "candidate_predictions": len(candidates),
             "deduplicated_predictions": len(selected),
             "cells_with_multiple_patch_candidates": int((counts > 1).sum()),
             "max_patch_candidates_per_cell": int(counts.max()),
             "per_fold": per_fold,
             "note": "Geometry-only audit; model logits and cell-level metrics not yet evaluated."}
    (ROOT / "qc/overlap_geometry_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


class OverlapPatchDataset(Dataset):
    """Read native WSI patches and rasterize all patch-wise CellViT contours."""

    def __init__(self, selected_cells: pd.DataFrame, mean, std):
        patches = pd.read_csv(ROOT / "metrics/patch_manifest.csv.gz")
        origins = {r.image: (int(r.x), int(r.y)) for r in patches.itertuples(index=False)}
        self.polygons = []
        self.centroids = []
        self.gid_to_cell = {}
        self.cell_to_gid = {}
        max_radius = 0.0
        with gzip.open(ROOT / "data_manifest/patch_contours.jsonl.gz", "rt") as source:
            for line in source:
                record = json.loads(line)
                ox, oy = origins[record["image"]]
                for inst in record["instances"]:
                    polygon = np.asarray(inst["contour"], dtype=np.int32) + np.asarray([ox, oy], dtype=np.int32)
                    center = np.asarray(inst["centroid"], dtype=np.float64) + np.asarray([ox, oy], dtype=np.float64)
                    gid = len(self.polygons) + 1
                    self.polygons.append(polygon)
                    self.centroids.append(center)
                    if len(polygon):
                        max_radius = max(max_radius, float(np.linalg.norm(polygon - center, axis=1).max()))
                    cell_id = inst["cell_id"]
                    if cell_id is not None:
                        if cell_id in self.cell_to_gid:
                            raise RuntimeError(f"Repeated canonical contour: {cell_id}")
                        self.gid_to_cell[gid] = cell_id
                        self.cell_to_gid[cell_id] = gid
        self.centroids = np.asarray(self.centroids)
        self.tree = cKDTree(self.centroids)
        self.query_radius = (2**0.5) * 128 + max_radius + 1
        selected = selected_cells.copy()
        selected["global_instance_id"] = selected.cell_id.map(self.cell_to_gid)
        if selected.global_instance_id.isna().any():
            raise RuntimeError("Selected validation cell has no contour")
        selected.global_instance_id = selected.global_instance_id.astype(int)
        self.selected_by_patch = {
            (int(x), int(y)): dict(zip(group.global_instance_id.astype(int), group.cell_id))
            for (x, y), group in selected.groupby(["patch_x", "patch_y"])
        }
        self.patch_coords = sorted(self.selected_by_patch)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.wsi = None

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, index):
        if self.wsi is None:
            self.wsi = pyvips.Image.new_from_file("/data/lf_data/xenium_data/ID0060276.ome.tif", access="random")
        x, y = self.patch_coords[index]
        crop = self.wsi.crop(x, y, PATCH_SIZE, PATCH_SIZE)
        image = np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8,
                           shape=(PATCH_SIZE, PATCH_SIZE, crop.bands)).copy()
        if image.shape[2] < 3:
            raise RuntimeError("WSI crop has fewer than three channels")
        image = image[:, :, :3]
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.int32)
        nearby = self.tree.query_ball_point([x + 128, y + 128], self.query_radius)
        for idx in nearby:
            polygon = self.polygons[idx]
            if len(polygon) < 3:
                continue
            if polygon[:, 0].max() < x or polygon[:, 0].min() >= x + PATCH_SIZE:
                continue
            if polygon[:, 1].max() < y or polygon[:, 1].min() >= y + PATCH_SIZE:
                continue
            cv2.fillPoly(mask, [(polygon - [x, y]).reshape(-1, 1, 2)], idx + 1)
        selected = self.selected_by_patch[(x, y)]
        if not set(selected) <= set(np.unique(mask)):
            missing = set(selected) - set(np.unique(mask))
            raise RuntimeError(f"Selected canonical contour absent from overlap patch {(x, y)}: {len(missing)}")
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                torch.from_numpy(mask.astype(np.int64)), selected, (x, y))


def overlap_collate(batch):
    images, masks, selected, coords = zip(*batch)
    return torch.stack(images), torch.stack(masks), selected, coords


def evaluate_model(model, dataset: OverlapPatchDataset, true_labels: dict, device,
                   batch_size=8, return_predictions=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=overlap_collate)
    model.eval()
    records = []
    with torch.inference_mode():
        for images, masks, selected, coords in loader:
            output = model(images.to(device), masks.to(device))
            probabilities = torch.softmax(output["cell_logits"].float(), dim=1).cpu().numpy()
            embeddings = output["embeddings"].float().cpu().numpy() if return_predictions else None
            for i, (batch_idx, instance_id) in enumerate(output["ordered_ids"]):
                if instance_id in selected[batch_idx]:
                    cell_id = selected[batch_idx][instance_id]
                    records.append((cell_id, int(true_labels[cell_id]), probabilities[i],
                                    embeddings[i] if embeddings is not None else None))
    expected = sum(len(sub) for sub in dataset.selected_by_patch.values())
    if len(records) != expected:
        raise RuntimeError(f"Validation prediction count mismatch: {len(records)}")
    ids = [r[0] for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Repeated validation cell after largest-area deduplication")
    y = np.asarray([r[1] for r in records], dtype=int)
    p = np.stack([r[2] for r in records])
    if not np.isfinite(p).all():
        raise FloatingPointError("Non-finite validation probabilities")
    pred = p.argmax(axis=1)
    macro_f1 = float(f1_score(y, pred, labels=list(range(7)), average="macro", zero_division=0))
    n_ap = float(average_precision_score(y == 3, p[:, 3]))
    ce = float(log_loss(y, p, labels=list(range(7))))
    model.train()
    metrics = {"n_cells": len(ids), "macro_f1": macro_f1,
               "neutrophil_auprc": n_ap, "cross_entropy": ce,
               "cell_ids_unique": True, "probabilities_finite": True}
    return (metrics, records) if return_predictions else metrics


def overlap_mask_presence_qc():
    selected = pd.read_csv(ROOT / "metrics/overlap_largest_area_manifest.csv.gz")
    dataset = OverlapPatchDataset(selected, [0, 0, 0], [1, 1, 1])
    checked = 0
    for i in range(len(dataset)):
        _, _, cells, _ = dataset[i]
        checked += len(cells)
    if checked != 96044:
        raise RuntimeError(f"Expected 96,044 visible selected cells, got {checked}")
    audit = {"status": "PASS", "selected_canonical_cells": checked,
             "overlap_patches_checked": len(dataset),
             "selected_instance_visible_in_raster_mask": True}
    (ROOT / "qc/overlap_mask_presence_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


def true_binary_probe(z_train, y_train, z_val, y_val, fold, positive_class, negative_class):
    """Exact corrected-Task011-style training-only linear probe on GHIST embeddings.

    `y_train` and `y_val` are seven-class IDs; only the specified two classes
    enter the binary fit/evaluation. No outer-validation data enters scaling,
    weighting, fitting, or threshold selection.
    """
    train_keep = np.isin(y_train, [positive_class, negative_class])
    val_keep = np.isin(y_val, [positive_class, negative_class])
    ztr = np.asarray(z_train[train_keep], dtype=np.float32)
    zva = np.asarray(z_val[val_keep], dtype=np.float32)
    ytr = (np.asarray(y_train)[train_keep] == positive_class).astype(np.int64)
    yva = (np.asarray(y_val)[val_keep] == positive_class).astype(np.int64)
    if len(np.unique(ytr)) != 2 or len(np.unique(yva)) != 2:
        raise RuntimeError("True binary probe needs both classes in train and validation")
    scaler = StandardScaler().fit(ztr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(scaler.transform(ztr).astype(np.float32)).to(device)
    xv = torch.from_numpy(scaler.transform(zva).astype(np.float32)).to(device)
    yt = torch.from_numpy(ytr).to(device)
    torch.manual_seed(20260923 + 5000 + int(fold) * 10 + positive_class + negative_class)
    w = torch.zeros((2, xt.shape[1]), device=device, requires_grad=True)
    b = torch.zeros(2, device=device, requires_grad=True)
    counts = torch.bincount(yt, minlength=2).float()
    weights = len(yt) / (2 * torch.clamp(counts, min=1.0))
    optimizer = torch.optim.LBFGS([w, b], lr=1.0, max_iter=40, history_size=10,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-5)
    regularization = 0.5 / max(len(yt), 1)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = xt @ w.T + b
        loss = torch.nn.functional.cross_entropy(logits, yt, weight=weights)
        loss = loss + regularization * torch.sum(w * w)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        probabilities = torch.softmax(xv @ w.T + b, dim=1)[:, 1].float().cpu().numpy()
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yva, predicted, labels=[1], zero_division=0)
    tn = int(((predicted == 0) & (yva == 0)).sum())
    return {"auroc": float(roc_auc_score(yva, probabilities)),
            "auprc": float(average_precision_score(yva, probabilities)),
            "f1": float(f1[0]), "sensitivity": float(recall[0]),
            "specificity": float(tn / max(int((yva == 0).sum()), 1)),
            "precision": float(precision[0]), "n_val": int(len(yva))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-qc", action="store_true")
    parser.add_argument("--mask-presence-qc", action="store_true")
    args = parser.parse_args()
    if args.mask_presence_qc:
        overlap_mask_presence_qc()
    elif args.overlap_qc:
        overlap_qc()
    else:
        raise SystemExit("Only geometry QC is implemented; classification evaluation remains gated")
