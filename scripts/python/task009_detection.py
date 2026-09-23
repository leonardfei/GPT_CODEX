#!/usr/bin/env python3
"""Audit CellViT nucleus detection recall under the frozen Task009 labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/data/lf_data/CellViT-plus-plus")

from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.training.evaluate.inference_cellvit_experiment_detection import CellViTInfExpDetection
from cellvit.training.utils.tools import pair_coordinates


CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]


class DetectionOnlyExperiment(CellViTInfExpDetection):
    def __init__(self, *args, split: str = "train", **kwargs):
        self._split = split
        super().__init__(*args, **kwargs)

    def _load_model(self, checkpoint_path):
        return None, {
            "data": {"num_classes": 7},
            "training": {"mixed_precision": True},
            "transformations": {"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        }

    def _load_dataset(self, transforms, normalize_stains):
        dataset = DetectionDataset(
            dataset_path=self.dataset_path,
            split=self._split,
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()
        return dataset


def audit_condition(result: Path, condition: str, cellvit: Path, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = result / "work" / f"CellViT_dataset_v3_{condition}"
    exp = DetectionOnlyExperiment(
        logdir=result / "logs" / "detection" / condition,
        cellvit_path=cellvit,
        dataset_path=dataset,
        input_shape=[256, 256],
        normalize_stains=False,
        gpu=0,
        comment=f"task009_detection_{condition}",
        split="train",
    )
    loader = DataLoader(exp.inference_dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=exp.inference_dataset.collate_batch)
    postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
    pair_rows, image_rows = [], []
    with torch.no_grad():
        for images, gt_batch, types_batch, image_names in loader:
            _, _, pred_dicts, _, _, _ = exp._get_cellvit_result(
                images=images,
                cell_gt_batch=gt_batch,
                types_batch=types_batch,
                image_names=image_names,
                postprocessor=postprocessor,
            )
            for true_centroids, cell_types, image_name, pred_dict in zip(gt_batch, types_batch, image_names, pred_dicts.values()):
                image_name = image_name.decode() if isinstance(image_name, bytes) else str(image_name)
                true_centroids = np.asarray(true_centroids, dtype=float).reshape(-1, 2)
                cell_types = np.asarray(cell_types, dtype=int).reshape(-1)
                pred_centroids = np.asarray([v["centroid"] for v in pred_dict.values()], dtype=float).reshape(-1, 2)
                if pred_centroids.size == 0:
                    pred_centroids = np.empty((0, 2), dtype=float)
                if true_centroids.size == 0:
                    true_centroids = np.empty((0, 2), dtype=float)
                paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroids, pred_centroids, 15)
                paired_true = set(int(x) for x in paired[:, 0]) if len(paired) else set()
                paired_order = {int(gt_idx): int(order) for order, (gt_idx, _pred_idx) in enumerate(paired)}
                image_rows.append({
                    "condition": f"V3_{condition}",
                    "image": image_name,
                    "gt_cells": len(true_centroids),
                    "detected_cells": len(pred_centroids),
                    "matched_cells": len(paired),
                    "unmatched_gt": len(unpaired_true),
                    "unmatched_detections": len(unpaired_pred),
                })
                for gt_idx, gt in enumerate(true_centroids):
                    class_id = int(cell_types[gt_idx])
                    pair_rows.append({
                        "condition": f"V3_{condition}",
                        "image": image_name,
                        "gt_x": float(gt[0]),
                        "gt_y": float(gt[1]),
                        "class_id": class_id,
                        "class_name": CLASSES[class_id],
                        "detected": int(gt_idx in paired_true),
                        "match_order": paired_order.get(gt_idx, -1),
                    })
    pairs = pd.DataFrame(pair_rows)
    images = pd.DataFrame(image_rows)
    pairs.to_csv(result / "metrics" / f"detection_pairs_{condition}.csv", index=False)
    images.to_csv(result / "metrics" / f"detection_by_image_{condition}.csv", index=False)
    return pairs, images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task009_v3_retraining"))
    parser.add_argument("--cellvit", type=Path, default=Path("/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    all_rows = []
    for condition in ("CORE", "EXTENDED"):
        pairs, _ = audit_condition(args.result, condition, args.cellvit, args.batch_size)
        for (condition_name, class_id, class_name), sub in pairs.groupby(["condition", "class_id", "class_name"], sort=False):
            all_rows.append({
                "condition": condition_name,
                "class_id": int(class_id),
                "class_name": class_name,
                "gt_cells": int(len(sub)),
                "detected_cells": int(sub["detected"].sum()),
                "detection_recall": float(sub["detected"].mean()),
            })
    out = pd.DataFrame(all_rows)
    out.to_csv(args.result / "metrics/detection_recall_by_class.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
