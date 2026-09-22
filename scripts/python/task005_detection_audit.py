#!/usr/bin/env python3
"""Run the official CellViT detector on train images and audit class recall."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.training.evaluate.inference_cellvit_experiment_detection import CellViTInfExpDetection
from cellvit.training.utils.tools import pair_coordinates


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]


class TrainDetectionExperiment(CellViTInfExpDetection):
    def _load_model(self, checkpoint_path):
        """Detection audit does not need a classifier head checkpoint."""
        return None, {
            "data": {"num_classes": 7},
            "training": {"mixed_precision": True},
            "transformations": {"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        }

    def _load_dataset(self, transforms, normalize_stains):
        dataset = DetectionDataset(
            dataset_path=self.dataset_path,
            split="train",
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()
        return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task005_backbone_domain"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cellvit", type=Path, required=True)
    parser.add_argument("--classifier-run", type=Path, required=True)
    parser.add_argument("--normalize-stains", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    result = args.result.resolve()
    qc = result / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    exp = TrainDetectionExperiment(
        logdir=args.classifier_run,
        cellvit_path=args.cellvit,
        dataset_path=args.dataset,
        input_shape=[256, 256],
        normalize_stains=args.normalize_stains,
        gpu=args.gpu,
        comment=f"task005_detection_{args.condition}",
    )
    postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
    loader = DataLoader(
        exp.inference_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=exp.inference_dataset.collate_batch,
    )
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
            for true_centroids, cell_types, image_name, pred_dict in zip(
                gt_batch, types_batch, image_names, pred_dicts.values()
            ):
                image_name = image_name.decode() if isinstance(image_name, bytes) else str(image_name)
                true_centroids = np.asarray(true_centroids, dtype=float).reshape(-1, 2)
                cell_types = np.asarray(cell_types, dtype=int).reshape(-1)
                pred_centroids = np.asarray([v["centroid"] for v in pred_dict.values()], dtype=float).reshape(-1, 2)
                if pred_centroids.size == 0:
                    pred_centroids = np.empty((0, 2), dtype=float)
                if true_centroids.size == 0:
                    true_centroids = np.empty((0, 2), dtype=float)
                paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroids, pred_centroids, 15)
                image_rows.append(
                    {
                        "condition": args.condition,
                        "image": image_name,
                        "gt_cells": len(true_centroids),
                        "detected_cells": len(pred_centroids),
                        "matched_cells": len(paired),
                        "unmatched_gt": len(unpaired_true),
                        "unmatched_detections": len(unpaired_pred),
                    }
                )
                for gt_idx, det_idx in paired:
                    gt = true_centroids[gt_idx]
                    det = pred_centroids[det_idx]
                    pair_rows.append(
                        {
                            "condition": args.condition,
                            "image": image_name,
                            "gt_x": float(gt[0]),
                            "gt_y": float(gt[1]),
                            "det_x": float(det[0]),
                            "det_y": float(det[1]),
                            "distance_px": float(np.linalg.norm(gt - det)),
                            "class_id": int(cell_types[gt_idx]),
                            "class_name": CLASS_NAMES[int(cell_types[gt_idx])],
                        }
                    )
    pairs = pd.DataFrame(pair_rows)
    images = pd.DataFrame(image_rows)
    pairs.to_csv(qc / f"detection_pairs_{args.condition}.csv", index=False)
    images.to_csv(qc / f"detection_by_image_{args.condition}.csv", index=False)
    print({"condition": args.condition, "images": len(images), "gt": int(images.gt_cells.sum()), "detected": int(images.detected_cells.sum()), "matched": int(images.matched_cells.sum())})


if __name__ == "__main__":
    main()
