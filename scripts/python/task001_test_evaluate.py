#!/usr/bin/env python3
"""Save per-cell test predictions from the native CellViT++ inference path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.evaluate.inference_cellvit_experiment_detection import (
    CellViTInfExpDetection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--cellvit-path", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    metrics_dir = result_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    experiment = CellViTInfExpDetection(
        logdir=args.logdir,
        cellvit_path=args.cellvit_path,
        dataset_path=args.dataset_path,
        input_shape=[256, 256],
        gpu=args.gpu,
    )
    postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
    loader = DataLoader(
        experiment.inference_dataset,
        batch_size=4,
        num_workers=8,
        shuffle=False,
        collate_fn=experiment.inference_dataset.collate_batch,
    )

    extracted_cells_cleaned = []
    detection_scores = {"F1": [], "Prec": [], "Rec": []}
    with torch.no_grad():
        for images, cell_gt_batch, types_batch, image_names in loader:
            cleaned, _, _, f1s, recs, precs = experiment._get_cellvit_result(
                images=images,
                cell_gt_batch=cell_gt_batch,
                types_batch=types_batch,
                image_names=image_names,
                postprocessor=postprocessor,
            )
            extracted_cells_cleaned.extend(cleaned)
            detection_scores["F1"].extend(f1s)
            detection_scores["Prec"].extend(precs)
            detection_scores["Rec"].extend(recs)

    output = experiment._get_classifier_result(extracted_cells_cleaned)
    np.save(metrics_dir / "test_predictions.npy", output["predictions"].numpy())
    np.save(metrics_dir / "test_probabilities.npy", output["probabilities"].numpy())
    np.save(metrics_dir / "test_ground_truth.npy", output["gt"].numpy())
    (metrics_dir / "test_metadata.json").write_text(
        json.dumps(output["metadata"], indent=2), encoding="utf-8"
    )
    summary = {
        "n_test_cells_after_detection_pairing": int(len(output["gt"])),
        "detection_f1_mean": float(np.mean(detection_scores["F1"])),
        "detection_precision_mean": float(np.mean(detection_scores["Prec"])),
        "detection_recall_mean": float(np.mean(detection_scores["Rec"])),
        "native_inference_result_dir": str(experiment.test_result_dir),
    }
    (metrics_dir / "test_extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
