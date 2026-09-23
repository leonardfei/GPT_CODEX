#!/usr/bin/env python3
"""Compute detection, conditional-classification and end-to-end metrics."""

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
from cellvit.models.classifier.linear_classifier import LinearClassifier
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.training.evaluate.inference_cellvit_experiment_detection import CellViTInfExpDetection
from cellvit.training.utils.tools import pair_coordinates
from cellvit.utils.tools import unflatten_dict


CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]


class CombinedExperiment(CellViTInfExpDetection):
    def __init__(self, *args, filelist: Path, **kwargs):
        self._filelist = filelist
        super().__init__(*args, **kwargs)

    def _load_dataset(self, transforms, normalize_stains):
        dataset = DetectionDataset(
            dataset_path=self.dataset_path,
            split="train",
            filelist_path=self._filelist,
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()
        return dataset

    def _load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        run_conf = unflatten_dict(checkpoint["config"], ".")
        model = LinearClassifier(
            embed_dim=checkpoint["model_state_dict"]["fc1.weight"].shape[1],
            hidden_dim=run_conf["model"].get("hidden_dim", 100),
            num_classes=run_conf["data"]["num_classes"],
            drop_rate=0,
        )
        self.logger.info(model.load_state_dict(checkpoint["model_state_dict"]))
        model = model.to(self.device)
        model.eval()
        return model, run_conf


def evaluate_fold(result: Path, condition: str, fold: int, cellvit: Path, batch_size: int) -> pd.DataFrame:
    dataset = result / "work" / f"CellViT_dataset_v3_{condition}"
    filelist = dataset / "splits" / f"fold_{fold}" / "val.csv"
    run_root = result / "runs" / f"V3_{condition}" / f"fold_{fold}"
    run_dirs = sorted(run_root.glob("*/"))
    if len(run_dirs) != 1:
        raise RuntimeError(f"Expected one run directory: {run_root}")
    run_dir = run_dirs[0]
    exp = CombinedExperiment(
        logdir=run_dir,
        cellvit_path=cellvit,
        dataset_path=dataset,
        input_shape=[256, 256],
        normalize_stains=False,
        gpu=0,
        comment=f"task009_end_to_end_{condition}_{fold}",
        filelist=filelist,
    )
    loader = DataLoader(exp.inference_dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=exp.inference_dataset.collate_batch)
    postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
    rows = []
    with torch.no_grad():
        for images, gt_batch, types_batch, image_names in loader:
            matched, overall, pred_dicts, _, _, _ = exp._get_cellvit_result(
                images=images,
                cell_gt_batch=gt_batch,
                types_batch=types_batch,
                image_names=image_names,
                postprocessor=postprocessor,
            )
            matched_out = exp._get_classifier_result(matched) if matched else {"predictions": torch.empty(0, dtype=torch.long), "metadata": []}
            overall_out = exp._get_classifier_result(overall) if overall else {"predictions": torch.empty(0, dtype=torch.long), "metadata": []}
            matched_pred = matched_out["predictions"].numpy().astype(int) if len(matched) else np.empty(0, dtype=int)
            overall_pred = overall_out["predictions"].numpy().astype(int) if len(overall) else np.empty(0, dtype=int)
            matched_cursor = 0
            overall_cursor = 0
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
                gt_to_pred = {int(gt_i): int(pred_i) for gt_i, pred_i in paired}
                pred_to_gt = {int(pred_i): int(gt_i) for gt_i, pred_i in paired}
                image_matched_pred = matched_pred[matched_cursor:matched_cursor + len(paired)]
                matched_cursor += len(paired)
                image_overall_pred = overall_pred[overall_cursor:overall_cursor + len(pred_centroids)]
                overall_cursor += len(pred_centroids)
                for gt_i, class_id in enumerate(cell_types):
                    det = gt_i in gt_to_pred
                    pred_class = int(image_matched_pred[list(gt_to_pred).index(gt_i)]) if det else -1
                    rows.append({
                        "condition": f"V3_{condition}", "fold": fold, "image": image_name,
                        "gt_index": gt_i, "class_id": int(class_id), "class_name": CLASSES[int(class_id)],
                        "detected": int(det), "predicted_class_id": pred_class,
                        "correct_end_to_end": int(det and pred_class == int(class_id)),
                    })
                # Keep all predicted detections for end-to-end precision. An
                # unmatched prediction is a false positive for every class.
                for pred_i, pred_class in enumerate(image_overall_pred):
                    gt_i = pred_to_gt.get(pred_i, -1)
                    gt_class = int(cell_types[gt_i]) if gt_i >= 0 else -1
                    rows.append({
                        "condition": f"V3_{condition}", "fold": fold, "image": image_name,
                        "gt_index": -1, "class_id": gt_class, "class_name": CLASSES[gt_class] if gt_class >= 0 else "UNMATCHED_PREDICTION",
                        "detected": 1, "predicted_class_id": int(pred_class),
                        "correct_end_to_end": int(gt_class >= 0 and int(pred_class) == gt_class),
                        "prediction_record": True,
                    })
    out = pd.DataFrame(rows)
    out.to_csv(result / "metrics" / f"end_to_end_rows_{condition}_fold_{fold}.csv", index=False)
    return out


def summarize(result: Path, rows: list[pd.DataFrame]) -> None:
    all_rows = pd.concat(rows, ignore_index=True)
    summaries = []
    for (condition, fold), sub in all_rows.groupby(["condition", "fold"], sort=False):
        gt = sub[sub["gt_index"] >= 0].drop_duplicates(["image", "gt_index"])
        pred = sub[sub.get("prediction_record", pd.Series(False, index=sub.index)).fillna(False)]
        neut = gt[gt.class_id == 3]
        matched_neut = neut[neut.detected == 1]
        tp_neut = int((neut.correct_end_to_end == 1).sum())
        pred_neut = int((pred.predicted_class_id == 3).sum())
        pred_neut_tp = int(((pred.predicted_class_id == 3) & (pred.class_id == 3)).sum())
        summaries.append({
            "condition": condition, "fold": fold,
            "detection_recall": float(neut.detected.mean()) if len(neut) else np.nan,
            "conditional_recall": float((matched_neut.correct_end_to_end == 1).mean()) if len(matched_neut) else np.nan,
            "end_to_end_recall": float(tp_neut / len(neut)) if len(neut) else np.nan,
            "end_to_end_precision": float(pred_neut_tp / pred_neut) if pred_neut else np.nan,
            "neutrophil_gt_cells": int(len(neut)),
            "neutrophil_detected_cells": int(len(matched_neut)),
            "neutrophil_end_to_end_tp": tp_neut,
        })
    fold_df = pd.DataFrame(summaries)
    summary_rows = []
    for condition, sub in fold_df.groupby("condition", sort=False):
        row = {"condition": condition, "fold": "MEAN", "neutrophil_gt_cells": int(sub.neutrophil_gt_cells.sum()), "neutrophil_detected_cells": int(sub.neutrophil_detected_cells.sum()), "neutrophil_end_to_end_tp": int(sub.neutrophil_end_to_end_tp.sum())}
        for metric in ["detection_recall", "conditional_recall", "end_to_end_recall", "end_to_end_precision"]:
            row[metric] = float(sub[metric].mean())
            row[f"{metric}_sd"] = float(sub[metric].std(ddof=1))
        summary_rows.append(row)
    fold_df.to_csv(result / "metrics/neutrophil_end_to_end_by_fold.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(result / "metrics/neutrophil_end_to_end.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task009_v3_retraining"))
    parser.add_argument("--cellvit", type=Path, default=Path("/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    rows = []
    for condition in ("CORE", "EXTENDED"):
        for fold in range(5):
            rows.append(evaluate_fold(args.result, condition, fold, args.cellvit, args.batch_size))
    summarize(args.result, rows)


if __name__ == "__main__":
    main()
