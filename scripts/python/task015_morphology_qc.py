#!/usr/bin/env python3
"""Full outer-fold pixel-head QC on largest-area GHIST validation patches.

Only the selected outer-validation nuclei receive class targets; unlabeled or
non-selected nuclei are ignored. True background is mask==0 in those patches.
This diagnostic is separate from the primary cell-level endpoint.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from task015_evaluate_ghist_celltype import ROOT, OverlapPatchDataset, overlap_collate
from task015_ghist_celltype_model import GHISTCellType


def run_one(variant, fold, device):
    result = json.loads((ROOT / "metrics" / f"{variant}_fold{fold}_result.json").read_text())
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    model = GHISTCellType().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    folds = pd.read_csv("/data/lf_data/result/task010_representation_benchmark/metrics/fold_manifest.csv")
    val = folds[(folds.fold == fold) & folds.role.eq("val")]
    label = dict(zip(val.cell_id, val.class_id))
    selected = pd.read_csv(ROOT / "metrics/overlap_largest_area_manifest.csv.gz")
    selected = selected[selected.cell_id.isin(label)].copy()
    if len(selected) != len(val):
        raise RuntimeError("Outer validation contour selection incomplete")
    ds = OverlapPatchDataset(selected, checkpoint["mean"], checkpoint["std"])
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=overlap_collate)
    confusion = np.zeros((8, 8), dtype=np.int64)
    supervised = 0
    with torch.inference_mode():
        for step, (images, masks, selected_batch, _) in enumerate(loader, start=1):
            output = model(images.to(device), masks.to(device))
            pred = output["pixel_logits"].argmax(dim=1).cpu().numpy().astype(np.uint8)
            m = masks.numpy()
            for j, chosen in enumerate(selected_batch):
                target = np.full(m[j].shape, -1, dtype=np.int16)
                target[m[j] == 0] = 0
                for instance_id, cell_id in chosen.items():
                    pix = m[j] == int(instance_id)
                    if not pix.any():
                        raise RuntimeError("Selected nucleus missing from raster mask")
                    target[pix] = int(label[cell_id]) + 1
                    supervised += 1
                valid = target >= 0
                hist = np.bincount(target[valid].astype(np.int32) * 8 + pred[j][valid].astype(np.int32),
                                   minlength=64).reshape(8, 8)
                confusion += hist
            if step % 100 == 0:
                print(f"morphology variant={variant} fold={fold} patches={min(step*8,len(ds))}/{len(ds)}", flush=True)
    if supervised != len(val):
        raise RuntimeError(f"Expected {len(val)} unique supervised cells, got {supervised}")
    np.save(ROOT / "qc" / f"morphology_confusion_{variant}_fold{fold}.npy", confusion)
    foreground_intersection = int(confusion[1:, 1:].sum())
    foreground_true = int(confusion[1:, :].sum())
    foreground_pred = int(confusion[:, 1:].sum())
    fg_iou = foreground_intersection / (foreground_true + foreground_pred - foreground_intersection)
    fg_dice = 2 * foreground_intersection / (foreground_true + foreground_pred)
    nucleus_correct = int(np.trace(confusion[1:, 1:]))
    nucleus_acc = nucleus_correct / foreground_true
    row = {"variant": variant, "fold": fold, "n_val_cells": supervised,
           "n_eval_patches": len(ds), "n_supervised_nucleus_pixels": foreground_true,
           "n_true_background_pixels": int(confusion[0].sum()),
           "foreground_dice": fg_dice, "foreground_iou": fg_iou,
           "supervised_nucleus_pixel_accuracy": nucleus_acc}
    for k in range(1, 8):
        tp = int(confusion[k, k])
        denominator = int(confusion[k, :].sum() + confusion[:, k].sum() - tp)
        row[f"class_{k-1}_pixel_iou"] = tp / denominator if denominator else float("nan")
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    rows = []
    for fold in range(5):
        for variant in ("GHIST_CT_OFFICIAL", "GHIST_CT_BALANCED"):
            rows.append(run_one(variant, fold, device))
            pd.DataFrame(rows).to_csv(ROOT / "metrics/ghist_morphology_qc.csv", index=False)
    if len(rows) != 10 or not np.isfinite(pd.DataFrame(rows).select_dtypes(include="number").to_numpy()).all():
        raise RuntimeError("Incomplete or nonfinite morphology QC")
    print("MORPHOLOGY_QC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
