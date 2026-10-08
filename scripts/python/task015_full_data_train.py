#!/usr/bin/env python3
"""Train the gated Task015 GHIST model on all frozen canonical cells.

This does not replace any Task012 or production checkpoint.  The model is
classification-only and the epoch count is the median of inner-selected CV
epochs, never chosen with outer-validation performance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task015_ghist_celltype_model import GHISTCellType
from task015_train_ghist_celltype import (CANON, ROOT, SEED, load_contours,
                                          make_loader, seed_all, train_one_epoch,
                                          training_normalization)


def main():
    decision = json.loads((ROOT / "metrics/decision_summary.json").read_text())
    variant = decision["best_ghist_variant"]
    if not decision["full_data_export_eligible"] or variant not in ("GHIST_CT_OFFICIAL", "GHIST_CT_BALANCED"):
        raise RuntimeError("Full-data model has not passed the predefined CV gate")
    selected = decision["variants"][variant]["selected_epochs"]
    if len(selected) != 5 or any(not 1 <= int(x) <= 50 for x in selected):
        raise RuntimeError("Five selected epochs are required")
    epochs = int(np.median(selected))
    seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    canonical = pd.read_csv(CANON)
    if len(canonical) != 96044 or not canonical.cell_id.is_unique or set(canonical.class_id) != set(range(7)):
        raise RuntimeError("Canonical cell cohort is invalid")
    active = set(canonical.cell_id)
    labels = dict(zip(canonical.cell_id, canonical.class_id))
    patches = pd.read_csv(ROOT / "metrics/patch_manifest.csv.gz")
    patches = patches[patches.n_canonical > 0].copy()
    if patches.empty or patches.image.duplicated().any():
        raise RuntimeError("Invalid training patches")
    contours = load_contours()
    work = ROOT / "work/final_full_data"
    work.mkdir(parents=True, exist_ok=True)
    mean, std = training_normalization(patches, work / "full_data_rgb.json")
    loader = make_loader(patches, contours, labels, active, mean, std, shuffle=True)
    counts = canonical.class_id.value_counts().reindex(range(7), fill_value=0)
    if (counts == 0).any():
        raise RuntimeError("Missing canonical class")
    if variant == "GHIST_CT_BALANCED":
        weight = len(canonical) / (7 * counts.to_numpy(dtype=float))
        cell_weights = torch.tensor(weight, dtype=torch.float32, device=device)
        pixel_weights = torch.tensor([1.0, *weight], dtype=torch.float32, device=device)
    else:
        cell_weights = pixel_weights = None
    model = GHISTCellType().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        row = train_one_epoch(model, loader, optimizer, device, pixel_weights, cell_weights, epoch)
        history.append(row)
        (work / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"phase": "final_full_data", "variant": variant, **row}), flush=True)
    final = ROOT / "models/ghist_hcc_7class_final.pth"
    torch.save({"model": model.state_dict(), "variant": variant, "epochs": epochs,
                "mean": mean, "std": std,
                "classes": ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"],
                "ghist_commit": "917456be305fc82e92293ea272812e79675e821c",
                "seed": SEED}, final)
    digest = hashlib.sha256()
    with final.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    cfg = {"status": "EXPORTED_NOT_PROMOTED", "checkpoint": str(final), "sha256": digest.hexdigest(),
           "variant": variant, "selected_epochs_by_fold": selected, "median_epochs": epochs,
           "training_cells": len(canonical), "training_patches": len(patches),
           "classes": ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"],
           "patch_pixels": 256, "native_mpp": 0.2125, "inference_overlap_pixels": 30,
           "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
           "source_ghist_commit": "917456be305fc82e92293ea272812e79675e821c",
           "seed": SEED, "command": "/data/lf_data/task010_env/bin/python code/task015_full_data_train.py",
           "production_model_replaced": False}
    (ROOT / "config/ghist_hcc_7class_final.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps(cfg, indent=2), flush=True)


if __name__ == "__main__":
    main()
