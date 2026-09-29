#!/usr/bin/env python3
"""Shared Task012 crop inference utilities for trusted local checkpoints."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
OUTPUT_NAMES = ["p_Endothelial", "p_Mesenchymal", "p_Myeloid", "p_Neutrophil", "p_Plasma", "p_T_and_B", "p_Tumor"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def samples(path: Path):
    path = path.expanduser().resolve()
    if path.is_dir():
        entries = [(p.stem, p) for p in sorted(path.iterdir()) if p.suffix.lower() in EXTENSIONS]
    elif path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or "crop_path" not in rows[0]:
            raise ValueError("Manifest must have a crop_path column and at least one row")
        entries = []
        for row in rows:
            crop = Path(row["crop_path"])
            if not crop.is_absolute(): crop = path.parent / crop
            entries.append((row.get("sample_id") or row.get("cell_id") or crop.stem, crop.resolve()))
    elif path.suffix.lower() in EXTENSIONS:
        entries = [(path.stem, path)]
    else:
        raise ValueError(f"Expected crop image, directory, or CSV manifest: {path}")
    if not entries:
        raise ValueError("No crop images found")
    names = [n for n, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate sample identifiers")
    for _, crop in entries:
        if not crop.is_file(): raise FileNotFoundError(crop)
    return entries


def load_model(path: Path, device: torch.device):
    from transformers import AutoModel

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["class_names"] != CLASS_NAMES or checkpoint["feature_dim"] != 3072:
        raise ValueError(f"Unexpected checkpoint class mapping or feature dimension: {path}")
    if checkpoint["native_crop_px"] != 57 or checkpoint["input_size"] != 224:
        raise ValueError(f"Unexpected checkpoint crop geometry: {path}")
    model = AutoModel.from_pretrained(checkpoint["source_model_path"], local_files_only=True)
    model.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    head = torch.nn.Linear(3072, 7)
    head.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    model.to(device).eval(); head.to(device).eval()
    return model, head


def probabilities(model, head, entries, device: torch.device, batch_size: int = 32):
    output = []
    with torch.inference_mode():
        for start in range(0, len(entries), batch_size):
            batch = entries[start:start + batch_size]
            arrays = []
            for _, path in batch:
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                    if rgb.size != (57, 57):
                        raise ValueError(f"Expected a 57 x 57 native H&E crop: {path}: {rgb.size}")
                    arrays.append(np.array(rgb, dtype=np.uint8, copy=True))
            x = torch.from_numpy(np.stack(arrays)).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
            x = (x - 0.5) / 0.5
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = model(x).last_hidden_state
                z = torch.cat([hidden[:, 0, :], hidden[:, 1:, :].mean(dim=1)], dim=1)
                logits = head(z)
            p = torch.softmax(logits, dim=1).float().cpu().numpy()
            if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=2e-3, rtol=0):
                raise RuntimeError("Non-finite or unnormalized model probabilities")
            # Task011's fp16 softmax has small rounding error in the row sum.
            # Normalize exported inference probabilities without changing argmax.
            p /= p.sum(axis=1, keepdims=True)
            output.append(p)
    return np.concatenate(output, axis=0)


def save_predictions(entries, probs, output: Path):
    import pandas as pd

    ids = [ident for ident, _ in entries]
    pred = probs.argmax(axis=1)
    out = pd.DataFrame({"sample_id": ids, "predicted_class_id": pred,
                        "predicted_class_name": [CLASS_NAMES[i] for i in pred]})
    for i, name in enumerate(OUTPUT_NAMES): out[name] = probs[:, i]
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out
