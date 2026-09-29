#!/usr/bin/env python3
"""Average five fold-specific Midnight probabilities for native 57x57 crops."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from task012_inference_common import load_model, probabilities, samples, save_predictions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="57x57 crop, crop directory, or CSV with crop_path")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--ensemble-dir", type=Path, default=Path("/data/lf_data/result/final_model/ensemble"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if args.batch_size < 1: raise ValueError("batch-size must be positive")
    entries = samples(args.input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = np.zeros((len(entries), 7), dtype=np.float64)
    for fold in range(5):
        path = args.ensemble_dir / f"fold{fold}_midnight_fov12_finalblock_7class.pth"
        model, head = load_model(path, device)
        total += probabilities(model, head, entries, device, args.batch_size)
        del model, head
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    save_predictions(entries, (total / 5).astype(np.float32), args.output)
    print(f"Wrote {len(entries)} ensemble predictions to {args.output}")


if __name__ == "__main__": main()
