#!/usr/bin/env python3
"""Infer seven H&E cell classes from native 57x57 crops with the final model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from task012_inference_common import load_model, probabilities, samples, save_predictions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="57x57 crop, crop directory, or CSV with crop_path")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if args.batch_size < 1: raise ValueError("batch-size must be positive")
    entries = samples(args.input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head = load_model(args.checkpoint, device)
    probs = probabilities(model, head, entries, device, args.batch_size)
    save_predictions(entries, probs, args.output)
    print(f"Wrote {len(entries)} predictions to {args.output}")


if __name__ == "__main__": main()
