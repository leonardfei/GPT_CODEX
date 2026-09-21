#!/usr/bin/env python3
"""Write a configuration accepted by CellViT++ train_cell_classifier_head.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--cellvit-path", required=True)
    parser.add_argument("--train-filelist", required=True)
    parser.add_argument("--val-filelist", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--wandb-dir", required=True)
    parser.add_argument("--log-comment", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cache-cell-dataset", action="store_true")
    args = parser.parse_args()

    config = {
        "gpu": args.gpu,
        "random_seed": 42,
        "cellvit_path": args.cellvit_path,
        "data": {
            "dataset": "DetectionDataset",
            "dataset_path": args.dataset_path,
            "num_classes": 7,
            "input_shape": 256,
            "train_filelist": args.train_filelist,
            "val_filelist": args.val_filelist,
            "normalize_stains_train": False,
            "normalize_stains_val": False,
        },
        "model": {
            "hidden_dim": args.hidden_dim,
        },
        "training": {
            "optimizer": args.optimizer,
            "optimizer_hyperparameter": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {
                "scheduler_type": "cosine",
                "eta_min": 1e-6,
            },
            "drop_rate": args.drop_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "eval_every": 1,
            "mixed_precision": True,
            "cache_cell_dataset": args.cache_cell_dataset,
            "early_stopping_patience": args.patience,
        },
        "transformations": {
            "RandomRotate90": {"p": 0.5},
            "HorizontalFlip": {"p": 0.5},
            "VerticalFlip": {"p": 0.5},
        },
        "logging": {
            "project": "task001_cellvit_head",
            "notes": "Task 001 local/offline run; no W&B authentication required.",
            "mode": "disabled",
            "level": "INFO",
            "log_dir": args.log_dir,
            "wandb_dir": args.wandb_dir,
            "log_comment": args.log_comment,
            "tags": ["task001", "classification_head_only"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()
