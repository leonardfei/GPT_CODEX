#!/usr/bin/env python3
"""Prepare and validate the strict official CellViT++ Task 003 layout.

This helper does not train a model. Training remains inside the official
``cellvit/train_cell_classifier_head.py`` entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


CLASS_NAMES = [
    "Endothelial",
    "Mesenchymal",
    "Myeloid",
    "Neutrophil",
    "Plasma cell",
    "T and B",
    "Tumor",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_names(path: Path) -> list[str]:
    values: list[str] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip() and row[0].strip().lower() != "image":
                values.append(Path(row[0].strip()).stem)
    return values


def image_hashes(paths: list[Path]) -> dict[str, str]:
    return {path.stem: sha256(path) for path in paths}


def label_counts(label_paths: list[Path], class_count: int) -> tuple[Counter, int, list[str]]:
    counts: Counter = Counter()
    rows = 0
    invalid: list[str] = []
    for path in label_paths:
        with path.open(newline="") as handle:
            for line_no, row in enumerate(csv.reader(handle), start=1):
                if not row:
                    continue
                if len(row) < 3:
                    invalid.append(f"{path.name}:{line_no}: fewer than 3 columns")
                    continue
                try:
                    int(row[0])
                    int(row[1])
                    label = int(row[2])
                except ValueError:
                    invalid.append(f"{path.name}:{line_no}: non-integer annotation")
                    continue
                rows += 1
                counts[label] += 1
                if label < 0 or label >= class_count:
                    invalid.append(f"{path.name}:{line_no}: class={label}")
    return counts, rows, invalid


def read_metadata(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "train":
                mapping[row["image"].strip()] = row["batch"].strip()
    return mapping


def ensure_link(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        raise RuntimeError(f"Refusing to replace existing derivative path: {link}")
    link.symlink_to(target, target_is_directory=True)


def config(
    result: Path,
    dataset: Path,
    cellvit: Path,
    name: str,
    train_filelist: Path,
    val_filelist: Path,
    hidden_dim: int,
    drop_rate: float,
    lr: float,
    weight_decay: float,
    epochs: int = 12,
    early_stopping_patience: int | None = 5,
) -> dict:
    return {
        "gpu": 0,
        "random_seed": 42,
        "cellvit_path": str(cellvit),
        "data": {
            "dataset": "DetectionDataset",
            "dataset_path": str(dataset),
            "num_classes": 7,
            "label_map": {idx: label for idx, label in enumerate(CLASS_NAMES)},
            "label_map_path": str(dataset / "label_map.yaml"),
            "input_shape": 256,
            "train_filelist": str(train_filelist),
            "val_filelist": str(val_filelist),
            "normalize_stains_train": False,
            "normalize_stains_val": False,
        },
        "model": {"hidden_dim": hidden_dim},
        "training": {
            "optimizer": "AdamW",
            "optimizer_hyperparameter": {"lr": lr, "weight_decay": weight_decay},
            "scheduler": {"scheduler_type": "cosine", "eta_min": 1.0e-6},
            "drop_rate": drop_rate,
            "batch_size": 2048,
            "epochs": epochs,
            "mixed_precision": True,
            "early_stopping_patience": early_stopping_patience,
            "eval_every": 1,
            "cache_cell_dataset": True,
        },
        "transformations": {
            "normalize": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            }
        },
        "logging": {
            "log_dir": str(result / "runs" / name),
            "wandb_dir": str(result / "logs" / "wandb"),
            "mode": "disabled",
            "project": "task003_official",
            "level": "INFO",
            "log_comment": name,
            "notes": "Task 003 strict official CellViT++ training; WandB disabled because no API key is configured.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/data/lf_data/xenium_data/CellViT_dataset"))
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task003_official"))
    parser.add_argument("--prior-result", type=Path, default=Path("/data/lf_data/result"))
    parser.add_argument("--repo", type=Path, default=Path("/data/lf_data/CellViT-plus-plus"))
    args = parser.parse_args()

    source = args.source.resolve()
    result = args.result.resolve()
    prior = args.prior_result.resolve()
    repo = args.repo.resolve()
    result.mkdir(parents=True, exist_ok=True)
    for name in ["config", "qc", "metrics", "figure_data", "figures", "code", "logs", "runs", "work"]:
        (result / name).mkdir(parents=True, exist_ok=True)

    production = prior / "model_best.pth"
    protected = prior / "task001_model_best_baseline.pth"
    if not protected.exists():
        shutil.copy2(production, protected)
    if sha256(production) != sha256(protected):
        raise RuntimeError("Task 001 production model and protected baseline differ")

    label_map_path = source / "label_map.yaml"
    label_map = yaml.safe_load(label_map_path.read_text())
    observed_names = [name for name, _ in sorted(label_map.items(), key=lambda item: int(item[1]))]
    if observed_names != CLASS_NAMES:
        raise RuntimeError(f"Unexpected label map: {observed_names}")

    split_dirs = [prior / "splits" / f"fold_{idx}" for idx in range(5)]
    if not all(path.is_dir() for path in split_dirs):
        raise RuntimeError("Leakage-safe grouped Task 001 splits are missing")

    train_images = sorted((source / "train" / "images").glob("*"))
    test_images = sorted((source / "test" / "images").glob("*"))
    train_labels = sorted((source / "train" / "labels").glob("*.csv"))
    test_labels = sorted((source / "test" / "labels").glob("*.csv"))
    train_stems = {path.stem for path in train_images}
    test_stems = {path.stem for path in test_images}
    train_label_stems = {path.stem for path in train_labels}
    test_label_stems = {path.stem for path in test_labels}
    if train_stems != train_label_stems or test_stems != test_label_stems:
        raise RuntimeError("Image-label pairing mismatch")

    train_shapes = Counter(Image.open(path).size for path in train_images)
    test_shapes = Counter(Image.open(path).size for path in test_images)
    train_counts, train_rows, train_invalid = label_counts(train_labels, 7)
    test_counts, test_rows, test_invalid = label_counts(test_labels, 7)
    if train_invalid or test_invalid:
        raise RuntimeError(f"Invalid labels: {train_invalid[:3]} {test_invalid[:3]}")

    train_hash = image_hashes(train_images)
    test_hash = image_hashes(test_images)
    duplicate_hashes = sorted(set(train_hash.values()) & set(test_hash.values()))

    metadata_path = prior / "data" / "patch_metadata.csv"
    batches = read_metadata(metadata_path)
    split_qc = []
    all_train = sorted(train_stems)
    for fold in range(5):
        fold_dir = split_dirs[fold]
        train_names = read_names(fold_dir / "train.csv")
        val_names = read_names(fold_dir / "val.csv")
        train_set, val_set = set(train_names), set(val_names)
        if train_set | val_set != train_stems or train_set & val_set:
            raise RuntimeError(f"Invalid fold {fold} image partition")
        train_batches = sorted({batches[name] for name in train_set if name in batches})
        val_batches = sorted({batches[name] for name in val_set if name in batches})
        split_qc.append(
            {
                "fold": fold,
                "train_images": len(train_set),
                "val_images": len(val_set),
                "train_batches": train_batches,
                "val_batches": val_batches,
                "batch_overlap": sorted(set(train_batches) & set(val_batches)),
            }
        )
        if set(train_batches) & set(val_batches):
            raise RuntimeError(f"Batch leakage detected in fold {fold}")

    derivative = result / "work" / "CellViT_dataset_official"
    derivative.mkdir(parents=True, exist_ok=True)
    ensure_link(derivative / "train", source / "train")
    ensure_link(derivative / "test", source / "test")
    (derivative / "splits").mkdir(exist_ok=True)
    (derivative / "train_configs").mkdir(exist_ok=True)
    shutil.copy2(label_map_path, derivative / "label_map.yaml")
    for fold in range(5):
        out_fold = derivative / "splits" / f"fold_{fold}"
        out_fold.mkdir(exist_ok=True)
        for filename in ["train.csv", "val.csv"]:
            shutil.copy2(split_dirs[fold] / filename, out_fold / filename)
    all_train_path = derivative / "splits" / "all_train.csv"
    with all_train_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        writer.writerows([[name] for name in all_train])

    source_files = [
        repo / "README.md",
        repo / "cellvit/train_cell_classifier_head.py",
        repo / "cellvit/training/experiments/experiment_cell_classifier.py",
        repo / "cellvit/training/trainer/trainer_cell_classifier.py",
        repo / "cellvit/models/classifier/linear_classifier.py",
        repo / "cellvit/training/evaluate/inference_cellvit_experiment_detection.py",
    ]
    with (result / "config" / "official_source_hashes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256"])
        writer.writerows([[str(path), sha256(path)] for path in source_files])

    configs: dict[str, dict] = {}
    candidates = {
        "baseline": (100, 0.0, 1.0e-3, 1.0e-4),
        "task001_like": (128, 0.2, 3.0e-4, 1.0e-4),
        "official_wider": (256, 0.2, 5.0e-4, 1.0e-4),
    }
    for name, (hidden, drop, lr, wd) in candidates.items():
        config_name = f"fold_0_official_{name}"
        cfg = config(
            result,
            derivative,
            repo / "checkpoints/CellViT-SAM-H-x40-AMP.pth",
            config_name,
            derivative / "splits/fold_0/train.csv",
            derivative / "splits/fold_0/val.csv",
            hidden,
            drop,
            lr,
            wd,
        )
        configs[config_name] = cfg
        (result / "config" / f"{config_name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    final_name = "final_official_all_train"
    final_cfg = config(
        result,
        derivative,
        repo / "checkpoints/CellViT-SAM-H-x40-AMP.pth",
        final_name,
        all_train_path,
        all_train_path,
        128,
        0.2,
        3.0e-4,
        1.0e-4,
        epochs=12,
        early_stopping_patience=None,
    )
    configs[final_name] = final_cfg
    (result / "config" / f"{final_name}.yaml").write_text(yaml.safe_dump(final_cfg, sort_keys=False))

    qc = {
        "source": str(source),
        "label_map_path": str(label_map_path),
        "label_map": label_map,
        "expected": {"train_images": 5002, "test_images": 6107, "classes": 7},
        "observed": {
            "train_images": len(train_images),
            "test_images": len(test_images),
            "train_labels": len(train_labels),
            "test_labels": len(test_labels),
            "train_annotation_rows": train_rows,
            "test_annotation_rows": test_rows,
            "train_class_counts": dict(sorted(train_counts.items())),
            "test_class_counts": dict(sorted(test_counts.items())),
            "train_shapes": {str(k): v for k, v in train_shapes.items()},
            "test_shapes": {str(k): v for k, v in test_shapes.items()},
            "train_test_duplicate_image_hashes": len(duplicate_hashes),
            "missing_batch_metadata": sorted(train_stems - set(batches)),
        },
        "folds": split_qc,
        "protected_baseline_sha256": sha256(protected),
    }
    (result / "qc" / "dataset_qc.json").write_text(json.dumps(qc, indent=2))
    with (result / "qc" / "split_qc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "train_images", "val_images", "train_batches", "val_batches", "batch_overlap"])
        writer.writeheader()
        for row in split_qc:
            writer.writerow({**row, "train_batches": ";".join(row["train_batches"]), "val_batches": ";".join(row["val_batches"]), "batch_overlap": ";".join(row["batch_overlap"])})

    commands = result / "logs" / "commands.sh"
    command_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "conda activate cellvit_env", "cd /data/lf_data/CellViT-plus-plus"]
    for config_name in configs:
        command_lines.append(f"python3 ./cellvit/train_cell_classifier_head.py --config {result}/config/{config_name}.yaml")
    commands.write_text("\n".join(command_lines) + "\n")
    commands.chmod(0o755)
    print(json.dumps({"status": "prepared", "result": str(result), "qc": qc, "configs": sorted(configs)}, indent=2))


if __name__ == "__main__":
    main()
