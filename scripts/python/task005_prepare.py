#!/usr/bin/env python3
"""Prepare the Task 005 official backbone/stain benchmark on the remote host."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
EXPECTED = {
    "ViT256": {"embedding_dim": 384, "patterns": ["vit256", "vit_256"]},
    "SAM-H": {"embedding_dim": 1280, "patterns": ["sam-h", "sam_h", "samh"]},
    "UNI": {"embedding_dim": 1024, "patterns": ["uni"]},
    "Virchow": {"embedding_dim": 1280, "patterns": ["virchow"]},
    "Virchow2": {"embedding_dim": 1280, "patterns": ["virchow2"]},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_name(path: Path) -> str:
    name = path.name.lower()
    if "virchow2" in name:
        return "Virchow2"
    if "virchow" in name:
        return "Virchow"
    if "sam-h" in name or "sam_h" in name or "samh" in name:
        return "SAM-H"
    if "uni" in name:
        return "UNI"
    if "vit256" in name or "vit_256" in name:
        return "ViT256"
    return "UNKNOWN"


def official_probe(path: Path, result: Path) -> tuple[bool, str, str, int]:
    """Load through the installed ExperimentCellVitClassifier architecture path."""
    from cellvit.training.experiments.experiment_cell_classifier import ExperimentCellVitClassifier
    from cellvit.utils.tools import unflatten_dict

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = unflatten_dict(checkpoint["config"], ".")
        experiment = ExperimentCellVitClassifier(
            {
                "random_seed": 42,
                "logging": {
                    "log_dir": str(result / "logs" / "compatibility_probe"),
                    "wandb_dir": str(result / "logs" / "wandb"),
                }
            }
        )
        model = experiment._get_cellvit_architecture(checkpoint["arch"], config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return True, "", str(checkpoint["arch"]), int(config["model"].get("embed_dim", EXPECTED.get(config["model"].get("backbone"), {}).get("embedding_dim", 0)))
    except Exception as exc:  # inventory must preserve the failure reason
        return False, f"{type(exc).__name__}: {exc}", "", 0


def make_dataset_links(source: Path, work: Path) -> None:
    for split in ("train", "test"):
        split_root = work / split
        split_root.mkdir(parents=True, exist_ok=True)
        for sub in ("images", "labels"):
            target = split_root / sub
            if not target.exists():
                target.symlink_to(source / split / sub, target_is_directory=True)
    (work / "cache").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task005_backbone_domain"))
    parser.add_argument("--source", type=Path, default=Path("/data/lf_data/xenium_data/CellViT_dataset"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/data/lf_data/CellViT-plus-plus/checkpoints"))
    parser.add_argument("--split-root", type=Path, default=Path("/data/lf_data/result/splits"))
    args = parser.parse_args()
    result = args.result.resolve()
    result.mkdir(parents=True, exist_ok=True)
    for sub in ("config", "code", "work", "qc", "metrics", "figures", "figure_data", "logs", "models", "cache"):
        (result / sub).mkdir(parents=True, exist_ok=True)

    inventory = []
    checkpoints = sorted(args.checkpoint_root.rglob("*.pth"))
    for path in checkpoints:
        backbone = infer_name(path)
        compatible, error, arch, dim = official_probe(path, result)
        expected = EXPECTED.get(backbone, {})
        inventory.append(
            {
                "backbone_name": backbone,
                "checkpoint_path": str(path),
                "sha256": sha256(path),
                "file_size_bytes": path.stat().st_size,
                "inferred_architecture": arch,
                "embedding_dim": dim or expected.get("embedding_dim", ""),
                "compatible_input_resolution": 256,
                "official_load_success": compatible,
                "official_load_error": error,
            }
        )
    available = [row for row in inventory if row["official_load_success"] and row["backbone_name"] != "UNKNOWN"]
    with (result / "metrics" / "backbone_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]) if inventory else ["backbone_name"])
        writer.writeheader()
        writer.writerows(inventory)

    if not available:
        raise RuntimeError("No official-compatible checkpoint was found")
    label_map = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    (result / "config" / "label_map.yaml").write_text(yaml.safe_dump(label_map, sort_keys=False))
    conditions = []
    for row in available:
        backbone = row["backbone_name"]
        condition_specs = [("RAW", False), ("STAIN_NORMALIZED", True)]
        for stain_name, normalized in condition_specs:
            condition = f"{backbone}_{stain_name}"
            work = result / "work" / condition / "dataset"
            make_dataset_links(args.source, work)
            conditions.append({"condition": condition, "backbone": backbone, "stain_condition": stain_name, "normalize_stains": normalized, "checkpoint_path": row["checkpoint_path"], "embedding_dim": row["embedding_dim"]})
            for fold in range(5):
                config = {
                    "gpu": 0,
                    "random_seed": 42,
                    "cellvit_path": row["checkpoint_path"],
                    "data": {
                        "dataset": "DetectionDataset",
                        "dataset_path": str(work),
                        "num_classes": 7,
                        "label_map": label_map,
                        "label_map_path": str(result / "config" / "label_map.yaml"),
                        "input_shape": 256,
                        "train_filelist": str(args.split_root / f"fold_{fold}" / "train.csv"),
                        "val_filelist": str(args.split_root / f"fold_{fold}" / "val.csv"),
                        "normalize_stains_train": normalized,
                        "normalize_stains_val": normalized,
                    },
                    "model": {"hidden_dim": 100},
                    "training": {
                        "optimizer": "AdamW",
                        "optimizer_hyperparameter": {"lr": 0.001, "weight_decay": 0.0001},
                        "scheduler": {"scheduler_type": "cosine", "eta_min": 1e-6},
                        "drop_rate": 0.0,
                        "batch_size": 2048,
                        "epochs": 12,
                        "mixed_precision": True,
                        "early_stopping_patience": 5,
                        "eval_every": 1,
                        "cache_cell_dataset": True,
                    },
                    "transformations": {"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
                    "logging": {
                        "log_dir": str(result / "runs" / condition / f"fold_{fold}"),
                        "wandb_dir": str(result / "logs" / "wandb"),
                        "mode": "disabled",
                        "project": "task005_backbone_domain",
                        "level": "INFO",
                        "log_comment": f"task005_{condition}_fold_{fold}",
                        "notes": "Task 005 fixed official recipe; backbone and stain condition are the only primary factors.",
                    },
                }
                (result / "config" / f"{condition}_fold_{fold}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    with (result / "config" / "conditions.json").open("w") as handle:
        json.dump(conditions, handle, indent=2)
    (result / "logs" / "commands_cv.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd /data/lf_data/CellViT-plus-plus\n" + "".join(
            f'python3 ./cellvit/train_cell_classifier_head.py --config /data/lf_data/result/task005_backbone_domain/config/{c["condition"]}_fold_{fold}.yaml\n'
            for c in conditions for fold in range(5)
        )
    )
    print(json.dumps({"inventory": inventory, "conditions": conditions}, indent=2))


if __name__ == "__main__":
    main()
