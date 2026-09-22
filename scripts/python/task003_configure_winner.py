#!/usr/bin/env python3
"""Create official grouped-fold and all-training configs from one official winner."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task003_official"))
    parser.add_argument("--winner", default="fold_0_official_baseline.yaml")
    args = parser.parse_args()
    result = args.result.resolve()
    winner_path = result / "config" / args.winner
    winner = yaml.safe_load(winner_path.read_text())
    derivative = result / "work" / "CellViT_dataset_official"

    generated = []
    for fold in range(1, 5):
        cfg = copy.deepcopy(winner)
        name = f"fold_{fold}_official_winner"
        cfg["data"]["train_filelist"] = str(derivative / "splits" / f"fold_{fold}" / "train.csv")
        cfg["data"]["val_filelist"] = str(derivative / "splits" / f"fold_{fold}" / "val.csv")
        cfg["logging"]["log_dir"] = str(result / "runs" / name)
        cfg["logging"]["log_comment"] = name
        cfg["logging"]["notes"] = "Task 003 strict official winner configuration applied to leakage-safe grouped fold."
        path = result / "config" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        generated.append(path)

    final_cfg = copy.deepcopy(winner)
    final_name = "final_official_all_train"
    all_train = derivative / "splits" / "all_train.csv"
    final_cfg["data"]["train_filelist"] = str(all_train)
    final_cfg["data"]["val_filelist"] = str(all_train)
    final_cfg["logging"]["log_dir"] = str(result / "runs" / final_name)
    final_cfg["logging"]["log_comment"] = final_name
    final_cfg["logging"]["notes"] = "Task 003 final strict official model; all training images; fixed epoch count from grouped CV."
    final_cfg["training"]["early_stopping_patience"] = None
    final_path = result / "config" / f"{final_name}.yaml"
    final_path.write_text(yaml.safe_dump(final_cfg, sort_keys=False))
    generated.append(final_path)

    commands = result / "logs" / "commands.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "conda activate cellvit_env",
        "cd /data/lf_data/CellViT-plus-plus",
        f"python3 ./cellvit/train_cell_classifier_head.py --config {result}/config/{args.winner}",
        f"python3 ./cellvit/train_cell_classifier_head.py --config {result}/config/fold_0_official_task001_like.yaml",
        f"python3 ./cellvit/train_cell_classifier_head.py --config {result}/config/fold_0_official_official_wider.yaml",
    ]
    for fold in range(1, 5):
        lines.append(f"python3 ./cellvit/train_cell_classifier_head.py --config {result}/config/fold_{fold}_official_winner.yaml")
    lines.append(f"python3 ./cellvit/train_cell_classifier_head.py --config {result}/config/{final_name}.yaml")
    lines.append("python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py --logdir <FINAL_OFFICIAL_LOGDIR> --dataset_path /data/lf_data/xenium_data/CellViT_dataset --cellvit_path /data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth --input_shape 256 256")
    commands.write_text("\n".join(lines) + "\n")
    commands.chmod(0o755)
    print("\n".join(str(path) for path in generated))


if __name__ == "__main__":
    main()
