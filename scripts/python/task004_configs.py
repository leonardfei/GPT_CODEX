#!/usr/bin/env python3
"""Generate fixed official CellViT++ configs for Task 004 tier CV."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


TIERS = ("ALL_MATCHED", "HIGH_CONFIDENCE", "ULTRA_HIGH_CONFIDENCE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task004_high_confidence"))
    parser.add_argument("--base", type=Path, default=Path("/data/lf_data/result/task003_official/config/fold_0_official_baseline.yaml"))
    args = parser.parse_args()
    result = args.result.resolve()
    base = yaml.safe_load(args.base.read_text())
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "cd /data/lf_data/CellViT-plus-plus"]
    for tier in TIERS:
        dataset = result / "work" / tier
        for fold in range(5):
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg["data"]["dataset_path"] = str(dataset)
            cfg["data"]["train_filelist"] = str(dataset / "splits" / f"fold_{fold}" / "train.csv")
            cfg["data"]["val_filelist"] = str(dataset / "splits" / f"fold_{fold}" / "val.csv")
            cfg["logging"]["log_dir"] = str(result / "runs" / tier / f"fold_{fold}")
            cfg["logging"]["log_comment"] = f"task004_{tier}_fold_{fold}"
            cfg["logging"]["notes"] = "Task 004 fixed official recipe; tier differs only by retained training labels"
            path = result / "config" / f"{tier}_fold_{fold}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            commands.append(f"python3 ./cellvit/train_cell_classifier_head.py --config {path} > {result / 'logs' / f'{tier}_fold_{fold}.stdout'} 2>&1")
    (result / "logs" / "commands_cv.sh").write_text("\n".join(commands) + "\n")
    print(f"generated {len(TIERS) * 5} official configs")


if __name__ == "__main__":
    main()
