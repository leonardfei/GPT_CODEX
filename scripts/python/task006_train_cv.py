#!/usr/bin/env python3
"""Run the pre-specified Task-006 SAM-H RAW five-fold CV for two tiers."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path("/data/lf_data/result/task006_xenium_reannotation")
REPO = Path("/data/lf_data/CellViT-plus-plus")
BACKBONE = REPO / "checkpoints/CellViT-SAM-H-x40-AMP.pth"
CLASSES = [
    "Endothelial",
    "Mesenchymal",
    "Myeloid",
    "Neutrophil",
    "Plasma cell",
    "T and B",
    "Tumor",
]


def yaml_config(tier: str, fold: int) -> str:
    dataset = ROOT / "work" / f"CellViT_dataset_v2_{tier}"
    run_dir = ROOT / "runs" / f"XENIUM_V2_HQ_{tier}" / f"fold_{fold}"
    labels = "\n".join(f"    {i}: {name}" for i, name in enumerate(CLASSES))
    return f"""gpu: 0
random_seed: 42
cellvit_path: {BACKBONE}
data:
  dataset: DetectionDataset
  dataset_path: {dataset}
  num_classes: 7
  label_map:
{labels}
  label_map_path: {dataset / 'label_map.yaml'}
  input_shape: 256
  train_filelist: {dataset / f'splits/fold_{fold}/train.csv'}
  val_filelist: {dataset / f'splits/fold_{fold}/val.csv'}
  normalize_stains_train: false
  normalize_stains_val: false
model:
  hidden_dim: 100
training:
  optimizer: AdamW
  optimizer_hyperparameter:
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    scheduler_type: cosine
    eta_min: 1.0e-06
  drop_rate: 0.0
  batch_size: 2048
  epochs: 12
  mixed_precision: true
  early_stopping_patience: 5
  eval_every: 1
  cache_cell_dataset: true
transformations:
  normalize:
    mean:
    - 0.5
    - 0.5
    - 0.5
    std:
    - 0.5
    - 0.5
    - 0.5
logging:
  log_dir: {run_dir}
  wandb_dir: {ROOT / 'logs/wandb'}
  mode: disabled
  project: task006_xenium_reannotation
  level: INFO
  log_comment: task006_XENIUM_V2_HQ_{tier}_fold_{fold}
  notes: Frozen independent Xenium reannotation; official SAM-H RAW recipe; no CellViT predictions used for labels or QC.
"""


def main() -> None:
    (ROOT / "config").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    commands = []
    for tier in ("CORE", "EXTENDED"):
        for fold in range(5):
            cfg = ROOT / "config" / f"XENIUM_V2_HQ_{tier}_fold_{fold}.yaml"
            cfg.write_text(yaml_config(tier, fold))
            log = ROOT / "logs" / f"cv_XENIUM_V2_HQ_{tier}_fold_{fold}.stdout"
            cmd = ["python3", "./cellvit/train_cell_classifier_head.py", "--config", str(cfg)]
            commands.append("cd /data/lf_data/CellViT-plus-plus && " + " ".join(cmd))
            if log.exists() and (ROOT / "runs" / f"XENIUM_V2_HQ_{tier}" / f"fold_{fold}").exists():
                # A completed run has a val_results directory; otherwise rerun.
                results = list((ROOT / "runs" / f"XENIUM_V2_HQ_{tier}" / f"fold_{fold}").glob("*/val_results"))
                if results:
                    continue
            with log.open("w") as handle:
                subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
    (ROOT / "logs" / "commands_cv_task006.sh").write_text("\n".join(commands) + "\n")


if __name__ == "__main__":
    main()
