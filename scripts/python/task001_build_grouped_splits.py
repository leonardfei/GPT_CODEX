#!/usr/bin/env python3
"""Create leakage-safe grouped CV splits for the CellViT training set."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from sklearn.model_selection import StratifiedGroupKFold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    result = args.result_root.resolve()

    label_map = {str(k): int(v) for k, v in yaml.safe_load((root / "label_map.yaml").read_text()).items()}
    valid_classes = set(label_map.values())
    with (root / "patch_records.pkl").open("rb") as handle:
        patch_records = pickle.load(handle)

    stems: list[str] = []
    majority_labels: list[int] = []
    groups: list[str] = []
    cell_counts: dict[str, Counter[int]] = {}
    for label_path in sorted((root / "train" / "labels").glob("*.csv")):
        stem = label_path.stem
        patch_id = int(stem.split("_")[-1])
        batch = str(patch_records[patch_id]["batch"])
        counts: Counter[int] = Counter()
        with label_path.open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row) == 3 and row[2].strip().lstrip("-").isdigit() and int(row[2]) in valid_classes:
                    counts[int(row[2])] += 1
        if not counts:
            raise RuntimeError(f"No valid labels in {label_path}")
        # StratifiedGroupKFold accepts one target per patch. Use the dominant
        # class only for fold balancing; all labels remain in the files.
        majority = min(((-count, class_id) for class_id, count in counts.items()))[1]
        stems.append(stem)
        majority_labels.append(majority)
        groups.append(batch)
        cell_counts[stem] = counts

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    placeholder = [[0] for _ in stems]
    split_dir = result / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    fold_summary = []
    seen_validation: set[str] = set()
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(placeholder, majority_labels, groups)):
        train_names = sorted(stems[i] for i in train_idx)
        val_names = sorted(stems[i] for i in val_idx)
        train_groups = sorted({groups[i] for i in train_idx})
        val_groups = sorted({groups[i] for i in val_idx})
        overlap = sorted(set(train_groups) & set(val_groups))
        image_overlap = sorted(set(train_names) & set(val_names))
        val_class_counts = Counter()
        train_class_counts = Counter()
        for name in train_names:
            train_class_counts.update(cell_counts[name])
        for name in val_names:
            val_class_counts.update(cell_counts[name])
        if overlap or image_overlap:
            raise RuntimeError(f"Fold {fold_idx} has leakage: groups={overlap}, images={image_overlap[:5]}")
        if seen_validation & set(val_names):
            raise RuntimeError(f"Validation images overlap across folds at fold {fold_idx}")
        seen_validation.update(val_names)
        fold_path = split_dir / f"fold_{fold_idx}"
        fold_path.mkdir(parents=True, exist_ok=True)
        for name, values in [("train.csv", train_names), ("val.csv", val_names)]:
            with (fold_path / name).open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image"])
                writer.writerows([[value] for value in values])
        fold_summary.append({
            "fold": fold_idx,
            "train_images": len(train_names),
            "val_images": len(val_names),
            "train_groups": train_groups,
            "val_groups": val_groups,
            "group_overlap": overlap,
            "image_overlap": image_overlap,
            "train_cell_counts": dict(sorted(train_class_counts.items())),
            "val_cell_counts": dict(sorted(val_class_counts.items())),
            "val_majority_class_counts": dict(Counter(majority_labels[i] for i in val_idx)),
        })

    if seen_validation != set(stems):
        raise RuntimeError("Validation folds do not partition the training images")

    summary = {
        "seed": args.seed,
        "n_splits": args.n_splits,
        "group_field": "batch from patch_records.pkl",
        "stratification_target": "dominant cell class per patch; all cell labels retained",
        "folds": fold_summary,
    }
    qc_dir = result / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "grouped_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Grouped Cross-Validation Split QC",
        "",
        "The pre-existing source folds were not used because every source fold mixed the same batch groups between train and validation.",
        "",
        f"- Splitter: `StratifiedGroupKFold(n_splits={args.n_splits}, shuffle=True, random_state={args.seed})`.",
        "- Group: `batch` recovered from `patch_records.pkl`.",
        "- Stratification target: dominant cell class per patch; all cell-level labels remain unchanged.",
        "- Validation partition: each training image appears in exactly one validation fold.",
        "",
        "## Folds",
        "",
    ]
    for fold in fold_summary:
        lines.append(f"- fold_{fold['fold']}: train={fold['train_images']} images ({', '.join(fold['train_groups'])}); val={fold['val_images']} images ({', '.join(fold['val_groups'])}); group overlap=none.")
    (qc_dir / "grouped_split_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
