#!/usr/bin/env python3
"""Dataset inventory and leakage/QC checks for CellViT DetectionDataset data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


ID_RE = re.compile(r"(?:train|test|patch)_(\d+)$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        return [row[0].strip() for row in reader if row and row[0].strip() != "image"]


def parse_id(stem: str) -> int | None:
    match = ID_RE.fullmatch(stem)
    return int(match.group(1)) if match else None


def inspect_split(dataset_root: Path, split: str, label_map: dict[str, int], patch_records: list[dict], batch_map: dict[str, str], file_hashes: dict[str, str]) -> dict:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    images = sorted(image_dir.glob("*.png"))
    image_stems = {path.stem for path in images}
    label_files = sorted(label_dir.glob("*.csv"))
    label_stems = {path.stem for path in label_files}
    valid_ids = set(label_map.values())
    class_counts: Counter[int] = Counter()
    class_image_counts: Counter[int] = Counter()
    batch_counts: Counter[str] = Counter()
    invalid_rows: list[str] = []
    empty_labels: list[str] = []
    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    unreadable: list[str] = []
    dimension_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    hashes: dict[str, str] = {}

    for image_path in images:
        stem = image_path.stem
        patch_id = parse_id(stem)
        if patch_id is not None and patch_id < len(patch_records):
            batch = str(patch_records[patch_id].get("batch", "UNKNOWN"))
            batch_counts[batch] += 1
        else:
            batch = "UNKNOWN"
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                dimension_counts[f"{image.width}x{image.height}"] += 1
                mode_counts[image.mode] += 1
        except Exception as exc:  # pragma: no cover - depends on input corruption
            unreadable.append(f"{stem}: {type(exc).__name__}: {exc}")
        hashes[stem] = sha256_file(image_path)
        file_hashes[f"{split}/{stem}"] = hashes[stem]

        label_path = label_dir / f"{stem}.csv"
        if not label_path.exists():
            continue
        rows = []
        with label_path.open(newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle), start=1):
                if not row:
                    continue
                rows.append(row)
                if len(row) != 3:
                    invalid_rows.append(f"{stem}:{row_number}: expected 3 columns, got {len(row)}")
                    continue
                try:
                    x, y, class_id = map(int, row)
                except ValueError:
                    invalid_rows.append(f"{stem}:{row_number}: non-integer row {row}")
                    continue
                if not (0 <= x < 256 and 0 <= y < 256):
                    invalid_rows.append(f"{stem}:{row_number}: coordinate ({x},{y}) outside 256x256")
                if class_id not in valid_ids:
                    invalid_rows.append(f"{stem}:{row_number}: invalid class id {class_id}")
                else:
                    class_counts[class_id] += 1
        if not rows:
            empty_labels.append(stem)
        for class_id in set(
            int(row[2]) for row in rows if len(row) == 3 and row[2].strip().lstrip("-").isdigit() and int(row[2]) in valid_ids
        ):
            class_image_counts[class_id] += 1

    return {
        "split": split,
        "image_count": len(images),
        "label_count": len(label_files),
        "cell_count": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "class_image_counts": dict(class_image_counts),
        "batch_counts": dict(batch_counts),
        "batches": sorted(batch_counts),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "empty_labels": empty_labels,
        "invalid_rows": invalid_rows,
        "unreadable": unreadable,
        "dimension_counts": dict(dimension_counts),
        "mode_counts": dict(mode_counts),
        "hashes": hashes,
    }


def validate_folds(dataset_root: Path, patch_records: list[dict], label_map: dict[str, int], train_stems: set[str]) -> list[dict]:
    reports: list[dict] = []
    valid_ids = set(label_map.values())
    for fold_dir in sorted((dataset_root / "splits").glob("fold_*")):
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        if not train_path.exists() or not val_path.exists():
            reports.append({"fold": fold_dir.name, "status": "FAIL", "details": "missing train.csv or val.csv"})
            continue
        train_names = set(read_rows(train_path))
        val_names = set(read_rows(val_path))
        train_batches = {str(patch_records[parse_id(name)].get("batch", "UNKNOWN")) for name in train_names if parse_id(name) is not None and parse_id(name) < len(patch_records)}
        val_batches = {str(patch_records[parse_id(name)].get("batch", "UNKNOWN")) for name in val_names if parse_id(name) is not None and parse_id(name) < len(patch_records)}
        class_counts = Counter()
        missing_label_files = 0
        for name in train_names | val_names:
            label_path = dataset_root / "train" / "labels" / f"{name}.csv"
            if not label_path.exists():
                missing_label_files += 1
                continue
            with label_path.open(newline="") as handle:
                for row in csv.reader(handle):
                    if len(row) == 3 and row[2].strip().lstrip("-").isdigit() and int(row[2]) in valid_ids:
                        class_counts[int(row[2])] += 1
        overlap = train_names & val_names
        unknown = (train_names | val_names) - train_stems
        group_overlap = train_batches & val_batches
        status = "PASS" if not overlap and not unknown and not group_overlap and not missing_label_files else "FAIL"
        reports.append({
            "fold": fold_dir.name,
            "status": status,
            "train_images": len(train_names),
            "val_images": len(val_names),
            "image_overlap": len(overlap),
            "unknown_images": len(unknown),
            "train_batches": ",".join(sorted(train_batches)),
            "val_batches": ",".join(sorted(val_batches)),
            "batch_overlap": ",".join(sorted(group_overlap)),
            "missing_label_files": missing_label_files,
            "class_counts": dict(class_counts),
        })
    return reports


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    with (dataset_root / "label_map.yaml").open() as handle:
        label_map = {str(k): int(v) for k, v in yaml.safe_load(handle).items()}
    with (dataset_root / "patch_records.pkl").open("rb") as handle:
        patch_records = pickle.load(handle)
    with (dataset_root / "batch_train_test_split.csv").open(newline="") as handle:
        batch_map = {row[0]: row[1] for row in csv.reader(handle) if row and row[0] != "batch"}

    file_hashes: dict[str, str] = {}
    train = inspect_split(dataset_root, "train", label_map, patch_records, batch_map, file_hashes)
    test = inspect_split(dataset_root, "test", label_map, patch_records, batch_map, file_hashes)

    train_hashes = set(train["hashes"].values())
    test_hashes = set(test["hashes"].values())
    train_test_duplicate_hashes = sorted(train_hashes & test_hashes)
    train_stems = set(train["hashes"])
    fold_reports = validate_folds(dataset_root, patch_records, label_map, train_stems)

    mapping_rows = []
    for split_name, split_report in [("train", train), ("test", test)]:
        for stem, image_hash in sorted(split_report["hashes"].items()):
            patch_id = parse_id(stem)
            batch = str(patch_records[patch_id].get("batch", "UNKNOWN")) if patch_id is not None and patch_id < len(patch_records) else "UNKNOWN"
            expected_split = batch_map.get(batch, "UNKNOWN")
            mapping_rows.append({"split": split_name, "image": stem, "patch_id": patch_id, "batch": batch, "expected_split_from_batch": expected_split, "sha256": image_hash})
    write_csv(result_root / "data" / "patch_metadata.csv", mapping_rows, list(mapping_rows[0]))

    class_rows = []
    total_train_cells = train["cell_count"] or 1
    total_test_cells = test["cell_count"] or 1
    inverse_map = {value: key for key, value in label_map.items()}
    for class_id in sorted(inverse_map):
        class_rows.append({
            "class_index": class_id,
            "class_name": inverse_map[class_id],
            "training_count": train["class_counts"].get(class_id, 0),
            "test_count": test["class_counts"].get(class_id, 0),
            "training_percentage": train["class_counts"].get(class_id, 0) / total_train_cells,
            "test_percentage": test["class_counts"].get(class_id, 0) / total_test_cells,
            "training_image_count": train["class_image_counts"].get(class_id, 0),
            "test_image_count": test["class_image_counts"].get(class_id, 0),
            "training_batches": ";".join(sorted(train["batches"])),
            "test_batches": ";".join(sorted(test["batches"])),
        })
    write_csv(result_root / "data" / "class_distribution.csv", class_rows, list(class_rows[0]))

    config_dir = result_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_root / "label_map.yaml", config_dir / "label_map.yaml")
    split_out = result_root / "splits"
    split_out.mkdir(parents=True, exist_ok=True)
    for fold_dir in sorted((dataset_root / "splits").glob("fold_*")):
        target = split_out / fold_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for file_name in ("train.csv", "val.csv"):
            source = fold_dir / file_name
            if source.exists():
                shutil.copy2(source, target / file_name)

    qc_rows = [
        {"metric": "train_image_count", "value": train["image_count"], "status": "PASS", "details": "PNG files"},
        {"metric": "test_image_count", "value": test["image_count"], "status": "PASS", "details": "PNG files"},
        {"metric": "train_cell_count", "value": train["cell_count"], "status": "PASS", "details": "valid parsed annotation rows"},
        {"metric": "test_cell_count", "value": test["cell_count"], "status": "PASS", "details": "valid parsed annotation rows"},
        {"metric": "train_missing_labels", "value": len(train["missing_labels"]), "status": "PASS" if not train["missing_labels"] else "FAIL", "details": ";".join(train["missing_labels"][:20])},
        {"metric": "test_missing_labels", "value": len(test["missing_labels"]), "status": "PASS" if not test["missing_labels"] else "FAIL", "details": ";".join(test["missing_labels"][:20])},
        {"metric": "train_orphan_labels", "value": len(train["orphan_labels"]), "status": "PASS" if not train["orphan_labels"] else "FAIL", "details": ";".join(train["orphan_labels"][:20])},
        {"metric": "test_orphan_labels", "value": len(test["orphan_labels"]), "status": "PASS" if not test["orphan_labels"] else "FAIL", "details": ";".join(test["orphan_labels"][:20])},
        {"metric": "train_empty_labels", "value": len(train["empty_labels"]), "status": "PASS" if not train["empty_labels"] else "FAIL", "details": ";".join(train["empty_labels"][:20])},
        {"metric": "test_empty_labels", "value": len(test["empty_labels"]), "status": "PASS" if not test["empty_labels"] else "FAIL", "details": ";".join(test["empty_labels"][:20])},
        {"metric": "train_invalid_annotation_rows", "value": len(train["invalid_rows"]), "status": "PASS" if not train["invalid_rows"] else "FAIL", "details": ";".join(train["invalid_rows"][:20])},
        {"metric": "test_invalid_annotation_rows", "value": len(test["invalid_rows"]), "status": "PASS" if not test["invalid_rows"] else "FAIL", "details": ";".join(test["invalid_rows"][:20])},
        {"metric": "train_unreadable_images", "value": len(train["unreadable"]), "status": "PASS" if not train["unreadable"] else "FAIL", "details": ";".join(train["unreadable"][:20])},
        {"metric": "test_unreadable_images", "value": len(test["unreadable"]), "status": "PASS" if not test["unreadable"] else "FAIL", "details": ";".join(test["unreadable"][:20])},
        {"metric": "train_test_exact_duplicate_images", "value": len(train_test_duplicate_hashes), "status": "PASS" if not train_test_duplicate_hashes else "FAIL", "details": "SHA-256 intersection"},
        {"metric": "train_dimensions", "value": json.dumps(train["dimension_counts"], sort_keys=True), "status": "PASS"},
        {"metric": "test_dimensions", "value": json.dumps(test["dimension_counts"], sort_keys=True), "status": "PASS"},
        {"metric": "train_batches", "value": ",".join(sorted(train["batches"])), "status": "PASS"},
        {"metric": "test_batches", "value": ",".join(sorted(test["batches"])), "status": "PASS"},
        {"metric": "fold_reports", "value": json.dumps(fold_reports, sort_keys=True), "status": "PASS" if all(x.get("status") == "PASS" for x in fold_reports) else "FAIL"},
    ]
    write_csv(result_root / "qc" / "dataset_qc.csv", qc_rows, ["metric", "value", "status", "details"])

    summary_lines = [
        "# Dataset QC Summary",
        "",
        "## Scope",
        "",
        f"Dataset root: `{dataset_root}`",
        "Original images and labels were read only; no source files were modified.",
        "",
        "## Dataset inventory",
        "",
        f"- Train images: {train['image_count']}; parsed cells: {train['cell_count']}",
        f"- Test images: {test['image_count']}; parsed cells: {test['cell_count']}",
        f"- Train batches: {', '.join(sorted(train['batches']))}",
        f"- Test batches: {', '.join(sorted(test['batches']))}",
        f"- Train dimensions: {train['dimension_counts']}",
        f"- Test dimensions: {test['dimension_counts']}",
        "",
        "## Label map",
        "",
        "```yaml",
        yaml.safe_dump(label_map, sort_keys=False).rstrip(),
        "```",
        "",
        "## QC findings",
        "",
        f"- Exact train/test image hash overlap: {len(train_test_duplicate_hashes)}.",
        f"- Train invalid annotation rows: {len(train['invalid_rows'])}; test invalid annotation rows: {len(test['invalid_rows'])}.",
        f"- Train unreadable images: {len(train['unreadable'])}; test unreadable images: {len(test['unreadable'])}.",
        f"- Fold/group QC status: {'PASS' if all(x.get('status') == 'PASS' for x in fold_reports) else 'FAIL'}.",
        "",
        "## Fold details",
        "",
    ]
    for fold in fold_reports:
        summary_lines.append(f"- {fold['fold']}: {fold['status']}; train={fold.get('train_images', 'NA')}, val={fold.get('val_images', 'NA')}, batch overlap={fold.get('batch_overlap', '') or 'none'}.")
    (result_root / "qc" / "dataset_qc_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (result_root / "qc" / "dataset_qc_summary.md").write_text("\n".join(summary_lines) + "\n")

    payload = {
        "dataset_root": str(dataset_root),
        "label_map": label_map,
        "train": {k: v for k, v in train.items() if k != "hashes"},
        "test": {k: v for k, v in test.items() if k != "hashes"},
        "train_test_exact_duplicate_hashes": train_test_duplicate_hashes,
        "fold_reports": fold_reports,
    }
    (result_root / "qc" / "dataset_qc.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"train": train["image_count"], "test": test["image_count"], "train_cells": train["cell_count"], "test_cells": test["cell_count"], "duplicate_hashes": len(train_test_duplicate_hashes), "fold_status": [x.get("status") for x in fold_reports]}, indent=2))


if __name__ == "__main__":
    main()
