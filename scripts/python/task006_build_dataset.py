#!/usr/bin/env python3
"""Build frozen Task-006 CellViT datasets from independent Xenium labels.

The source H&E patches and patch_records.pkl are treated as immutable.  Only
cells in the frozen quality tiers are written to the new label files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path("/data/lf_data/result/task006_xenium_reannotation")
SOURCE_DATASET = Path("/data/lf_data/xenium_data/CellViT_dataset")
SOURCE_H5AD = Path("/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad")
ANNOTATIONS = ROOT / "metrics/xenium_v2_cell_annotations.csv.gz"
SPLITS = Path("/data/lf_data/result/splits")
PATCH_SIZE = 256
CLASS_NAMES = [
    "Endothelial",
    "Mesenchymal",
    "Myeloid",
    "Neutrophil",
    "Plasma cell",
    "T and B",
    "Tumor",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
TRAIN_BATCHES = {"s01A", "s01B", "s02A", "s04B", "s06A", "s11", "s22", "s93"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in values
    ])


def write_label(path: Path, rows: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def make_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def load_annotations() -> pd.DataFrame:
    cols = ["cell_id", "batch", "new_celltype", "quality_tier", "he_x_px", "he_y_px"]
    ann = pd.read_csv(ANNOTATIONS, usecols=cols)
    ann["cell_id"] = ann["cell_id"].astype(str)
    with h5py.File(SOURCE_H5AD, "r") as f:
        cell_ids = decode_array(f["obs"]["cell_id"][:])
        object_ids = np.asarray(f["obs"]["cell_labels"][:], dtype=np.int64)
    cell_to_object = dict(zip(cell_ids.tolist(), object_ids.tolist()))
    ann["object_id"] = ann["cell_id"].map(cell_to_object).astype("Int64")
    ann = ann.dropna(subset=["object_id"]).copy()
    ann["object_id"] = ann["object_id"].astype(np.int64)
    ann["class_id"] = ann["new_celltype"].map(CLASS_TO_ID)
    ann = ann[ann["class_id"].notna()].copy()
    ann["class_id"] = ann["class_id"].astype(int)
    ann["patch_x"] = (np.floor(ann["he_x_px"] / PATCH_SIZE) * PATCH_SIZE).astype(int)
    ann["patch_y"] = (np.floor(ann["he_y_px"] / PATCH_SIZE) * PATCH_SIZE).astype(int)
    return ann.set_index("object_id", drop=False)


def load_records() -> list[dict]:
    with (SOURCE_DATASET / "patch_records.pkl").open("rb") as f:
        return pickle.load(f)


def build_tier(tier_name: str, ann: pd.DataFrame, records: list[dict]) -> dict:
    if tier_name == "CORE":
        keep_tiers = {"HQ_CORE"}
    elif tier_name == "EXTENDED":
        keep_tiers = {"HQ_CORE", "HQ_EXTENDED"}
    else:
        raise ValueError(tier_name)

    selected = ann[ann["quality_tier"].isin(keep_tiers)].copy()
    selected_by_object = selected
    selected_by_patch: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    # These coordinates are independently computed from the frozen inverse
    # registration.  CellViT predictions are not consulted here.
    for row in selected.itertuples(index=False):
        lx = int(np.rint(row.he_x_px - row.patch_x))
        ly = int(np.rint(row.he_y_px - row.patch_y))
        if 0 <= lx < PATCH_SIZE and 0 <= ly < PATCH_SIZE:
            selected_by_patch[(int(row.patch_x), int(row.patch_y))].append(
                (lx, ly, int(row.class_id))
            )

    out = ROOT / "work" / f"CellViT_dataset_v2_{tier_name}"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    split_records = {"train": [], "test": []}
    for record in records:
        split = "train" if record["batch"] in TRAIN_BATCHES else "test"
        split_records[split].append(record)

    summary_rows = []
    all_label_hashes = {}
    for split, split_recs in split_records.items():
        for record in split_recs:
            patch_id = int(record["patch_id"])
            stem = f"{split}_{patch_id:06d}"
            src_img = SOURCE_DATASET / split / "images" / f"{stem}.png"
            if not src_img.exists():
                raise FileNotFoundError(src_img)
            make_link(src_img, out / split / "images" / src_img.name)
            rows = selected_by_patch.get((int(record["x"]), int(record["y"])), [])
            # Stable ordering makes the files reproducible and easy to audit.
            rows = sorted(set(rows), key=lambda x: (x[1], x[0], x[2]))
            label_path = out / split / "labels" / f"{stem}.csv"
            write_label(label_path, rows)
            all_label_hashes[str(label_path.relative_to(out))] = sha256_file(label_path)
            counts = Counter(row[2] for row in rows)
            summary_rows.append(
                {
                    "tier": tier_name,
                    "split": split,
                    "patch_id": patch_id,
                    "batch": record["batch"],
                    "image": stem,
                    "n_labels": len(rows),
                    "n_empty": int(len(rows) == 0),
                    **{f"n_{name}": counts.get(i, 0) for i, name in enumerate(CLASS_NAMES)},
                }
            )

    split_dir = out / "splits"
    for fold in range(5):
        (split_dir / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
        for name in ("train.csv", "val.csv"):
            shutil.copy2(SPLITS / f"fold_{fold}" / name, split_dir / f"fold_{fold}" / name)

    (out / "label_map.yaml").write_text(
        "\n".join([f"{i}: {name}" for i, name in enumerate(CLASS_NAMES)]) + "\n"
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "metrics" / f"dataset_summary_{tier_name}.csv", index=False)
    selected_counts = (
        selected.groupby(["quality_tier", "new_celltype"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    selected_counts.to_csv(ROOT / "metrics" / f"selected_label_counts_{tier_name}.csv", index=False)
    manifest = {
        "tier": tier_name,
        "quality_tiers": sorted(keep_tiers),
        "source_dataset": str(SOURCE_DATASET),
        "patch_size": PATCH_SIZE,
        "n_source_records": len(records),
        "n_train_records": len(split_records["train"]),
        "n_test_records": len(split_records["test"]),
        "n_selected_cells": int(len(selected)),
        "selected_class_counts": {
            k: int(v) for k, v in selected["new_celltype"].value_counts().to_dict().items()
        },
        "label_file_sha256": all_label_hashes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    ROOT.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    ann = load_annotations()
    records = load_records()
    manifests = [build_tier(tier, ann, records) for tier in ("CORE", "EXTENDED")]
    (ROOT / "metrics" / "dataset_build_manifest.json").write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False)
    )
    print(json.dumps(manifests, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
