#!/usr/bin/env python3
"""Regenerate the Task009 CellViT datasets from the original H&E source.

This script deliberately does not read the historical CellViT_dataset.  It
reconstructs patches from the original OME-TIFF and frozen Task007/Task008
tables, then writes two new dataset roots and the audit artefacts required by
Task009.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyvips
import yaml
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold


CLASSES = [
    "Endothelial",
    "Mesenchymal",
    "Myeloid",
    "Neutrophil",
    "Plasma cell",
    "T and B",
    "Tumor",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
NEUTROPHIL_ORIGINALS = {"Neutrophil", "Neutrophil_CXCR4"}
TRAIN_BATCHES = {"s01A", "s01B", "s02A", "s04B", "s06A", "s11", "s22", "s93"}
TEST_BATCHES = {"s02B", "s03A", "s03B", "s05A", "s05B", "s100", "s33", "s98"}
ALL_BATCHES = sorted(TRAIN_BATCHES | TEST_BATCHES)
PATCH_SIZE = 256
PATCH_STRIDE = 256
MAX_CELLS_PER_CLASS = 50_000
CAP_SEED = 1234
CV_SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_mkdirs(result: Path) -> None:
    forbidden = Path("/data/lf_data/xenium_data/CellViT_dataset").resolve()
    if forbidden == result or forbidden in result.parents:
        raise RuntimeError("Task009 output is inside the forbidden historical dataset")
    for name in ("config", "code", "qc", "metrics", "work", "models", "cache", "figures", "figure_data", "logs", "runs"):
        (result / name).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_sources(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, int, int, str]:
    ann = pd.read_csv(args.annotation)
    required_ann = {
        "cell_index", "cell_id", "batch", "original_cl1", "original_cl1_7class",
        "v3_label", "v3_action", "cell_quality_status", "technical_fail",
        "qc_nucleus_present_xenium", "he_registration_status",
    }
    missing = sorted(required_ann - set(ann.columns))
    if missing:
        raise RuntimeError(f"Task007 annotation missing columns: {missing}")
    elig = pd.read_csv(args.eligibility)
    required_elig = {
        "cell_index", "cell_id", "batch", "task008_status",
        "trainable_core_task008", "trainable_extended_task008",
    }
    missing = sorted(required_elig - set(elig.columns))
    if missing:
        raise RuntimeError(f"Task008 eligibility missing columns: {missing}")
    if ann["cell_index"].duplicated().any() or elig["cell_index"].duplicated().any():
        raise RuntimeError("Duplicate cell_index in frozen annotation/eligibility")
    if len(ann) != 990_850:
        raise RuntimeError(f"Unexpected Task007 row count: {len(ann)}")
    elig_indexed = elig.set_index("cell_index")
    ann["task008_status"] = "NOT_APPLICABLE"
    ann["trainable_core_task008"] = False
    ann["trainable_extended_task008"] = False
    ann["task008_registration"] = "NOT_APPLICABLE"
    common = ann["cell_index"].isin(elig_indexed.index)
    common_idx = ann.loc[common, "cell_index"]
    ann.loc[common, "task008_status"] = common_idx.map(elig_indexed["task008_status"]).to_numpy()
    ann.loc[common, "trainable_core_task008"] = common_idx.map(elig_indexed["trainable_core_task008"]).fillna(False).to_numpy()
    ann.loc[common, "trainable_extended_task008"] = common_idx.map(elig_indexed["trainable_extended_task008"]).fillna(False).to_numpy()
    # Registration status is a frozen Task007 field; Task008 eligibility only
    # adds the accepted H&E status and training-tier flags.
    ann.loc[common, "task008_registration"] = ann.loc[common, "he_registration_status"].astype(str).to_numpy()
    if (ann.loc[common, "cell_id"].to_numpy() != common_idx.map(elig_indexed["cell_id"]).to_numpy()).any():
        raise RuntimeError("Task007 and Task008 cell_id mismatch")
    if (ann.loc[common, "batch"].to_numpy() != common_idx.map(elig_indexed["batch"]).to_numpy()).any():
        raise RuntimeError("Task007 and Task008 batch mismatch")

    source = ad.read_h5ad(args.adata, backed="r")
    if source.n_obs != len(ann) or "spatial_HE" not in source.obsm:
        raise RuntimeError("Derived AnnData shape/spatial_HE does not match Task007")
    obs_ids = np.asarray(source.obs["cell_id"].astype(str))
    if not np.array_equal(obs_ids[:10], ann["cell_id"].astype(str).to_numpy()[:10]):
        raise RuntimeError("Derived AnnData and Task007 row order mismatch at first rows")
    coords = np.asarray(source.obsm["spatial_HE"][:], dtype=np.float64)
    if coords.shape != (len(ann), 2) or not np.isfinite(coords).all():
        raise RuntimeError("Invalid spatial_HE coordinates")
    slide = pyvips.Image.new_from_file(str(args.he), access="random")
    matrix_text = args.matrix.read_text().strip()
    ann["he_x"] = coords[:, 0]
    ann["he_y"] = coords[:, 1]
    return ann, coords, int(slide.width), int(slide.height), matrix_text


def frozen_eligibility(df: pd.DataFrame, condition: str) -> tuple[pd.Series, pd.Series]:
    if condition not in {"CORE", "EXTENDED"}:
        raise ValueError(condition)
    base_quality = (
        df["cell_quality_status"].eq("Pass")
        & ~df["technical_fail"].astype(bool)
        & df["qc_nucleus_present_xenium"].astype(bool)
        & df["v3_action"].ne("REVIEW")
    )
    labelable = df["v3_label"].isin(CLASSES)
    non_neut = base_quality & labelable & ~df["original_cl1"].isin(NEUTROPHIL_ORIGINALS) & df["v3_label"].ne("Neutrophil")
    neut_col = "trainable_core_task008" if condition == "CORE" else "trainable_extended_task008"
    neut = base_quality & df["original_cl1"].isin(NEUTROPHIL_ORIGINALS) & df[neut_col].astype(bool)
    # Task008 flags are already target-centered and manually accepted.  Keep
    # the explicit status/registration guard visible in the audit.
    neut &= df["task008_status"].eq("TARGET_NUCLEUS_PRESENT")
    neut &= df["task008_registration"].eq("inbounds")
    return non_neut | neut, base_quality


def historical_cap(df: pd.DataFrame, eligible: pd.Series) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    rng = np.random.RandomState(CAP_SEED)
    selected: list[int] = []
    uncapped: dict[str, int] = {}
    capped: dict[str, int] = {}
    for cls in CLASSES:
        if cls == "Neutrophil":
            mask = eligible & (df["original_cl1"].isin(NEUTROPHIL_ORIGINALS))
        else:
            mask = eligible & df["v3_label"].eq(cls)
        idx = np.flatnonzero(mask.to_numpy())
        uncapped[cls] = int(len(idx))
        if len(idx) > MAX_CELLS_PER_CLASS:
            idx = rng.choice(idx, size=MAX_CELLS_PER_CLASS, replace=False)
        selected.extend(int(x) for x in idx)
        capped[cls] = int(len(idx))
    return np.asarray(selected, dtype=np.int64), uncapped, capped


def link_or_replace(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        raise RuntimeError(f"Refusing to replace existing dataset file: {link}")
    link.symlink_to(os.path.relpath(target, link.parent))


def label_for_row(row: pd.Series) -> str:
    if row["original_cl1"] in NEUTROPHIL_ORIGINALS:
        return "Neutrophil"
    return str(row["v3_label"])


def generate_dataset(
    args: argparse.Namespace,
    result: Path,
    df: pd.DataFrame,
    coords: np.ndarray,
    width: int,
    height: int,
    condition: str,
) -> dict:
    dataset = result / "work" / f"CellViT_dataset_v3_{condition}"
    if dataset.exists():
        raise RuntimeError(f"Refusing to overwrite existing Task009 dataset: {dataset}")
    for sub in ("all/images", "all/labels", "train/images", "train/labels", "train/overlay", "test/images", "test/labels", "test/overlay", "splits"):
        (dataset / sub).mkdir(parents=True, exist_ok=True)
    (dataset / "label_map.yaml").write_text(yaml.safe_dump(CLASS_TO_ID, sort_keys=False))

    eligible, base_quality = frozen_eligibility(df, condition)
    selected_idx, uncapped, capped = historical_cap(df, eligible)
    selected = df.iloc[selected_idx].copy()
    selected_coords = coords[selected_idx]
    half = PATCH_SIZE // 2
    in_bounds = (
        np.isfinite(selected_coords).all(axis=1)
        & (selected_coords[:, 0] >= 0) & (selected_coords[:, 0] < width)
        & (selected_coords[:, 1] >= 0) & (selected_coords[:, 1] < height)
    )
    selected = selected.loc[in_bounds].copy()
    selected_coords = selected_coords[in_bounds]
    edge_ok = (
        (selected_coords[:, 0] >= half) & (selected_coords[:, 0] < width - half)
        & (selected_coords[:, 1] >= half) & (selected_coords[:, 1] < height - half)
    )
    selected = selected.loc[edge_ok].copy()
    selected_coords = selected_coords[edge_ok]
    selected.reset_index(drop=True, inplace=True)
    selected_coords = np.asarray(selected_coords, dtype=np.float64)
    if not len(selected):
        raise RuntimeError(f"No selected cells remain for {condition}")
    geometry_counts = {}
    for cls in CLASSES:
        geometry_counts[cls] = int(
            (
                selected["original_cl1"].isin(NEUTROPHIL_ORIGINALS)
                if cls == "Neutrophil"
                else selected["v3_label"].eq(cls)
            ).sum()
        )

    tree = cKDTree(selected_coords)
    cell_batches = selected["batch"].astype(str).to_numpy()
    slide = pyvips.Image.new_from_file(str(args.he), access="random")
    radius = PATCH_SIZE * np.sqrt(2.0) / 2.0
    xs = np.arange(0, width - PATCH_SIZE + 1, PATCH_STRIDE, dtype=int)
    ys = np.arange(0, height - PATCH_SIZE + 1, PATCH_STRIDE, dtype=int)
    patch_records: list[dict] = []
    cell_map_rows: list[dict] = []
    mixed_batch = 0
    patch_id = 0
    for y0 in ys:
        for x0 in xs:
            cx = x0 + PATCH_SIZE / 2.0
            cy = y0 + PATCH_SIZE / 2.0
            idx = tree.query_ball_point([cx, cy], radius)
            if not idx:
                continue
            pts = selected_coords[np.asarray(idx)]
            inside = (
                (pts[:, 0] >= x0) & (pts[:, 0] < x0 + PATCH_SIZE)
                & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + PATCH_SIZE)
            )
            idx = np.asarray(idx, dtype=np.int64)[inside]
            if not len(idx):
                continue
            batches = sorted(set(cell_batches[idx]))
            if len(batches) != 1:
                mixed_batch += 1
                continue
            batch = batches[0]
            split = "train" if batch in TRAIN_BATCHES else "test" if batch in TEST_BATCHES else "unknown"
            if split == "unknown":
                raise RuntimeError(f"Unexpected batch {batch}")
            image_name = f"patch_{patch_id:06d}"
            patch = slide.crop(int(x0), int(y0), PATCH_SIZE, PATCH_SIZE)
            img = np.ndarray(buffer=patch.write_to_memory(), dtype=np.uint8, shape=[patch.height, patch.width, patch.bands])
            if img.shape[2] >= 4:
                img = img[:, :, :3]
            from PIL import Image
            Image.fromarray(img[:, :, :3]).save(dataset / "all/images" / f"{image_name}.png")
            counts = Counter()
            label_rows = []
            for local_pos in idx:
                row = selected.iloc[int(local_pos)]
                cls = label_for_row(row)
                class_id = CLASS_TO_ID[cls]
                lx = int(np.clip(np.rint(selected_coords[local_pos, 0] - x0), 0, PATCH_SIZE - 1))
                ly = int(np.clip(np.rint(selected_coords[local_pos, 1] - y0), 0, PATCH_SIZE - 1))
                label_rows.append((lx, ly, class_id))
                counts[cls] += 1
                cell_map_rows.append({
                    "condition": condition,
                    "patch_id": patch_id,
                    "image": image_name,
                    "split": split,
                    "batch": batch,
                    "cell_index": int(row["cell_index"]),
                    "cell_id": str(row["cell_id"]),
                    "original_cl1": str(row["original_cl1"]),
                    "v3_label": str(row["v3_label"]),
                    "class_name": cls,
                    "class_id": class_id,
                    "he_x": float(selected_coords[local_pos, 0]),
                    "he_y": float(selected_coords[local_pos, 1]),
                    "local_x": lx,
                    "local_y": ly,
                    "task008_status": str(row["task008_status"]),
                    "task008_eligibility": f"TRAINABLE_{condition}" if cls == "Neutrophil" else "NOT_APPLICABLE",
                })
            with (dataset / "all/labels" / f"{image_name}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(label_rows)
            patch_records.append({
                "patch_id": patch_id,
                "image": image_name,
                "x": int(x0),
                "y": int(y0),
                "batch": batch,
                "split": split,
                "n_cells": len(label_rows),
                "class_counts": dict(sorted(counts.items())),
            })
            patch_id += 1

    if not patch_records:
        raise RuntimeError(f"No patches generated for {condition}")
    # Keep newly generated source images in all/, and expose split-specific
    # names as symlinks inside the same new dataset. No old file is referenced.
    for record in patch_records:
        prefix = "train" if record["split"] == "train" else "test"
        split_name = f"{prefix}_{record['patch_id']:06d}"
        for sub in ("images", "labels"):
            link_or_replace(dataset / record["split"] / sub / f"{split_name}{'.png' if sub == 'images' else '.csv'}", dataset / "all" / sub / f"{record['image']}{'.png' if sub == 'images' else '.csv'}")
    with (dataset / "patch_records.pkl").open("wb") as handle:
        pickle.dump(patch_records, handle, protocol=4)
    patch_rows = []
    for record in patch_records:
        patch_rows.append({
            "condition": condition,
            "patch_id": record["patch_id"],
            "image": f"{'train' if record['split'] == 'train' else 'test'}_{record['patch_id']:06d}",
            "source_image": record["image"],
            "split": record["split"],
            "batch": record["batch"],
            "x": record["x"],
            "y": record["y"],
            "n_cells": record["n_cells"],
            **{f"n_{cls}": int(record["class_counts"].get(cls, 0)) for cls in CLASSES},
        })
    write_csv(dataset / "patch_metadata.csv", patch_rows)
    write_csv(dataset / "cell_to_patch.csv", cell_map_rows)
    write_csv(dataset / "batch_train_test_split.csv", [{"batch": b, "split": "train" if b in TRAIN_BATCHES else "test"} for b in ALL_BATCHES])

    # Split files use the same dominant-class stratification and batch grouping
    # as the project’s previous controlled benchmarks.
    train_records = [r for r in patch_rows if r["split"] == "train"]
    stems = [str(r["image"]) for r in train_records]
    groups = [str(r["batch"]) for r in train_records]
    majority = []
    for r in train_records:
        counts = {CLASS_TO_ID[cls]: int(r[f"n_{cls}"]) for cls in CLASSES if int(r[f"n_{cls}"]) > 0}
        majority.append(min(((-n, cid) for cid, n in counts.items()))[1])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    split_rows = []
    seen_val = set()
    dummy = np.zeros((len(stems), 1), dtype=np.uint8)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(dummy, majority, groups)):
        train_names = sorted(stems[i] for i in train_idx)
        val_names = sorted(stems[i] for i in val_idx)
        train_groups = sorted({groups[i] for i in train_idx})
        val_groups = sorted({groups[i] for i in val_idx})
        overlap = sorted(set(train_groups) & set(val_groups))
        if overlap or seen_val.intersection(val_names):
            raise RuntimeError(f"Grouped split leakage in {condition} fold {fold}: {overlap}")
        seen_val.update(val_names)
        fold_dir = dataset / "splits" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name, values in (("train.csv", train_names), ("val.csv", val_names)):
            with (fold_dir / name).open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image"])
                writer.writerows([[value] for value in values])
        split_rows.append({
            "condition": condition,
            "fold": fold,
            "train_images": len(train_names),
            "val_images": len(val_names),
            "train_batches": ";".join(train_groups),
            "val_batches": ";".join(val_groups),
            "batch_overlap": "",
            "train_cells": sum(next(r for r in train_records if r["image"] == name)["n_cells"] for name in train_names),
            "val_cells": sum(next(r for r in train_records if r["image"] == name)["n_cells"] for name in val_names),
        })
    if seen_val != set(stems):
        raise RuntimeError(f"Validation folds do not partition {condition} training images")
    write_csv(result / "metrics" / f"split_manifest_{condition}.csv", split_rows)

    comp_rows = []
    map_df = pd.DataFrame(cell_map_rows)
    for cls in CLASSES:
        selected_class = map_df[map_df["class_name"].eq(cls)]
        comp_rows.append({
            "condition": condition,
            "class_name": cls,
            "class_id": CLASS_TO_ID[cls],
            "eligible_uncapped": uncapped[cls],
            "selected_after_50000_cap": capped[cls],
            "selected_after_cap_pre_geometry": capped[cls],
            "retained_after_geometry": geometry_counts[cls],
            "assigned_to_patches": int(len(selected_class)),
            "train_cells": int((selected_class["split"] == "train").sum()),
            "test_cells": int((selected_class["split"] == "test").sum()),
        })
    write_csv(result / "metrics" / f"dataset_composition_{condition}.csv", comp_rows)

    # Official CellViT++ head configuration, identical across V3 conditions.
    for fold in range(5):
        run_dir = result / "runs" / f"V3_{condition}" / f"fold_{fold}"
        cfg = {
            "gpu": 0,
            "random_seed": 42,
            "cellvit_path": str(args.cellvit),
            "data": {
                "dataset": "DetectionDataset",
                "dataset_path": str(dataset),
                "num_classes": 7,
                "label_map": {idx: name for name, idx in CLASS_TO_ID.items()},
                "label_map_path": str(dataset / "label_map.yaml"),
                "input_shape": 256,
                "train_filelist": str(dataset / "splits" / f"fold_{fold}" / "train.csv"),
                "val_filelist": str(dataset / "splits" / f"fold_{fold}" / "val.csv"),
                "normalize_stains_train": False,
                "normalize_stains_val": False,
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
                "log_dir": str(run_dir),
                "wandb_dir": str(result / "logs" / "wandb"),
                "mode": "disabled",
                "project": "task009_v3_retraining",
                "level": "INFO",
                "log_comment": f"task009_V3_{condition}_fold_{fold}",
                "notes": "Frozen Task007 biological labels and Task008 accepted H&E eligibility; official SAM-H RAW recipe.",
            },
        }
        (result / "config" / f"V3_{condition}_fold_{fold}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    manifest = {
        "condition": condition,
        "dataset_root": str(dataset),
        "historical_dataset_reused": False,
        "source_he": str(args.he),
        "source_matrix": str(args.matrix),
        "source_task007_annotation": str(args.annotation),
        "source_task008_eligibility": str(args.eligibility),
        "source_adata_for_spatial_he": str(args.adata),
        "new_image_patches": len(patch_records),
        "new_label_files": len(patch_records),
        "cells_assigned_to_patches": len(cell_map_rows),
        "cells_excluded_after_cap_or_geometry_or_patch": int(len(selected_idx) - len(cell_map_rows)),
        "mixed_batch_patches_dropped": mixed_batch,
        "grid": {"width": width, "height": height, "n_x": len(xs), "n_y": len(ys), "patch_size": PATCH_SIZE, "stride": PATCH_STRIDE},
        "train_batches": sorted(TRAIN_BATCHES),
        "test_batches": sorted(TEST_BATCHES),
        "uncapped_eligible": uncapped,
        "capped_eligible": capped,
    }
    (dataset / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"dataset": str(dataset), "manifest": manifest, "composition": comp_rows, "split_rows": split_rows}


def write_audits(args: argparse.Namespace, result: Path, width: int, height: int, matrix_text: str, outputs: dict[str, dict]) -> None:
    source_hashes = {str(path): sha256(path) for path in (args.he, args.matrix, args.annotation, args.eligibility)}
    (result / "config" / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    audit = f"""# Task009 dataset reconstruction audit

- Historical notebook: `{args.notebook}` (used only to reconstruct logic).
- Coordinate source: `spatial_HE` from `{args.adata}`; registration matrix text was read from `{args.matrix}`.
- H&E source: `{args.he}`; level-0 dimensions observed as `{width} x {height}` (width × height).
- Patch geometry: 256 × 256 pixels, stride 256; grid starts at `(0, 0)` and retains complete crops only.
- Historical order preserved: remove out-of-WSI coordinates, remove cells within 128 px of the WSI edge, query grid patches by KD-tree radius `256*sqrt(2)/2`, retain cells inside the crop, and drop patches containing more than one batch.
- Patch-edge handling: cells are assigned only when their registered centroid is inside `[x0, x0+256) × [y0, y0+256)`; local coordinates are rounded and clipped to 0–255 as in the historical notebook.
- Multiple labels per patch: allowed; every retained cell is written as one row in its patch label CSV.
- Per-class cap: historical 50,000-cell cap, applied before geometry filtering, with fixed NumPy seed {CAP_SEED}; classes below the cap are retained in full.
- Batch split: fixed historical 8-train/8-test split; train=`{', '.join(sorted(TRAIN_BATCHES))}`, test=`{', '.join(sorted(TEST_BATCHES))}`.
- Grouped CV: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state={CV_SEED})`, grouped by batch and stratified by dominant patch class.

The old image/label/split dataset was not used as a training source. New patches are exported from the original OME-TIFF for both V3 conditions.

## Registration matrix

```text
{matrix_text}
```

## Generated datasets

"""
    for condition, item in outputs.items():
        m = item["manifest"]
        audit += f"- {condition}: {m['new_image_patches']} image patches, {m['new_label_files']} label files, {m['cells_assigned_to_patches']} assigned cells, {m['mixed_batch_patches_dropped']} mixed-batch patches dropped.\n"
    (result / "qc" / "dataset_reconstruction_audit.md").write_text(audit)
    provenance = f"""# Task009 dataset regeneration provenance

1. Historical `/data/lf_data/xenium_data/CellViT_dataset` files reused: **No old CellViT_dataset files were reused.**
2. Source H&E: `{args.he}`
3. Registration matrix: `{args.matrix}`
4. Task007 annotation table: `{args.annotation}`
5. Task008 eligibility table: `{args.eligibility}`
6. Source coordinate AnnData: `{args.adata}` (`spatial_HE` only; no expression matrix was used for labels)

The following datasets were regenerated under Task009 from the original OME-TIFF and frozen tables:

| condition | new image patches | new label files | cells assigned to patches | cells excluded after cap/geometry/patch | manifest |
|---|---:|---:|---:|---:|---|
"""
    for condition, item in outputs.items():
        m = item["manifest"]
        provenance += f"| V3_{condition} | {m['new_image_patches']} | {m['new_label_files']} | {m['cells_assigned_to_patches']} | {m['cells_excluded_after_cap_or_geometry_or_patch']} | `{result / 'work' / f'CellViT_dataset_v3_{condition}' / 'dataset_manifest.json'}` |\n"
    provenance += "\nDataset manifests and source SHA256 values are recorded under `config/` and `metrics/`. No checkpoint or cache is included in Git.\n"
    (result / "qc" / "dataset_regeneration_provenance.md").write_text(provenance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task009_v3_retraining"))
    parser.add_argument("--annotation", type=Path, default=Path("/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz"))
    parser.add_argument("--eligibility", type=Path, default=Path("/data/lf_data/result/task008_neutrophil_he_recalibration/metrics/neutrophil_training_eligibility_task008.csv.gz"))
    parser.add_argument("--adata", type=Path, default=Path("/data/lf_data/result/task007_xenium5k_panelaware/adata_xenium_v3_panelaware.h5ad"))
    parser.add_argument("--he", type=Path, default=Path("/data/lf_data/xenium_data/ID0060276.ome.tif"))
    parser.add_argument("--matrix", type=Path, default=Path("/data/lf_data/xenium_data/matrix.csv"))
    parser.add_argument("--notebook", type=Path, default=Path("/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb"))
    parser.add_argument("--cellvit", type=Path, default=Path("/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth"))
    args = parser.parse_args()
    for path in (args.annotation, args.eligibility, args.adata, args.he, args.matrix, args.notebook, args.cellvit):
        if not path.exists():
            raise FileNotFoundError(path)
    result = args.result.resolve()
    safe_mkdirs(result)
    df, coords, width, height, matrix_text = load_sources(args)
    outputs = {}
    for condition in ("CORE", "EXTENDED"):
        outputs[condition] = generate_dataset(args, result, df, coords, width, height, condition)
    all_comp = outputs["CORE"]["composition"] + outputs["EXTENDED"]["composition"]
    write_csv(result / "metrics" / "dataset_composition.csv", all_comp)
    all_splits = outputs["CORE"]["split_rows"] + outputs["EXTENDED"]["split_rows"]
    write_csv(result / "metrics" / "split_manifest.csv", all_splits)
    write_audits(args, result, width, height, matrix_text, outputs)
    (result / "config" / "prepare_parameters.json").write_text(json.dumps({
        "classes": CLASSES,
        "cap_seed": CAP_SEED,
        "cv_seed": CV_SEED,
        "max_cells_per_class": MAX_CELLS_PER_CLASS,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "historical_dataset_reused": False,
    }, indent=2) + "\n")
    print(json.dumps({k: v["manifest"] for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
