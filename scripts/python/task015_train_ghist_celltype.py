#!/usr/bin/env python3
"""Split-safe GHIST cell-typing data and one-batch loss smoke test.

The full pilot/CV entry point will be enabled only after overlap-aware inner
validation is implemented and audited. This script never uses expression data.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from task015_ghist_celltype_model import GHISTCellType
from task015_evaluate_ghist_celltype import OverlapPatchDataset, evaluate_model

ROOT = Path("/data/lf_data/result/task015_ghist_celltyping")
FOLDS = Path("/data/lf_data/result/task010_representation_benchmark/metrics/fold_manifest.csv")
CANON = Path("/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz")
IGNORE = -100
SEED = 20260923


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_contours():
    result = {}
    path = ROOT / "data_manifest/patch_contours.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        for line in handle:
            obj = json.loads(line)
            image = obj["image"]
            if image in result:
                raise RuntimeError(f"Repeated contour patch: {image}")
            result[image] = obj["instances"]
    return result


def fold_roles(fold: int):
    df = pd.read_csv(FOLDS)
    df = df[df.fold == fold].copy()
    if len(df) != 96044 or df.cell_id.duplicated().any():
        raise RuntimeError("Fold manifest does not cover 96,044 unique cells")
    train_batches = sorted(df.loc[df.role == "train", "batch"].unique())
    val_batches = sorted(df.loc[df.role == "val", "batch"].unique())
    if set(train_batches) & set(val_batches):
        raise RuntimeError("Outer train/val batch leakage")
    inner_batch = train_batches[-1]
    inner_train_batches = set(train_batches) - {inner_batch}
    return df, inner_train_batches, inner_batch, val_batches


def training_normalization(patches: pd.DataFrame, path: Path):
    if path.exists():
        obj = json.loads(path.read_text())
        if set(obj["training_images"]) != set(patches.image):
            raise RuntimeError("Cached normalization was computed from different patches")
        return np.asarray(obj["mean"], dtype=np.float32), np.asarray(obj["std"], dtype=np.float32)
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for row in patches.itertuples(index=False):
        image = np.asarray(Image.open(row.image_path).convert("RGB"), dtype=np.float64) / 255.0
        if image.shape != (256, 256, 3):
            raise RuntimeError(f"Invalid input patch size: {row.image}")
        total += image.sum(axis=(0, 1))
        total_sq += np.square(image).sum(axis=(0, 1))
        n_pixels += image.shape[0] * image.shape[1]
    mean = total / n_pixels
    std = np.sqrt(np.maximum(total_sq / n_pixels - mean**2, 1e-12))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(),
                                "training_images": sorted(patches.image.tolist()),
                                "method": "global RGB pixel moments from inner-training patches only"}, indent=2) + "\n")
    return mean.astype(np.float32), std.astype(np.float32)


class GHISTPatchDataset(Dataset):
    def __init__(self, patches, contours, cell_labels, active_cells, mean, std, augment=False):
        self.patches = patches.reset_index(drop=True)
        self.contours = contours
        self.cell_labels = cell_labels
        self.active_cells = active_cells
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        row = self.patches.iloc[index]
        image = np.asarray(Image.open(row.image_path).convert("RGB"), dtype=np.uint8)
        if image.shape != (256, 256, 3):
            raise RuntimeError(f"Invalid image shape: {row.image}")
        instance_mask = np.zeros((256, 256), dtype=np.int32)
        label_map = {}
        cell_map = {}
        for obj in self.contours[row.image]:
            local_id = int(obj["instance_id"])
            polygon = np.asarray(obj["contour"], dtype=np.int32).reshape(-1, 1, 2)
            if len(polygon) < 3:
                continue
            cv2.fillPoly(instance_mask, [polygon], local_id)
            cell_id = obj["cell_id"]
            if cell_id is not None:
                cell_map[local_id] = cell_id
                if cell_id in self.active_cells:
                    label_map[local_id] = int(self.cell_labels[cell_id])
        target = np.zeros((256, 256), dtype=np.int64)
        for local_id in np.unique(instance_mask):
            if local_id == 0:
                continue
            target[instance_mask == local_id] = label_map.get(int(local_id), IGNORE) + (1 if int(local_id) in label_map else 0)
        if self.augment:
            if random.random() < 0.5:
                image, instance_mask, target = (np.flip(x, axis=1) for x in (image, instance_mask, target))
            if random.random() < 0.5:
                image, instance_mask, target = (np.flip(x, axis=0) for x in (image, instance_mask, target))
            k = random.randrange(4)
            image, instance_mask, target = (np.rot90(x, k=k) for x in (image, instance_mask, target))
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(instance_mask.astype(np.int64))),
                torch.from_numpy(np.ascontiguousarray(target)),
                cell_map, label_map, row.image)


def collate(batch):
    images, masks, targets, cell_maps, label_maps, names = zip(*batch)
    return torch.stack(images), torch.stack(masks), torch.stack(targets), cell_maps, label_maps, names


def compute_losses(output, pixel_targets, label_maps):
    pixel_loss = F.cross_entropy(output["pixel_logits"], pixel_targets, ignore_index=IGNORE)
    ids = output["ordered_ids"]
    cell_targets = torch.tensor([label_maps[b].get(local_id, IGNORE) for b, local_id in ids],
                                dtype=torch.long, device=pixel_targets.device)
    if not bool((cell_targets != IGNORE).any()):
        raise RuntimeError("No supervised cells in training batch")
    cell_loss = F.cross_entropy(output["cell_logits"], cell_targets, ignore_index=IGNORE)
    if not torch.isfinite(pixel_loss + cell_loss):
        raise FloatingPointError("Non-finite GHIST loss")
    return pixel_loss, cell_loss, cell_targets


def smoke(fold: int):
    seed_all()
    fold_df, train_batches, inner_batch, val_batches = fold_roles(fold)
    patches = pd.read_csv(ROOT / "metrics/patch_manifest.csv.gz")
    canonical = pd.read_csv(CANON)
    labels = dict(zip(canonical.cell_id, canonical.class_id))
    active = set(fold_df.loc[fold_df.batch.isin(train_batches), "cell_id"])
    assert not active & set(fold_df.loc[fold_df.role == "val", "cell_id"])
    assert not active & set(fold_df.loc[fold_df.batch == inner_batch, "cell_id"])
    train_patches = patches[(patches.batch.isin(train_batches)) & (patches.n_canonical > 0)].copy()
    mean, std = training_normalization(train_patches, ROOT / f"config/fold{fold}_inner_training_rgb.json")
    contours = load_contours()
    ds = GHISTPatchDataset(train_patches, contours, labels, active, mean, std, augment=True)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate)
    images, masks, targets, _, label_maps, names = next(iter(loader))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GHISTCellType().to(device)
    model.train()
    images, masks, targets = images.to(device), masks.to(device), targets.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(images, masks)
    pixel_loss, cell_loss, cell_targets = compute_losses(output, targets, label_maps)
    (pixel_loss + cell_loss).backward()
    optimizer.step()
    result = {"status": "LOSS_SMOKE_PASS", "fold": fold, "inner_validation_batch": inner_batch,
              "outer_validation_batches": val_batches, "training_batches": sorted(train_batches),
              "batch_images": list(names), "batch_size": len(names),
              "supervised_cells": int((cell_targets != IGNORE).sum()),
              "pixel_loss": float(pixel_loss), "cell_loss": float(cell_loss),
              "finite_logits": bool(torch.isfinite(output["cell_logits"]).all() and torch.isfinite(output["pixel_logits"]).all()),
              "supervised_ids_present_after_augmentation": bool(all(set(label_maps[i]) <= (set(torch.unique(masks[i]).tolist()) - {0}) for i in range(len(names)))),
              "hed_stain_augmentation": "disabled; official stainlib absent from installed environment",
              "note": "One-batch loss test only; not the required five-epoch pilot or overlapping validation."}
    (ROOT / "qc/loss_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def pilot(fold: int, epochs: int = 5):
    if fold != 0 or epochs < 5:
        raise ValueError("The predefined pilot is fold 0 for at least five epochs")
    seed_all()
    fold_df, train_batches, inner_batch, outer_val_batches = fold_roles(fold)
    patches = pd.read_csv(ROOT / "metrics/patch_manifest.csv.gz")
    canonical = pd.read_csv(CANON)
    labels = dict(zip(canonical.cell_id, canonical.class_id))
    active = set(fold_df.loc[fold_df.batch.isin(train_batches), "cell_id"])
    inner_cells = canonical[canonical.batch == inner_batch]
    outer_cells = set(fold_df.loc[fold_df.role == "val", "cell_id"])
    assert not active & set(inner_cells.cell_id)
    assert not active & outer_cells
    assert not set(inner_cells.cell_id) & outer_cells
    train_patches = patches[(patches.batch.isin(train_batches)) & (patches.n_canonical > 0)].copy()
    mean, std = training_normalization(train_patches, ROOT / "config/fold0_inner_training_rgb.json")
    contours = load_contours()
    train_ds = GHISTPatchDataset(train_patches, contours, labels, active, mean, std, augment=True)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4,
                              collate_fn=collate, generator=generator, pin_memory=True)
    overlap_selected = pd.read_csv(ROOT / "metrics/overlap_largest_area_manifest.csv.gz")
    overlap_selected = overlap_selected[overlap_selected.cell_id.isin(set(inner_cells.cell_id))].copy()
    assert len(overlap_selected) == len(inner_cells)
    val_ds = OverlapPatchDataset(overlap_selected, mean, std)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GHISTCellType().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    epoch_results = []
    checkpoint_dir = ROOT / "models/pilot_fold0_official"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.param_groups[0]["lr"] = 1e-3 * (1 - (epoch - 1) / 50)
        total_loss = total_pixel = total_cell = 0.0
        n_steps = 0
        for images, masks, targets, _, label_maps, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images, masks)
            pixel_loss, cell_loss, _ = compute_losses(output, targets, label_maps)
            loss = pixel_loss + cell_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            total_pixel += float(pixel_loss.detach())
            total_cell += float(cell_loss.detach())
            n_steps += 1
            if n_steps % 100 == 0:
                print(f"epoch={epoch} step={n_steps}/{len(train_loader)} loss={total_loss/n_steps:.4f}", flush=True)
        inner_metrics = evaluate_model(model, val_ds, labels, device, batch_size=8)
        if inner_metrics["n_cells"] != len(inner_cells):
            raise RuntimeError("Inner validation prediction count mismatch")
        row = {"epoch": epoch, "train_total_loss": total_loss / n_steps,
               "train_pixel_loss": total_pixel / n_steps,
               "train_cell_loss": total_cell / n_steps,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "max_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
               **inner_metrics}
        epoch_results.append(row)
        torch.save({"model": model.state_dict(), "epoch": epoch, "fold": fold,
                    "variant": "GHIST_CT_OFFICIAL", "mean": mean, "std": std,
                    "ghist_commit": "917456be305fc82e92293ea272812e79675e821c"},
                   checkpoint_dir / f"epoch_{epoch:02d}.pth")
        (ROOT / "qc/pilot_epoch_metrics.json").write_text(json.dumps(epoch_results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if not epoch_results[-1]["train_total_loss"] < epoch_results[0]["train_total_loss"]:
        raise RuntimeError("Pilot loss did not decrease across five epochs")
    summary = {"status": "PILOT_GATE_PASS", "fold": fold,
               "variant": "GHIST_CT_OFFICIAL", "epochs": epochs,
               "inner_validation_batch": inner_batch,
               "outer_validation_batches_untouched": outer_val_batches,
               "training_cells": len(active), "inner_validation_cells": len(inner_cells),
               "hed_stain_augmentation": "disabled; official stainlib unavailable",
               "mixed_precision": "disabled for initial numerical-stability pilot",
               "epoch_metrics": epoch_results}
    (ROOT / "qc/pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.pilot:
        pilot(args.fold, args.epochs)
    elif args.smoke:
        smoke(args.fold)
    else:
        raise SystemExit("Full training is gated on audited overlapping validation; use --smoke for now")
