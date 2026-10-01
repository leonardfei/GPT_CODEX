#!/usr/bin/env python3
"""Split-safe GHIST cell-typing data and one-batch loss smoke test.

The full pilot/CV entry point will be enabled only after overlap-aware inner
validation is implemented and audited. This script never uses expression data.
"""
from __future__ import annotations

import argparse
import copy
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, average_precision_score, roc_auc_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

from task015_ghist_celltype_model import GHISTCellType
from task015_evaluate_ghist_celltype import OverlapPatchDataset, evaluate_model, true_binary_probe

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


def mask_qc():
    canonical = pd.read_csv(CANON)
    patches = pd.read_csv(ROOT / "metrics/patch_manifest.csv.gz")
    labels = dict(zip(canonical.cell_id, canonical.class_id))
    contours = load_contours()
    ds = GHISTPatchDataset(patches, contours, labels, set(labels),
                           np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32),
                           augment=False)
    missing = []
    n_visible = n_labeled_pixels = 0
    for idx in range(len(ds)):
        _, mask, target, cell_map, label_map, image = ds[idx]
        visible = set(torch.unique(mask).tolist()) - {0}
        for local_id, cell_id in cell_map.items():
            if local_id not in visible:
                missing.append({"cell_id": cell_id, "image": image, "instance_id": local_id})
            else:
                n_visible += 1
        n_labeled_pixels += int((target > 0).sum())
    audit = {"canonical_cells": len(canonical), "visible_canonical_instances": n_visible,
             "missing_or_overwritten_instances": len(missing),
             "missing_examples": missing[:20], "labeled_nucleus_pixels": n_labeled_pixels,
             "status": "PASS" if not missing and n_visible == len(canonical) else "FAIL"}
    (ROOT / "qc/raster_mask_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    if audit["status"] != "PASS":
        raise RuntimeError("Not all canonical cells remain visible after contour rasterization")


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


def train_one_epoch(model, loader, optimizer, device, pixel_weights, cell_weights, epoch):
    model.train()
    optimizer.param_groups[0]["lr"] = 1e-3 * (1 - (epoch - 1) / 50)
    total = pixel_total = cell_total = 0.0
    for step, (images, masks, targets, _, label_maps, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(images, masks)
        cell_targets = torch.tensor([label_maps[b].get(local_id, IGNORE)
                                     for b, local_id in output["ordered_ids"]],
                                    dtype=torch.long, device=device)
        if not bool((cell_targets != IGNORE).any()):
            raise RuntimeError("Training batch has no supervised cells")
        pixel_loss = F.cross_entropy(output["pixel_logits"], targets,
                                     weight=pixel_weights, ignore_index=IGNORE)
        cell_loss = F.cross_entropy(output["cell_logits"], cell_targets,
                                    weight=cell_weights, ignore_index=IGNORE)
        loss = pixel_loss + cell_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CV loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        pixel_total += float(pixel_loss.detach())
        cell_total += float(cell_loss.detach())
        if step % 100 == 0:
            print(f"train_epoch={epoch} step={step}/{len(loader)} mean_loss={total/step:.5f}", flush=True)
    return {"epoch": epoch, "total_loss": total / len(loader),
            "pixel_loss": pixel_total / len(loader),
            "cell_loss": cell_total / len(loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "max_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None}


def make_loader(patches, contours, labels, active_ids, mean, std, shuffle, workers=4):
    ds = GHISTPatchDataset(patches, contours, labels, active_ids, mean, std,
                           augment=shuffle)
    return DataLoader(ds, batch_size=8, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate, pin_memory=True,
                      generator=torch.Generator().manual_seed(SEED))


def extract_training_embeddings(model, loader, active_ids, labels, device):
    model.eval()
    rows = {}
    with torch.inference_mode():
        for images, masks, _, cell_maps, _, _ in loader:
            output = model(images.to(device), masks.to(device))
            emb = output["embeddings"].float().cpu().numpy()
            for idx, (batch_idx, instance_id) in enumerate(output["ordered_ids"]):
                cell_id = cell_maps[batch_idx].get(instance_id)
                if cell_id not in active_ids:
                    continue
                if cell_id in rows:
                    raise RuntimeError(f"Repeated training embedding: {cell_id}")
                rows[cell_id] = (int(labels[cell_id]), emb[idx])
    if set(rows) != active_ids:
        raise RuntimeError(f"Training embeddings missing {len(active_ids - set(rows))} cells")
    ids = sorted(rows)
    return ids, np.asarray([rows[i][0] for i in ids], dtype=int), np.stack([rows[i][1] for i in ids])


def seven_class_metrics(y, p, variant, fold):
    pred = p.argmax(axis=1)
    per = []
    for k, name in enumerate(["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]):
        precision, recall, ff, _ = precision_recall_fscore_support(y, pred, labels=[k], zero_division=0)
        per.append({"variant": variant, "fold": fold, "class_id": k, "class_name": name,
                    "precision": float(precision[0]), "recall": float(recall[0]),
                    "f1": float(ff[0]), "auroc": float(roc_auc_score(y == k, p[:, k])),
                    "auprc": float(average_precision_score(y == k, p[:, k])),
                    "support": int((y == k).sum())})
    f = np.asarray([r["f1"] for r in per])
    row = {"variant": variant, "fold": fold,
           "accuracy": float(accuracy_score(y, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f.mean()),
           "macro_auprc": float(np.mean([r["auprc"] for r in per])),
           "macro_auroc": float(np.mean([r["auroc"] for r in per])),
           "weighted_f1": float(f1_score(y, pred, average="weighted", labels=list(range(7)), zero_division=0)),
           "lowest_three_f1": float(np.sort(f)[:3].mean()),
           "neutrophil_precision": per[3]["precision"],
           "neutrophil_recall": per[3]["recall"],
           "neutrophil_f1": per[3]["f1"],
           "neutrophil_auroc": per[3]["auroc"],
           "neutrophil_auprc": per[3]["auprc"]}
    return row, per


def run_cv(fold: int, variant: str, gpu: int):
    gate_path = ROOT / "qc/pilot_summary.json"
    if not gate_path.exists() or json.loads(gate_path.read_text()).get("status") != "PILOT_GATE_PASS":
        raise RuntimeError("Full CV is blocked until the five-epoch pilot passes")
    if variant not in {"GHIST_CT_OFFICIAL", "GHIST_CT_BALANCED"} or fold not in range(5):
        raise ValueError("Unknown variant or fold")
    seed_all(SEED + fold)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    fold_df, inner_train_batches, inner_batch, outer_val_batches = fold_roles(fold)
    outer_train_df = fold_df[fold_df.role == "train"]
    outer_train_ids = set(outer_train_df.cell_id)
    inner_train_ids = set(fold_df.loc[fold_df.batch.isin(inner_train_batches), "cell_id"])
    inner_val_ids = set(fold_df.loc[fold_df.batch == inner_batch, "cell_id"])
    outer_val_ids = set(fold_df.loc[fold_df.role == "val", "cell_id"])
    if (inner_train_ids & inner_val_ids) or (outer_train_ids & outer_val_ids):
        raise RuntimeError("Fold split leakage")
    canonical = pd.read_csv(CANON)
    labels = dict(zip(canonical.cell_id, canonical.class_id))
    patches = pd.read_csv(ROOT / "metrics/patch_manifest.csv.gz")
    contours = load_contours()
    balanced = variant == "GHIST_CT_BALANCED"
    counts = outer_train_df.class_id.value_counts().reindex(range(7), fill_value=0)
    if (counts == 0).any():
        raise RuntimeError("An outer-training class is missing")
    weights = len(outer_train_df) / (7 * counts.to_numpy(dtype=np.float64))
    cell_weights = torch.tensor(weights, dtype=torch.float32, device=device) if balanced else None
    pixel_weights = torch.tensor([1.0] + weights.tolist(), dtype=torch.float32, device=device) if balanced else None
    work = ROOT / "work" / f"{variant}_fold{fold}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "class_weights.json").write_text(json.dumps({"outer_training_counts": counts.tolist(),
        "cell_weights": weights.tolist() if balanced else [1.0] * 7,
        "pixel_weights": [1.0] + (weights.tolist() if balanced else [1.0] * 7),
        "source": "outer-training cells only"}, indent=2) + "\n")
    selected = pd.read_csv(ROOT / "metrics/overlap_largest_area_manifest.csv.gz")
    inner_patches = patches[(patches.batch.isin(inner_train_batches)) & (patches.n_canonical > 0)].copy()
    inner_mean, inner_std = training_normalization(inner_patches, work / "inner_training_rgb.json")
    inner_loader = make_loader(inner_patches, contours, labels, inner_train_ids,
                               inner_mean, inner_std, shuffle=True)
    inner_selected = selected[selected.cell_id.isin(inner_val_ids)].copy()
    if len(inner_selected) != len(inner_val_ids):
        raise RuntimeError("Incomplete inner-validation cell selection")
    inner_ds = OverlapPatchDataset(inner_selected, inner_mean, inner_std)
    model = GHISTCellType().to(device)
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    history = []
    checkpoint_dir = work / "selection_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 51):
        train_row = train_one_epoch(model, inner_loader, optimizer, device,
                                    pixel_weights, cell_weights, epoch)
        val_row = evaluate_model(model, inner_ds, labels, device, batch_size=8)
        if val_row["n_cells"] != len(inner_val_ids):
            raise RuntimeError("Inner-validation prediction count mismatch")
        history.append({**train_row, **{f"inner_{k}": v for k, v in val_row.items()}})
        torch.save({"model": model.state_dict(), "epoch": epoch, "fold": fold,
                    "variant": variant, "mean": inner_mean, "std": inner_std},
                   checkpoint_dir / f"epoch_{epoch:02d}.pth")
        (work / "selection_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"phase": "selection", "fold": fold, "variant": variant,
                          **history[-1]}), flush=True)
    best = max(history, key=lambda r: (r["inner_macro_f1"],
                                       r["inner_neutrophil_auprc"],
                                       -r["inner_cross_entropy"]))
    selected_epochs = best["epoch"]
    (work / "selected_epoch.json").write_text(json.dumps({"epoch": selected_epochs,
         "criterion": "inner Macro-F1, then Neutrophil AUPRC, then lower cross-entropy",
         "inner_validation_batch": inner_batch, "selected_history": best}, indent=2) + "\n")
    del inner_loader, inner_ds, model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Refit begins with the exact saved initialization and uses all outer-training batches.
    outer_patches = patches[(patches.batch.isin(set(outer_train_df.batch))) & (patches.n_canonical > 0)].copy()
    outer_mean, outer_std = training_normalization(outer_patches, work / "outer_training_rgb.json")
    outer_loader = make_loader(outer_patches, contours, labels, outer_train_ids,
                               outer_mean, outer_std, shuffle=True)
    model = GHISTCellType().to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    refit_history = []
    for epoch in range(1, selected_epochs + 1):
        row = train_one_epoch(model, outer_loader, optimizer, device,
                              pixel_weights, cell_weights, epoch)
        refit_history.append(row)
        (work / "refit_history.json").write_text(json.dumps(refit_history, indent=2) + "\n")
        print(json.dumps({"phase": "refit", "fold": fold, "variant": variant, **row}), flush=True)
    checkpoint = ROOT / "models" / f"{variant}_fold{fold}_refit.pth"
    torch.save({"model": model.state_dict(), "variant": variant, "fold": fold,
                "selected_epochs": selected_epochs, "mean": outer_mean, "std": outer_std,
                "ghist_commit": "917456be305fc82e92293ea272812e79675e821c"}, checkpoint)
    outer_selected = selected[selected.cell_id.isin(outer_val_ids)].copy()
    if len(outer_selected) != len(outer_val_ids):
        raise RuntimeError("Incomplete outer-validation cell selection")
    outer_ds = OverlapPatchDataset(outer_selected, outer_mean, outer_std)
    outer_metrics, records = evaluate_model(model, outer_ds, labels, device,
                                             batch_size=8, return_predictions=True)
    if outer_metrics["n_cells"] != len(outer_val_ids):
        raise RuntimeError("Outer-validation prediction count mismatch")
    val_ids = [r[0] for r in records]
    y_val = np.asarray([r[1] for r in records], dtype=int)
    p_val = np.stack([r[2] for r in records])
    z_val = np.stack([r[3] for r in records])
    metrics, per_class = seven_class_metrics(y_val, p_val, variant, fold)
    train_eval_loader = make_loader(outer_patches, contours, labels, outer_train_ids,
                                    outer_mean, outer_std, shuffle=False)
    train_ids, y_train, z_train = extract_training_embeddings(model, train_eval_loader,
                                                               outer_train_ids, labels, device)
    binary = []
    for comparison, pos, neg in [("neutrophil_vs_myeloid", 3, 2),
                                 ("neutrophil_vs_tb", 3, 5)]:
        binary.append({"variant": variant, "fold": fold, "comparison": comparison,
                       **true_binary_probe(z_train, y_train, z_val, y_val, fold, pos, neg)})
    prediction_path = ROOT / "metrics" / f"{variant}_fold{fold}_oof_predictions.csv.gz"
    prediction = pd.DataFrame({"cell_id": val_ids, "class_id": y_val})
    for k in range(7):
        prediction[f"prob_{k}"] = p_val[:, k]
    prediction.to_csv(prediction_path, index=False, compression="gzip")
    np.savez_compressed(ROOT / "work" / f"{variant}_fold{fold}_embeddings.npz",
                        train_cell_ids=np.asarray(train_ids), train_y=y_train, train_z=z_train,
                        val_cell_ids=np.asarray(val_ids), val_y=y_val, val_z=z_val)
    result = {"status": "FOLD_COMPLETE", "variant": variant, "fold": fold,
              "selected_epochs": selected_epochs, "outer_validation_batches": outer_val_batches,
              "outer_val_metrics": metrics, "per_class": per_class,
              "true_binary": binary, "outer_eval_qc": outer_metrics,
              "checkpoint": str(checkpoint), "predictions": str(prediction_path)}
    (ROOT / "metrics" / f"{variant}_fold{fold}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mask-qc", action="store_true")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--variant", choices=["GHIST_CT_OFFICIAL", "GHIST_CT_BALANCED"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.cv:
        if not args.variant:
            parser.error("--cv requires --variant")
        run_cv(args.fold, args.variant, args.gpu)
    elif args.mask_qc:
        mask_qc()
    elif args.pilot:
        pilot(args.fold, args.epochs)
    elif args.smoke:
        smoke(args.fold)
    else:
        raise SystemExit("Full training is gated on audited overlapping validation; use --smoke for now")
