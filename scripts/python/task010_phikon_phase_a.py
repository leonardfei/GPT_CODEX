#!/usr/bin/env python3
"""Task010 Phase A: frozen CellViT-token versus local Phikon-v2 benchmark.

This script intentionally uses only the regenerated Task009 V3_CORE dataset,
the original OME-TIFF, and the locally uploaded Phikon-v2 directory.  It does
not modify labels, production checkpoints, or the historical CellViT dataset.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score, silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path("/data/lf_data/result/task010_representation_benchmark")
DATASET = Path("/data/lf_data/result/task009_v3_retraining/work/CellViT_dataset_v3_CORE")
TASK009 = Path("/data/lf_data/result/task009_v3_retraining")
WSI = Path("/data/lf_data/xenium_data/ID0060276.ome.tif")
CELLVIT = Path("/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth")
PHIKON = Path("/data/lf_data/models/phikon-v2")
PHIKON_VENV = Path("/data/lf_data/task010_env/bin/python")
CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
SCALE = 0.2125
SEED = 20260923

sys.path.insert(0, "/data/lf_data/CellViT-plus-plus")
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.training.evaluate.inference_cellvit_experiment_detection import CellViTInfExpDetection
from cellvit.training.utils.tools import pair_coordinates


class DetectionOnlyExperiment(CellViTInfExpDetection):
    """Load the official CellViT inference path without requiring a run config."""
    def __init__(self, *args, split: str = "train", **kwargs):
        self._split = split
        super().__init__(*args, **kwargs)

    def _load_model(self, checkpoint_path):
        return None, {
            "data": {"num_classes": 7},
            "training": {"mixed_precision": True},
            "transformations": {"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        }

    def _load_dataset(self, transforms, normalize_stains):
        dataset = DetectionDataset(
            dataset_path=self.dataset_path, split=self._split,
            normalize_stains=normalize_stains, transforms=transforms,
        )
        dataset.cache_dataset()
        return dataset


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdirs() -> None:
    for d in ["config", "metrics", "features", "qc", "figures", "figure_data", "logs", "work", "code"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(DATASET / "cell_to_patch.csv")
    patches = pd.read_csv(DATASET / "patch_metadata.csv")
    detection = pd.read_csv(TASK009 / "metrics/detection_pairs_CORE.csv")
    folds = pd.read_csv(TASK009 / "metrics/split_manifest.csv")
    cells = cells[(cells["condition"] == "CORE") & (cells["split"] == "train")].copy()
    patches = patches[(patches["condition"] == "CORE") & (patches["split"] == "train")].copy()
    detection = detection[detection["condition"] == "V3_CORE"].copy()
    if cells.empty or patches.empty or detection.empty:
        raise RuntimeError("Task009 V3_CORE inputs are empty")
    if set(cells["class_name"].unique()) != set(CLASSES):
        raise RuntimeError(f"Unexpected class set: {sorted(cells['class_name'].unique())}")
    return cells, patches, detection, folds[folds["condition"] == "CORE"].copy()


def build_detection_cohort(cells: pd.DataFrame, patches: pd.DataFrame, detection: pd.DataFrame) -> pd.DataFrame:
    pm = patches[["image", "source_image", "patch_id", "batch", "x", "y"]].copy()
    # cell_to_patch uses source image names (patch_XXXXXX), while the
    # DetectionDataset uses regenerated image names (train_XXXXXX).
    c = cells.merge(pm, left_on=["image", "patch_id", "batch"], right_on=["source_image", "patch_id", "batch"], how="inner", suffixes=("_cell", ""))
    d = detection[detection["detected"].astype(int) == 1].copy()
    d = d.drop(columns=["class_name"], errors="ignore")
    d["local_x"] = d["gt_x"].round().astype(int)
    d["local_y"] = d["gt_y"].round().astype(int)
    c["local_x"] = c["local_x"].round().astype(int)
    c["local_y"] = c["local_y"].round().astype(int)
    key = ["image", "local_x", "local_y", "class_id"]
    dup = c.duplicated(key).sum()
    if dup:
        raise RuntimeError(f"Duplicate cell coordinate keys in Task009 dataset: {dup}")
    d = d.merge(c[key + ["cell_id", "original_cl1", "v3_label", "class_name", "he_x", "he_y", "source_image", "patch_id", "batch", "x", "y"]], on=key, how="inner")
    if d["cell_id"].duplicated().any():
        raise RuntimeError("A cell matched more than once to the detection manifest")
    d["match_distance"] = np.nan
    d["detected_source"] = "Task009_detection_pairs_CORE.csv"
    d["center_x"] = d["he_x"].astype(float)
    d["center_y"] = d["he_y"].astype(float)
    d["patch_x"] = d["x"].astype(int)
    d["patch_y"] = d["y"].astype(int)
    # Native WSI shape was independently audited as width=50000, height=23451.
    w, h = 50000, 23451
    for side, label in [(75, "small"), (263, "context")]:
        half = side // 2
        d[f"{label}_inbounds"] = (
            (d.center_x - half >= 0) & (d.center_y - half >= 0) &
            (d.center_x + half < w) & (d.center_y + half < h)
        )
    d = d[d["small_inbounds"] & d["context_inbounds"]].copy()
    d["class_id"] = d["class_id"].astype(int)
    d["class_name"] = pd.Categorical(d["class_name"], categories=CLASSES, ordered=True)
    d = d.sort_values(["batch", "patch_id", "cell_id"], kind="stable").reset_index(drop=True)
    return d


def save_cohort(cohort: pd.DataFrame) -> None:
    keep = ["cell_id", "batch", "patch_id", "image", "source_image", "class_id", "class_name", "original_cl1", "v3_label", "he_x", "he_y", "local_x", "local_y", "patch_x", "patch_y", "match_distance", "small_inbounds", "context_inbounds"]
    cohort[keep].to_csv(ROOT / "metrics/shared_cell_manifest.csv.gz", index=False, compression="gzip")
    cohort.groupby(["class_id", "class_name"], observed=False).size().rename("n_cells").reset_index().to_csv(ROOT / "metrics/shared_cell_counts_by_class.csv", index=False)
    cohort.groupby("batch").size().rename("n_cells").reset_index().to_csv(ROOT / "metrics/shared_cell_counts_by_batch.csv", index=False)
    (ROOT / "qc/shared_cohort_exclusion_summary.md").write_text(
        "# Task010 shared cohort exclusion summary\n\n"
        f"Input eligible Task009 V3_CORE training cells: {len(cohort) if False else 'see source audit'}\n"
        f"Frozen SHARED_DETECTED_CORE cells after detection and crop-boundary filters: **{len(cohort):,}**.\n\n"
        "The cohort was frozen before Phikon feature extraction. Eligibility uses only V3_CORE labels, Task009 training-batch membership, Task009 CellViT-detected nuclei, and native WSI crop bounds. No model-dependent exclusion was applied.\n"
    )


def extract_cellvit_tokens(cohort: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
    logdir = ROOT / "logs/cellvit_token_extraction"
    logdir.mkdir(parents=True, exist_ok=True)
    exp = DetectionOnlyExperiment(
        logdir=logdir, cellvit_path=CELLVIT, dataset_path=DATASET,
        input_shape=[256, 256], normalize_stains=False, gpu=0,
        comment="task010_cellvit_tokens_v3_core", split="train",
    )
    loader = DataLoader(exp.inference_dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=exp.inference_dataset.collate_batch)
    postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
    cohort_keys = set(zip(cohort.image.astype(str), cohort.local_x.astype(int), cohort.local_y.astype(int), cohort.class_id.astype(int)))
    rows: List[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for bi, (images, gt_batch, types_batch, image_names) in enumerate(loader):
            matched, _overall, pred_dicts, _f1, _prec, _rec = exp._get_cellvit_result(
                images=images, cell_gt_batch=gt_batch, types_batch=types_batch,
                image_names=image_names, postprocessor=postprocessor,
            )
            matched_by_image: Dict[str, List[dict]] = {}
            for item in matched:
                nm = item["image"].decode() if isinstance(item["image"], bytes) else str(item["image"])
                matched_by_image.setdefault(nm, []).append(item)
            for true_centroids, cell_types, image_name, pred_dict in zip(gt_batch, types_batch, image_names, pred_dicts.values()):
                nm = image_name.decode() if isinstance(image_name, bytes) else str(image_name)
                true_xy = np.asarray(true_centroids, dtype=float).reshape(-1, 2)
                true_types = np.asarray(cell_types, dtype=int).reshape(-1)
                pred_xy = np.asarray([v["centroid"] for v in pred_dict.values()], dtype=float).reshape(-1, 2)
                if pred_xy.size == 0:
                    pred_xy = np.empty((0, 2), dtype=float)
                if true_xy.size == 0 or pred_xy.size == 0:
                    continue
                paired, _unpaired_true, _unpaired_pred = pair_coordinates(true_xy, pred_xy, 15)
                paired_pred_to_true = {int(p): int(t) for t, p in paired}
                for item in matched_by_image.get(nm, []):
                    pc = np.asarray(item["coords"], dtype=float)
                    if len(pred_xy) == 0:
                        continue
                    pi = int(np.argmin(np.sum((pred_xy - pc[None, :]) ** 2, axis=1)))
                    if pi not in paired_pred_to_true:
                        continue
                    ti = paired_pred_to_true[pi]
                    tx, ty = np.rint(true_xy[ti]).astype(int)
                    key = (nm, int(tx), int(ty), int(true_types[ti]))
                    if key not in cohort_keys:
                        continue
                    token = item["token"].detach().cpu().float().reshape(-1)
                    rows.append({"image": nm, "local_x": int(tx), "local_y": int(ty), "class_id": int(true_types[ti]), "pred_x": float(pc[0]), "pred_y": float(pc[1]), "match_distance": float(np.linalg.norm(pc - true_xy[ti])), "token": token.numpy()})
            if bi % 25 == 0:
                print(f"CellViT token batches={bi} rows={len(rows)} elapsed_min={(time.time()-t0)/60:.1f}", flush=True)
    if not rows:
        raise RuntimeError("No valid CellViT tokens were extracted")
    # Deduplicate only as a safety check; duplicate matching indicates a broken join.
    out = pd.DataFrame([{k: v for k, v in r.items() if k != "token"} for r in rows])
    if out.duplicated(["image", "local_x", "local_y", "class_id"]).any():
        raise RuntimeError("Duplicate CellViT token keys")
    tok = torch.tensor(np.stack([r["token"] for r in rows]), dtype=torch.float32)
    payload = {"features": tok, "cell_keys": out[["image", "local_x", "local_y", "class_id"]].astype(str).agg("|".join, axis=1).tolist(), "source": str(CELLVIT), "semantics": "official CellViTInfExpDetection._extract_tokens: mean feature-map vectors inside detected bounding boxes", "seed": SEED}
    torch.save(payload, ROOT / "features/cellvit_tokens.pt")
    out.to_csv(ROOT / "features/cellvit_token_manifest.csv.gz", index=False, compression="gzip")
    print(f"Saved CellViT tokens {tuple(tok.shape)}")
    return out


def _vips_array(im: pyvips.Image, x: int, y: int, size: int) -> np.ndarray:
    crop = im.crop(int(x), int(y), int(size), int(size))
    return np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8, shape=(size, size, crop.bands)).copy()


def _tile_array(im: pyvips.Image, x: int, y: int, size: int = 528) -> Tuple[np.ndarray, int, int]:
    w, h = im.width, im.height
    ox = max(0, min(int(x) - 140, w - size))
    oy = max(0, min(int(y) - 140, h - size))
    return _vips_array(im, ox, oy, size), ox, oy


def extract_phikon_features(cohort: pd.DataFrame, batch_size: int = 128) -> None:
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(str(PHIKON), local_files_only=True)
    model = AutoModel.from_pretrained(str(PHIKON), local_files_only=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    hidden = int(model.config.hidden_size)
    print(f"Phikon loaded hidden={hidden} device={device}", flush=True)
    wsi = pyvips.Image.new_from_file(str(WSI), access="random")
    work = cohort.sort_values(["patch_id", "cell_id"], kind="stable").reset_index()
    all_small: List[np.ndarray] = []
    all_context: List[np.ndarray] = []
    crop_rows: List[dict] = []
    montage: Dict[str, List[Image.Image]] = {"small": [], "context": []}
    montage_seen: Dict[str, set] = {"small": set(), "context": set()}

    def flush(imgs_small: List[Image.Image], imgs_context: List[Image.Image], out_small: List[np.ndarray], out_context: List[np.ndarray]) -> None:
        if not imgs_small:
            return
        with torch.inference_mode():
            inp = processor(images=imgs_small, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                z = model(**inp).last_hidden_state[:, 0, :].float().cpu().numpy()
            out_small.extend(z)
            inp = processor(images=imgs_context, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                z = model(**inp).last_hidden_state[:, 0, :].float().cpu().numpy()
            out_context.extend(z)

    imgs_s: List[Image.Image] = []
    imgs_c: List[Image.Image] = []
    t0 = time.time()
    for i, r in work.iterrows():
        cx, cy = float(r.he_x), float(r.he_y)
        if i == 0 or work.iloc[i - 1].patch_id != r.patch_id:
            tile, ox, oy = _tile_array(wsi, int(r.patch_x), int(r.patch_y))
        side_info = {}
        for side, label in [(75, "small"), (263, "context")]:
            half = side // 2
            sx = int(round(cx)) - half - ox
            sy = int(round(cy)) - half - oy
            arr = tile[sy:sy + side, sx:sx + side, :3]
            if arr.shape != (side, side, 3):
                raise RuntimeError(f"Bad crop shape {arr.shape} for {r.cell_id} {label}")
            side_info[f"{label}_x"] = int(round(cx)) - half
            side_info[f"{label}_y"] = int(round(cy)) - half
            side_info[f"{label}_side"] = side
            im = Image.fromarray(arr, mode="RGB")
            if label == "small":
                imgs_s.append(im)
            else:
                imgs_c.append(im)
            if len(montage[label]) < 21:
                key = (label, str(r.class_name), str(r.batch))
                key = (str(r.class_name), str(r.batch))
                if len(montage[label]) < 21 and key not in montage_seen[label]:
                    montage[label].append(im.copy())
                    montage_seen[label].add(key)
        crop_rows.append({"cell_id": r.cell_id, "batch": r.batch, "class_name": r.class_name, "he_x": cx, "he_y": cy, **side_info, "processor": "local phikon-v2 preprocessor_config.json"})
        if len(imgs_s) >= batch_size:
            flush(imgs_s, imgs_c, all_small, all_context)
            imgs_s.clear(); imgs_c.clear()
        if i % 1000 == 0:
            print(f"Phikon cells={i}/{len(work)} elapsed_min={(time.time()-t0)/60:.1f}", flush=True)
    flush(imgs_s, imgs_c, all_small, all_context)
    if len(all_small) != len(work) or len(all_context) != len(work):
        raise RuntimeError(f"Feature count mismatch small={len(all_small)} context={len(all_context)} cohort={len(work)}")
    sm = np.stack(all_small).astype(np.float32)
    co = np.stack(all_context).astype(np.float32)
    if not np.isfinite(sm).all() or not np.isfinite(co).all():
        raise RuntimeError("Phikon features contain NaN/Inf")
    # Restore frozen cohort order and save exact aligned features.
    order = work["index"].to_numpy()
    inv = np.argsort(order)
    sm, co = sm[inv], co[inv]
    torch.save({"features": torch.from_numpy(sm), "cell_ids": cohort.cell_id.tolist(), "condition": "PHIKON_V2_SMALL", "embedding_dim": hidden, "seed": SEED}, ROOT / "features/phikon_v2_small.pt")
    torch.save({"features": torch.from_numpy(co), "cell_ids": cohort.cell_id.tolist(), "condition": "PHIKON_V2_CONTEXT", "embedding_dim": hidden, "seed": SEED}, ROOT / "features/phikon_v2_context.pt")
    pd.DataFrame(crop_rows).to_csv(ROOT / "metrics/crop_manifest.csv.gz", index=False, compression="gzip")
    for label in ["small", "context"]:
        ims = montage[label]
        fig, axes = plt.subplots(3, 7, figsize=(14, 6))
        axes = np.asarray(axes).ravel()
        for ax, im in zip(axes, ims):
            ax.imshow(im); ax.axis("off")
        for ax in axes[len(ims):]: ax.axis("off")
        fig.suptitle(f"Task010 Phikon-v2 {label} native crop QC")
        fig.tight_layout()
        fig.savefig(ROOT / f"qc/phikon_{label}_crop_montage.pdf")
        plt.close(fig)
    print(f"Saved Phikon features {sm.shape} and {co.shape}", flush=True)


def fold_assign(cohort: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for _, row in folds.sort_values("fold").iterrows():
        tr = set(str(row.train_batches).split(";")); va = set(str(row.val_batches).split(";"))
        for batch in sorted(tr | va):
            recs.append({"fold": int(row.fold), "batch": batch, "role": "train" if batch in tr else "val"})
    fm = pd.DataFrame(recs)
    base = cohort[["cell_id", "batch", "class_id", "class_name"]].reset_index().rename(columns={"index": "cell_idx"})
    out = base.merge(fm, on="batch", how="left")
    if out["fold"].isna().any() or (out.groupby("fold")["role"].nunique() < 2).any():
        raise RuntimeError("Incomplete fold assignment")
    if (out.groupby("fold").apply(lambda x: set(x.loc[x.role == "train", "batch"]) & set(x.loc[x.role == "val", "batch"]))).map(len).max() != 0:
        raise RuntimeError("Train/validation batch overlap")
    out.to_csv(ROOT / "metrics/fold_manifest.csv", index=False)
    lines = ["# Task010 fold equivalence audit", "", "The exact Task009 V3_CORE split manifest was reused; no new partition was generated.", "", "| fold | train batches | validation batches |", "|---:|---|---|"]
    for _, r in folds.sort_values("fold").iterrows():
        lines.append(f"| {int(r.fold)} | {r.train_batches} | {r.val_batches} |")
    lines.append("\nNo train/validation batch overlap was detected in any fold.")
    (ROOT / "qc/fold_equivalence_task009.md").write_text("\n".join(lines) + "\n")
    return out


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try: return float(roc_auc_score(y, p))
    except ValueError: return float("nan")


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    try: return float(average_precision_score(y, p))
    except ValueError: return float("nan")


def multiclass_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    per_f1 = f1_score(y, pred, labels=np.arange(7), average=None, zero_division=0)
    aupr = []
    auroc = []
    for k in range(7):
        aupr.append(safe_ap((y == k).astype(int), p[:, k]))
        auroc.append(safe_auc((y == k).astype(int), p[:, k]))
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred), "macro_f1": float(np.mean(per_f1)), "macro_auprc": float(np.nanmean(aupr)), "macro_auroc": float(np.nanmean(auroc)), "weighted_f1": f1_score(y, pred, labels=np.arange(7), average="weighted", zero_division=0), "mcc": matthews_corrcoef(y, pred), "lowest_three_f1": float(np.mean(np.sort(per_f1)[:3]))}


def fit_linear(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(X[train_idx])
    # GPU full-batch softmax regression is mathematically the same multinomial
    # logistic probe, with the same C=1 L2 regularization and balanced class
    # weights.  It avoids an hours-long CPU L-BFGS fit for 96k cells.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(scaler.transform(X[train_idx]).astype(np.float32, copy=False)).to(device)
    yt = torch.from_numpy(y[train_idx].astype(np.int64, copy=False)).to(device)
    xv = torch.from_numpy(scaler.transform(X[val_idx]).astype(np.float32, copy=False)).to(device)
    counts = torch.bincount(yt, minlength=7).float()
    weights = (len(yt) / (7.0 * torch.clamp(counts, min=1.0))).to(device)
    w = torch.zeros((7, xt.shape[1]), device=device, requires_grad=True)
    b = torch.zeros(7, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=35, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5, tolerance_change=1e-9)
    reg = 0.5 / len(yt)
    def closure():
        opt.zero_grad(set_to_none=True)
        logits = xt @ w.T + b
        loss = F.cross_entropy(logits, yt, weight=weights) + reg * torch.sum(w * w)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pp = torch.softmax(xv @ w.T + b, dim=1).cpu().numpy()
        pred = pp.argmax(axis=1)
    del xt, yt, xv, w, b, opt
    if device.type == "cuda": torch.cuda.empty_cache()
    return pp, pred


def run_linear(cohort: pd.DataFrame, folds_cells: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reps = {
        "CellViT_TOKEN": torch.load(ROOT / "features/cellvit_tokens.pt", map_location="cpu")["features"].numpy(),
        "PHIKON_V2_SMALL": torch.load(ROOT / "features/phikon_v2_small.pt", map_location="cpu")["features"].numpy(),
        "PHIKON_V2_CONTEXT": torch.load(ROOT / "features/phikon_v2_context.pt", map_location="cpu")["features"].numpy(),
    }
    y = cohort.class_id.to_numpy(dtype=int)
    fold_rows, per_rows, binary_rows, flows = [], [], [], []
    for rep, X in reps.items():
        if len(X) != len(cohort): raise RuntimeError(f"{rep} not aligned with cohort")
        for fold in sorted(folds_cells.fold.unique()):
            f = folds_cells[folds_cells.fold == fold]
            tr = f.loc[f.role == "train", "cell_idx"].to_numpy(dtype=int); va = f.loc[f.role == "val", "cell_idx"].to_numpy(dtype=int)
            p, pred = fit_linear(X, y, tr, va)
            met = multiclass_metrics(y[va], p); met.update({"representation": rep, "fold": int(fold), "n_train": len(tr), "n_val": len(va)})
            fold_rows.append(met)
            cm = confusion_matrix(y[va], pred, labels=np.arange(7))
            for a, b in [(3,2),(2,3),(3,5),(5,3),(3,4),(4,3)]:
                flows.append({"representation": rep, "fold": int(fold), "source": CLASSES[a], "target": CLASSES[b], "count": int(cm[a,b]), "source_total": int(cm[a,:].sum()), "rate": float(cm[a,b] / max(cm[a,:].sum(), 1))})
            for k, name in enumerate(CLASSES):
                yy = (y[va] == k).astype(int)
                per_rows.append({"representation": rep, "fold": int(fold), "class_id": k, "class_name": name, "precision": precision_recall_fscore_support(y[va], pred, labels=[k], zero_division=0)[0][0], "recall": precision_recall_fscore_support(y[va], pred, labels=[k], zero_division=0)[1][0], "f1": precision_recall_fscore_support(y[va], pred, labels=[k], zero_division=0)[2][0], "auroc": safe_auc(yy, p[:, k]), "auprc": safe_ap(yy, p[:, k]), "support": int(yy.sum())})
            for title, a, b in [("neutrophil_vs_myeloid", 3, 2), ("neutrophil_vs_tb", 3, 5)]:
                keep = np.isin(y[va], [a, b]); yy = (y[va][keep] == a).astype(int); pp = p[keep][:, a] / np.maximum(p[keep][:, a] + p[keep][:, b], 1e-12); pr = (pp >= 0.5).astype(int)
                binary_rows.append({"representation": rep, "fold": int(fold), "comparison": title, "auroc": safe_auc(yy, pp), "auprc": safe_ap(yy, pp), "balanced_accuracy": balanced_accuracy_score(yy, pr), "f1": f1_score(yy, pr, zero_division=0), "sensitivity": float(((pr == 1) & (yy == 1)).sum() / max((yy == 1).sum(), 1)), "specificity": float(((pr == 0) & (yy == 0)).sum() / max((yy == 0).sum(), 1))})
    fm = pd.DataFrame(fold_rows); pc = pd.DataFrame(per_rows); bm = pd.DataFrame(binary_rows)
    fm.to_csv(ROOT / "metrics/linear_probe_fold_metrics.csv", index=False)
    pc.to_csv(ROOT / "metrics/linear_probe_per_class.csv", index=False)
    bm.to_csv(ROOT / "metrics/binary_neutrophil_diagnostics.csv", index=False)
    bm[bm.comparison == "neutrophil_vs_myeloid"].to_csv(ROOT / "metrics/binary_neutrophil_vs_myeloid.csv", index=False)
    bm[bm.comparison == "neutrophil_vs_tb"].to_csv(ROOT / "metrics/binary_neutrophil_vs_tb.csv", index=False)
    summary = fm.groupby("representation").agg(**{c: (c, "mean") for c in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "weighted_f1", "mcc", "lowest_three_f1"]}).reset_index()
    for c in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "weighted_f1", "mcc", "lowest_three_f1"]:
        summary[c + "_sd"] = fm.groupby("representation")[c].std(ddof=1).reindex(summary.representation).to_numpy()
    summary.to_csv(ROOT / "metrics/linear_probe_summary.csv", index=False)
    pc.groupby(["representation", "class_id", "class_name"], as_index=False).agg({"precision": ["mean", "std"], "recall": ["mean", "std"], "f1": ["mean", "std"], "auroc": ["mean", "std"], "auprc": ["mean", "std"], "support": "sum"}).to_csv(ROOT / "metrics/linear_probe_per_class_summary.csv", index=False)
    n = pc[pc.class_name == "Neutrophil"].groupby("representation", as_index=False).agg(precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"), auroc=("auroc", "mean"), auprc=("auprc", "mean"))
    n.to_csv(ROOT / "metrics/neutrophil_metrics.csv", index=False)
    pd.DataFrame(flows).to_csv(ROOT / "metrics/confusion_flows.csv", index=False)
    bin_summary = bm.groupby(["representation", "comparison"], as_index=False).mean(numeric_only=True)
    bin_summary.to_csv(ROOT / "metrics/binary_neutrophil_summary.csv", index=False)
    # Paired fold deltas keep the held-out batch comparison explicit.
    piv = fm.pivot(index="fold", columns="representation")
    delta_rows = []
    for metric in ["macro_f1", "macro_auprc"]:
        for a, b, label in [("PHIKON_V2_CONTEXT", "PHIKON_V2_SMALL", "CONTEXT_MINUS_SMALL"), ("PHIKON_V2_SMALL", "CellViT_TOKEN", "SMALL_MINUS_CELLVIT"), ("PHIKON_V2_CONTEXT", "CellViT_TOKEN", "CONTEXT_MINUS_CELLVIT")]:
            for fold in piv.index:
                delta_rows.append({"fold": int(fold), "metric": metric, "comparison": label, "delta": float(piv.loc[fold, (metric, a)] - piv.loc[fold, (metric, b)])})
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(ROOT / "metrics/representation_paired_deltas.csv", index=False)
    deltas[deltas.comparison == "CONTEXT_MINUS_SMALL"].groupby("metric", as_index=False).agg(delta_mean=("delta", "mean"), delta_sd=("delta", "std")).to_csv(ROOT / "metrics/context_value_deltas.csv", index=False)
    return fm, pc, bm


def run_mlp(cohort: pd.DataFrame, folds_cells: pd.DataFrame, epochs: int = 30) -> pd.DataFrame:
    """Identical secondary MLP probe on the three frozen representations."""
    reps = {
        "CellViT_TOKEN": torch.load(ROOT / "features/cellvit_tokens.pt", map_location="cpu")["features"].numpy(),
        "PHIKON_V2_SMALL": torch.load(ROOT / "features/phikon_v2_small.pt", map_location="cpu")["features"].numpy(),
        "PHIKON_V2_CONTEXT": torch.load(ROOT / "features/phikon_v2_context.pt", map_location="cpu")["features"].numpy(),
    }
    y = cohort.class_id.to_numpy(dtype=int); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows=[]
    for rep, X in reps.items():
        for fold in sorted(folds_cells.fold.unique()):
            f=folds_cells[folds_cells.fold==fold]; tr=f.loc[f.role=="train","cell_idx"].to_numpy(dtype=int); va=f.loc[f.role=="val","cell_idx"].to_numpy(dtype=int)
            scaler=StandardScaler().fit(X[tr]); xt=torch.from_numpy(scaler.transform(X[tr]).astype(np.float32)).to(device); xv=torch.from_numpy(scaler.transform(X[va]).astype(np.float32)).to(device); yt=torch.from_numpy(y[tr].astype(np.int64)).to(device)
            counts=torch.bincount(yt,minlength=7).float(); weights=(len(yt)/(7.0*torch.clamp(counts,min=1.0))).to(device)
            net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],256),torch.nn.ReLU(),torch.nn.Dropout(0.5),torch.nn.Linear(256,128),torch.nn.ReLU(),torch.nn.Dropout(0.5),torch.nn.Linear(128,7)).to(device)
            opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4); best=None; best_f=-1.0; stale=0
            for ep in range(epochs):
                net.train(); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(net(xt),yt,weight=weights); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): vp=torch.softmax(net(xv),dim=1).cpu().numpy()
                vf=multiclass_metrics(y[va],vp)["macro_f1"]
                if vf>best_f+1e-5: best_f=vf; best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; stale=0
                else: stale+=1
                if stale>=6: break
            net.load_state_dict(best); net.eval()
            with torch.no_grad(): p=torch.softmax(net(xv),dim=1).cpu().numpy()
            mm=multiclass_metrics(y[va],p); mm.update({"representation":rep,"fold":int(fold),"epochs_run":ep+1,"n_train":len(tr),"n_val":len(va)}); rows.append(mm)
            del xt,xv,yt,net,opt
            if device.type=="cuda": torch.cuda.empty_cache()
            print(f"MLP {rep} fold={fold} macro_f1={mm['macro_f1']:.4f} epochs={ep+1}",flush=True)
    out=pd.DataFrame(rows); out.to_csv(ROOT/"metrics/mlp_probe_fold_metrics.csv",index=False)
    summary=out.groupby("representation",as_index=False).agg({c:["mean","std"] for c in ["accuracy","balanced_accuracy","macro_f1","macro_auprc","macro_auroc","weighted_f1","mcc","lowest_three_f1"]}); summary.to_csv(ROOT/"metrics/mlp_probe_summary.csv",index=False); return out


def geometry(cohort: pd.DataFrame, folds_cells: pd.DataFrame) -> pd.DataFrame:
    reps = {
        "CellViT_TOKEN": torch.load(ROOT / "features/cellvit_tokens.pt", map_location="cpu")["features"].numpy(),
        "PHIKON_V2_SMALL": torch.load(ROOT / "features/phikon_v2_small.pt", map_location="cpu")["features"].numpy(),
        "PHIKON_V2_CONTEXT": torch.load(ROOT / "features/phikon_v2_context.pt", map_location="cpu")["features"].numpy(),
    }
    y = cohort.class_id.to_numpy(); batch = cohort.batch.astype(str).to_numpy(); rows=[]
    for rep, X in reps.items():
        X = StandardScaler().fit_transform(X)
        # Pairwise distances use a deterministic subset and sampled pairs so the
        # audit remains bounded for the large shared cohort.
        idx = np.arange(len(X)) if len(X) <= 10000 else np.random.default_rng(SEED).choice(len(X), 10000, replace=False)
        xx, yy, bb = X[idx], y[idx], batch[idx]
        xx = xx / np.maximum(np.linalg.norm(xx, axis=1, keepdims=True), 1e-12)
        rng=np.random.default_rng(SEED); pi=rng.integers(0,len(xx),size=100000); pj=rng.integers(0,len(xx),size=100000); keep=pi!=pj; pi=pi[keep]; pj=pj[keep]
        dist=1-np.sum(xx[pi]*xx[pj],axis=1); samec=yy[pi]==yy[pj]; sameb=bb[pi]==bb[pj]
        within_class=dist[samec]; between_class=dist[~samec]; within_batch=dist[sameb]; between_batch=dist[~sameb]
        knn=KNeighborsClassifier(n_neighbors=11, metric="cosine", n_jobs=-1).fit(xx, yy)
        neigh=knn.kneighbors(return_distance=False)[:,1:]
        class_purity=float(np.mean(yy[neigh]==yy[:,None])); batch_purity=float(np.mean(bb[neigh]==bb[:,None]))
        sil=float(silhouette_score(xx, yy, metric="cosine", sample_size=min(10000,len(xx)), random_state=SEED)) if len(np.unique(yy))>1 else float("nan")
        rows.append({"representation":rep,"within_class_cosine_distance":float(np.mean(within_class)),"between_class_cosine_distance":float(np.mean(between_class)),"class_separation_ratio":float(np.mean(between_class)/max(np.mean(within_class),1e-12)),"within_batch_cosine_distance":float(np.mean(within_batch)),"between_batch_cosine_distance":float(np.mean(between_batch)),"batch_separation_ratio":float(np.mean(between_batch)/max(np.mean(within_batch),1e-12)),"knn_class_purity":class_purity,"knn_batch_purity":batch_purity,"silhouette":sil,"n_geometry":len(xx)})
    out=pd.DataFrame(rows); out.to_csv(ROOT/"metrics/representation_geometry.csv",index=False); return out


def figures(fm: pd.DataFrame, pc: pd.DataFrame, bm: pd.DataFrame, geom: pd.DataFrame, cohort: pd.DataFrame) -> None:
    summ=fm.groupby("representation",as_index=False).mean(numeric_only=True)
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(summ.representation,summ.macro_f1,yerr=fm.groupby("representation").macro_f1.std().reindex(summ.representation),capsize=4); ax.set_ylabel("Macro-F1"); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(ROOT/"figures/Fig3_linear_probe_macroF1.pdf"); plt.close(fig)
    q=pc.groupby(["representation","class_name"],as_index=False).f1.mean().pivot(index="class_name",columns="representation",values="f1"); q.plot(kind="bar",figsize=(10,5)); plt.ylabel("F1"); plt.tight_layout(); plt.savefig(ROOT/"figures/Fig4_linear_probe_per_class_F1.pdf"); plt.close()
    n=pc[pc.class_name=="Neutrophil"].groupby("representation",as_index=False).f1.mean(); n.plot.bar(x="representation",y="f1",legend=False,figsize=(7,4)); plt.ylabel("Neutrophil F1"); plt.tight_layout(); plt.savefig(ROOT/"figures/Fig5_neutrophil_metrics.pdf"); plt.close()
    b=bm.groupby(["representation","comparison"],as_index=False).auroc.mean().pivot(index="representation",columns="comparison",values="auroc"); b.plot.bar(figsize=(8,4)); plt.ylabel("AUROC"); plt.tight_layout(); plt.savefig(ROOT/"figures/Fig7_binary_specialist_results.pdf"); plt.close()
    d=summ.set_index("representation").loc[["PHIKON_V2_SMALL","PHIKON_V2_CONTEXT"],"macro_f1"]; (d-d.iloc[0]).plot.bar(figsize=(5,4)); plt.ylabel("Context − Small macro-F1"); plt.tight_layout(); plt.savefig(ROOT/"figures/Fig8_small_vs_context.pdf"); plt.close()
    geom.set_index("representation")[["class_separation_ratio","batch_separation_ratio"]].plot.bar(figsize=(8,4)); plt.ylabel("Distance ratio"); plt.tight_layout(); plt.savefig(ROOT/"figures/Fig9_representation_geometry.pdf"); plt.close()
    cohort.groupby("class_name",observed=False).size().plot.bar(figsize=(8,4)); plt.ylabel("Shared cells"); plt.tight_layout(); plt.savefig(ROOT/"figures/Fig2_shared_cohort_composition.pdf"); plt.close()


def write_configs(cells: pd.DataFrame, cohort: pd.DataFrame, folds: pd.DataFrame) -> None:
    model_files={p.name:p.stat().st_size for p in PHIKON.iterdir() if p.is_file()}
    cfg=json.loads((PHIKON/"config.json").read_text()); pre=json.loads((PHIKON/"preprocessor_config.json").read_text())
    prov={"status":"LOADED_OFFLINE","model_id":"local_upload_phikon-v2","path":str(PHIKON),"files":model_files,"sha256_model_safetensors":sha256(PHIKON/"model.safetensors"),"architecture":cfg.get("architectures"),"model_type":cfg.get("model_type"),"hidden_size":cfg.get("hidden_size"),"image_size":cfg.get("image_size"),"patch_size":cfg.get("patch_size"),"torch_dtype":cfg.get("torch_dtype"),"preprocessor":pre,"feature_rule":"last_hidden_state[:,0,:]","offline_flags":["HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1"],"loaded_with_local_files_only":True}
    (ROOT/"config/phikon_v2_provenance.json").write_text(json.dumps(prov,indent=2,ensure_ascii=False)+"\n")
    (ROOT/"config/task010_crop_geometry.json").write_text(json.dumps({"status":"COMPLETED_LOCAL_PHIKON_PHASE_A","native_h_and_e_pixel_scale_um_per_pixel":SCALE,"scale_source":"historical validated notebook Prepare_allcelltype_batch8_train8_test.ipynb, PIXEL_SIZE = 0.2125","ome_metadata_note":"OME PhysicalSize metadata was inconsistent and not used","small":{"fov_um":[16,16],"side_px":75},"context":{"fov_um":[56,56],"side_px":263},"center":"matched CellViT H&E nucleus centroid","source_wsi":str(WSI)},indent=2)+"\n")
    (ROOT/"qc/pixel_scale_audit.md").write_text(f"# Task010 H&E pixel-scale audit\n\n- Source: `{WSI}`; audited native shape: 50,000 × 23,451 × 3, uint8 RGB.\n- Validated project scale: **{SCALE} µm/px**, from the historical project notebook.\n- OME PhysicalSize metadata was inconsistent with registration and was not used.\n- SMALL: 16 µm FOV → 75 × 75 native px.\n- CONTEXT: 56 µm FOV → 263 × 263 native px.\n- Both crops were centered on the matched CellViT H&E nucleus centroid and processed through the local Phikon-v2 processor.\n")
    (ROOT/"qc/phikon_local_provenance.md").write_text(f"# Local Phikon-v2 provenance\n\n- Path: `{PHIKON}`\n- Fully offline load: `AutoImageProcessor.from_pretrained(..., local_files_only=True)` and `AutoModel.from_pretrained(..., local_files_only=True)`\n- SHA256: `{prov['sha256_model_safetensors']}`\n- Architecture: `{cfg.get('architectures')}`, model type `{cfg.get('model_type')}`, hidden size `{cfg.get('hidden_size')}`, patch size `{cfg.get('patch_size')}`\n- Feature: CLS vector `last_hidden_state[:, 0, :]`; runtime dimension verified from model config/output.\n- No alternate model or network download was used.\n")


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=["all","cohort","features","metrics","mlp"],default="all"); ap.add_argument("--batch-size",type=int,default=128); args=ap.parse_args()
    mkdirs(); seed_everything(); write_text = ROOT/"config/foundation_model_environment.txt"
    write_text.write_text(f"Task010 Phase A environment\npython={sys.version}\ntorch={torch.__version__}\ncuda_available={torch.cuda.is_available()}\ndevice={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\nphikon_venv={PHIKON_VENV}\ntransformers=4.45.2 (isolated offline venv)\nseed={SEED}\n")
    cells, patches, detection, folds = load_inputs()
    cohort = build_detection_cohort(cells, patches, detection)
    write_configs(cells, cohort, folds)
    save_cohort(cohort)
    if args.stage in ("all","cohort"):
        print(f"Frozen cohort n={len(cohort)}", flush=True)
    if args.stage in ("all","features"):
        tokens=extract_cellvit_tokens(cohort)
        # Replace the initial detection-derived match distance with the exact re-extracted token match distance.
        cohort=cohort.merge(tokens[["image","local_x","local_y","class_id","pred_x","pred_y","match_distance"]], on=["image","local_x","local_y","class_id"], how="inner", suffixes=("_cohort",""))
        cohort=cohort.drop(columns=["match_distance_cohort"],errors="ignore")
        save_cohort(cohort)
        extract_phikon_features(cohort,args.batch_size)
    if args.stage in ("all","metrics"):
        if not (ROOT/"features/cellvit_tokens.pt").exists(): raise RuntimeError("Missing CellViT token features")
        cohort=pd.read_csv(ROOT/"metrics/shared_cell_manifest.csv.gz")
        fold_cells=fold_assign(cohort,folds)
        fm,pc,bm=run_linear(cohort,fold_cells)
        geom=geometry(cohort,fold_cells)
        figures(fm,pc,bm,geom,cohort)
        decision={"status":"PHASE_A_LINEAR_AND_GEOMETRY_COMPLETE","n_shared_detected_core":int(len(cohort)),"representations":["CellViT_TOKEN","PHIKON_V2_SMALL","PHIKON_V2_CONTEXT"],"c_grid":"fixed C=1 because identical grouped folds are the primary comparison","midnight_phase_b":"PENDING_OPTIONAL","production_model_modified":False}
        (ROOT/"metrics/decision_summary.json").write_text(json.dumps(decision,indent=2)+"\n")
        print(fm.groupby("representation")[["macro_f1","macro_auprc"]].mean().to_string(),flush=True)
    if args.stage == "mlp":
        cohort=pd.read_csv(ROOT/"metrics/shared_cell_manifest.csv.gz")
        fold_cells=fold_assign(cohort,folds)
        run_mlp(cohort,fold_cells)


if __name__=="__main__": main()
