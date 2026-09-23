#!/usr/bin/env python3
"""Task011: frozen Midnight local-FOV optimization and immune hierarchy.

This implementation reuses the Task010 canonical cohort and exact Task009
CORE grouped folds.  Large feature tensors and model artifacts stay on the
server; only small reviewable summaries are intended for Git.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

T10 = Path("/data/lf_data/result/task010_representation_benchmark")
ROOT = Path("/data/lf_data/result/task011_midnight_local_optimization")
TASK009 = Path("/data/lf_data/result/task009_v3_retraining")
WSI = Path("/data/lf_data/xenium_data/ID0060276.ome.tif")
MIDNIGHT = Path("/data/lf_data/models/midnight-12k")
SEED = 20260923
SCALE_UM_PER_PX = 0.2125
CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
FOVS = {12: 57, 16: 75, 20: 95, 24: 113, 32: 151}

sys.path.insert(0, str(T10 / "code"))
import task010_phikon_phase_a as base  # noqa: E402


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdirs() -> None:
    for d in ["config", "code", "features", "metrics", "models", "figures", "qc", "logs", "work"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, p))
    except ValueError:
        return float("nan")


def metric7(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(axis=1)
    f = f1_score(y, pred, labels=np.arange(7), average=None, zero_division=0)
    ap = [safe_ap((y == k).astype(int), p[:, k]) for k in range(7)]
    auc = [safe_auc((y == k).astype(int), p[:, k]) for k in range(7)]
    return {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(np.mean(f)),
        "macro_auprc": float(np.nanmean(ap)),
        "macro_auroc": float(np.nanmean(auc)),
        "weighted_f1": float(f1_score(y, pred, labels=np.arange(7), average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "lowest_three_f1": float(np.sort(f)[:3].mean()),
    }


def per_class_rows(candidate: str, fold: int, y: np.ndarray, p: np.ndarray) -> list[dict]:
    pred = p.argmax(axis=1)
    rows = []
    for k, name in enumerate(CLASSES):
        yy = (y == k).astype(int)
        pr, rc, f, _ = precision_recall_fscore_support(y, pred, labels=[k], zero_division=0)
        rows.append({
            "candidate": candidate, "fold": fold, "class_id": k, "class_name": name,
            "precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f[0]),
            "auroc": safe_auc(yy, p[:, k]), "auprc": safe_ap(yy, p[:, k]), "support": int(yy.sum()),
        })
    return rows


def binary_rows(candidate: str, fold: int, y: np.ndarray, p: np.ndarray, comparison: str, pos: int, neg: int) -> dict:
    keep = np.isin(y, [pos, neg])
    yy = (y[keep] == pos).astype(int)
    score = p[keep, 1]
    pred = (score >= 0.5).astype(int)
    pr, rc, f, _ = precision_recall_fscore_support(yy, pred, labels=[1], zero_division=0)
    return {
        "candidate": candidate, "fold": fold, "comparison": comparison,
        "auroc": safe_auc(yy, score), "auprc": safe_ap(yy, score), "f1": float(f[0]),
        "balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
        "sensitivity": float(rc[0]), "precision": float(pr[0]),
        "specificity": float(((pred == 0) & (yy == 0)).sum() / max((yy == 0).sum(), 1)),
        "n_val": int(keep.sum()),
    }


def immune_binary_row(candidate: str, fold: int, y: np.ndarray, score: np.ndarray) -> dict:
    keep = np.isin(y, [2, 3, 4, 5])
    yy = (y[keep] == 3).astype(int)
    s = score[keep]
    pred = (s >= 0.5).astype(int)
    pr, rc, f, _ = precision_recall_fscore_support(yy, pred, labels=[1], zero_division=0)
    return {"candidate": candidate, "fold": fold, "comparison": "neutrophil_vs_all_immune", "auroc": safe_auc(yy, s), "auprc": safe_ap(yy, s), "f1": float(f[0]), "balanced_accuracy": float(balanced_accuracy_score(yy, pred)), "sensitivity": float(rc[0]), "precision": float(pr[0]), "specificity": float(((pred == 0) & (yy == 0)).sum() / max((yy == 0).sum(), 1)), "n_val": int(keep.sum())}


def load_canonical() -> pd.DataFrame:
    path = T10 / "metrics/canonical_cell_order.csv.gz"
    c = pd.read_csv(path)
    required = ["canonical_index", "cell_id", "batch", "patch_id", "patch_x", "patch_y", "he_x", "he_y", "class_id", "class_name"]
    missing = [x for x in required if x not in c.columns]
    if missing:
        raise RuntimeError(f"canonical manifest missing {missing}")
    if len(c) != 96044 or c.cell_id.duplicated().any():
        raise RuntimeError(f"unexpected canonical cohort: n={len(c)} duplicate_ids={c.cell_id.duplicated().sum()}")
    if set(c.class_name.astype(str)) != set(CLASSES):
        raise RuntimeError("canonical class set changed")
    if not np.array_equal(c.canonical_index.to_numpy(), np.arange(len(c))):
        raise RuntimeError("canonical_index is not contiguous and ordered")
    return c.reset_index(drop=True)


def load_folds(c: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    sm = pd.read_csv(TASK009 / "metrics/split_manifest.csv")
    sm = sm[sm.condition == "CORE"].sort_values("fold")
    if len(sm) != 5:
        raise RuntimeError(f"expected five CORE folds, found {len(sm)}")
    out = {}
    for _, row in sm.iterrows():
        trb = set(str(row.train_batches).split(";"))
        vab = set(str(row.val_batches).split(";"))
        if trb & vab:
            raise RuntimeError(f"fold {row.fold} has train/validation batch overlap")
        tr = np.flatnonzero(c.batch.astype(str).isin(trb))
        va = np.flatnonzero(c.batch.astype(str).isin(vab))
        if len(tr) == 0 or len(va) == 0 or set(tr) & set(va):
            raise RuntimeError(f"fold {row.fold} has invalid cell assignment")
        out[int(row.fold)] = (tr, va)
    lines = ["# Task011 fold equivalence audit", "", "Exact Task009 CORE split_manifest.csv was reused.", "", "| fold | n_train | n_val | train batches | validation batches |", "|---:|---:|---:|---|---|"]
    for _, row in sm.iterrows():
        tr, va = out[int(row.fold)]
        lines.append(f"| {int(row.fold)} | {len(tr):,} | {len(va):,} | {row.train_batches} | {row.val_batches} |")
    (ROOT / "qc/fold_equivalence_task009.md").write_text("\n".join(lines) + "\n")
    return out


def check_payload(path: Path, ids: Sequence[str]) -> np.ndarray:
    p = torch.load(path, map_location="cpu")
    got = [str(x) for x in p["cell_ids"]]
    if got != list(ids):
        raise RuntimeError(f"feature IDs are not canonical: {path}")
    x = p["features"].float().numpy()
    if x.shape[0] != len(ids) or not np.isfinite(x).all():
        raise RuntimeError(f"invalid feature payload: {path} {x.shape}")
    return x


def load_existing_features(c: pd.DataFrame) -> dict[str, np.ndarray]:
    ids = c.cell_id.astype(str).tolist()
    files = {
        "CELLVIT_TOKEN_ALIGNED": T10 / "features/cellvit_tokens_aligned.pt",
        "PHIKON_V2_SMALL": T10 / "features/phikon_v2_small_aligned.pt",
        "MIDNIGHT16": T10 / "features/midnight12k_small.pt",
    }
    out = {k: check_payload(v, ids) for k, v in files.items()}
    dims = {k: int(v.shape[1]) for k, v in out.items()}
    (ROOT / "qc/input_feature_alignment.md").write_text(
        "# Task011 input-feature alignment\n\n"
        f"Canonical cells: **{len(c):,}**. Exact cell_id order was asserted for Task010 aligned CellViT, Phikon SMALL, and Midnight 16 μm tensors.\n\n"
        + "\n".join(f"- {k}: shape={v.shape}, finite=True" for k, v in out.items())
        + "\n"
    )
    (ROOT / "config/input_feature_provenance.json").write_text(json.dumps({"canonical": str(T10 / "metrics/canonical_cell_order.csv.gz"), "features": {k: {"path": str(files[k]), "shape": list(out[k].shape), "dtype": "float32"} for k in out}, "fold_source": str(TASK009 / "metrics/split_manifest.csv"), "seed": SEED}, indent=2) + "\n")
    return out


def load_midnight_model():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(MIDNIGHT), local_files_only=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def crop_tensor(arr: np.ndarray) -> torch.Tensor:
    im = Image.fromarray(arr, mode="RGB").resize((224, 224), Image.Resampling.BICUBIC)
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((x - 0.5) / 0.5)


def encode_batch(model, device, xs: list[torch.Tensor]) -> np.ndarray:
    x = torch.stack(xs).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            h = model(x).last_hidden_state
            z = torch.cat([h[:, 0, :], h[:, 1:, :].mean(dim=1)], dim=1).float()
    out = z.cpu().numpy()
    del x, h, z
    return out


def extract_fov(c: pd.DataFrame, model, device, um: int, side: int, batch_size: int = 96) -> np.ndarray:
    out_path = ROOT / f"features/midnight_fov{um}.pt"
    ids = c.cell_id.astype(str).tolist()
    if out_path.exists():
        try:
            return check_payload(out_path, ids)
        except Exception:
            out_path.unlink()
    if um == 16:
        x = check_payload(T10 / "features/midnight12k_small.pt", ids)
        torch.save({"features": torch.from_numpy(x), "cell_ids": ids, "fov_um": 16, "native_side_px": side, "source": "Task010 exact Midnight SMALL payload", "preprocessing": "resize 224, center crop 224, mean/std 0.5", "feature_rule": "CLS + mean patch tokens", "seed": SEED}, out_path)
        return x
    wsi = pyvips.Image.new_from_file(str(WSI), access="random")
    work = c.sort_values(["patch_id", "cell_id"], kind="stable").reset_index()
    rows: list[np.ndarray] = []
    xs: list[torch.Tensor] = []
    tile = None
    ox = oy = 0
    current = None
    t0 = time.time()
    for i, r in work.iterrows():
        if current != r.patch_id:
            tile, ox, oy = base._tile_array(wsi, int(r.patch_x), int(r.patch_y))
            current = r.patch_id
        half = side // 2
        sx = int(round(float(r.he_x))) - half - ox
        sy = int(round(float(r.he_y))) - half - oy
        arr = tile[sy:sy + side, sx:sx + side, :3]
        if arr.shape != (side, side, 3):
            raise RuntimeError(f"invalid FOV{um} crop {arr.shape} for {r.cell_id}")
        xs.append(crop_tensor(arr))
        if len(xs) >= batch_size:
            rows.append(encode_batch(model, device, xs))
            xs.clear()
        if i % 5000 == 0:
            print(f"FOV{um}: {i:,}/{len(work):,} elapsed_min={(time.time() - t0) / 60:.1f}", flush=True)
    if xs:
        rows.append(encode_batch(model, device, xs))
    z = np.concatenate(rows, axis=0)
    z = z[np.argsort(work["index"].to_numpy())].astype(np.float32)
    if z.shape != (len(c), 3072) or not np.isfinite(z).all():
        raise RuntimeError(f"FOV{um} feature QC failed: {z.shape}")
    torch.save({"features": torch.from_numpy(z), "cell_ids": ids, "fov_um": um, "native_side_px": side, "source_wsi": str(WSI), "scale_um_per_px": SCALE_UM_PER_PX, "preprocessing": "resize 224, center crop 224, mean/std 0.5", "feature_rule": "CLS + mean patch tokens", "seed": SEED}, out_path)
    return z


def fit_softmax_scaled(xtr: np.ndarray, xva: np.ndarray, ytr: np.ndarray, n_classes: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(np.asarray(xtr, dtype=np.float32)).to(device)
    xv = torch.from_numpy(np.asarray(xva, dtype=np.float32)).to(device)
    yt = torch.from_numpy(np.asarray(ytr, dtype=np.int64)).to(device)
    counts = torch.bincount(yt, minlength=n_classes).float()
    weights = len(ytr) / (n_classes * torch.clamp(counts, min=1.0))
    w = torch.zeros((n_classes, xt.shape[1]), device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=32, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5)
    reg = 0.5 / max(len(ytr), 1)
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(xt @ w.T + b, yt, weight=weights) + reg * torch.sum(w * w)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        p = torch.softmax(xv @ w.T + b, dim=1).cpu().numpy()
    del xt, xv, yt, w, b, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return p


def fit_probe(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, n_classes: int) -> np.ndarray:
    sc = StandardScaler().fit(X[tr])
    return fit_softmax_scaled(sc.transform(X[tr]).astype(np.float32), sc.transform(X[va]).astype(np.float32), y[tr], n_classes)


def fit_binary_scores(X: np.ndarray, y7: np.ndarray, fit_idx: np.ndarray, pred_idx: np.ndarray, pos: int, neg: int) -> np.ndarray:
    fit_idx = np.asarray(fit_idx)[np.isin(y7[np.asarray(fit_idx)], [pos, neg])]
    yb = (y7[fit_idx] == pos).astype(int)
    sc = StandardScaler().fit(X[fit_idx])
    p = fit_softmax_scaled(sc.transform(X[fit_idx]).astype(np.float32), sc.transform(X[pred_idx]).astype(np.float32), yb, 2)
    return p[:, 1]


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "weighted_f1", "mcc", "lowest_three_f1"]
    rows = []
    for candidate, g in fold_df.groupby("candidate", sort=False):
        row = {"candidate": candidate, "n_folds": len(g)}
        for col in metric_cols:
            row[col] = float(g[col].mean())
            row[col + "_sd"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def geometry_one(candidate: str, X: np.ndarray, c: pd.DataFrame, subset: np.ndarray | None = None) -> dict:
    if subset is None:
        ids = pd.read_csv(T10 / "metrics/geometry_subset_cell_ids.csv")
        lookup = {str(v): i for i, v in enumerate(c.cell_id.astype(str))}
        subset = np.asarray([lookup[str(v)] for v in ids.cell_id], dtype=int)
    y = c.class_id.to_numpy(int)[subset]
    batch = c.batch.astype(str).to_numpy()[subset]
    spatial = c.patch_id.astype(str).to_numpy()[subset]
    z = StandardScaler().fit_transform(X[subset]).astype(np.float32)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    knn = KNeighborsClassifier(n_neighbors=11, metric="cosine", n_jobs=-1).fit(z, y)
    nn = knn.kneighbors(return_distance=False)[:, 1:]
    rng = np.random.default_rng(SEED)
    i = rng.integers(0, len(subset), size=100000)
    j = rng.integers(0, len(subset), size=100000)
    keep = i != j
    i, j = i[keep], j[keep]
    dist = 1 - np.sum(z[i] * z[j], axis=1)
    same_class = y[i] == y[j]
    same_batch = batch[i] == batch[j]
    same_spatial = spatial[i] == spatial[j]
    return {
        "candidate": candidate, "n_geometry": len(subset),
        "class_separation_ratio": float(dist[~same_class].mean() / max(dist[same_class].mean(), 1e-12)),
        "batch_separation_ratio": float(dist[~same_batch].mean() / max(dist[same_batch].mean(), 1e-12)),
        "spatial_separation_ratio": float(dist[~same_spatial].mean() / max(dist[same_spatial].mean(), 1e-12)),
        "knn_class_purity": float(np.mean(y[nn] == y[:, None])),
        "knn_batch_purity": float(np.mean(batch[nn] == batch[:, None])),
        "spatial_knn_purity": float(np.mean(spatial[nn] == spatial[:, None])),
    }


def benchmark_candidates(c: pd.DataFrame, folds: dict, arrays: dict[str, np.ndarray], candidates: dict[str, np.ndarray], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    y = c.class_id.to_numpy(int)
    fold_rows: list[dict] = []
    per_rows: list[dict] = []
    bin_rows: list[dict] = []
    preds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for name, X in candidates.items():
        preds[name] = {}
        for fold, (tr, va) in folds.items():
            p = fit_probe(X, y, tr, va, 7)
            m = metric7(y[va], p); m.update({"candidate": name, "fold": fold, "n_train": len(tr), "n_val": len(va)})
            fold_rows.append(m)
            per_rows.extend(per_class_rows(name, fold, y[va], p))
            # Independent binary probes, not probability ratios from the 7-way model.
            for comparison, pos, neg in [("neutrophil_vs_myeloid", 3, 2), ("neutrophil_vs_tb", 3, 5)]:
                fit_idx = tr[np.isin(y[tr], [pos, neg])]
                score = fit_binary_scores(X, y, tr, va, pos, neg)
                bp = np.column_stack([1 - score, score])
                bin_rows.append(binary_rows(name, fold, y[va], bp, comparison, pos, neg))
            preds[name][fold] = (va, p)
            print(f"{prefix} {name} fold={fold} macroF1={m['macro_f1']:.4f}", flush=True)
    fd = pd.DataFrame(fold_rows)
    pd_ = pd.DataFrame(per_rows)
    bd = pd.DataFrame(bin_rows)
    gd = pd.DataFrame([geometry_one(name, X, c) for name, X in candidates.items()])
    summary = summarize_fold_metrics(fd)
    summary.to_csv(ROOT / f"metrics/{prefix}_summary.csv", index=False)
    fd.to_csv(ROOT / f"metrics/{prefix}_fold_metrics.csv", index=False)
    pd_.to_csv(ROOT / f"metrics/{prefix}_per_class.csv", index=False)
    bd.to_csv(ROOT / f"metrics/{prefix}_true_binary.csv", index=False)
    gd.to_csv(ROOT / f"metrics/{prefix}_geometry.csv", index=False)
    return fd, pd_, bd, gd, preds


def pca_branch(x: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    scaler = StandardScaler().fit(x[tr])
    xt = scaler.transform(x[tr]).astype(np.float32)
    xv = scaler.transform(x[va]).astype(np.float32)
    nmax = min(256, xt.shape[1], xt.shape[0] - 1)
    pca = PCA(n_components=nmax, svd_solver="randomized", random_state=SEED, whiten=False)
    zt = pca.fit_transform(xt).astype(np.float32)
    zv = pca.transform(xv).astype(np.float32)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1) if cumulative[-1] >= 0.95 else nmax
    k = min(nmax, n95)
    return zt[:, :k], zv[:, :k], k


def raw_branch(x: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    scaler = StandardScaler().fit(x[tr])
    return scaler.transform(x[tr]).astype(np.float32), scaler.transform(x[va]).astype(np.float32), x.shape[1]


def run_fusion(c: pd.DataFrame, folds: dict, branches: dict[str, list[tuple[str, np.ndarray]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = c.class_id.to_numpy(int)
    fold_rows: list[dict] = []
    per_rows: list[dict] = []
    bin_rows: list[dict] = []
    geometry_rows: list[dict] = []
    dimensions: dict[str, dict[str, list[int]]] = {}
    for bname, parts in branches.items():
        for mode in ["raw", "pca_controlled"]:
            candidate = f"{bname}_{mode}"
            dimensions[candidate] = {}
            for fold, (tr, va) in folds.items():
                train_parts = []
                val_parts = []
                dims = []
                for label, x in parts:
                    if mode == "raw":
                        xt, xv, k = raw_branch(x, tr, va)
                    else:
                        xt, xv, k = pca_branch(x, tr, va)
                    train_parts.append(xt); val_parts.append(xv); dims.append(k)
                xtr = np.concatenate(train_parts, axis=1)
                xva = np.concatenate(val_parts, axis=1)
                dimensions[candidate][str(fold)] = dims
                p = fit_softmax_scaled(xtr, xva, y[tr], 7)
                m = metric7(y[va], p); m.update({"candidate": candidate, "fold": fold, "n_train": len(tr), "n_val": len(va), "mode": mode, "branch": bname})
                fold_rows.append(m)
                per_rows.extend(per_class_rows(candidate, fold, y[va], p))
                for comparison, pos, neg in [("neutrophil_vs_myeloid", 3, 2), ("neutrophil_vs_tb", 3, 5)]:
                    # A true independently trained binary probe on the fused representation.
                    ybtr = y[tr]
                    # Reuse the already fitted fold representation; binary fitting is on the same train-only transform.
                    # Build the transformed binary training subset from the rows already in xtr.
                    tr_pos_mask = np.isin(y[tr], [pos, neg])
                    val_x_bin = xva
                    score = fit_softmax_scaled(xtr[tr_pos_mask], val_x_bin, y[tr][tr_pos_mask] == pos, 2)[:, 1]
                    bp = np.column_stack([1 - score, score])
                    bin_rows.append(binary_rows(candidate, fold, y[va], bp, comparison, pos, neg))
                del xtr, xva, train_parts, val_parts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Fusion {candidate} fold={fold} macroF1={m['macro_f1']:.4f}", flush=True)
            # Geometry is descriptive and uses full-cohort branch-wise standardization only.
            geom_parts = []
            for _, x in parts:
                z = StandardScaler().fit_transform(x)
                if mode == "pca_controlled":
                    nmax = min(256, z.shape[1], len(z) - 1)
                    pc = PCA(n_components=nmax, svd_solver="randomized", random_state=SEED)
                    z = pc.fit_transform(z)
                    cum = np.cumsum(pc.explained_variance_ratio_)
                    k = int(np.searchsorted(cum, 0.95) + 1) if cum[-1] >= 0.95 else nmax
                    z = z[:, :min(nmax, k)]
                geom_parts.append(z.astype(np.float32))
            geometry_rows.append(geometry_one(candidate, np.concatenate(geom_parts, axis=1), c))
    fd = pd.DataFrame(fold_rows); pd_ = pd.DataFrame(per_rows); bd = pd.DataFrame(bin_rows); gd = pd.DataFrame(geometry_rows)
    summarize_fold_metrics(fd).to_csv(ROOT / "metrics/fusion_linear_summary.csv", index=False)
    fd.to_csv(ROOT / "metrics/fusion_linear_fold_metrics.csv", index=False)
    pd_.to_csv(ROOT / "metrics/fusion_per_class.csv", index=False)
    bd.to_csv(ROOT / "metrics/fusion_true_binary.csv", index=False)
    gd.to_csv(ROOT / "metrics/fusion_geometry.csv", index=False)
    (ROOT / "config/fusion_pca_dimensions.json").write_text(json.dumps(dimensions, indent=2) + "\n")
    return fd, pd_, bd, gd


def compose_hierarchy(pcoarse: np.ndarray, pstruct: np.ndarray, pimmune: np.ndarray) -> np.ndarray:
    p = np.zeros((len(pcoarse), 7), dtype=np.float32)
    p[:, 6] = pcoarse[:, 0]
    p[:, 0] = pcoarse[:, 1] * pstruct[:, 0]
    p[:, 1] = pcoarse[:, 1] * pstruct[:, 1]
    p[:, 2:6] = pcoarse[:, 2, None] * pimmune
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p


def hierarchy_probs(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yc = np.where(y == 6, 0, np.where(np.isin(y, [0, 1]), 1, 2)).astype(int)
    ys = np.where(y == 1, 1, 0).astype(int)
    yi = np.select([y == 3, y == 4, y == 5], [1, 2, 3], default=0).astype(int)
    str_tr = tr[np.isin(y[tr], [0, 1])]
    imm_tr = tr[np.isin(y[tr], [2, 3, 4, 5])]
    pcoarse = fit_probe(X, yc, tr, va, 3)
    pstruct = fit_probe(X, ys, str_tr, va, 2)
    pimmune = fit_probe(X, yi, imm_tr, va, 4)
    return pcoarse, pstruct, pimmune


def run_hierarchy(c: pd.DataFrame, folds: dict, X: np.ndarray, candidate: str, prefix: str = "hierarchical") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = c.class_id.to_numpy(int)
    fold_rows: list[dict] = []; per_rows: list[dict] = []; bin_rows: list[dict] = []
    for fold, (tr, va) in folds.items():
        pc, ps, pi = hierarchy_probs(X, y, tr, va)
        p = compose_hierarchy(pc, ps, pi)
        m = metric7(y[va], p); m.update({"candidate": candidate, "fold": fold, "n_train": len(tr), "n_val": len(va)})
        fold_rows.append(m); per_rows.extend(per_class_rows(candidate, fold, y[va], p))
        for comparison, pos, neg in [("neutrophil_vs_myeloid", 3, 2), ("neutrophil_vs_tb", 3, 5)]:
            score = fit_binary_scores(X, y, tr, va, pos, neg); bp = np.column_stack([1 - score, score])
            bin_rows.append(binary_rows(candidate, fold, y[va], bp, comparison, pos, neg))
        fit_all = tr[np.isin(y[tr], [2, 3, 4, 5])]
        y_all = (y[fit_all] == 3).astype(int)
        sc_all = StandardScaler().fit(X[fit_all])
        score_all = fit_softmax_scaled(sc_all.transform(X[fit_all]).astype(np.float32), sc_all.transform(X[va]).astype(np.float32), y_all, 2)[:, 1]
        bin_rows.append(immune_binary_row(candidate, fold, y[va], score_all))
        print(f"Hierarchy fold={fold} macroF1={m['macro_f1']:.4f}", flush=True)
    fd = pd.DataFrame(fold_rows); pd_ = pd.DataFrame(per_rows); bd = pd.DataFrame(bin_rows); gd = pd.DataFrame([geometry_one(candidate, X, c)])
    summarize_fold_metrics(fd).to_csv(ROOT / f"metrics/{prefix}_summary.csv", index=False)
    fd.to_csv(ROOT / f"metrics/{prefix}_fold_metrics.csv", index=False)
    pd_.to_csv(ROOT / f"metrics/{prefix}_per_class.csv", index=False)
    bd.to_csv(ROOT / f"metrics/{prefix}_true_binary.csv", index=False)
    gd.to_csv(ROOT / f"metrics/{prefix}_geometry.csv", index=False)
    flows = []
    for fold, (tr, va) in folds.items():
        row = fd[fd.fold == fold].iloc[0]
        # The exact flow is rebuilt from a held-out hierarchical prediction for auditability.
        pc, ps, pi = hierarchy_probs(X, y, tr, va); pred = compose_hierarchy(pc, ps, pi).argmax(1)
        cm = confusion_matrix(y[va], pred, labels=np.arange(7))
        for a in range(7):
            for b in range(7):
                if cm[a, b]:
                    flows.append({"candidate": candidate, "fold": fold, "true_class": CLASSES[a], "predicted_class": CLASSES[b], "count": int(cm[a, b]), "rate_among_true": float(cm[a, b] / max(cm[a].sum(), 1))})
    pd.DataFrame(flows).to_csv(ROOT / f"metrics/{prefix}_confusion.csv", index=False)
    return fd, pd_, bd, gd


def fit_meta_calibrator(x_inner: np.ndarray, y_inner: np.ndarray, x_outer: np.ndarray) -> np.ndarray:
    sc = StandardScaler().fit(x_inner)
    model = LogisticRegression(max_iter=500, class_weight="balanced", multi_class="multinomial", random_state=SEED, solver="lbfgs")
    model.fit(sc.transform(x_inner), y_inner)
    p = model.predict_proba(sc.transform(x_outer))
    out = np.zeros((len(x_outer), 4), dtype=np.float32)
    for j, cls in enumerate(model.classes_.astype(int)):
        out[:, cls] = p[:, j]
    return out


def run_specialist(c: pd.DataFrame, folds: dict, X: np.ndarray, candidate: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = c.class_id.to_numpy(int)
    fold_rows: list[dict] = []; per_rows: list[dict] = []; bin_rows: list[dict] = []
    for fold, (tr, va) in folds.items():
        train_batches = sorted(c.batch.astype(str).to_numpy()[tr].tolist())
        unique_batches = sorted(set(train_batches))
        inner_batch = unique_batches[-1]
        inner_va = tr[c.batch.astype(str).to_numpy()[tr] == inner_batch]
        inner_tr = tr[c.batch.astype(str).to_numpy()[tr] != inner_batch]
        if len(inner_va) == 0 or len(inner_tr) == 0:
            raise RuntimeError(f"empty inner split in fold {fold}")

        # Training-only inner hierarchy and specialist predictions for meta calibration.
        pc_i, ps_i, pi_i = hierarchy_probs(X, y, inner_tr, inner_va)
        specialist_inner = np.column_stack([
            fit_binary_scores(X, y, inner_tr, inner_va, 3, 2),
            fit_binary_scores(X, y, inner_tr, inner_va, 3, 5),
            np.zeros(len(inner_va), dtype=np.float32),
        ])
        # The all-immune specialist uses a synthetic negative class: all non-neutrophil immune cells.
        # fit_binary_scores accepts one negative class, so replace its score with an explicit probe.
        fit_all = inner_tr[np.isin(y[inner_tr], [2, 3, 4, 5])]
        y_all = (y[fit_all] == 3).astype(int)
        sc_all = StandardScaler().fit(X[fit_all])
        specialist_inner[:, 2] = fit_softmax_scaled(sc_all.transform(X[fit_all]).astype(np.float32), sc_all.transform(X[inner_va]).astype(np.float32), y_all, 2)[:, 1]
        x_inner = np.column_stack([pi_i, specialist_inner])
        yi_inner = np.select([y[inner_va] == 3, y[inner_va] == 4, y[inner_va] == 5], [1, 2, 3], default=0).astype(int)

        # Outer predictions: fit every component on all outer-training cells.
        pc_o, ps_o, pi_o = hierarchy_probs(X, y, tr, va)
        specialist_outer = np.column_stack([
            fit_binary_scores(X, y, tr, va, 3, 2),
            fit_binary_scores(X, y, tr, va, 3, 5),
            np.zeros(len(va), dtype=np.float32),
        ])
        fit_all = tr[np.isin(y[tr], [2, 3, 4, 5])]
        y_all = (y[fit_all] == 3).astype(int)
        sc_all = StandardScaler().fit(X[fit_all])
        specialist_outer[:, 2] = fit_softmax_scaled(sc_all.transform(X[fit_all]).astype(np.float32), sc_all.transform(X[va]).astype(np.float32), y_all, 2)[:, 1]
        x_outer = np.column_stack([pi_o, specialist_outer])
        pmeta = fit_meta_calibrator(x_inner, yi_inner, x_outer)
        p = compose_hierarchy(pc_o, ps_o, pmeta)
        m = metric7(y[va], p); m.update({"candidate": candidate, "fold": fold, "inner_val_batch": inner_batch, "n_train": len(tr), "n_val": len(va)})
        fold_rows.append(m); per_rows.extend(per_class_rows(candidate, fold, y[va], p))
        for comparison, pos, neg in [("neutrophil_vs_myeloid", 3, 2), ("neutrophil_vs_tb", 3, 5)]:
            score = fit_binary_scores(X, y, tr, va, pos, neg); bp = np.column_stack([1 - score, score])
            bin_rows.append(binary_rows(candidate, fold, y[va], bp, comparison, pos, neg))
        bin_rows.append(immune_binary_row(candidate, fold, y[va], specialist_outer[:, 2]))
        print(f"Specialist fold={fold} macroF1={m['macro_f1']:.4f} neutrophilF1={per_rows[-4]['f1']:.4f}", flush=True)
    fd = pd.DataFrame(fold_rows); pd_ = pd.DataFrame(per_rows); bd = pd.DataFrame(bin_rows); gd = pd.DataFrame([geometry_one(candidate, X, c)])
    summarize_fold_metrics(fd).to_csv(ROOT / "metrics/hierarchical_specialist_summary.csv", index=False)
    fd.to_csv(ROOT / "metrics/hierarchical_specialist_fold_metrics.csv", index=False)
    pd_.to_csv(ROOT / "metrics/hierarchical_specialist_per_class.csv", index=False)
    bd.to_csv(ROOT / "metrics/hierarchical_specialist_true_binary.csv", index=False)
    gd.to_csv(ROOT / "metrics/hierarchical_specialist_geometry.csv", index=False)
    return fd, pd_, bd, gd


def tta_transform(arr: np.ndarray, view: int) -> np.ndarray:
    flip = view >= 4
    k = view % 4
    z = np.rot90(arr, k=k)
    if flip:
        z = np.fliplr(z)
    return np.ascontiguousarray(z)


def extract_tta(c: pd.DataFrame, model, device, um: int, side: int, views: int, batch_size: int = 64) -> np.ndarray:
    out_path = ROOT / f"features/midnight_fov{um}_tta{views}.pt"
    ids = c.cell_id.astype(str).tolist()
    if out_path.exists():
        try:
            return check_payload(out_path, ids)
        except Exception:
            out_path.unlink()
    wsi = pyvips.Image.new_from_file(str(WSI), access="random")
    work = c.sort_values(["patch_id", "cell_id"], kind="stable").reset_index()
    rows: list[np.ndarray] = []
    arr_batch: list[np.ndarray] = []
    tile = None; ox = oy = 0; current = None; t0 = time.time()
    def flush() -> None:
        if not arr_batch:
            return
        total = np.zeros((len(arr_batch), 3072), dtype=np.float32)
        for view in range(views):
            xs = [crop_tensor(tta_transform(a, view)) for a in arr_batch]
            z = encode_batch(model, device, xs)
            z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
            total += z
        total /= float(views)
        total /= np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-12)
        rows.append(total.astype(np.float32))
    for i, r in work.iterrows():
        if current != r.patch_id:
            tile, ox, oy = base._tile_array(wsi, int(r.patch_x), int(r.patch_y)); current = r.patch_id
        half = side // 2
        sx = int(round(float(r.he_x))) - half - ox; sy = int(round(float(r.he_y))) - half - oy
        arr = tile[sy:sy + side, sx:sx + side, :3]
        if arr.shape != (side, side, 3):
            raise RuntimeError(f"invalid TTA crop {arr.shape} for {r.cell_id}")
        arr_batch.append(arr)
        if len(arr_batch) >= batch_size:
            flush(); arr_batch.clear()
        if i % 5000 == 0:
            print(f"TTA{views} FOV{um}: {i:,}/{len(work):,} elapsed_min={(time.time() - t0) / 60:.1f}", flush=True)
    flush()
    z = np.concatenate(rows, axis=0)[np.argsort(work["index"].to_numpy())]
    if z.shape != (len(c), 3072) or not np.isfinite(z).all():
        raise RuntimeError("TTA feature QC failed")
    torch.save({"features": torch.from_numpy(z), "cell_ids": ids, "fov_um": um, "views": views, "normalization": "L2 per view, mean, L2 final", "preprocessing": "Midnight official 224/0.5", "seed": SEED}, out_path)
    return z


def select_fov(summary: pd.DataFrame, geometry: pd.DataFrame, binary: pd.DataFrame) -> tuple[str, dict]:
    s = summary.copy()
    g = geometry.set_index("candidate")
    b = binary[binary.comparison == "neutrophil_vs_myeloid"].groupby("candidate", as_index=True).auprc.mean()
    s["neutrophil_auprc"] = s.candidate.map(b)
    s["knn_batch_purity"] = s.candidate.map(g.knn_batch_purity)
    s["spatial_knn_purity"] = s.candidate.map(g.spatial_knn_purity)
    best_f1 = s.macro_f1.max()
    close = s[s.macro_f1 >= best_f1 - 0.01].copy()
    close = close.sort_values(["neutrophil_auprc", "knn_batch_purity", "spatial_knn_purity", "macro_f1"], ascending=[False, True, True, False], kind="stable")
    chosen = str(close.iloc[0].candidate)
    return chosen, {"selection_rule": ["highest mean macro-F1", "within 0.01: higher true N-vs-Myeloid AUPRC", "then lower batch/spatial kNN purity"], "candidate": chosen, "best_macro_f1": float(best_f1), "close_candidates": close.candidate.tolist()}


def choose_fusion(summary: pd.DataFrame, fov_summary: pd.DataFrame, fov_name: str) -> tuple[str, dict]:
    base_row = fov_summary[fov_summary.candidate == fov_name].iloc[0]
    raw = summary[summary.candidate.str.endswith("_raw")].copy()
    raw["delta_macro_f1"] = raw.macro_f1 - float(base_row.macro_f1)
    raw["positive_fold_count"] = 0
    # Fold-positive count is filled by the caller from paired fold metrics if available.
    # Candidate selection remains primary-mode and deterministic.
    best = raw.sort_values("macro_f1", ascending=False).iloc[0]
    selected = str(best.candidate)
    return selected, {"candidate": selected, "baseline": fov_name, "delta_macro_f1": float(best.delta_macro_f1), "criterion": "primary raw fusion; secondary PCA-controlled candidates reported separately"}


def ladder_row(model: str, key: str, fold_df: pd.DataFrame, per_df: pd.DataFrame, bin_df: pd.DataFrame, geom_df: pd.DataFrame, key_col: str = "candidate") -> dict:
    g = fold_df[fold_df[key_col] == key]
    p = per_df[per_df[key_col] == key]
    b = bin_df[bin_df[key_col] == key]
    z = geom_df[geom_df[key_col] == key].iloc[0]
    n = p[p.class_id == 3]
    row = {"model": model, "candidate": key}
    for col in ["macro_f1", "macro_auprc", "lowest_three_f1", "accuracy", "balanced_accuracy"]:
        row[col] = float(g[col].mean())
    row["neutrophil_f1"] = float(n.f1.mean())
    row["neutrophil_auprc"] = float(n.auprc.mean())
    for comp, label in [("neutrophil_vs_myeloid", "n_vs_myeloid"), ("neutrophil_vs_tb", "n_vs_tb"), ("neutrophil_vs_all_immune", "n_vs_all_immune")]:
        q = b[b.comparison == comp]
        row[label + "_auroc"] = float(q.auroc.mean()) if len(q) else float("nan")
        row[label + "_auprc"] = float(q.auprc.mean()) if len(q) else float("nan")
    row["knn_batch_purity"] = float(z.get("knn_batch_purity", np.nan))
    row["spatial_knn_purity"] = float(z.get("spatial_knn_purity", np.nan))
    row["class_knn_purity"] = float(z.get("knn_class_purity", np.nan))
    return row


def ladder_row_t10(model: str, key: str, representation: str) -> dict:
    s = pd.read_csv(T10 / "metrics/linear_probe_summary_corrected.csv")
    p = pd.read_csv(T10 / "metrics/linear_probe_per_class_corrected.csv")
    b = pd.read_csv(T10 / "metrics/true_binary_summary.csv")
    g = pd.read_csv(T10 / "metrics/representation_geometry_corrected.csv")
    g = g[g.representation == representation].iloc[0]
    return ladder_row(model, representation, s.rename(columns={"representation": "candidate"}), p.rename(columns={"representation": "candidate"}), b.rename(columns={"representation": "candidate"}), g.to_frame().T.rename(columns={"representation": "candidate"}), key_col="candidate")


def build_ladder(c: pd.DataFrame, fov_fd: pd.DataFrame, fov_pd: pd.DataFrame, fov_bd: pd.DataFrame, fov_gd: pd.DataFrame, fov_best: str, fusion_fd: pd.DataFrame, fusion_pd: pd.DataFrame, fusion_bd: pd.DataFrame, fusion_gd: pd.DataFrame, fusion_best: str, hier_fd: pd.DataFrame, hier_pd: pd.DataFrame, hier_bd: pd.DataFrame, hier_gd: pd.DataFrame, spec_fd: pd.DataFrame, spec_pd: pd.DataFrame, spec_bd: pd.DataFrame, spec_gd: pd.DataFrame, tta_fd: pd.DataFrame, tta_pd: pd.DataFrame, tta_bd: pd.DataFrame, tta_gd: pd.DataFrame, tta_best: str) -> pd.DataFrame:
    rows = []
    # Convert Task010 summary tables to the local candidate-column convention.
    t10s = pd.read_csv(T10 / "metrics/linear_probe_summary_corrected.csv").rename(columns={"representation": "candidate"})
    t10p = pd.read_csv(T10 / "metrics/linear_probe_per_class_corrected.csv").rename(columns={"representation": "candidate"})
    t10b = pd.read_csv(T10 / "metrics/true_binary_summary.csv").rename(columns={"representation": "candidate"})
    t10g = pd.read_csv(T10 / "metrics/representation_geometry_corrected.csv").rename(columns={"representation": "candidate"})
    rows.append(ladder_row("Task010 corrected CellViT", "CELLVIT_TOKEN_ALIGNED", t10s, t10p, t10b, t10g))
    rows.append(ladder_row("Task010 Midnight 16 μm", "MIDNIGHT12K_SMALL", t10s, t10p, t10b, t10g))
    rows.append(ladder_row("Task011 BEST_FOV", fov_best, fov_fd, fov_pd, fov_bd, fov_gd))
    rows.append(ladder_row("Task011 BEST_FUSION", fusion_best, fusion_fd, fusion_pd, fusion_bd, fusion_gd))
    rows.append(ladder_row("Task011 HIERARCHICAL", "HIERARCHICAL", hier_fd, hier_pd, hier_bd, hier_gd))
    rows.append(ladder_row("Task011 HIERARCHICAL + N specialist", "HIERARCHICAL_SPECIALIST", spec_fd, spec_pd, spec_bd, spec_gd))
    rows.append(ladder_row("Task011 TTA best", tta_best, tta_fd, tta_pd, tta_bd, tta_gd))
    return pd.DataFrame(rows)


def write_figures(fov_summary: pd.DataFrame, fusion_summary: pd.DataFrame, hier_summary: pd.DataFrame, spec_summary: pd.DataFrame, tta_summary: pd.DataFrame, ladder: pd.DataFrame, robustness: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(fov_summary.candidate, fov_summary.macro_f1, yerr=fov_summary.macro_f1_sd, marker="o", capsize=3)
    ax.set_xlabel("Native local FOV"); ax.set_ylabel("Macro-F1"); ax.set_title("Task011 Midnight local-FOV sweep"); ax.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(ROOT / "figures/Fig_fov_sweep.pdf"); plt.close(fig)
    stages = pd.concat([
        fusion_summary[["candidate", "macro_f1"]], hier_summary[["candidate", "macro_f1"]], spec_summary[["candidate", "macro_f1"]], tta_summary[["candidate", "macro_f1"]]
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 5)); ax.bar(stages.candidate, stages.macro_f1); ax.set_ylabel("Macro-F1"); ax.tick_params(axis="x", rotation=45); ax.set_title("Task011 frozen optimization stages"); fig.tight_layout(); fig.savefig(ROOT / "figures/Fig_optimization_stages.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(ladder.model, ladder.macro_f1, marker="o", label="Macro-F1"); ax.plot(ladder.model, ladder.neutrophil_f1, marker="o", label="Neutrophil F1"); ax.set_ylabel("Score"); ax.tick_params(axis="x", rotation=55); ax.legend(); ax.set_title("Task011 model ladder"); fig.tight_layout(); fig.savefig(ROOT / "figures/Fig_model_ladder.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5));
    for metric, label in [("macro_f1", "Outer-fold macro-F1"), ("knn_batch_purity", "Batch kNN purity")]:
        q = robustness[robustness.metric == metric]
        if len(q): ax.plot(q.candidate, q.value, marker="o", label=label)
    ax.set_ylabel("Value"); ax.tick_params(axis="x", rotation=55); ax.legend(); ax.set_title("Spatial/batch robustness across candidates"); fig.tight_layout(); fig.savefig(ROOT / "figures/Fig_spatial_fold_robustness.pdf"); plt.close(fig)


def main() -> None:
    seed_everything()
    mkdirs()
    c = load_canonical()
    folds = load_folds(c)
    existing = load_existing_features(c)
    model, device = load_midnight_model()
    fov_arrays = {}
    for um, side in FOVS.items():
        fov_arrays[f"FOV{um}"] = extract_fov(c, model, device, um, side)
    fov_fd, fov_pd, fov_bd, fov_gd, _ = benchmark_candidates(c, folds, fov_arrays, fov_arrays, "fov_sweep")
    fov_summary = summarize_fold_metrics(fov_fd)
    fov_best, fov_decision = select_fov(fov_summary, fov_gd, fov_bd)
    (ROOT / "config/best_fov.json").write_text(json.dumps({**fov_decision, "physical_fov_um": int(fov_best.replace("FOV", "")), "native_side_px": int(FOVS[int(fov_best.replace("FOV", ""))])}, indent=2) + "\n")
    best_fov = fov_arrays[fov_best]

    # Primary fusion uses branch-wise scaled raw concatenation. PCA-controlled results are secondary.
    branches = {
        "B0": [("MIDNIGHT_BEST_FOV", best_fov)],
        "B1": [("MIDNIGHT_BEST_FOV", best_fov), ("CELLVIT_TOKEN_ALIGNED", existing["CELLVIT_TOKEN_ALIGNED"])],
        "B2": [("MIDNIGHT_BEST_FOV", best_fov), ("PHIKON_V2_SMALL", existing["PHIKON_V2_SMALL"])],
        "B3": [("MIDNIGHT_BEST_FOV", best_fov), ("CELLVIT_TOKEN_ALIGNED", existing["CELLVIT_TOKEN_ALIGNED"]), ("PHIKON_V2_SMALL", existing["PHIKON_V2_SMALL"])],
    }
    fusion_fd, fusion_pd, fusion_bd, fusion_gd = run_fusion(c, folds, branches)
    fov_best_row = fov_fd[fov_fd.candidate == fov_best].set_index("fold")
    fov_best_per = fov_pd[(fov_pd.candidate == fov_best) & (fov_pd.class_id == 3)].set_index("fold")
    qualifying = []
    for cand in fusion_fd.candidate.unique():
        if not cand.endswith("_raw"):
            continue
        q = fusion_fd[fusion_fd.candidate == cand].set_index("fold")
        delta = q.macro_f1 - fov_best_row.macro_f1
        n = fusion_pd[(fusion_pd.candidate == cand) & (fusion_pd.class_id == 3)].set_index("fold")
        nbase = fov_best_per
        n_f1_delta = n.f1 - nbase.f1
        nb = fusion_bd[(fusion_bd.candidate == cand) & (fusion_bd.comparison == "neutrophil_vs_myeloid")].set_index("fold")
        nbb = fov_bd[(fov_bd.candidate == fov_best) & (fov_bd.comparison == "neutrophil_vs_myeloid")].set_index("fold")
        auprc_delta = nb.auprc - nbb.auprc
        gain_rule = float(q.macro_f1.mean() - fov_best_row.macro_f1.mean()) >= 0.015 or float(n_f1_delta.mean()) >= 0.03 or float(auprc_delta.mean()) >= 0.03
        if gain_rule and int((delta > 0).sum()) >= 4:
            qualifying.append((cand, float(q.macro_f1.mean())))
    fusion_best = max(qualifying, key=lambda z: z[1])[0] if qualifying else "B0_raw"
    fusion_decision = {"candidate": fusion_best, "baseline": fov_best, "qualifying_raw_fusions": [x[0] for x in qualifying], "criterion": "macro-F1 +0.015 OR Neutrophil F1/AUPRC +0.03, with positive macro-F1 delta in >=4/5 folds", "primary_mode": "raw branch-wise concatenation"}
    (ROOT / "config/best_fusion.json").write_text(json.dumps(fusion_decision, indent=2) + "\n")
    fusion_X = {"B0_raw": best_fov, "B1_raw": np.concatenate([best_fov, existing["CELLVIT_TOKEN_ALIGNED"]], axis=1), "B2_raw": np.concatenate([best_fov, existing["PHIKON_V2_SMALL"]], axis=1), "B3_raw": np.concatenate([best_fov, existing["CELLVIT_TOKEN_ALIGNED"], existing["PHIKON_V2_SMALL"]], axis=1)}[fusion_best]

    hier_fd, hier_pd, hier_bd, hier_gd = run_hierarchy(c, folds, fusion_X, "HIERARCHICAL")
    spec_fd, spec_pd, spec_bd, spec_gd = run_specialist(c, folds, fusion_X, "HIERARCHICAL_SPECIALIST")
    hier_summary = summarize_fold_metrics(hier_fd); spec_summary = summarize_fold_metrics(spec_fd)
    flat_summary = fusion_fd[fusion_fd.candidate == fusion_best].copy()
    hier_gain = float(hier_summary.macro_f1.iloc[0] - flat_summary.macro_f1.mean())
    (ROOT / "config/hierarchical_mlp_status.json").write_text(json.dumps({"status": "NOT_RUN", "reason": "Secondary MLP heads are only authorized when the linear hierarchy improves or is within 0.01 macro-F1 of the flat best; observed hierarchy delta will be recorded in the report.", "hierarchy_delta_vs_flat": hier_gain}, indent=2) + "\n")

    best_um = int(fov_best.replace("FOV", "")); best_side = FOVS[best_um]
    tta_arrays = {"SINGLE": best_fov, "TTA4": extract_tta(c, model, device, best_um, best_side, 4, batch_size=1024), "TTA8": extract_tta(c, model, device, best_um, best_side, 8, batch_size=1024)}
    tta_fd, tta_pd, tta_bd, tta_gd, _ = benchmark_candidates(c, folds, tta_arrays, tta_arrays, "tta")
    tta_summary = summarize_fold_metrics(tta_fd)
    base_tta = tta_summary[tta_summary.candidate == "SINGLE"].iloc[0]
    tta_decisions = []
    for cand in ["TTA4", "TTA8"]:
        row = tta_summary[tta_summary.candidate == cand].iloc[0]; geom = tta_gd[tta_gd.candidate == cand].iloc[0]; base_geom = tta_gd[tta_gd.candidate == "SINGLE"].iloc[0]
        n = tta_pd[(tta_pd.candidate == cand) & (tta_pd.class_id == 3)].f1.mean(); n0 = tta_pd[(tta_pd.candidate == "SINGLE") & (tta_pd.class_id == 3)].f1.mean()
        nb = tta_bd[(tta_bd.candidate == cand) & (tta_bd.comparison == "neutrophil_vs_myeloid")].auprc.mean(); nb0 = tta_bd[(tta_bd.candidate == "SINGLE") & (tta_bd.comparison == "neutrophil_vs_myeloid")].auprc.mean()
        eligible = (row.macro_f1 - base_tta.macro_f1 >= 0.01 or nb - nb0 >= 0.02) and (geom.knn_batch_purity - base_geom.knn_batch_purity <= 0.05) and (geom.spatial_knn_purity - base_geom.spatial_knn_purity <= 0.05)
        tta_decisions.append({"candidate": cand, "macro_f1_delta": float(row.macro_f1 - base_tta.macro_f1), "neutrophil_auprc_delta": float(nb - nb0), "batch_purity_delta": float(geom.knn_batch_purity - base_geom.knn_batch_purity), "spatial_purity_delta": float(geom.spatial_knn_purity - base_geom.spatial_knn_purity), "eligible": bool(eligible)})
    eligible_tta = [x for x in tta_decisions if x["eligible"]]
    tta_best = max(eligible_tta, key=lambda x: x["macro_f1_delta"])["candidate"] if eligible_tta else "SINGLE"
    (ROOT / "config/tta_selection.json").write_text(json.dumps({"best": tta_best, "base": "SINGLE", "decisions": tta_decisions, "normalization": "L2-normalize each view, average, L2-normalize final"}, indent=2) + "\n")

    ladder = build_ladder(c, fov_fd, fov_pd, fov_bd, fov_gd, fov_best, fusion_fd, fusion_pd, fusion_bd, fusion_gd, fusion_best, hier_fd, hier_pd, hier_bd, hier_gd, spec_fd, spec_pd, spec_bd, spec_gd, tta_fd, tta_pd, tta_bd, tta_gd, tta_best)
    ladder.to_csv(ROOT / "metrics/model_ladder.csv", index=False)
    robustness_rows = []
    for name, fd, gd in [(fov_best, fov_fd, fov_gd), (fusion_best, fusion_fd, fusion_gd), ("HIERARCHICAL", hier_fd, hier_gd), ("HIERARCHICAL_SPECIALIST", spec_fd, spec_gd), (tta_best, tta_fd, tta_gd)]:
        q = fd[fd.candidate == name] if name in set(fd.candidate) else fd
        z = gd[gd.candidate == name].iloc[0] if name in set(gd.candidate) else gd.iloc[0]
        robustness_rows.append({"candidate": name, "metric": "macro_f1", "value": float(q.macro_f1.mean())})
        robustness_rows.append({"candidate": name, "metric": "knn_batch_purity", "value": float(z.knn_batch_purity)})
        robustness_rows.append({"candidate": name, "metric": "spatial_knn_purity", "value": float(z.spatial_knn_purity)})
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(ROOT / "metrics/spatial_robustness_summary.csv", index=False)
    write_figures(fov_summary, summarize_fold_metrics(fusion_fd), hier_summary, spec_summary, tta_summary, ladder, robustness)

    midnight16 = float(ladder.loc[ladder.model == "Task010 Midnight 16 μm", "macro_f1"].iloc[0])
    best_frozen = float(ladder.loc[ladder.model == "Task011 TTA best", "macro_f1"].iloc[0])
    best_neut = float(ladder.loc[ladder.model == "Task011 TTA best", "neutrophil_f1"].iloc[0])
    plateaued = (best_frozen - midnight16 < 0.015) and (best_neut - float(ladder.loc[ladder.model == "Task010 Midnight 16 μm", "neutrophil_f1"].iloc[0]) < 0.03)
    gate = {"status": "NOT_TRIGGERED_STAGE_A_E_NOT_PLATEAUED" if not plateaued else "TRIGGERED_REVIEW_REQUIRED", "best_frozen_macro_f1": best_frozen, "task010_midnight16_macro_f1": midnight16, "best_frozen_neutrophil_f1": best_neut, "plateau_definition": "no >=0.015 macro-F1 or >=0.03 Neutrophil F1 frozen improvement over Task010 Midnight 16 μm", "production_model_modified": False, "fine_tuning_executed": False}
    (ROOT / "config/fine_tuning_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    decision = {"status": "TASK011_FROZEN_OPTIMIZATION_COMPLETE", "n_canonical": len(c), "best_fov": fov_best, "best_fusion": fusion_best, "hierarchy": "HIERARCHICAL", "specialist": "HIERARCHICAL_SPECIALIST", "tta_best": tta_best, "fine_tuning_gate": gate, "production_model_modified": False, "ground_truth_modified": False, "seed": SEED}
    (ROOT / "metrics/decision_summary.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(ladder[["model", "macro_f1", "neutrophil_f1", "neutrophil_auprc"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
