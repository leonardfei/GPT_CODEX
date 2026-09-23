#!/usr/bin/env python3
"""Task011 Stage F1: limited Midnight final-block fine-tuning gate.

Only the last Midnight transformer block and a seven-class head are trained.
Outer validation batches remain untouched; one training-only batch is used for
epoch selection, followed by an outer-training refit at the selected epoch.
Large crops/embeddings stay on the server.
"""
from __future__ import annotations

import copy
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

T10 = Path("/data/lf_data/result/task010_representation_benchmark")
ROOT = Path("/data/lf_data/result/task011_midnight_local_optimization")
TASK009 = Path("/data/lf_data/result/task009_v3_retraining")
WSI = Path("/data/lf_data/xenium_data/ID0060276.ome.tif")
MIDNIGHT = Path("/data/lf_data/models/midnight-12k")
SEED = 20260923
CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
sys.path.insert(0, str(T10 / "code"))


def seed_all(s: int = SEED) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def safe_auc(y, p):
    try: return float(roc_auc_score(y, p))
    except ValueError: return float("nan")


def safe_ap(y, p):
    try: return float(average_precision_score(y, p))
    except ValueError: return float("nan")


def load_canonical():
    c = pd.read_csv(T10 / "metrics/canonical_cell_order.csv.gz")
    if len(c) != 96044 or c.cell_id.duplicated().any(): raise RuntimeError("canonical cohort changed")
    return c.reset_index(drop=True)


def load_folds(c):
    sm = pd.read_csv(TASK009 / "metrics/split_manifest.csv")
    sm = sm[sm.condition == "CORE"].sort_values("fold")
    out = {}
    for _, r in sm.iterrows():
        trb = set(str(r.train_batches).split(";")); vab = set(str(r.val_batches).split(";"))
        out[int(r.fold)] = (np.flatnonzero(c.batch.astype(str).isin(trb)), np.flatnonzero(c.batch.astype(str).isin(vab)))
    return out


def load_model(device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(MIDNIGHT), local_files_only=True)
    for p in model.parameters(): p.requires_grad = False
    for p in model.encoder.layer[-1].parameters(): p.requires_grad = True
    model.to(device)
    return model


def crop_memmap(c):
    path = ROOT / "work/midnight_fov12_crops_uint8.npy"
    if path.exists():
        mm = np.load(path, mmap_mode="r")
        if mm.shape == (len(c), 57, 57, 3): return mm
        raise RuntimeError(f"crop cache shape mismatch {mm.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(c), 57, 57, 3))
    wsi = pyvips.Image.new_from_file(str(WSI), access="random")
    work = c.sort_values(["patch_id", "cell_id"], kind="stable").reset_index()
    tile = None; ox = oy = 0; current = None; t0 = time.time()
    for i, r in work.iterrows():
        if current != r.patch_id:
            from task010_phikon_phase_a import _tile_array
            tile, ox, oy = _tile_array(wsi, int(r.patch_x), int(r.patch_y)); current = r.patch_id
        half = 28; sx = int(round(float(r.he_x))) - half - ox; sy = int(round(float(r.he_y))) - half - oy
        arr = tile[sy:sy + 57, sx:sx + 57, :3]
        if arr.shape != (57, 57, 3): raise RuntimeError(f"bad FOV12 crop {arr.shape}")
        mm[int(r["index"])] = arr
        if i % 5000 == 0: print(f"crop cache {i:,}/{len(work):,} elapsed_min={(time.time() - t0) / 60:.1f}", flush=True)
    mm.flush()
    if not np.isfinite(np.asarray(mm[::1000], dtype=np.float32)).all(): raise RuntimeError("crop cache QC failed")
    return np.load(path, mmap_mode="r")


def batch_x(crops, idx, device, train=False):
    x = torch.from_numpy(np.asarray(crops[idx])).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
    if train:
        # Deterministic seeded batch-level orientation and mild photometric augmentation.
        k = int(torch.randint(0, 4, (1,), device=device).item())
        x = torch.rot90(x, k=k, dims=(2, 3))
        if bool(torch.rand((), device=device) < 0.5): x = torch.flip(x, dims=(3,))
        if bool(torch.rand((), device=device) < 0.5):
            contrast = 0.9 + 0.2 * torch.rand((), device=device)
            brightness = 0.9 + 0.2 * torch.rand((), device=device)
            x = ((x - 0.5) * contrast + 0.5) * brightness
    return (x - 0.5) / 0.5


def forward(model, head, x):
    h = model(x).last_hidden_state
    z = torch.cat([h[:, 0, :], h[:, 1:, :].mean(dim=1)], dim=1)
    return head(z), z


def class_weights(y, idx, device):
    counts = torch.bincount(torch.from_numpy(y[idx].astype(np.int64)), minlength=7).float().to(device)
    return len(idx) / (7 * torch.clamp(counts, min=1.0))


def evaluate(model, head, crops, idx, y, device, batch_size=1024, keep_z=False):
    model.eval(); head.eval(); probs = []; zs = []
    with torch.inference_mode():
        for start in range(0, len(idx), batch_size):
            ii = idx[start:start + batch_size]
            x = batch_x(crops, ii, device, train=False)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, z = forward(model, head, x)
            probs.append(torch.softmax(logits, dim=1).float().cpu().numpy())
            if keep_z: zs.append(z.float().cpu().numpy())
    p = np.concatenate(probs, axis=0)
    return p, (np.concatenate(zs, axis=0) if keep_z else None)


def train_epochs(model, head, crops, train_idx, val_idx, y, device, epochs, batch_size=1024):
    weights = class_weights(y, train_idx, device)
    params = [{"params": [p for p in model.encoder.layer[-1].parameters() if p.requires_grad], "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -1.0; best_epoch = 1
    rng = np.random.default_rng(SEED + len(train_idx) + (len(val_idx) if val_idx is not None else 0))
    for ep in range(1, epochs + 1):
        model.train(); head.train(); order = rng.permutation(train_idx)
        for start in range(0, len(order), batch_size):
            ii = order[start:start + batch_size]
            x = batch_x(crops, ii, device, train=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = forward(model, head, x)
                loss = F.cross_entropy(logits, torch.from_numpy(y[ii].astype(np.int64)).to(device), weight=weights)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if val_idx is None:
            print(f"fine-tune refit epoch={ep} complete", flush=True)
        else:
            p, _ = evaluate(model, head, crops, val_idx, y, device, batch_size=batch_size)
            score = float(f1_score(y[val_idx], p.argmax(1), labels=np.arange(7), average="macro", zero_division=0))
            print(f"fine-tune epoch={ep} inner_macroF1={score:.4f}", flush=True)
            if score > best_score:
                best_score = score; best_epoch = ep
    return best_epoch, best_score


def metric_rows(candidate, fold, y, p):
    pred = p.argmax(1); f = f1_score(y, pred, labels=np.arange(7), average=None, zero_division=0)
    ap = [safe_ap((y == k).astype(int), p[:, k]) for k in range(7)]
    auc = [safe_auc((y == k).astype(int), p[:, k]) for k in range(7)]
    fold_row = {"candidate": candidate, "fold": fold, "accuracy": float(np.mean(pred == y)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "macro_f1": float(f.mean()), "macro_auprc": float(np.nanmean(ap)), "macro_auroc": float(np.nanmean(auc)), "weighted_f1": float(f1_score(y, pred, labels=np.arange(7), average="weighted", zero_division=0)), "lowest_three_f1": float(np.sort(f)[:3].mean())}
    per = []
    for k, name in enumerate(CLASSES):
        pr, rc, ff, _ = precision_recall_fscore_support(y, pred, labels=[k], zero_division=0)
        per.append({"candidate": candidate, "fold": fold, "class_id": k, "class_name": name, "precision": float(pr[0]), "recall": float(rc[0]), "f1": float(ff[0]), "auroc": auc[k], "auprc": ap[k], "support": int((y == k).sum())})
    return fold_row, per


def binary_row(candidate, fold, y, score, comp, pos, neg):
    keep = np.isin(y, [pos, neg]); yy = (y[keep] == pos).astype(int); s = score[keep]; pred = (s >= .5).astype(int)
    pr, rc, f, _ = precision_recall_fscore_support(yy, pred, labels=[1], zero_division=0)
    return {"candidate": candidate, "fold": fold, "comparison": comp, "auroc": safe_auc(yy, s), "auprc": safe_ap(yy, s), "f1": float(f[0]), "sensitivity": float(rc[0]), "precision": float(pr[0]), "n_val": int(keep.sum())}


def geometry(candidate, z, c):
    ids = pd.read_csv(T10 / "metrics/geometry_subset_cell_ids.csv")
    lookup = {str(v): i for i, v in enumerate(c.cell_id.astype(str))}; idx = np.asarray([lookup[str(v)] for v in ids.cell_id], dtype=int)
    x = StandardScaler().fit_transform(z[idx]).astype(np.float32); x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = c.class_id.to_numpy(int)[idx]; b = c.batch.astype(str).to_numpy()[idx]; s = c.patch_id.astype(str).to_numpy()[idx]
    nn = KNeighborsClassifier(n_neighbors=11, metric="cosine", n_jobs=-1).fit(x, y).kneighbors(return_distance=False)[:, 1:]
    return {"candidate": candidate, "n_geometry": len(idx), "knn_class_purity": float(np.mean(y[nn] == y[:, None])), "knn_batch_purity": float(np.mean(b[nn] == b[:, None])), "spatial_knn_purity": float(np.mean(s[nn] == s[:, None]))}


def main():
    seed_all(); ROOT.joinpath("work").mkdir(parents=True, exist_ok=True)
    c = load_canonical(); folds = load_folds(c); y = c.class_id.to_numpy(int); crops = crop_memmap(c)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []; per_rows = []; bin_rows = []; all_z = np.zeros((len(c), 3072), dtype=np.float32); selected_epochs = []
    for fold, (tr, va) in folds.items():
        batches = sorted(c.batch.astype(str).to_numpy()[tr].tolist()); inner_batch = sorted(set(batches))[-1]
        inner_va = tr[c.batch.astype(str).to_numpy()[tr] == inner_batch]; inner_tr = tr[c.batch.astype(str).to_numpy()[tr] != inner_batch]
        # Inner model for epoch selection.
        m = load_model(device); h = torch.nn.Linear(3072, 7).to(device)
        best_epoch, inner_score = train_epochs(m, h, crops, inner_tr, inner_va, y, device, epochs=3)
        del m, h; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
        # Outer-training refit from the frozen Midnight checkpoint at selected epoch count.
        m = load_model(device); h = torch.nn.Linear(3072, 7).to(device)
        selected_epochs.append({"fold": fold, "inner_val_batch": inner_batch, "selected_epoch": best_epoch, "inner_macro_f1": inner_score})
        train_epochs(m, h, crops, tr, None, y, device, epochs=best_epoch)
        p, z = evaluate(m, h, crops, va, y, device, keep_z=True)
        all_z[va] = z
        fr, pr = metric_rows("F1_FINAL_BLOCK", fold, y[va], p); fold_rows.append(fr); per_rows.extend(pr)
        for comp, pos, neg in [("neutrophil_vs_myeloid", 3, 2), ("neutrophil_vs_tb", 3, 5)]:
            score = p[:, 3] / np.maximum(p[:, 3] + p[:, neg], 1e-12); bin_rows.append(binary_row("F1_FINAL_BLOCK", fold, y[va], score, comp, pos, neg))
        print(f"outer fold={fold} macroF1={fr['macro_f1']:.4f} selected_epoch={best_epoch}", flush=True)
        del m, h; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    fd = pd.DataFrame(fold_rows); pd_ = pd.DataFrame(per_rows); bd = pd.DataFrame(bin_rows)
    summary = fd.groupby("candidate", as_index=False).agg(**{col: (col, "mean") for col in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "weighted_f1", "lowest_three_f1"]})
    for col in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "weighted_f1", "lowest_three_f1"]:
        summary[col + "_sd"] = fd.groupby("candidate")[col].std(ddof=1).to_numpy()
    gd = pd.DataFrame([geometry("F1_FINAL_BLOCK", all_z, c)])
    fd.to_csv(ROOT / "metrics/fine_tune_fold_metrics.csv", index=False); summary.to_csv(ROOT / "metrics/fine_tune_summary.csv", index=False); pd_.to_csv(ROOT / "metrics/fine_tune_per_class.csv", index=False); bd.to_csv(ROOT / "metrics/fine_tune_true_binary.csv", index=False); gd.to_csv(ROOT / "metrics/fine_tune_geometry.csv", index=False); pd.DataFrame(selected_epochs).to_csv(ROOT / "qc/fine_tune_inner_validation.csv", index=False)
    torch.save({"features": torch.from_numpy(all_z), "cell_ids": c.cell_id.tolist(), "source": "Task011 F1 final transformer block outer-validation embeddings", "fov_um": 12, "seed": SEED}, ROOT / "features/fine_tuned_f1_embeddings.pt")
    old_gate = json.loads((ROOT / "config/fine_tuning_gate.json").read_text())
    old_gate.update({"status": "EXECUTED_F1_FINAL_BLOCK", "fine_tuning_executed": True, "f1_candidate": "F1_FINAL_BLOCK", "lora_stage": "NOT_RUN_NO_LOCAL_IMPLEMENTATION", "production_model_modified": False})
    (ROOT / "config/fine_tuning_gate.json").write_text(json.dumps(old_gate, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__": main()
