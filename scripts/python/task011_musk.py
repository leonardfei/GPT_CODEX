#!/usr/bin/env python3
"""Task011-MUSK: frozen official MUSK benchmark on the canonical H&E cohort.

Large MUSK feature tensors stay on the server.  This script is deliberately
offline: it loads the already-uploaded official source tree and safetensors
checkpoint and never calls Hugging Face.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch
from PIL import Image, ImageDraw

T10 = Path("/data/lf_data/result/task010_representation_benchmark")
T11 = Path("/data/lf_data/result/task011_midnight_local_optimization")
TASK009 = Path("/data/lf_data/result/task009_v3_retraining")
ROOT = Path("/data/lf_data/result/task011_musk_benchmark")
WSI = Path("/data/lf_data/xenium_data/ID0060276.ome.tif")
MUSK_CODE = Path("/data/lf_data/models/MUSK-code")
MUSK_MODEL = Path("/data/lf_data/models/musk")
MUSK_WEIGHTS = MUSK_MODEL / "model.safetensors"
SEED = 20260923
SCALE_UM_PER_PX = 0.2125
CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
FOVS = {8: 37, 12: 57, 16: 75, 24: 113, 56: 263}

sys.path.insert(0, str(T11 / "code"))
sys.path.insert(0, str(T10 / "code"))
import task011_midnight_optimization as old  # noqa: E402
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD  # noqa: E402


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdirs() -> None:
    for d in ["config", "metrics", "features", "figures", "qc", "logs", "code", "work"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def version(name: str) -> str:
    try:
        m = __import__(name)
        return str(getattr(m, "__version__", "unknown"))
    except Exception as e:
        return f"unavailable:{type(e).__name__}"


def load_canonical() -> pd.DataFrame:
    p = T10 / "metrics/canonical_cell_order.csv.gz"
    c = pd.read_csv(p)
    required = ["canonical_index", "cell_id", "batch", "patch_id", "patch_x", "patch_y", "he_x", "he_y", "class_id", "class_name"]
    missing = [x for x in required if x not in c.columns]
    if missing or len(c) != 96044 or c.cell_id.duplicated().any():
        raise RuntimeError(f"canonical QC failed: missing={missing}, n={len(c)}, duplicate_ids={int(c.cell_id.duplicated().sum())}")
    if not np.array_equal(c.canonical_index.to_numpy(), np.arange(len(c))):
        raise RuntimeError("canonical_index is not contiguous")
    if set(c.class_name.astype(str)) != set(CLASSES):
        raise RuntimeError("canonical class set changed")
    return c.reset_index(drop=True)


def load_folds(c: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    sm = pd.read_csv(TASK009 / "metrics/split_manifest.csv")
    sm = sm[sm.condition.eq("CORE")].sort_values("fold")
    if len(sm) != 5:
        raise RuntimeError(f"expected five CORE folds, found {len(sm)}")
    out = {}
    lines = ["# Task011-MUSK fold audit", "", "Exact Task009 V3_CORE split_manifest.csv was reused.", "", "| fold | n_train | n_val | train batches | validation batches |", "|---:|---:|---:|---|---|"]
    for _, row in sm.iterrows():
        trb, vab = set(str(row.train_batches).split(";")), set(str(row.val_batches).split(";"))
        if trb & vab:
            raise RuntimeError(f"fold {row.fold} has train/validation batch overlap")
        tr = np.flatnonzero(c.batch.astype(str).isin(trb))
        va = np.flatnonzero(c.batch.astype(str).isin(vab))
        if not len(tr) or not len(va) or set(tr) & set(va):
            raise RuntimeError(f"fold {row.fold} has invalid cell assignment")
        out[int(row.fold)] = (tr, va)
        lines.append(f"| {int(row.fold)} | {len(tr):,} | {len(va):,} | {row.train_batches} | {row.val_batches} |")
    (ROOT / "qc/fold_equivalence_task009.md").write_text("\n".join(lines) + "\n")
    return out


def write_geometry_config() -> None:
    payload = {
        "source_wsi": str(WSI),
        "scale_um_per_px": SCALE_UM_PER_PX,
        "fovs": {str(um): {"physical_fov_um": [um, um], "native_side_px": side, "crop_center": "matched H&E nucleus center he_x/he_y"} for um, side in FOVS.items()},
        "preprocessing": "official MUSK resize 384, center crop 384, ImageNet inception normalization",
        "canonical_n": 96044,
        "seed": SEED,
    }
    (ROOT / "config/musk_crop_geometry.json").write_text(json.dumps(payload, indent=2) + "\n")


def crop_from_tile(tile: np.ndarray, row: pd.Series, side: int, ox: int, oy: int) -> np.ndarray:
    half = side // 2
    sx = int(round(float(row.he_x))) - half - ox
    sy = int(round(float(row.he_y))) - half - oy
    arr = tile[sy:sy + side, sx:sx + side, :3]
    if arr.shape != (side, side, 3):
        raise RuntimeError(f"invalid crop FOV side={side}, cell={row.cell_id}, shape={arr.shape}")
    return arr


def write_crop_manifest(c: pd.DataFrame) -> None:
    path = ROOT / "metrics/musk_crop_manifest.csv.gz"
    if path.exists():
        q = pd.read_csv(path, usecols=["cell_id", "fov_um", "native_side_px"])
        if len(q) == len(c) * len(FOVS):
            return
    rows = []
    for um, side in FOVS.items():
        half = side // 2
        for r in c.itertuples(index=False):
            rows.append({
                "canonical_index": int(r.canonical_index), "cell_id": str(r.cell_id), "batch": str(r.batch), "patch_id": str(r.patch_id),
                "class_id": int(r.class_id), "class_name": str(r.class_name), "he_x": float(r.he_x), "he_y": float(r.he_y),
                "fov_um": um, "native_side_px": side, "native_x0": int(round(float(r.he_x))) - half, "native_y0": int(round(float(r.he_y))) - half,
            })
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


def write_montages(c: pd.DataFrame) -> None:
    wsi = pyvips.Image.new_from_file(str(WSI), access="random")
    sample_idx = np.linspace(0, len(c) - 1, 16, dtype=int)
    for um, side in FOVS.items():
        out = ROOT / f"figures/musk_crop_montage_fov{um}.jpg"
        if out.exists():
            continue
        work = c.iloc[sample_idx]
        tile = None
        ox = oy = 0
        current = None
        ims = []
        for _, r in work.iterrows():
            if current != r.patch_id:
                tile, ox, oy = old.base._tile_array(wsi, int(r.patch_x), int(r.patch_y))
                current = r.patch_id
            arr = crop_from_tile(tile, r, side, ox, oy)
            ims.append(Image.fromarray(arr).resize((192, 192), Image.Resampling.BICUBIC))
        sheet = Image.new("RGB", (4 * 192, 4 * 212), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (im, (_, r)) in enumerate(zip(ims, work.iterrows())):
            x, y = (i % 4) * 192, (i // 4) * 212
            sheet.paste(im, (x, y))
            draw.text((x + 3, y + 193), f"{r.cell_id}  {r.class_name}", fill="black")
        sheet.save(out, quality=90)


def load_musk_model() -> tuple[torch.nn.Module, torch.device, int, dict]:
    if not MUSK_CODE.is_dir() or not MUSK_WEIGHTS.is_file():
        raise RuntimeError("BLOCKED_NEEDS_LOCAL_MUSK_UPLOAD: official local code or model.safetensors missing")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(MUSK_CODE))
    from musk import modeling, utils  # noqa: F401
    from timm.models import create_model
    model = create_model("musk_large_patch16_384")
    utils.load_model_and_may_interpolate(str(MUSK_WEIGHTS), model, "model|module", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        x = torch.zeros((1, 3, 384, 384), device=device, dtype=dtype)
        z_ms = model(image=x, with_head=False, out_norm=False, ms_aug=True, return_global=True)[0]
        z_single = model(image=x, with_head=False, out_norm=False, ms_aug=False, return_global=True)[0]
    dim_ms, dim_single = int(z_ms.shape[1]), int(z_single.shape[1])
    del x, z_ms, z_single
    if device.type == "cuda":
        torch.cuda.empty_cache()
    provenance = {
        "status": "LOADED_OFFLINE",
        "source_repository": "lilab-stanford/MUSK",
        "source_path": str(MUSK_CODE),
        "source_commit": subprocess.check_output(["git", "-C", str(MUSK_CODE), "rev-parse", "HEAD"], text=True).strip(),
        "model_id": "xiangjx/musk",
        "model_architecture": "musk_large_patch16_384",
        "weight_path": str(MUSK_WEIGHTS),
        "weight_sha256": sha256(MUSK_WEIGHTS),
        "image_tokenizer_path": str(MUSK_MODEL / "image_tokenizer.pth"),
        "tokenizer_path": str(MUSK_MODEL / "tokenizer.spm"),
        "input_size": 384,
        "feature_flags": {"with_head": False, "out_norm": False, "return_global": True, "ms_aug_primary": True, "ms_aug_sensitivity": False},
        "runtime_output_dimension_ms_aug_true": dim_ms,
        "runtime_output_dimension_ms_aug_false": dim_single,
        "device": str(device),
        "dtype": str(dtype),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "preprocessing": {"resize": 384, "center_crop": [384, 384], "mean": list(IMAGENET_INCEPTION_MEAN), "std": list(IMAGENET_INCEPTION_STD)},
        "license": "CC-BY-NC-ND-4.0; academic non-commercial use only",
        "loaded_entirely_offline": True,
        "hf_token_used": False,
        "python": sys.version,
        "packages": {"torch": version("torch"), "timm": version("timm"), "torchvision": version("torchvision"), "fairscale": version("fairscale"), "pandas": version("pandas"), "sklearn": version("sklearn"), "PIL": version("PIL"), "pyvips": version("pyvips")},
    }
    (ROOT / "config/musk_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=True) + "\n")
    (ROOT / "qc/musk_access_audit.md").write_text(
        "# MUSK access audit\n\n"
        f"- Official source: `lilab-stanford/MUSK`; local path: `{MUSK_CODE}`\n"
        f"- Official model: `xiangjx/musk`; local weights: `{MUSK_WEIGHTS}`\n"
        f"- Source commit: `{provenance['source_commit']}`\n"
        f"- Weight SHA256: `{provenance['weight_sha256']}`\n"
        "- Access mode: local files only; Hugging Face was not contacted and no token was used.\n"
        f"- Runtime feature dimensions: ms_aug=True `{dim_ms}`, ms_aug=False `{dim_single}`.\n"
        "- Model weights are not copied into this repository.\n"
    )
    return model, device, dim_ms, provenance


def transform_crop(arr: np.ndarray) -> torch.Tensor:
    im = Image.fromarray(arr, mode="RGB").resize((384, 384), Image.Resampling.BICUBIC)
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray(IMAGENET_INCEPTION_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_INCEPTION_STD, dtype=np.float32)[:, None, None]
    return torch.from_numpy((x - mean) / std)


def encode_batch(model: torch.nn.Module, device: torch.device, xs: list[torch.Tensor], ms_aug: bool) -> np.ndarray:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    x = torch.stack(xs).to(device=device, dtype=dtype, non_blocking=True)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = model(image=x, with_head=False, out_norm=False, ms_aug=ms_aug, return_global=True)[0]
        else:
            z = model(image=x, with_head=False, out_norm=False, ms_aug=ms_aug, return_global=True)[0]
    out = z.float().cpu().numpy()
    del x, z
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def valid_payload(path: Path, ids: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        p = torch.load(path, map_location="cpu")
        got = [str(x) for x in p["cell_ids"]]
        x = p["features"].float().numpy()
        if got != ids or x.shape[0] != len(ids) or not np.isfinite(x).all():
            return None
        return x
    except Exception:
        return None


def extract_features(c: pd.DataFrame, model: torch.nn.Module, device: torch.device, um: int, side: int, ms_aug: bool, batch_size: int) -> np.ndarray:
    tag = "ms" if ms_aug else "single"
    out_path = ROOT / f"features/musk_fov{um}_{tag}.pt"
    ids = c.cell_id.astype(str).tolist()
    cached = valid_payload(out_path, ids)
    if cached is not None:
        print(f"reuse {out_path} shape={cached.shape}", flush=True)
        return cached
    wsi = pyvips.Image.new_from_file(str(WSI), access="random")
    work = c.sort_values(["patch_id", "cell_id"], kind="stable").reset_index()
    rows, xs = [], []
    tile = None
    ox = oy = 0
    current = None
    t0 = time.time()

    def flush() -> None:
        if xs:
            rows.append(encode_batch(model, device, xs, ms_aug))
            xs.clear()

    for i, r in work.iterrows():
        if current != r.patch_id:
            tile, ox, oy = old.base._tile_array(wsi, int(r.patch_x), int(r.patch_y))
            current = r.patch_id
        xs.append(transform_crop(crop_from_tile(tile, r, side, ox, oy)))
        if len(xs) >= batch_size:
            flush()
        if i % 1000 == 0:
            print(f"extract FOV{um} {tag}: {i:,}/{len(work):,} elapsed_min={(time.time()-t0)/60:.1f}", flush=True)
    flush()
    z = np.concatenate(rows, axis=0)[np.argsort(work["index"].to_numpy())].astype(np.float32)
    if z.shape[0] != len(c) or not np.isfinite(z).all():
        raise RuntimeError(f"MUSK feature QC failed: {z.shape}")
    torch.save({"features": torch.from_numpy(z), "cell_ids": ids, "fov_um": um, "native_side_px": side, "ms_aug": ms_aug, "embedding_dim": int(z.shape[1]), "source_wsi": str(WSI), "scale_um_per_px": SCALE_UM_PER_PX, "preprocessing": "official MUSK resize/center crop 384 with ImageNet inception normalization", "feature_flags": {"with_head": False, "out_norm": False, "return_global": True}, "seed": SEED}, out_path)
    print(f"saved {out_path} shape={z.shape}", flush=True)
    return z


def binary_summary(bd: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auroc", "auprc", "f1", "sensitivity", "specificity", "precision"]
    m = bd.groupby(["candidate", "comparison"], as_index=False)[metrics].mean()
    s = bd.groupby(["candidate", "comparison"], as_index=False)[metrics].std(ddof=1).rename(columns={x: x + "_sd" for x in metrics})
    n = bd.groupby(["candidate", "comparison"], as_index=False)["n_val"].mean()
    return m.merge(s, on=["candidate", "comparison"]).merge(n, on=["candidate", "comparison"])


def run_benchmark(c: pd.DataFrame, folds: dict[int, tuple[np.ndarray, np.ndarray]], arrays: dict[str, np.ndarray], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = c.class_id.to_numpy(int)
    fold_rows, per_rows, bin_rows = [], [], []
    geom_rows = []
    for candidate, X in arrays.items():
        for fold, (tr, va) in folds.items():
            p = old.fit_probe(X, y, tr, va, 7)
            m = old.metric7(y[va], p)
            m.update({"candidate": candidate, "fold": fold, "n_train": len(tr), "n_val": len(va)})
            fold_rows.append(m)
            per_rows.extend(old.per_class_rows(candidate, fold, y[va], p))
            for comparison, pos, neg in [("neutrophil_vs_myeloid", 3, 2), ("neutrophil_vs_tb", 3, 5)]:
                score = old.fit_binary_scores(X, y, tr, va, pos, neg)
                bin_rows.append(old.binary_rows(candidate, fold, y[va], np.column_stack([1-score, score]), comparison, pos, neg))
            print(f"benchmark {candidate} fold={fold} macroF1={m['macro_f1']:.4f}", flush=True)
        geom_rows.append(old.geometry_one(candidate, X, c))
        del X
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    fd, pc, bd, gd = pd.DataFrame(fold_rows), pd.DataFrame(per_rows), pd.DataFrame(bin_rows), pd.DataFrame(geom_rows)
    fd.to_csv(ROOT / f"metrics/{prefix}_fold_metrics.csv", index=False)
    old.summarize_fold_metrics(fd).to_csv(ROOT / f"metrics/{prefix}_summary.csv", index=False)
    pc.to_csv(ROOT / f"metrics/{prefix}_per_class.csv", index=False)
    pc[pc.class_id.eq(3)].to_csv(ROOT / f"metrics/{prefix}_neutrophil_metrics.csv", index=False)
    bd.to_csv(ROOT / f"metrics/{prefix}_true_binary.csv", index=False)
    binary_summary(bd).to_csv(ROOT / f"metrics/{prefix}_true_binary_summary.csv", index=False)
    gd.to_csv(ROOT / f"metrics/{prefix}_representation_geometry.csv", index=False)
    return fd, pc, bd, gd


def load_feature(path: Path, ids: list[str]) -> np.ndarray:
    x = valid_payload(path, ids)
    if x is None:
        raise RuntimeError(f"invalid feature payload: {path}")
    return x


def select_best(summary: pd.DataFrame, binary: pd.DataFrame, geometry: pd.DataFrame) -> tuple[str, dict]:
    s = summary.copy()
    g = geometry.set_index("candidate")
    b = binary[binary.comparison.eq("neutrophil_vs_myeloid")].set_index("candidate")
    s["neutrophil_auprc"] = s.candidate.map(b.auprc)
    s["batch_purity"] = s.candidate.map(g.knn_batch_purity)
    s["spatial_purity"] = s.candidate.map(g.spatial_knn_purity)
    best = float(s.macro_f1.max())
    close = s[s.macro_f1 >= best - 0.01].sort_values(["neutrophil_auprc", "batch_purity", "spatial_purity", "macro_f1"], ascending=[False, True, True, False], kind="stable")
    chosen = str(close.iloc[0].candidate)
    decision = {"selection_rule": ["highest mean macro-F1", "within 0.01: higher true N-vs-Myeloid AUPRC", "then lower batch/spatial kNN purity"], "candidate": chosen, "best_macro_f1": best, "close_candidates": close.candidate.tolist()}
    (ROOT / "config/best_musk_fov.json").write_text(json.dumps(decision, indent=2) + "\n")
    return chosen, decision


def baseline_table(best: str, primary_summary: pd.DataFrame, primary_pc: pd.DataFrame, primary_geom: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t10s = pd.read_csv(T10 / "metrics/linear_probe_summary_corrected.csv")
    t10p = pd.read_csv(T10 / "metrics/linear_probe_per_class_corrected.csv")
    for rep, label, note in [("CELLVIT_TOKEN_ALIGNED", "CellViT aligned", "frozen"), ("PHIKON_V2_SMALL", "Phikon SMALL", "frozen"), ("MIDNIGHT12K_SMALL", "Midnight FOV12", "frozen")]:
        s = t10s[t10s.representation.eq(rep)].iloc[0]
        p = t10p[(t10p.representation.eq(rep)) & (t10p.class_id.eq(3))]
        rows.append({"model": label, "candidate": rep, "model_type": note, "macro_f1": s.macro_f1, "macro_auprc": s.macro_auprc, "neutrophil_f1": p.f1.mean(), "neutrophil_auprc": p.auprc.mean(), "comparison_note": "frozen representation"})
    fs = pd.read_csv(T11 / "metrics/fine_tune_summary_corrected.csv").iloc[0]
    fp = pd.read_csv(T11 / "metrics/fine_tune_per_class_corrected.csv")
    fp = fp[fp.class_id.eq(3)]
    rows.append({"model": "Midnight F1 final-block", "candidate": "F1_FINAL_BLOCK", "model_type": "partially fine-tuned", "macro_f1": fs.macro_f1, "macro_auprc": fs.macro_auprc, "neutrophil_f1": fp.f1.mean(), "neutrophil_auprc": fp.auprc.mean(), "comparison_note": "corrected Task011 candidate; not directly frozen-comparable"})
    s = primary_summary[primary_summary.candidate.eq(best)].iloc[0]
    p = primary_pc[(primary_pc.candidate.eq(best)) & (primary_pc.class_id.eq(3))]
    rows.append({"model": f"MUSK {best}", "candidate": best, "model_type": "frozen", "macro_f1": s.macro_f1, "macro_auprc": s.macro_auprc, "neutrophil_f1": p.f1.mean(), "neutrophil_auprc": p.auprc.mean(), "comparison_note": "official MUSK frozen linear probe"})
    return pd.DataFrame(rows)


def paired_deltas(best: str, fd: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    base_fd = pd.read_csv(T11 / "metrics/fov_sweep_fold_metrics.csv")
    base_fd = base_fd[base_fd.candidate.eq("FOV12")].set_index("fold")
    base_pc = pd.read_csv(T11 / "metrics/fov_sweep_per_class.csv")
    base_pc = base_pc[(base_pc.candidate.eq("FOV12")) & (base_pc.class_id.eq(3))].set_index("fold")
    a = fd[fd.candidate.eq(best)].set_index("fold")
    b = pc[(pc.candidate.eq(best)) & (pc.class_id.eq(3))].set_index("fold")
    rows = []
    for fold in sorted(a.index):
        rows.append({"candidate": best, "fold": int(fold), "delta_macro_f1_vs_midnight_fov12": float(a.loc[fold, "macro_f1"] - base_fd.loc[fold, "macro_f1"]), "delta_neutrophil_f1_vs_midnight_fov12": float(b.loc[fold, "f1"] - base_pc.loc[fold, "f1"]), "delta_neutrophil_auprc_vs_midnight_fov12": float(b.loc[fold, "auprc"] - base_pc.loc[fold, "auprc"]), "macro_f1_improved": bool(a.loc[fold, "macro_f1"] > base_fd.loc[fold, "macro_f1"])})
    return pd.DataFrame(rows)


def write_sensitivity(primary_fd: pd.DataFrame, primary_pc: pd.DataFrame, primary_bd: pd.DataFrame, single_fd: pd.DataFrame, single_pc: pd.DataFrame, single_bd: pd.DataFrame, best: str) -> None:
    def row(fd: pd.DataFrame, pc: pd.DataFrame, bd: pd.DataFrame, candidate: str) -> dict:
        f = fd[fd.candidate.eq(candidate)]
        n = pc[(pc.candidate.eq(candidate)) & (pc.class_id.eq(3))]
        out = {"condition": candidate, "macro_f1": f.macro_f1.mean(), "macro_auprc": f.macro_auprc.mean(), "neutrophil_f1": n.f1.mean(), "neutrophil_auprc": n.auprc.mean()}
        for comp, short in [("neutrophil_vs_myeloid", "n_vs_myeloid"), ("neutrophil_vs_tb", "n_vs_tb")]:
            q = bd[(bd.candidate.eq(candidate)) & (bd.comparison.eq(comp))]
            out[short + "_auroc"] = q.auroc.mean(); out[short + "_auprc"] = q.auprc.mean()
        return out
    a, b = row(primary_fd, primary_pc, primary_bd, best), row(single_fd, single_pc, single_bd, "MUSK_BEST_FOV_SINGLE_SCALE")
    out = pd.DataFrame([a, b])
    for col in [x for x in out.columns if x not in ["condition"]]:
        out.loc[1, col + "_delta_vs_ms"] = out.loc[1, col] - out.loc[0, col]
    out.to_csv(ROOT / "metrics/musk_multiscale_sensitivity.csv", index=False)


def finalize(c: pd.DataFrame, folds: dict, model_prov: dict, best: str, primary_fd: pd.DataFrame, primary_pc: pd.DataFrame, primary_bd: pd.DataFrame, primary_gd: pd.DataFrame, single_fd: pd.DataFrame, single_pc: pd.DataFrame, single_bd: pd.DataFrame) -> dict:
    summary = pd.read_csv(ROOT / "metrics/musk_fov_summary.csv")
    base = pd.read_csv(T11 / "metrics/fov_sweep_summary.csv")
    base = base[base.candidate.eq("FOV12")].iloc[0]
    s = summary[summary.candidate.eq(best)].iloc[0]
    n = primary_pc[(primary_pc.candidate.eq(best)) & (primary_pc.class_id.eq(3))]
    g = primary_gd[primary_gd.candidate.eq(best)].iloc[0]
    b = primary_bd[(primary_bd.candidate.eq(best)) & (primary_bd.comparison.eq("neutrophil_vs_myeloid"))]
    improved = int((primary_fd[primary_fd.candidate.eq(best)].sort_values("fold").macro_f1.to_numpy() > pd.read_csv(T11 / "metrics/fov_sweep_fold_metrics.csv").query("candidate == 'FOV12'").sort_values("fold").macro_f1.to_numpy()).sum())
    promotion = {
        "macro_f1_gain_vs_frozen_midnight_fov12": float(s.macro_f1 - base.macro_f1),
        "neutrophil_f1_gain_vs_frozen_midnight_fov12": float(n.f1.mean() - base.neutrophil_f1),
        "neutrophil_auprc_gain_vs_frozen_midnight_fov12": float(n.auprc.mean() - base.neutrophil_auprc),
        "outer_folds_improved_macro_f1": improved,
        "batch_purity_delta_vs_frozen_midnight_fov12": float(g.knn_batch_purity - base.knn_batch_purity),
        "spatial_purity_delta_vs_frozen_midnight_fov12": float(g.spatial_knn_purity - base.spatial_knn_purity),
    }
    strong = bool(promotion["macro_f1_gain_vs_frozen_midnight_fov12"] >= 0.03 and (promotion["neutrophil_f1_gain_vs_frozen_midnight_fov12"] >= 0.03 or promotion["neutrophil_auprc_gain_vs_frozen_midnight_fov12"] >= 0.03) and improved >= 4 and promotion["batch_purity_delta_vs_frozen_midnight_fov12"] <= 0.05 and promotion["spatial_purity_delta_vs_frozen_midnight_fov12"] <= 0.05)
    moderate = bool(not strong and (promotion["macro_f1_gain_vs_frozen_midnight_fov12"] >= 0.01 or promotion["neutrophil_f1_gain_vs_frozen_midnight_fov12"] >= 0.02 or promotion["neutrophil_auprc_gain_vs_frozen_midnight_fov12"] >= 0.02) and improved >= 3)
    decision = {"status": "TASK011_MUSK_COMPLETE", "best_musk_fov": best, "best_musk_fov_um": int(best.replace("MUSK_FOV", "")), "best_musk_fov_native_side_px": FOVS[int(best.replace("MUSK_FOV", ""))], "primary_ms_aug": True, "multiscale_sensitivity_completed": True, "frozen_gain_category": "STRONG" if strong else ("MODERATE" if moderate else "NO_MEANINGFUL_GAIN"), "frozen_gain_vs_midnight_fov12": promotion, "true_binary_probes": True, "model_provenance": model_prov, "task012_primary_replacement_recommended": False, "task012_recommendation": "Retain corrected Midnight F1_FINAL_BLOCK as the primary validation candidate; MUSK is frozen and is a secondary representation candidate even if it strongly improves over frozen Midnight.", "license_next_step": "Any MUSK fine-tuning requires academic non-commercial use under CC-BY-NC-ND-4.0 and must not redistribute weights.", "production_model_modified": False, "ground_truth_modified": False}
    (ROOT / "metrics/decision_summary.json").write_text(json.dumps(decision, indent=2, allow_nan=True) + "\n")
    baseline_table(best, summary, primary_pc, primary_gd).to_csv(ROOT / "metrics/musk_vs_existing_baselines.csv", index=False)
    paired_deltas(best, primary_fd, primary_pc).to_csv(ROOT / "metrics/musk_paired_fold_deltas.csv", index=False)
    return decision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "extract"], default="all")
    ap.add_argument("--fovs", default=",".join(str(x) for x in FOVS), help="comma-separated primary FOVs for --stage extract")
    args = ap.parse_args()
    seed_everything(); mkdirs()
    c = load_canonical(); folds = load_folds(c); write_geometry_config(); write_crop_manifest(c); write_montages(c)
    model, device, dim_ms, prov = load_musk_model()
    batch_size = int(os.environ.get("MUSK_BATCH_SIZE", "16"))
    if args.stage == "extract":
        requested = [int(x) for x in args.fovs.split(",") if x.strip()]
        unknown = [x for x in requested if x not in FOVS]
        if unknown:
            raise RuntimeError(f"unknown FOVs: {unknown}")
        for um in requested:
            extract_features(c, model, device, um, FOVS[um], True, batch_size)
        print(json.dumps({"status": "PRIMARY_FEATURE_EXTRACTION_COMPLETE", "fovs": requested, "device": str(device), "batch_size": batch_size}, indent=2), flush=True)
        return
    primary_paths = {f"MUSK_FOV{um}": ROOT / f"features/musk_fov{um}_ms.pt" for um in FOVS}
    arrays = {name: extract_features(c, model, device, int(name.replace("MUSK_FOV", "")), FOVS[int(name.replace("MUSK_FOV", ""))], True, batch_size) for name in primary_paths}
    fd, pc, bd, gd = run_benchmark(c, folds, arrays, "musk_fov")
    primary_summary = pd.read_csv(ROOT / "metrics/musk_fov_summary.csv")
    primary_binary_summary = pd.read_csv(ROOT / "metrics/musk_fov_true_binary_summary.csv")
    best, selection = select_best(primary_summary, primary_binary_summary, gd)
    best_um = int(best.replace("MUSK_FOV", ""))
    single = {"MUSK_BEST_FOV_SINGLE_SCALE": extract_features(c, model, device, best_um, FOVS[best_um], False, batch_size)}
    sfd, spc, sbd, sgd = run_benchmark(c, folds, single, "musk_single_scale")
    write_sensitivity(fd, pc, bd, sfd, spc, sbd, best)
    decision = finalize(c, folds, prov, best, fd, pc, bd, gd, sfd, spc, sbd)
    (ROOT / "config/musk_run.json").write_text(json.dumps({"status": "COMPLETE", "canonical_n": len(c), "folds": sorted(folds), "batch_size": batch_size, "primary_feature_dimension": dim_ms, "best_selection": selection, "decision": decision}, indent=2, allow_nan=True) + "\n")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(decision, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
