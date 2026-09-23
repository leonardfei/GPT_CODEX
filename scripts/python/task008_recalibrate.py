#!/usr/bin/env python3
"""Task 008: target-centered Neutrophil H&E QC recalibration.

Task007 biological identity is read-only input.  This script recalibrates only
H&E nuclear evidence and Neutrophil training eligibility, using empirical
centroid-to-nucleus offsets from technically valid non-Neutrophil cells.
CellViT is intentionally not used or trained.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pyvips
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree


SEED = 42
RNG = np.random.default_rng(SEED)

INPUT_H5AD = Path("/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad")
INPUT_HE = Path("/data/lf_data/xenium_data/ID0060276.ome.tif")
INPUT_MATRIX = Path("/data/lf_data/xenium_data/matrix.csv")
TASK7_ANNOT = Path("/data/lf_data/result/task007_xenium5k_panelaware/metrics/xenium_v3_annotations.csv.gz")
TASK7_H5AD = Path("/data/lf_data/result/task007_xenium5k_panelaware/adata_xenium_v3_panelaware.h5ad")
TASK7_HE_QC = Path("/data/lf_data/result/task007_xenium5k_panelaware/metrics/neutrophil_he_qc_summary.csv")
OUT = Path("/data/lf_data/result/task008_neutrophil_he_recalibration")

SUBDIRS = ["config", "code", "qc", "metrics", "figures", "logs", "work"]
REF_CLASSES = ["T and B", "Myeloid", "Endothelial", "Mesenchymal", "Plasma cell", "Tumor"]
NEUT_SUBTYPES = ["Neutrophil", "Neutrophil_CXCR4"]
CROP_SIZE = 128
CENTER = 64
HE_VECTOR = np.asarray([0.65, 0.70, 0.29], dtype=np.float32)
HE_VECTOR /= np.linalg.norm(HE_VECTOR)
PRIMARY_H_PERCENTILE = 80.0
PRIMARY_MIN_COMPONENT_AREA = 12
PRIMARY_GROUP_DISTANCE = 18.0
UPPER_TOLERANCE_CAP_PX = 48.0


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def dec(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return x


def decode_arr(x):
    a = np.asarray(x)
    if a.dtype.kind in "SUO":
        return np.asarray([dec(v) for v in a], dtype=object)
    return a


def read_h5_col(obj):
    if isinstance(obj, h5py.Dataset):
        return decode_arr(obj[:])
    if "categories" in obj and "codes" in obj:
        cats = decode_arr(obj["categories"][:])
        codes = np.asarray(obj["codes"][:], dtype=np.int64)
        out = np.empty(codes.shape, dtype=object)
        out[:] = None
        good = codes >= 0
        out[good] = cats[codes[good]]
        return out
    raise TypeError(f"Unsupported HDF5 column: {obj.name}")


def ensure_dirs() -> None:
    for sub in SUBDIRS:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    (OUT / "qc/review_montages").mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, path: Path, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip" if gzip else None)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def he_optical_density(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb[..., :3], dtype=np.float32)
    od = -np.log((x + 1.0) / 256.0)
    return np.tensordot(od, HE_VECTOR, axes=([-1], [0])).astype(np.float32)


def _component_features(labels: np.ndarray, h_od: np.ndarray, min_area: int = 4) -> list[dict]:
    objs = ndimage.find_objects(labels)
    comps = []
    for label_id, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        sub = labels[slc] == label_id
        area = int(sub.sum())
        if area < min_area:
            continue
        yy, xx = np.where(sub)
        yy = yy + y0
        xx = xx + x0
        cy = float(yy.mean())
        cx = float(xx.mean())
        mask = np.zeros_like(labels, dtype=bool)
        mask[yy, xx] = True
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        perimeter = int(np.count_nonzero(mask ^ eroded))
        compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1))
        if len(xx) >= 3:
            cov = np.cov(np.vstack([xx, yy]))
            vals = np.linalg.eigvalsh(cov)
            vals = np.maximum(vals, 0)
            eccentricity = float(math.sqrt(max(0.0, 1.0 - vals[0] / max(vals[1], 1e-8))))
        else:
            eccentricity = 1.0
        comps.append({
            "area": area,
            "cx": cx,
            "cy": cy,
            "distance": float(math.hypot(cx - CENTER, cy - CENTER)),
            "mean_h_od": float(h_od[mask].mean()),
            "max_h_od": float(h_od[mask].max()),
            "x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1),
            "perimeter": perimeter,
            "compactness": compactness,
            "eccentricity": eccentricity,
        })
    return comps


def extract_components(rgb: np.ndarray, percentile: float, min_raw_area: int = 4) -> tuple[np.ndarray, list[dict], float]:
    h_od = he_optical_density(rgb)
    threshold = float(np.percentile(h_od, percentile))
    # A local percentile is used for stain-aware adaptation. Opening removes
    # isolated pixels; closing preserves nearby lobes without merging distant
    # components through the whole context patch.
    mask = h_od >= threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return h_od, _component_features(labels, h_od, min_area=min_raw_area), threshold


def crop_array(img: pyvips.Image, x: float, y: float, size: int = CROP_SIZE) -> np.ndarray:
    crop = img.crop(int(round(x)), int(round(y)), size, size)
    return np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8, shape=(crop.height, crop.width, crop.bands)).copy()


def inbounds(x: float, y: float, width: int, height: int, half: int = CENTER) -> bool:
    return x >= half and y >= half and x < width - half and y < height - half


def stratified_sample(df: pd.DataFrame, cls: str, target: int, seed: int = SEED) -> np.ndarray:
    sub = df[df["original_cl1_7class"].eq(cls)].copy()
    if sub.empty:
        return np.array([], dtype=np.int64)
    # Avoid Python's process-randomized hash so the stratified sample is
    # reproducible across runs and hosts.
    stable_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(cls)) % 10000
    rng = np.random.default_rng(seed + stable_offset)
    per_batch = {}
    for batch, part in sub.groupby("batch", sort=True):
        arr = part["cell_index"].to_numpy(dtype=np.int64).copy()
        rng.shuffle(arr)
        per_batch[str(batch)] = list(arr)
    selected = []
    batches = sorted(per_batch)
    while len(selected) < min(target, len(sub)) and batches:
        progressed = False
        for batch in batches:
            if per_batch[batch]:
                selected.append(per_batch[batch].pop())
                progressed = True
                if len(selected) >= min(target, len(sub)):
                    break
        if not progressed:
            break
    return np.asarray(selected, dtype=np.int64)


def nearest_center_component(comps: list[dict], min_area: int = 15) -> dict | None:
    valid = [c for c in comps if c["area"] >= min_area]
    return min(valid, key=lambda c: (c["distance"], -c["area"])) if valid else None


def group_components(comps: list[dict], tolerance: float, min_area: int, group_distance: float) -> tuple[list[dict], dict | None]:
    valid = [c for c in comps if c["area"] >= min_area and c["distance"] <= tolerance]
    if not valid:
        return [], None
    seed = min(valid, key=lambda c: (c["distance"], -c["area"]))
    group = [seed]
    changed = True
    while changed:
        changed = False
        for comp in valid:
            if comp in group:
                continue
            if any(math.hypot(comp["cx"] - g["cx"], comp["cy"] - g["cy"]) <= group_distance for g in group):
                group.append(comp)
                changed = True
    if len(group) > 5:
        group = sorted(group, key=lambda c: (c["distance"], -c["area"]))[:5]
    total_area = float(sum(c["area"] for c in group))
    cx = sum(c["cx"] * c["area"] for c in group) / max(total_area, 1)
    cy = sum(c["cy"] * c["area"] for c in group) / max(total_area, 1)
    span = 0.0
    for a in group:
        for b in group:
            span = max(span, math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))
    summary = {
        "n_components": len(group),
        "total_area": total_area,
        "centroid_distance": float(math.hypot(cx - CENTER, cy - CENTER)),
        "span": span,
        "mean_h_od": float(sum(c["mean_h_od"] * c["area"] for c in group) / max(total_area, 1)),
        "max_component_area": max(c["area"] for c in group),
        "small_fraction": float(sum(c["area"] < 40 for c in group) / len(group)),
        "group": group,
    }
    return group, summary


def context_features(comps: list[dict]) -> dict:
    small = sum(1 for c in comps if 4 <= c["area"] < 40)
    medium = sum(1 for c in comps if 40 <= c["area"] < 120)
    large = sum(1 for c in comps if c["area"] >= 120)
    n = max(len(comps), 1)
    fragmentation = min(1.0, (small + 0.5 * medium) / 18.0)
    loss_of_intact = 1.0 - min(1.0, large / 3.0)
    score = float(np.clip(0.65 * fragmentation + 0.35 * loss_of_intact, 0, 1))
    return {
        "context_fragment_density": float((small + medium) / n),
        "context_intact_nuclei_density": float(large / n),
        "context_necrosis_score": score,
        "context_component_count": len(comps),
    }


def raw_target_evidence(rgb: np.ndarray, comps: list[dict], tolerance: float, min_area: int, group_distance: float, ref_area_low: float) -> dict:
    h_od = he_optical_density(rgb)
    center32 = h_od[48:80, 48:80]
    center64 = h_od[32:96, 32:96]
    center_signal_fraction = float((center32 >= np.percentile(h_od, PRIMARY_H_PERCENTILE)).mean())
    group, gs = group_components(comps, tolerance, min_area, group_distance)
    ctx = context_features(comps)
    out = {
        "center_signal_fraction": center_signal_fraction,
        "center_component_count": int(sum(c["distance"] <= 16 for c in comps if c["area"] >= min_area)),
        "nearest_nuclear_component_distance_px": np.nan,
        "nearest_component_area_px": np.nan,
        "nearest_component_mean_H_OD": np.nan,
        "center_associated_n_components": 0,
        "center_group_total_area_px": 0.0,
        "center_group_centroid_distance_px": np.nan,
        "center_group_span_px": np.nan,
        "center_group_H_OD": np.nan,
        "target_evidence": "NO_TARGET_NUCLEUS",
        "strong_center_fragmentation": False,
        **ctx,
    }
    near = nearest_center_component(comps, min_area=min_area)
    if near is not None:
        out["nearest_nuclear_component_distance_px"] = near["distance"]
        out["nearest_component_area_px"] = near["area"]
        out["nearest_component_mean_H_OD"] = near["mean_h_od"]
    if gs is None:
        return out
    out["center_associated_n_components"] = int(gs["n_components"])
    out["center_group_total_area_px"] = float(gs["total_area"])
    out["center_group_centroid_distance_px"] = float(gs["centroid_distance"])
    out["center_group_span_px"] = float(gs["span"])
    out["center_group_H_OD"] = float(gs["mean_h_od"])
    very_small = gs["max_component_area"] < 40 and gs["small_fraction"] >= 0.75
    low_area = gs["total_area"] < max(ref_area_low, 24.0)
    fragmented = bool(low_area and (very_small or gs["n_components"] >= 2) and ctx["context_necrosis_score"] >= 0.35)
    if fragmented:
        out["target_evidence"] = "FRAGMENTED_TARGET_SUSPECT"
        out["strong_center_fragmentation"] = True
    else:
        out["target_evidence"] = "TARGET_NUCLEUS_PRESENT"
    return out


def agreement_category(target_evidence: str, nucleus_count: float, nucleus_area: float) -> str:
    xenium_present = nucleus_count >= 1 and nucleus_area > 0
    if target_evidence == "TARGET_NUCLEUS_PRESENT" and xenium_present:
        return "agree_present"
    if target_evidence in {"NO_TARGET_NUCLEUS", "FRAGMENTED_TARGET_SUSPECT"} and not xenium_present:
        return "agree_absent"
    if target_evidence == "TARGET_NUCLEUS_PRESENT" and not xenium_present:
        return "HE_present_Xenium_absent"
    if target_evidence in {"NO_TARGET_NUCLEUS", "FRAGMENTED_TARGET_SUSPECT"} and xenium_present:
        return "HE_absent_Xenium_present"
    return "ambiguous"


def finalize_status(raw: dict, agreement: str, registration_ok: bool) -> str:
    if not registration_ok:
        return "REGISTRATION_UNCERTAIN"
    if agreement in {"HE_present_Xenium_absent", "HE_absent_Xenium_present"}:
        return "MANUAL_REVIEW"
    if raw["target_evidence"] == "FRAGMENTED_TARGET_SUSPECT":
        return "FRAGMENTED_TARGET_SUSPECT"
    if raw["target_evidence"] == "NO_TARGET_NUCLEUS":
        return "NO_TARGET_NUCLEUS"
    return "TARGET_NUCLEUS_PRESENT"


def status_from_variant(raw: dict, tolerance: float, ref_area_low: float) -> str:
    if raw["target_evidence"] == "FRAGMENTED_TARGET_SUSPECT":
        return "FRAGMENTED_TARGET_SUSPECT"
    return raw["target_evidence"]


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def draw_review_tile(rgb: np.ndarray, row: pd.Series, tolerance: float) -> Image.Image:
    context = Image.fromarray(rgb[..., :3]).convert("RGB")
    context = context.resize((160, 160))
    draw = ImageDraw.Draw(context)
    scale = 160 / CROP_SIZE
    cx = CENTER * scale
    cy = CENTER * scale
    r = tolerance * scale
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 220, 0), width=2)
    draw.line((cx - 5, cy, cx + 5, cy), fill=(255, 0, 0), width=2)
    draw.line((cx, cy - 5, cx, cy + 5), fill=(255, 0, 0), width=2)
    center = Image.fromarray(rgb[32:96, 32:96, :3]).convert("RGB").resize((160, 160))
    tile = Image.new("RGB", (340, 235), "white")
    tile.paste(context, (5, 5))
    tile.paste(center, (170, 5))
    d = ImageDraw.Draw(tile)
    text = (f"{row.get('original_cl1','')} | {row.get('batch','')}\n"
            f"nuc={row.get('nucleus_count',np.nan):.0f}, area={row.get('nucleus_area',np.nan):.1f}, "
            f"counts={row.get('total_counts',np.nan):.0f}\n"
            f"T7={row.get('task007_status','')} -> T8={row.get('task008_status','')}\n"
            f"evidence={row.get('target_evidence','')}\n"
            f"necrosis={row.get('context_necrosis_score',np.nan):.2f}, "
            f"stab={row.get('status_stability_fraction',np.nan):.2f}")
    d.multiline_text((5, 170), text[:260], fill="black", spacing=2)
    return tile


def save_review_montage(df: pd.DataFrame, img: pyvips.Image, tolerance: float, path: Path) -> None:
    categories = {
        "T7 debris -> T8 present": (df.task007_status.eq("debris_suspect") & df.target_evidence.eq("TARGET_NUCLEUS_PRESENT")),
        "T7 debris -> T8 no target": (df.task007_status.eq("debris_suspect") & df.task008_status.eq("NO_TARGET_NUCLEUS")),
        "T7 debris -> T8 fragmented": (df.task007_status.eq("debris_suspect") & df.task008_status.eq("FRAGMENTED_TARGET_SUSPECT")),
        "T7 debris -> T8 review": (df.task007_status.eq("debris_suspect") & df.task008_status.eq("MANUAL_REVIEW")),
        "T7 intact -> T8 present": (df.task007_status.eq("intact_or_plausible") & df.task008_status.eq("TARGET_NUCLEUS_PRESENT")),
        "CXCR4 representatives": df.original_cl1.eq("Neutrophil_CXCR4"),
        "Conventional representatives": df.original_cl1.eq("Neutrophil"),
        "High necrosis context": df.context_necrosis_score >= df.context_necrosis_score.quantile(0.90),
        "Low necrosis context": df.context_necrosis_score <= df.context_necrosis_score.quantile(0.10),
    }
    pdf_path = Path(path)
    with PdfPages(pdf_path) as pdf:
        for title, mask in categories.items():
            sub = df.loc[mask].copy()
            if sub.empty:
                continue
            n_take = min(100, len(sub))
            sub = sub.sample(n=n_take, random_state=SEED).sort_values("cell_index")
            tiles = []
            for _, row in sub.iterrows():
                if not bool(row["registration_ok"]):
                    continue
                rgb = crop_array(img, row["x_he_level0"] - CENTER, row["y_he_level0"] - CENTER)
                tiles.append(draw_review_tile(rgb, row, tolerance))
            for start in range(0, len(tiles), 10):
                page_tiles = tiles[start:start + 10]
                fig, axes = plt.subplots(2, 5, figsize=(15, 7))
                axes = axes.ravel()
                fig.suptitle(f"Task008 manual review: {title} ({start + 1}-{start + len(page_tiles)} / {len(tiles)})")
                for ax in axes:
                    ax.axis("off")
                for ax, tile in zip(axes, page_tiles):
                    ax.imshow(tile)
                    ax.axis("off")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def main() -> int:
    ensure_dirs()
    log("Task008 started; Task007 biological identity is frozen")
    for p in [INPUT_H5AD, INPUT_HE, INPUT_MATRIX, TASK7_ANNOT, TASK7_H5AD, TASK7_HE_QC]:
        if not p.exists():
            raise FileNotFoundError(p)

    with h5py.File(INPUT_H5AD, "r") as f:
        obs = f["obs"]
        keys = ["cell_id", "batch", "cl1", "cell_area", "nucleus_area", "nucleus_count", "total_counts", "n_genes_by_counts", "x", "y"]
        obs_cols = {k: read_h5_col(obs[k]) for k in keys if k in obs}
        spatial = np.asarray(f["obsm"]["spatial"][:], dtype=np.float32)
        n_cells = spatial.shape[0]
    ann = pd.read_csv(TASK7_ANNOT)
    if len(ann) != n_cells or not np.array_equal(ann["cell_index"].to_numpy(dtype=np.int64), np.arange(n_cells)):
        raise ValueError("Task007 annotation row order does not match source H5AD cell_index")
    if "original_cl1" not in ann or "v3_action" not in ann:
        raise ValueError("Task007 annotation lacks frozen biological/action fields")
    df = ann.copy()
    for key, vals in obs_cols.items():
        if key == "cell_id" or key == "batch" or key == "cl1":
            continue
        df[key] = pd.to_numeric(vals, errors="coerce")
    df["batch"] = df["batch"].astype(str)
    df["original_cl1"] = df["original_cl1"].astype(str)
    df["original_cl1_7class"] = df["original_cl1_7class"].astype(str)
    for key in ["cell_area", "nucleus_area", "nucleus_count", "total_counts", "n_genes_by_counts"]:
        df[key] = pd.to_numeric(df[key], errors="coerce").fillna(0)

    matrix = np.loadtxt(INPUT_MATRIX, delimiter=",")
    he_px = spatial / 0.2125
    he_hom = np.c_[he_px, np.ones(n_cells, dtype=np.float32)]
    he_xy = (np.linalg.inv(matrix) @ he_hom.T).T[:, :2]
    df["x_he_level0"] = he_xy[:, 0]
    df["y_he_level0"] = he_xy[:, 1]

    old_he = pd.read_csv(TASK7_HE_QC)
    old_he = old_he.rename(columns={"he_neutrophil_nucleus_status": "task007_status", "he_component_count": "task007_full_patch_component_count", "he_largest_component_px": "task007_largest_component_px"})
    old_he = old_he[[c for c in old_he.columns if c in {"cell_index", "cell_id", "batch", "task007_status", "he_signal_fraction", "task007_full_patch_component_count", "task007_largest_component_px", "he_registration_status"}]]
    neut_mask = df["original_cl1_7class"].eq("Neutrophil")
    neut = df.loc[neut_mask].copy()
    neut = neut.merge(old_he, on=["cell_index", "cell_id", "batch"], how="left", validate="one_to_one")
    log(f"Frozen Task007 broad Neutrophil candidates: {len(neut):,}")

    # Diagnostic local registered-cell density, used only to audit the old
    # whole-patch rule rather than to define the new target-centered status.
    tree = cKDTree(he_xy)
    try:
        density = tree.query_ball_point(neut[["x_he_level0", "y_he_level0"]].to_numpy(), r=64.0, return_length=True)
    except TypeError:
        density = np.asarray([len(x) for x in tree.query_ball_point(neut[["x_he_level0", "y_he_level0"]].to_numpy(), r=64.0)], dtype=np.int32)
    neut["local_registered_cell_density_64px"] = np.asarray(density, dtype=np.int32)
    audit_cols = ["cell_index", "cell_id", "batch", "original_cl1", "v3_action", "cell_quality_status", "total_counts", "n_genes_by_counts", "nucleus_count", "nucleus_area", "cell_area", "x_he_level0", "y_he_level0", "task007_status", "he_signal_fraction", "task007_full_patch_component_count", "task007_largest_component_px", "local_registered_cell_density_64px"]
    write_df(neut[audit_cols], OUT / "metrics/task007_he_status_audit.csv")

    # Reference cells: frozen Task007 KEEP + technically valid + Xenium nucleus.
    ref_base = (
        df["original_cl1_7class"].isin(REF_CLASSES)
        & df["v3_action"].eq("KEEP")
        & df["cell_quality_status"].eq("Pass")
        & (df["nucleus_count"] >= 1)
        & (df["nucleus_area"] > 0)
    )
    # The sampler expects the full frame; apply the reference mask first.
    ref_df = df.loc[ref_base].copy()
    ref_indices_list = []
    for cls in REF_CLASSES:
        ref_indices_list.append(stratified_sample(ref_df, cls, 1500))
    ref_indices = np.concatenate(ref_indices_list) if ref_indices_list else np.array([], dtype=np.int64)
    ref_indices = np.unique(ref_indices)
    log(f"Reference cells selected: {len(ref_indices):,}")

    he_img = pyvips.Image.new_from_file(str(INPUT_HE), page=0, access="random")
    ref_rows = []
    primary_ref_components = []
    for j, idx in enumerate(ref_indices):
        row = df.iloc[int(idx)]
        ok = inbounds(float(row["x_he_level0"]), float(row["y_he_level0"]), he_img.width, he_img.height)
        result = {"cell_index": int(idx), "cell_id": str(row["cell_id"]), "class": str(row["original_cl1_7class"]), "batch": str(row["batch"]), "registration_ok": ok}
        if ok:
            rgb = crop_array(he_img, row["x_he_level0"] - CENTER, row["y_he_level0"] - CENTER)
            h_od, comps, thr = extract_components(rgb, PRIMARY_H_PERCENTILE)
            near = nearest_center_component(comps, min_area=PRIMARY_MIN_COMPONENT_AREA)
            result.update({
                "nearest_nuclear_component_distance_px": near["distance"] if near else np.nan,
                "nearest_component_area_px": near["area"] if near else np.nan,
                "nearest_component_mean_H_OD": near["mean_h_od"] if near else np.nan,
                "center_signal_fraction": float((h_od[48:80, 48:80] >= thr).mean()),
                "center_component_count": int(sum(c["distance"] <= 16 and c["area"] >= PRIMARY_MIN_COMPONENT_AREA for c in comps)),
                "component_count": len(comps),
                "threshold_H_OD": thr,
            })
            primary_ref_components.append((int(idx), comps))
        else:
            result.update({"nearest_nuclear_component_distance_px": np.nan, "nearest_component_area_px": np.nan, "nearest_component_mean_H_OD": np.nan, "center_signal_fraction": np.nan, "center_component_count": 0, "component_count": 0, "threshold_H_OD": np.nan})
        ref_rows.append(result)
        if j and j % 1000 == 0:
            log(f"Reference H&E scan: {j:,}/{len(ref_indices):,}")
    ref_qc = pd.DataFrame(ref_rows)
    write_df(ref_qc, OUT / "metrics/reference_cell_center_qc.csv.gz", gzip=True)
    valid_offsets = ref_qc.loc[ref_qc["nearest_nuclear_component_distance_px"].notna(), "nearest_nuclear_component_distance_px"].to_numpy(dtype=float)
    if len(valid_offsets) < 100:
        raise RuntimeError("Too few reference cells had detectable center-associated H&E components")
    qvals = {q: float(np.quantile(valid_offsets, q)) for q in [0.50, 0.75, 0.90, 0.95, 0.975]}
    selected_tolerance = float(min(UPPER_TOLERANCE_CAP_PX, max(8.0, qvals[0.95])))
    ref_area_values = ref_qc["nearest_component_area_px"].dropna().to_numpy(dtype=float)
    ref_area_low = float(np.quantile(ref_area_values, 0.05)) if len(ref_area_values) else 24.0
    ref_area_p25 = float(np.quantile(ref_area_values, 0.25)) if len(ref_area_values) else 50.0
    tol_rows = [{"scope": "global", "group": "all", "n": len(valid_offsets), **{f"q{int(q*1000):03d}_distance_px": val for q, val in qvals.items()}, "selected_center_tolerance_px": selected_tolerance, "upper_sanity_cap_px": UPPER_TOLERANCE_CAP_PX}]
    for col in ["class", "batch"]:
        for group, sub in ref_qc.dropna(subset=["nearest_nuclear_component_distance_px"]).groupby(col):
            vals = sub["nearest_nuclear_component_distance_px"].to_numpy(dtype=float)
            qs = {q: float(np.quantile(vals, q)) for q in [0.50, 0.75, 0.90, 0.95, 0.975]}
            tol_rows.append({"scope": col, "group": group, "n": len(vals), **{f"q{int(q*1000):03d}_distance_px": val for q, val in qs.items()}, "selected_center_tolerance_px": selected_tolerance, "upper_sanity_cap_px": UPPER_TOLERANCE_CAP_PX})
    write_df(pd.DataFrame(tol_rows), OUT / "metrics/registration_tolerance_summary.csv")
    write_df(ref_qc, OUT / "metrics/registration_tolerance_reference.csv")
    write_json({"seed": SEED, "hematoxylin_vector": HE_VECTOR.tolist(), "primary_h_percentile": PRIMARY_H_PERCENTILE, "primary_min_component_area_px": PRIMARY_MIN_COMPONENT_AREA, "primary_group_distance_px": PRIMARY_GROUP_DISTANCE, "selected_center_tolerance_px": selected_tolerance, "reference_offset_quantiles_px": qvals, "reference_nearest_area_p05_px": ref_area_low, "reference_nearest_area_p25_px": ref_area_p25, "upper_sanity_cap_px": UPPER_TOLERANCE_CAP_PX, "source_files_unchanged": True}, OUT / "config/task008_parameters.json")

    # Fig1: empirical centroid offset calibration.
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid_offsets, bins=40, color="#4477AA", alpha=0.85)
    ax.axvline(qvals[0.95], color="crimson", lw=2, label=f"95th percentile={qvals[0.95]:.2f}px")
    ax.axvline(selected_tolerance, color="black", ls="--", label=f"selected={selected_tolerance:.2f}px")
    ax.set(xlabel="nearest plausible nuclear component distance (px)", ylabel="reference cells", title="Task008 center-offset calibration")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir / "Fig1_center_offset_calibration.pdf"); plt.close(fig)

    # Target-centered classification and one-at-a-time sensitivity variants.
    variants = [
        ("primary", PRIMARY_H_PERCENTILE, selected_tolerance, PRIMARY_MIN_COMPONENT_AREA, PRIMARY_GROUP_DISTANCE),
        ("h_threshold_low", PRIMARY_H_PERCENTILE - 4, selected_tolerance, PRIMARY_MIN_COMPONENT_AREA, PRIMARY_GROUP_DISTANCE),
        ("h_threshold_high", PRIMARY_H_PERCENTILE + 4, selected_tolerance, PRIMARY_MIN_COMPONENT_AREA, PRIMARY_GROUP_DISTANCE),
        ("tolerance_low", PRIMARY_H_PERCENTILE, max(8.0, selected_tolerance * 0.85), PRIMARY_MIN_COMPONENT_AREA, PRIMARY_GROUP_DISTANCE),
        ("tolerance_high", PRIMARY_H_PERCENTILE, min(UPPER_TOLERANCE_CAP_PX, selected_tolerance * 1.15), PRIMARY_MIN_COMPONENT_AREA, PRIMARY_GROUP_DISTANCE),
        ("min_area_low", PRIMARY_H_PERCENTILE, selected_tolerance, max(5, PRIMARY_MIN_COMPONENT_AREA - 5), PRIMARY_GROUP_DISTANCE),
        ("min_area_high", PRIMARY_H_PERCENTILE, selected_tolerance, PRIMARY_MIN_COMPONENT_AREA + 5, PRIMARY_GROUP_DISTANCE),
        ("group_distance_low", PRIMARY_H_PERCENTILE, selected_tolerance, PRIMARY_MIN_COMPONENT_AREA, max(10.0, PRIMARY_GROUP_DISTANCE - 4)),
        ("group_distance_high", PRIMARY_H_PERCENTILE, selected_tolerance, PRIMARY_MIN_COMPONENT_AREA, PRIMARY_GROUP_DISTANCE + 4),
    ]
    variant_results = {name: [] for name, *_ in variants}
    rows = []
    n_neut = len(neut)
    for j, (idx, row) in enumerate(neut.set_index("cell_index").iterrows()):
        row = row
        ok = inbounds(float(row["x_he_level0"]), float(row["y_he_level0"]), he_img.width, he_img.height)
        base = {"cell_index": int(idx), "cell_id": str(row["cell_id"]), "batch": str(row["batch"]), "original_cl1": str(row["original_cl1"]), "original_cl1_7class": str(row["original_cl1_7class"]), "v3_label": str(row["v3_label"]), "v3_action": str(row["v3_action"]), "cell_quality_status": str(row["cell_quality_status"]), "nucleus_count": float(row["nucleus_count"]), "nucleus_area": float(row["nucleus_area"]), "cell_area": float(row["cell_area"]), "total_counts": float(row["total_counts"]), "n_genes_by_counts": float(row["n_genes_by_counts"]), "x_he_level0": float(row["x_he_level0"]), "y_he_level0": float(row["y_he_level0"]), "registration_ok": ok, "task007_status": str(row.get("task007_status", ""))}
        if not ok:
            base.update({"target_evidence": "REGISTRATION_UNCERTAIN", "task008_status": "REGISTRATION_UNCERTAIN", "xenium_he_nucleus_agreement": "ambiguous", "status_stability_fraction": 1.0, "raw_status_stability_fraction": 1.0})
            for key in ["nearest_nuclear_component_distance_px", "nearest_component_area_px", "nearest_component_mean_H_OD", "center_signal_fraction", "center_component_count", "center_associated_n_components", "center_group_total_area_px", "center_group_centroid_distance_px", "center_group_span_px", "center_group_H_OD", "context_necrosis_score", "context_fragment_density", "context_intact_nuclei_density", "context_component_count"]:
                base[key] = np.nan
            rows.append(base)
            continue
        rgb = crop_array(he_img, row["x_he_level0"] - CENTER, row["y_he_level0"] - CENTER)
        comps_by_pct = {}
        raw_by_variant = {}
        for name, h_pct, tol, min_area, group_dist in variants:
            if h_pct not in comps_by_pct:
                _, comps_by_pct[h_pct], _ = extract_components(rgb, h_pct)
            raw = raw_target_evidence(rgb, comps_by_pct[h_pct], tol, min_area, group_dist, ref_area_low)
            raw_by_variant[name] = raw
            variant_results[name].append(raw["target_evidence"])
        primary = raw_by_variant["primary"]
        agreement = agreement_category(primary["target_evidence"], row["nucleus_count"], row["nucleus_area"])
        status = finalize_status(primary, agreement, ok)
        statuses = [raw_by_variant[name]["target_evidence"] for name, *_ in variants]
        stability = float(np.mean(np.asarray(statuses, dtype=object) == primary["target_evidence"]))
        base.update({k: v for k, v in primary.items() if k not in {"target_evidence"}})
        base.update({"target_evidence": primary["target_evidence"], "task008_status": status, "xenium_he_nucleus_agreement": agreement, "status_stability_fraction": stability, "raw_status_stability_fraction": stability})
        base["center_core_signal_fraction"] = primary["center_signal_fraction"]
        rows.append(base)
        if j and j % 2000 == 0:
            log(f"Target-centered H&E scan: {j:,}/{n_neut:,}")
    qc = pd.DataFrame(rows)
    # Reorder and preserve frozen Task007 identifiers.
    qc = qc.sort_values("cell_index").reset_index(drop=True)
    write_df(qc, OUT / "metrics/neutrophil_centered_he_qc.csv.gz", gzip=True)

    # Parameter sensitivity summary.
    sensitivity_rows = []
    for name, *_ in variants:
        vals = pd.Series(variant_results[name])
        for status, count in vals.value_counts(dropna=False).items():
            sensitivity_rows.append({"variant": name, "status": status, "n_cells": int(count), "fraction": float(count / len(vals))})
    sensitivity_rows.append({"variant": "per_cell_primary_stability", "status": "mean", "n_cells": len(qc), "fraction": float(qc["status_stability_fraction"].mean())})
    sensitivity_rows.append({"variant": "per_cell_primary_stability", "status": "fraction_ge_0.80", "n_cells": int((qc["status_stability_fraction"] >= 0.80).sum()), "fraction": float((qc["status_stability_fraction"] >= 0.80).mean())})
    write_df(pd.DataFrame(sensitivity_rows), OUT / "metrics/parameter_sensitivity.csv")

    # Transition and status summaries.
    trans = pd.crosstab(qc["task007_status"], qc["target_evidence"], dropna=False).reset_index()
    write_df(trans, OUT / "metrics/task007_to_task008_transition.csv")
    status_summary = qc["task008_status"].value_counts(dropna=False).rename_axis("task008_status").reset_index(name="n_cells")
    write_df(status_summary, OUT / "metrics/neutrophil_status_summary.csv")
    agreement_summary = qc["xenium_he_nucleus_agreement"].value_counts(dropna=False).rename_axis("xenium_he_nucleus_agreement").reset_index(name="n_cells")
    write_df(agreement_summary, OUT / "metrics/xenium_he_nucleus_agreement.csv")
    subtype_summary = qc.groupby("original_cl1", dropna=False).agg(
        n_cells=("cell_index", "size"),
        target_present=("target_evidence", lambda x: int((x == "TARGET_NUCLEUS_PRESENT").sum())),
        no_target=("task008_status", lambda x: int((x == "NO_TARGET_NUCLEUS").sum())),
        fragmented=("task008_status", lambda x: int((x == "FRAGMENTED_TARGET_SUSPECT").sum())),
        manual_review=("task008_status", lambda x: int((x == "MANUAL_REVIEW").sum())),
        registration_uncertain=("task008_status", lambda x: int((x == "REGISTRATION_UNCERTAIN").sum())),
    ).reset_index()
    for subtype in NEUT_SUBTYPES:
        if subtype not in subtype_summary.original_cl1.values:
            subtype_summary = pd.concat([subtype_summary, pd.DataFrame([{ "original_cl1": subtype, "n_cells": 0, "target_present": 0, "no_target": 0, "fragmented": 0, "manual_review": 0, "registration_uncertain": 0}])], ignore_index=True)
    # Provisional training eligibility: identity remains frozen; REVIEW cells
    # with plausible target evidence are retained in extended candidates but
    # remain explicitly marked for manual review.
    technical_pass = qc["cell_quality_status"].eq("Pass")
    frozen_keep = qc["v3_action"].eq("KEEP")
    core = technical_pass & frozen_keep & qc["task008_status"].eq("TARGET_NUCLEUS_PRESENT") & qc["xenium_he_nucleus_agreement"].eq("agree_present") & qc["status_stability_fraction"].ge(0.80)
    extended = technical_pass & frozen_keep & qc["target_evidence"].eq("TARGET_NUCLEUS_PRESENT") & ~qc["task008_status"].isin(["NO_TARGET_NUCLEUS", "FRAGMENTED_TARGET_SUSPECT", "REGISTRATION_UNCERTAIN"]) & qc["status_stability_fraction"].ge(0.60)
    qc["trainable_core_task008"] = core
    qc["trainable_extended_task008"] = extended
    qc["training_exclusion_reason_task008"] = np.where(core, "", np.where(qc["task008_status"].eq("NO_TARGET_NUCLEUS"), "NO_TARGET_NUCLEUS", np.where(qc["task008_status"].eq("FRAGMENTED_TARGET_SUSPECT"), "FRAGMENTED_TARGET_SUSPECT", np.where(qc["task008_status"].eq("REGISTRATION_UNCERTAIN"), "REGISTRATION_UNCERTAIN", np.where(~technical_pass, "TECHNICAL_QC_FAIL", np.where(~frozen_keep, "TASK007_IDENTITY_REVIEW", np.where(qc["status_stability_fraction"] < 0.60, "PARAMETER_SENSITIVE", "MANUAL_REVIEW")))))))
    eligibility_cols = ["cell_index", "cell_id", "batch", "original_cl1", "original_cl1_7class", "v3_label", "v3_action", "cell_quality_status", "task007_status", "target_evidence", "task008_status", "xenium_he_nucleus_agreement", "trainable_core_task008", "trainable_extended_task008", "training_exclusion_reason_task008", "status_stability_fraction", "context_necrosis_score", "center_group_total_area_px", "center_group_centroid_distance_px", "nucleus_count", "nucleus_area", "total_counts", "n_genes_by_counts"]
    write_df(qc[eligibility_cols], OUT / "metrics/neutrophil_training_eligibility_task008.csv.gz", gzip=True)
    write_df(subtype_summary, OUT / "metrics/neutrophil_retention_by_subtype.csv")
    batch_summary = qc.groupby("batch", dropna=False).agg(
        n_cells=("cell_index", "size"),
        core_eligible=("trainable_core_task008", "sum"),
        extended_eligible=("trainable_extended_task008", "sum"),
        hard_excluded=("task008_status", lambda x: int(x.isin(["NO_TARGET_NUCLEUS", "FRAGMENTED_TARGET_SUSPECT", "REGISTRATION_UNCERTAIN"]).sum())),
        manual_review=("task008_status", lambda x: int((x == "MANUAL_REVIEW").sum())),
    ).reset_index()
    batch_summary["core_retention"] = batch_summary["core_eligible"] / batch_summary["n_cells"]
    batch_summary["extended_retention"] = batch_summary["extended_eligible"] / batch_summary["n_cells"]
    batch_summary["hard_exclusion_rate"] = batch_summary["hard_excluded"] / batch_summary["n_cells"]
    write_df(batch_summary, OUT / "metrics/neutrophil_retention_by_batch.csv")

    # Reference non-Neutrophil QC agreement and safety rails.
    ref_target_no = int((ref_qc["nearest_nuclear_component_distance_px"].isna()).sum())
    ref_no_rate = float(ref_target_no / len(ref_qc))
    hard = qc["task008_status"].isin(["NO_TARGET_NUCLEUS", "FRAGMENTED_TARGET_SUSPECT", "REGISTRATION_UNCERTAIN"])
    hard_rate = float(hard.mean())
    subtype_hard = qc.groupby("original_cl1")[hard.name if hasattr(hard, "name") else "task008_status"].mean() if False else qc.assign(_hard=hard).groupby("original_cl1")['_hard'].mean().to_dict()
    batch_hard = qc.assign(_hard=hard).groupby("batch")['_hard'].mean().to_dict()
    batch_spread = float(max(batch_hard.values()) - min(batch_hard.values())) if batch_hard else 0.0
    subtype_spread = float(max(subtype_hard.values()) - min(subtype_hard.values())) if subtype_hard else 0.0
    mean_stability = float(qc["status_stability_fraction"].mean())
    safety = {
        "reference_cells_n": int(len(ref_qc)),
        "reference_no_target_rate": ref_no_rate,
        "neutrophil_hard_exclusion_rate": hard_rate,
        "hard_exclusion_by_subtype": subtype_hard,
        "hard_exclusion_by_batch": batch_hard,
        "batch_hard_exclusion_spread": batch_spread,
        "subtype_hard_exclusion_spread": subtype_spread,
        "mean_status_stability_fraction": mean_stability,
        "rail_1_retained_neutrophil_hard_exclusion_gt_50pct": hard_rate > 0.50,
        "rail_2_reference_no_target_gt_25pct": ref_no_rate > 0.25,
        "rail_3_batch_spread_gt_20pp": batch_spread > 0.20,
        "rail_4_subtype_spread_gt_20pp": subtype_spread > 0.20,
        "rail_5_parameter_sensitive": mean_stability < 0.80,
        "any_safety_trigger": bool(hard_rate > 0.50 or ref_no_rate > 0.25 or batch_spread > 0.20 or subtype_spread > 0.20 or mean_stability < 0.80),
        "manual_review_required_before_final": True,
        "task_status": "PARTIAL_REVIEW_REQUIRED",
    }
    write_json(safety, OUT / "metrics/safety_rail_decision.json")

    # Fig2: reproduce the Task007 whole-patch failure mode.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for status, sub in neut.groupby("task007_status", dropna=False):
        axes[0].hist(sub["task007_full_patch_component_count"].dropna(), bins=35, alpha=0.6, label=str(status))
        axes[1].scatter(sub["local_registered_cell_density_64px"], sub["task007_full_patch_component_count"], s=2, alpha=0.12, label=str(status))
    axes[0].set(xlabel="Task007 full-patch component count", ylabel="cells", title="Old component-count rule"); axes[0].legend()
    axes[1].set(xlabel="registered cells within 64 px", ylabel="full-patch components", title="density confounding")
    axes[2].boxplot([neut.loc[neut.task007_status.eq(s), "task007_full_patch_component_count"].dropna() for s in ["intact_or_plausible", "debris_suspect"]], labels=["intact", "debris"])
    axes[2].set(title="Task007 status by component count", ylabel="components")
    fig.tight_layout(); fig.savefig(figdir / "Fig2_task007_failure_mode.pdf"); plt.close(fig)

    # Fig3–Fig6 summaries.
    fig, ax = plt.subplots(figsize=(8, 5)); sc = status_summary.set_index("task008_status"); ax.bar(sc.index, sc.n_cells); ax.set_ylabel("cells"); ax.set_title("Task008 target-centered status counts"); ax.tick_params(axis="x", rotation=60); fig.tight_layout(); fig.savefig(figdir / "Fig3_target_centered_status_counts.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); pt = pd.crosstab(qc.task007_status, qc.task008_status); ax.imshow(pt.to_numpy(), cmap="Blues", aspect="auto"); ax.set_xticks(range(len(pt.columns)), pt.columns, rotation=70); ax.set_yticks(range(len(pt.index)), pt.index); fig.colorbar(ax.images[0], ax=ax, label="cells"); ax.set_title("Task007 to Task008 transition"); fig.tight_layout(); fig.savefig(figdir / "Fig4_task007_to_task008_transition.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); sub = subtype_summary.set_index("original_cl1"); x = np.arange(len(sub)); w = 0.35; ax.bar(x-w/2, sub.target_present / sub.n_cells, w, label="target evidence"); ax.bar(x+w/2, qc.groupby("original_cl1").trainable_extended_task008.mean().reindex(sub.index), w, label="extended eligible"); ax.set_xticks(x, sub.index, rotation=45); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Neutrophil retention by subtype"); fig.tight_layout(); fig.savefig(figdir / "Fig5_retention_by_subtype.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); ax.bar(batch_summary.batch, batch_summary.extended_retention); ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=70); ax.set_ylabel("extended retention"); ax.set_title("Neutrophil retention by batch"); fig.tight_layout(); fig.savefig(figdir / "Fig6_retention_by_batch.pdf"); plt.close(fig)

    save_review_montage(qc, he_img, selected_tolerance, OUT / "figures/Fig7_task008_manual_review.pdf")
    fig, ax = plt.subplots(figsize=(10, 5)); sens = pd.DataFrame(sensitivity_rows); piv = sens[sens.variant != "per_cell_primary_stability"].pivot_table(index="variant", columns="status", values="fraction", fill_value=0); piv.plot(kind="bar", stacked=True, ax=ax); ax.set_ylabel("fraction of Neutrophils"); ax.set_title("Parameter sensitivity"); ax.tick_params(axis="x", rotation=70); fig.tight_layout(); fig.savefig(figdir / "Fig8_parameter_sensitivity.pdf"); plt.close(fig)

    checksums = {"task007_annotation_sha256": sha256_file(TASK7_ANNOT), "source_h5ad_sha256": sha256_file(INPUT_H5AD), "source_he_sha256": sha256_file(INPUT_HE), "source_files_modified": False}
    write_json(checksums, OUT / "config/checksums.json")

    # Human-readable remote report; it explicitly distinguishes computational
    # evidence from interpretation and keeps eligibility provisional pending
    # manual review of the generated montage.
    present = int((qc["target_evidence"] == "TARGET_NUCLEUS_PRESENT").sum())
    no_target = int((qc["task008_status"] == "NO_TARGET_NUCLEUS").sum())
    fragmented = int((qc["task008_status"] == "FRAGMENTED_TARGET_SUSPECT").sum())
    review = int((qc["task008_status"] == "MANUAL_REVIEW").sum())
    core_n = int(core.sum()); ext_n = int(extended.sum())
    subtype_lookup = subtype_summary.set_index("original_cl1")
    conv_core = int(qc.loc[qc.original_cl1.eq("Neutrophil"), "trainable_core_task008"].sum())
    conv_ext = int(qc.loc[qc.original_cl1.eq("Neutrophil"), "trainable_extended_task008"].sum())
    cxcr4_core = int(qc.loc[qc.original_cl1.eq("Neutrophil_CXCR4"), "trainable_core_task008"].sum())
    cxcr4_ext = int(qc.loc[qc.original_cl1.eq("Neutrophil_CXCR4"), "trainable_extended_task008"].sum())
    report = f"""# Task008 — Neutrophil nucleus-centered H&E QC recalibration

Status: **PARTIAL / REVIEW_REQUIRED**. Task007 biological identities were frozen. No CellViT predictions were used and no CellViT model was trained.

## A–O answers

**A. Why Task007 over-called debris.** Task007 counted hematoxylin-like connected components across the full 128×128 context crop. In dense immune regions, neighboring nuclei inflate that count. Task008 uses a center-associated component/group and treats the full patch only as context.

**B. Empirical offset.** Reference cells: **{len(ref_qc):,}** across {len(REF_CLASSES)} non-Neutrophil classes and all batches. Median **{qvals[0.50]:.2f}px**; 75th **{qvals[0.75]:.2f}px**; 90th **{qvals[0.90]:.2f}px**; 95th **{qvals[0.95]:.2f}px**; 97.5th **{qvals[0.975]:.2f}px**.

**C. Selected tolerance.** **{selected_tolerance:.2f}px**, based on the empirical 95th percentile with an upper sanity cap of {UPPER_TOLERANCE_CAP_PX:.1f}px and a documented 8px pixel-space floor.

**D. Task007 debris-suspect cells with plausible target-centered nucleus.** **{int(((qc.task007_status == 'debris_suspect') & (qc.target_evidence == 'TARGET_NUCLEUS_PRESENT')).sum()):,}**. This is target-centered evidence, not a biological identity conclusion.

**E. No-target-nucleus.** **{no_target:,}**.

**F. Center-fragmented suspect.** **{fragmented:,}**.

**G. Manual review.** **{review:,}** final `MANUAL_REVIEW` cells; the montage includes stratified panels and up to 100 examples per category where available.

**H. Conventional Neutrophil retention.** Provisional TRAINABLE_CORE **{conv_core:,}** and TRAINABLE_EXTENDED **{conv_ext:,}** of 14,647 original conventional Neutrophils; frozen biological labels were unchanged.

**I. Neutrophil_CXCR4 retention.** Provisional TRAINABLE_CORE **{cxcr4_core:,}** and TRAINABLE_EXTENDED **{cxcr4_ext:,}** of 9,520 original CXCR4 Neutrophils; subtype labels were not changed.

**J. Batch effects.** Hard-exclusion spread across batches: **{batch_spread:.2%}**. Full batch table is in `neutrophil_retention_by_batch.csv`.

**K. Xenium/H&E agreement.** See `neutrophil_centered_he_qc.csv.gz` and the `xenium_he_nucleus_agreement` categories; modality conflicts are sent to `MANUAL_REVIEW`.

**L. Stability.** Mean status stability across the predefined one-at-a-time sensitivity grid: **{mean_stability:.2%}**; per-cell stability is recorded in `neutrophil_centered_he_qc.csv.gz`.

**M. Safety rails.** **{'TRIGGERED' if safety['any_safety_trigger'] else 'No computational safety rail triggered'}**. Eligibility remains REVIEW_REQUIRED because manual review is mandatory before finalizing.

**N. Provisional eligibility.** TRAINABLE_CORE: **{core_n:,}**; TRAINABLE_EXTENDED: **{ext_n:,}**. These flags do not alter biological labels and remain provisional until review.

**O. Proceed to CellViT retraining?** **No.** Review `Fig7_task008_manual_review.pdf`, the safety file, and the eligibility table first.

## Inputs and reproducibility

- Source H5AD: `{INPUT_H5AD}`
- Source H&E: `{INPUT_HE}`
- Registration: `{INPUT_MATRIX}`
- Frozen Task007 annotation: `{TASK7_ANNOT}`
- Script: `task008_recalibrate.py`; command: `python3 /data/lf_data/result/task008_neutrophil_he_recalibration/code/task008_recalibrate.py`
- Runtime: Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, h5py {h5py.__version__}, scipy {__import__('scipy').__version__}, pyvips {pyvips.__version__}; seed {SEED}.
- Primary parameters: hematoxylin percentile {PRIMARY_H_PERCENTILE}; min component area {PRIMARY_MIN_COMPONENT_AREA}px; lobe-group distance {PRIMARY_GROUP_DISTANCE}px; center tolerance {selected_tolerance:.2f}px.

## Output paths

- `{OUT / 'metrics/task007_he_status_audit.csv'}`
- `{OUT / 'metrics/registration_tolerance_reference.csv'}` and `{OUT / 'metrics/registration_tolerance_summary.csv'}`
- `{OUT / 'metrics/neutrophil_centered_he_qc.csv.gz'}`
- `{OUT / 'metrics/neutrophil_training_eligibility_task008.csv.gz'}`
- `{OUT / 'metrics/safety_rail_decision.json'}`
- `{OUT / 'figures/Fig7_task008_manual_review.pdf'}`

Source and Task007 files were read-only. The context score is a review-prioritization feature; it does not override clear target-centered nuclear evidence and does not establish biological identity.
"""
    (OUT / "TASK008_REPORT.md").write_text(report)
    log(f"Task008 complete: provisional core={core_n:,}, extended={ext_n:,}, status=PARTIAL_REVIEW_REQUIRED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        raise
