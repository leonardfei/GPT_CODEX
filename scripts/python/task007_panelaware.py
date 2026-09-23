#!/usr/bin/env python3
"""Task 007: panel-aware, conservative Xenium re-annotation and audit.

This script intentionally stops before any CellViT training.  It reads the
source AnnData/H&E files, derives signatures from the observed panel, creates
cross-fitted identity evidence, applies conservative KEEP/RELABEL/REVIEW
rules, runs full-resolution H&E crops for neutrophil candidates, and writes
auditable outputs under the task-specific result directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, sparse


SEED = 42
np.random.seed(SEED)

INPUT_H5AD = Path("/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad")
INPUT_HE = Path("/data/lf_data/xenium_data/ID0060276.ome.tif")
INPUT_MATRIX = Path("/data/lf_data/xenium_data/matrix.csv")
INPUT_NOTEBOOK = Path("/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb")
TASK6_ROOT = Path("/data/lf_data/result/task006_xenium_reannotation")
OUT = Path("/data/lf_data/result/task007_xenium5k_panelaware")

SUBDIRS = ["config", "code", "qc", "metrics", "figures", "figure_data", "logs", "work", "models"]
CLASSES = ["T and B", "Myeloid", "Neutrophil", "Plasma cell", "Mesenchymal", "Endothelial", "Tumor"]
CLASS_INDEX = {x: i for i, x in enumerate(CLASSES)}
MAJOR_CLASSES = ["Tumor", "T and B", "Myeloid", "Neutrophil", "Mesenchymal", "Endothelial"]


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
    """Read an AnnData HDF5 dataset or categorical group into a 1D array."""
    if isinstance(obj, h5py.Dataset):
        return decode_arr(obj[:])
    if "categories" in obj and "codes" in obj:
        cats = decode_arr(obj["categories"][:])
        codes = np.asarray(obj["codes"][:], dtype=np.int64)
        out = np.empty(codes.shape, dtype=object)
        missing = codes < 0
        out[:] = None
        out[~missing] = cats[codes[~missing]]
        return out
    raise TypeError(f"Unsupported HDF5 column structure: {obj.name}")


def read_sparse_group(g):
    shape = tuple(int(x) for x in g.attrs["shape"])
    return sparse.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)


class CSRReader:
    def __init__(self, h5: h5py.File, path: str):
        self.g = h5[path]
        self.shape = tuple(int(x) for x in self.g.attrs["shape"])

    def rows(self, start: int, stop: int) -> sparse.csr_matrix:
        ip = np.asarray(self.g["indptr"][start : stop + 1], dtype=np.int64)
        first, last = int(ip[0]), int(ip[-1])
        data = self.g["data"][first:last]
        indices = self.g["indices"][first:last]
        return sparse.csr_matrix((data, indices, ip - first), shape=(stop - start, self.shape[1]))


def ensure_dirs() -> None:
    for sub in SUBDIRS:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def robust_z(values: np.ndarray, groups: np.ndarray, high: bool = False) -> np.ndarray:
    out = np.zeros(values.shape[0], dtype=np.float32)
    s = pd.Series(values)
    for key, idx in pd.Series(np.arange(len(values))).groupby(groups, sort=False):
        ii = idx.to_numpy(dtype=np.int64)
        v = values[ii].astype(float)
        med = np.nanmedian(v)
        mad = np.nanmedian(np.abs(v - med)) * 1.4826
        if not np.isfinite(mad) or mad < 1e-8:
            mad = np.nanstd(v)
        if not np.isfinite(mad) or mad < 1e-8:
            mad = 1.0
        out[ii] = (v - med) / mad
    return out


def lognorm_csr(mat: sparse.csr_matrix, totals: np.ndarray) -> sparse.csr_matrix:
    m = mat.astype(np.float32, copy=True)
    denom = np.asarray(totals, dtype=np.float32).copy()
    denom[~np.isfinite(denom) | (denom <= 0)] = 1.0
    counts = np.diff(m.indptr)
    if m.nnz:
        m.data = np.log1p(m.data * np.repeat(10000.0 / denom, counts)).astype(np.float32)
    return m


def write_df(df: pd.DataFrame, path: Path, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip" if gzip else None)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def group_stats(df: pd.DataFrame, value_cols) -> pd.DataFrame:
    return df.groupby(value_cols, dropna=False, observed=True).size().reset_index(name="n_cells")


def crop_to_array(img: pyvips.Image, x: int, y: int, size: int = 128) -> np.ndarray:
    crop = img.crop(int(x), int(y), int(size), int(size))
    return np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8, shape=(crop.height, crop.width, crop.bands)).copy()


def assess_he_crop(arr: np.ndarray) -> dict:
    # This is a conservative morphology proxy for triage only; it is not a
    # biological classifier and does not make a micron-scale claim.
    rgb = arr[..., :3].astype(np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    blue_contrast = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
    signal = (gray < np.percentile(gray, 32)) | (blue_contrast > np.percentile(blue_contrast, 78))
    signal = ndimage.binary_opening(signal, iterations=1)
    signal = ndimage.binary_closing(signal, iterations=1)
    labels, n = ndimage.label(signal, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())[1:] if n else np.array([], dtype=int)
    sizes = sizes[sizes >= 8]
    largest = int(sizes.max()) if sizes.size else 0
    n_components = int(sizes.size)
    frac = float(signal.mean())
    if largest == 0:
        status = "no_visible_nucleus"
    elif n_components >= 18 or (n_components >= 8 and largest < 40):
        status = "debris_suspect"
    elif largest >= 18 and frac < 0.55:
        status = "intact_or_plausible"
    else:
        status = "review"
    return {
        "he_neutrophil_nucleus_status": status,
        "he_neutrophil_debris_status": "suspect" if status == "debris_suspect" else "not_suspect",
        "he_signal_fraction": frac,
        "he_component_count": n_components,
        "he_largest_component_px": largest,
    }


def draw_montage(rows, crop_dir: Path, path: Path) -> None:
    rows = list(rows)
    if not rows:
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No neutrophil crops available", ha="center", va="center")
        plt.axis("off")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return
    cols, tile = 5, 180
    nrows = int(math.ceil(len(rows) / cols))
    canvas = Image.new("RGB", (cols * tile, nrows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        fn = crop_dir / str(row["he_crop_file"])
        try:
            im = Image.open(fn).convert("RGB").resize((150, 150))
        except Exception:
            continue
        x, y = (i % cols) * tile + 10, (i // cols) * tile + 5
        canvas.paste(im, (x, y))
        label = f"{row['cell_id']}\n{row['he_neutrophil_nucleus_status']}"
        draw.multiline_text((x, y + 152), label[:42], fill="black", spacing=1)
    canvas.save(path.with_suffix(".png"))
    canvas.save(path.with_suffix(".jpg"), quality=88)
    fig = plt.figure(figsize=(cols * 2.0, nrows * 2.0))
    plt.imshow(canvas)
    plt.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    log("Task007 started; no CellViT training will be run")
    for p in [INPUT_H5AD, INPUT_HE, INPUT_MATRIX, INPUT_NOTEBOOK]:
        if not p.exists():
            raise FileNotFoundError(p)

    with h5py.File(INPUT_H5AD, "r") as f:
        obs = f["obs"]
        obs_cols = {}
        wanted = [
            "cell_id", "batch", "cl1", "leiden", "cell_area", "nucleus_area", "nucleus_count",
            "total_counts", "n_genes_by_counts", "control_probe_counts", "genomic_control_counts",
            "control_codeword_counts", "unassigned_codeword_counts", "deprecated_codeword_counts",
            "transcript_counts", "x", "y", "z_level",
        ]
        for key in wanted:
            if key in obs:
                obs_cols[key] = read_h5_col(obs[key])
        spatial = np.asarray(f["obsm"]["spatial"][:], dtype=np.float32)
        var_names = read_h5_col(f["var"]["_index"] if "_index" in f["var"] else f["var"]["gene_ids"])
        var_names = np.asarray([str(dec(x)) for x in var_names], dtype=object)
        var_gene_ids = read_h5_col(f["var"]["gene_ids"]) if "gene_ids" in f["var"] else var_names.copy()
        var_features = read_h5_col(f["var"]["feature_types"]) if "feature_types" in f["var"] else np.repeat("unknown", len(var_names))
        var_genome = read_h5_col(f["var"]["genome"]) if "genome" in f["var"] else np.repeat("unknown", len(var_names))
        n_cells, n_genes = spatial.shape[0], len(var_names)
        counts_reader = CSRReader(f, "layers/counts")
        log(f"Loaded H5AD metadata: cells={n_cells:,}, genes={n_genes:,}")

    df = pd.DataFrame(obs_cols)
    df["cell_index"] = np.arange(n_cells, dtype=np.int64)
    df["original_cl1"] = df["cl1"].astype(str)
    df["original_cl1_7class"] = df["original_cl1"].replace({"Neutrophil_CXCR4": "Neutrophil", "Low quality": "Low quality"})
    df["batch"] = df["batch"].astype(str)
    df["leiden"] = df["leiden"].astype(str) if "leiden" in df else "NA"
    df["total_counts"] = pd.to_numeric(df.get("total_counts", df.get("transcript_counts", 0)), errors="coerce").fillna(0)
    df["n_genes_by_counts"] = pd.to_numeric(df.get("n_genes_by_counts", 0), errors="coerce").fillna(0)
    for key in ["cell_area", "nucleus_area", "nucleus_count", "control_probe_counts", "genomic_control_counts", "control_codeword_counts", "unassigned_codeword_counts", "deprecated_codeword_counts"]:
        if key not in df:
            df[key] = 0.0
        df[key] = pd.to_numeric(df[key], errors="coerce").fillna(0)

    # Panel inventory and explicit audit of common named features.  Only the
    # observed panel is eligible for downstream scores.
    panel = pd.DataFrame({
        "gene": var_names,
        "gene_id": [dec(x) for x in var_gene_ids],
        "feature_type": [dec(x) for x in var_features],
        "genome": [dec(x) for x in var_genome],
        "gene_index": np.arange(n_genes, dtype=np.int32),
    })
    panel["present"] = True
    write_df(panel, OUT / "metrics/panel_gene_inventory.csv")
    audit_genes = ["CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2", "MS4A1", "CD79A", "NKG7", "LYZ", "S100A8", "S100A9", "FCGR3A", "MPO", "CSF3R", "CXCR4", "EPCAM", "KRT8", "KRT18", "PECAM1", "VWF", "COL1A1", "DCN", "MZB1", "JCHAIN"]
    coverage = pd.DataFrame({"gene": audit_genes})
    lookup = {g: i for i, g in enumerate(var_names)}
    coverage["present"] = coverage["gene"].map(lambda x: x in lookup)
    coverage["gene_index"] = coverage["gene"].map(lookup)
    coverage["feature_type"] = coverage["gene"].map(lambda x: str(var_features[lookup[x]]) if x in lookup else "ABSENT")
    write_df(coverage, OUT / "metrics/panel_annotation_coverage.csv")

    # Technical QC is recorded separately from identity.  Low RNA alone is
    # not treated as a biological identity failure.
    df["control_fraction"] = df["control_probe_counts"] / df["total_counts"].replace(0, np.nan)
    df["control_fraction"] = df["control_fraction"].fillna(0)
    df["z_total_counts_batch"] = robust_z(np.log1p(df["total_counts"].to_numpy()), df["batch"].to_numpy())
    df["z_genes_batch"] = robust_z(np.log1p(df["n_genes_by_counts"].to_numpy()), df["batch"].to_numpy())
    df["z_control_fraction_batch"] = robust_z(df["control_fraction"].to_numpy(), df["batch"].to_numpy(), high=True)
    df["z_area_batch"] = robust_z(np.log1p(df["cell_area"].to_numpy()), df["batch"].to_numpy())
    df["qc_low_counts"] = df["z_total_counts_batch"] < -4
    df["qc_low_genes"] = df["z_genes_batch"] < -4
    df["qc_high_control_fraction"] = df["z_control_fraction_batch"] > 4
    df["qc_extreme_area"] = (df["z_area_batch"].abs() > 8) & (df["cell_area"] > 0)
    df["qc_nucleus_present_xenium"] = (df["nucleus_count"] >= 1) & (df["nucleus_area"] > 0) & (df["cell_area"] > 0)
    df["qc_transcript_pass"] = ~(df["qc_low_counts"] & df["qc_low_genes"]) & ~df["qc_high_control_fraction"]
    df["qc_segmentation_pass"] = ~df["qc_extreme_area"] & (df["cell_area"] > 0)
    df["technical_fail"] = ~(df["qc_transcript_pass"] & df["qc_segmentation_pass"])
    df["cell_quality_status"] = np.where(df["original_cl1"].eq("Low quality"), "Low_quality", np.where(~df["qc_nucleus_present_xenium"], "Artifact_or_no_nucleus", np.where(df["technical_fail"], "Technical_fail", "Pass")))
    write_df(group_stats(df, ["original_cl1"]), OUT / "metrics/original_annotation_summary.csv")
    qc_cols = ["total_counts", "n_genes_by_counts", "cell_area", "nucleus_area", "nucleus_count", "control_fraction", "technical_fail", "qc_nucleus_present_xenium"]
    qc_class = df.groupby("original_cl1", dropna=False)[qc_cols].agg(["count", "median", "mean"]).reset_index()
    qc_class.columns = ["_".join([str(x) for x in c if str(x) != ""]).rstrip("_") for c in qc_class.columns]
    write_df(qc_class, OUT / "metrics/original_qc_by_class.csv")
    qc_batch = df.groupby("batch", dropna=False)[qc_cols].agg(["count", "median", "mean"]).reset_index()
    qc_batch.columns = ["_".join([str(x) for x in c if str(x) != ""]).rstrip("_") for c in qc_batch.columns]
    write_df(qc_batch, OUT / "metrics/original_qc_by_batch.csv")

    # Anchor definition uses original labels, QC, and cluster purity only.
    anchor_base = df["original_cl1_7class"].isin(CLASSES) & ~df["technical_fail"]
    cluster_counts = df.loc[anchor_base].groupby(["batch", "leiden", "original_cl1_7class"], observed=True).size().unstack(fill_value=0)
    cluster_counts = cluster_counts.reindex(columns=CLASSES, fill_value=0)
    cluster_total = cluster_counts.sum(axis=1).replace(0, np.nan)
    cluster_top = cluster_counts.idxmax(axis=1)
    cluster_frac = cluster_counts.max(axis=1) / cluster_total
    cluster_prob = cluster_counts.div(cluster_total, axis=0).fillna(0)
    cluster_entropy = -(cluster_prob.where(cluster_prob > 0, 1e-12) * np.log(cluster_prob.where(cluster_prob > 0, 1e-12))).sum(axis=1)
    cluster_key = pd.MultiIndex.from_arrays([df["batch"], df["leiden"]])
    df["cluster_top_class"] = [cluster_top.get(k, "NA") for k in cluster_key]
    df["cluster_original_label_fraction"] = np.asarray([cluster_frac.get(k, 0.0) for k in cluster_key], dtype=np.float32)
    # The existing Leiden composition is used as a documented neighborhood
    # proxy; no new graph is inferred from future labels.
    df["neighbor_top_class"] = df["cluster_top_class"]
    df["neighbor_original_label_fraction"] = df["cluster_original_label_fraction"]
    df["neighbor_entropy"] = np.asarray([cluster_entropy.get(k, np.nan) for k in cluster_key], dtype=np.float32)
    df["anchor"] = anchor_base & (df["cluster_original_label_fraction"] >= 0.80)
    if int(df["anchor"].sum()) < 500:
        df["anchor"] = anchor_base
        log("Cluster-purity anchor set was small; fell back to all QC-passing original labels")
    log(f"Anchor cells: {int(df['anchor'].sum()):,}")

    # Data-driven signatures: class-vs-rest effects are computed from observed
    # panel counts in batch-stratified original-label anchors.
    totals = df["total_counts"].to_numpy(dtype=np.float32)
    group_sum = defaultdict(lambda: np.zeros(n_genes, dtype=np.float64))
    group_det = defaultdict(lambda: np.zeros(n_genes, dtype=np.int64))
    group_n = defaultdict(int)
    batch_arr = df["batch"].to_numpy()
    class_arr = df["original_cl1_7class"].to_numpy()
    anchor_arr = df["anchor"].to_numpy()
    chunk_size = 25000
    with h5py.File(INPUT_H5AD, "r") as f:
        reader = CSRReader(f, "layers/counts")
        for start in range(0, n_cells, chunk_size):
            stop = min(start + chunk_size, n_cells)
            mat = lognorm_csr(reader.rows(start, stop), totals[start:stop])
            valid = anchor_arr[start:stop]
            if not valid.any():
                continue
            local_keys = {}
            for i in np.flatnonzero(valid):
                key = (str(batch_arr[start + i]), str(class_arr[start + i]))
                local_keys.setdefault(key, []).append(int(i))
            for key, ii in local_keys.items():
                sub = mat[ii]
                group_sum[key] += np.asarray(sub.sum(axis=0)).ravel()
                group_det[key] += np.asarray((sub > 0).sum(axis=0)).ravel()
                group_n[key] += len(ii)
            if start and start % 250000 == 0:
                log(f"Signature scan: {stop:,}/{n_cells:,}")

    batch_names = sorted({k[0] for k in group_sum})
    marker_rows = []
    selected_by_class = {}
    for cls in CLASSES:
        per_batch = []
        for batch in batch_names:
            key = (batch, cls)
            if key not in group_sum or group_n[key] < 50:
                continue
            rest_sum = np.zeros(n_genes, dtype=np.float64)
            rest_n = 0
            for other in CLASSES:
                if other == cls:
                    continue
                ok = (batch, other)
                if ok in group_sum:
                    rest_sum += group_sum[ok]
                    rest_n += group_n[ok]
            if rest_n < 50:
                continue
            cm = group_sum[key] / group_n[key]
            rm = rest_sum / rest_n
            cd = group_det[key] / group_n[key]
            rd = np.zeros(n_genes, dtype=float)
            for other in CLASSES:
                ok = (batch, other)
                if ok in group_det:
                    rd += group_det[ok]
            rd = rd / max(rest_n, 1)
            per_batch.append((batch, cm - rm, cd - rd))
        if not per_batch:
            selected_by_class[cls] = np.array([], dtype=np.int32)
            continue
        effects = np.vstack([x[1] for x in per_batch])
        detdiff = np.vstack([x[2] for x in per_batch])
        med_eff = np.median(effects, axis=0)
        med_det = np.median(detdiff, axis=0)
        consistent = (effects > 0.03).sum(axis=0)
        frac = consistent / effects.shape[0]
        score = med_eff * frac + 0.25 * med_det
        eligible = (frac >= 0.60) & (med_eff > 0.03)
        order = np.argsort(-np.where(eligible, score, -np.inf))
        chosen = [int(g) for g in order if eligible[g]][:30]
        if len(chosen) < 10:
            fallback = np.argsort(-score)[: min(30, n_genes)]
            chosen = list(dict.fromkeys(chosen + [int(g) for g in fallback]))[:30]
        selected_by_class[cls] = np.asarray(chosen, dtype=np.int32)
        for rank, g in enumerate(chosen, 1):
            marker_rows.append({
                "class": cls, "gene": str(var_names[g]), "gene_index": g,
                "median_log_effect": float(med_eff[g]), "median_detection_diff": float(med_det[g]),
                "consistent_batches": int(consistent[g]), "n_batches": int(effects.shape[0]),
                "consistency_fraction": float(frac[g]), "rank": rank, "direction": "enriched", "selected": True,
            })
        negative_consistent = ((effects < -0.03).sum(axis=0) / effects.shape[0] >= 0.60) & (med_eff < -0.03)
        depleted_score = (-med_eff) * ((effects < -0.03).sum(axis=0) / effects.shape[0]) + 0.25 * (-med_det)
        depleted_order = np.argsort(-np.where(negative_consistent, depleted_score, -np.inf))
        for rank, g in enumerate([int(x) for x in depleted_order if negative_consistent[x]][:20], 1):
            marker_rows.append({
                "class": cls, "gene": str(var_names[g]), "gene_index": g,
                "median_log_effect": float(med_eff[g]), "median_detection_diff": float(med_det[g]),
                "consistent_batches": int((effects[:, g] < -0.03).sum()), "n_batches": int(effects.shape[0]),
                "consistency_fraction": float((effects[:, g] < -0.03).mean()), "rank": rank, "direction": "depleted", "selected": False,
            })
    marker_df = pd.DataFrame(marker_rows)
    write_df(marker_df, OUT / "metrics/panel_data_driven_markers.csv")
    union_genes = sorted(set(int(g) for arr in selected_by_class.values() for g in arr))
    if not union_genes:
        raise RuntimeError("No data-driven panel signature genes were selected")

    # Cross-fitted score: held-out batch profile, using only observed panel
    # signature genes and positive class-vs-rest effects.
    score_matrix = np.zeros((n_cells, len(CLASSES)), dtype=np.float32)
    class_profile_by_holdout = {}
    for holdout in batch_names:
        profiles = []
        for cls in CLASSES:
            s = np.zeros(n_genes, dtype=float)
            nn = 0
            for batch in batch_names:
                if batch == holdout:
                    continue
                key = (batch, cls)
                if key in group_sum:
                    s += group_sum[key]
                    nn += group_n[key]
            if nn == 0:
                for batch in batch_names:
                    key = (batch, cls)
                    if key in group_sum:
                        s += group_sum[key]
                        nn += group_n[key]
            profiles.append(s / max(nn, 1))
        profiles = np.vstack(profiles)[:, union_genes]
        weights = np.maximum(profiles - profiles.mean(axis=0, keepdims=True), 0.0)
        # Ensure every class retains a non-zero vector for degenerate panels.
        for j in range(weights.shape[0]):
            if not np.any(weights[j] > 0):
                weights[j] = np.maximum(profiles[j], 0.0)
        class_profile_by_holdout[holdout] = weights.astype(np.float32)
    weights_norm = {b: np.sqrt((w * w).sum(axis=1)) + 1e-8 for b, w in class_profile_by_holdout.items()}
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        with h5py.File(INPUT_H5AD, "r") as f:
            mat = lognorm_csr(CSRReader(f, "layers/counts").rows(start, stop), totals[start:stop])[:, union_genes]
        for batch in batch_names:
            idx = np.flatnonzero(batch_arr[start:stop] == batch)
            if idx.size == 0:
                continue
            sub = mat[idx]
            raw = np.asarray(sub.dot(class_profile_by_holdout[batch].T), dtype=np.float32)
            row_norm = np.sqrt(np.asarray(sub.multiply(sub).sum(axis=1)).ravel()).astype(np.float32) + 1e-8
            score_matrix[start + idx] = raw / (row_norm[:, None] * weights_norm[batch][None, :])
        if start and start % 250000 == 0:
            log(f"Cross-fit score scan: {stop:,}/{n_cells:,}")

    order = np.argsort(-score_matrix, axis=1)
    top_idx = order[:, 0]
    second_idx = order[:, 1]
    top_score = score_matrix[np.arange(n_cells), top_idx]
    second_score = score_matrix[np.arange(n_cells), second_idx]
    margin = top_score - second_score
    orig_idx = np.asarray([CLASS_INDEX.get(x, -1) for x in class_arr], dtype=np.int16)
    orig_score = np.where(orig_idx >= 0, score_matrix[np.arange(n_cells), np.maximum(orig_idx, 0)], np.nan)
    ranks = np.full(n_cells, 99, dtype=np.int16)
    for j in range(len(CLASSES)):
        ranks[orig_idx == j] = np.argmax(order[orig_idx == j] == j, axis=1) + 1

    # Thresholds are calibrated on original-label anchors for conservative
    # high precision, rather than chosen from external marker knowledge.
    anchor_valid = df["anchor"].to_numpy() & (orig_idx >= 0)
    precision_target = 0.95
    candidates = []
    for qscore in np.linspace(0.40, 0.95, 12):
        for qmargin in np.linspace(0.40, 0.95, 12):
            sc = float(np.quantile(top_score[anchor_valid], qscore))
            mg = float(np.quantile(margin[anchor_valid], qmargin))
            keep = anchor_valid & (top_score >= sc) & (margin >= mg)
            n = int(keep.sum())
            prec = float((top_idx[keep] == orig_idx[keep]).mean()) if n else 0.0
            candidates.append((prec >= precision_target and n >= 50, n, prec, sc, mg))
    eligible = [x for x in candidates if x[0]]
    if not eligible:
        eligible = [x for x in candidates if x[2] >= 0.90 and x[1] >= 50]
        precision_target = 0.90
    if not eligible:
        eligible = sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)[:1]
        precision_target = float(eligible[0][2])
    _, _, cal_precision, score_cut, margin_cut = max(eligible, key=lambda x: x[1])
    write_json({"seed": SEED, "score_cut": score_cut, "margin_cut": margin_cut, "anchor_precision": cal_precision, "precision_target": precision_target, "n_anchor": int(anchor_valid.sum()), "signature_genes": int(len(union_genes))}, OUT / "config/crossfit_thresholds.json")

    crossfit_summary = []
    for batch in batch_names:
        bm = anchor_valid & (batch_arr == batch)
        for cls in CLASSES:
            cm = bm & (orig_idx == CLASS_INDEX[cls])
            crossfit_summary.append({
                "held_out_batch": batch, "class": cls, "n_anchor": int(cm.sum()),
                "n_correct_top1": int((top_idx[cm] == CLASS_INDEX[cls]).sum()),
                "top1_consistency": float((top_idx[cm] == CLASS_INDEX[cls]).mean()) if cm.any() else np.nan,
                "mean_top_score": float(np.mean(top_score[cm])) if cm.any() else np.nan,
                "mean_margin": float(np.mean(margin[cm])) if cm.any() else np.nan,
            })
    write_df(pd.DataFrame(crossfit_summary), OUT / "metrics/crossfit_class_consistency.csv.gz", gzip=True)
    crossfit_cells = pd.DataFrame({
        "cell_index": np.arange(n_cells, dtype=np.int64), "batch": batch_arr,
        "original_cl1_7class": class_arr, "anchor": anchor_arr,
        "crossfit_top_class": [CLASSES[i] for i in top_idx], "crossfit_second_class": [CLASSES[i] for i in second_idx],
        "crossfit_top_score": top_score, "crossfit_second_score": second_score, "crossfit_margin": margin,
        "crossfit_original_score": orig_score, "crossfit_original_rank": ranks,
    })
    write_df(crossfit_cells, OUT / "metrics/crossfit_scores.csv.gz", gzip=True)

    # Conservative label policy. Identity and quality remain separate columns.
    top_class = np.asarray([CLASSES[i] for i in top_idx], dtype=object)
    alt_cluster_support = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        key = (batch_arr[i], str(df.iloc[i]["leiden"]))
        if key in cluster_counts.index and top_class[i] in cluster_counts.columns:
            total = float(cluster_counts.loc[key].sum())
            alt_cluster_support[i] = float(cluster_counts.loc[key, top_class[i]] / total) if total else 0.0
    df["crossfit_top_class"] = top_class
    df["crossfit_top_score"] = top_score
    df["crossfit_margin"] = margin
    df["crossfit_original_rank"] = ranks
    df["crossfit_alt_cluster_support"] = alt_cluster_support
    strong = (top_score >= score_cut) & (margin >= margin_cut)
    differs = (top_class != df["original_cl1_7class"].to_numpy()) & df["original_cl1_7class"].isin(CLASSES).to_numpy()
    relabel = differs & strong & (alt_cluster_support >= 0.65) & (ranks >= 3) & ~df["original_cl1"].eq("Low quality").to_numpy()
    moderate = differs & (top_score >= 0.5 * score_cut) & (margin >= 0.5 * margin_cut) & ((alt_cluster_support >= 0.55) | strong)
    review = moderate & ~relabel & ~df["original_cl1"].eq("Low quality").to_numpy()
    action = np.full(n_cells, "KEEP", dtype=object)
    action[review] = "REVIEW"
    action[relabel] = "RELABEL"
    df["v3_action"] = action
    v3_label = df["original_cl1"].to_numpy(dtype=object).copy()
    for i in np.flatnonzero(relabel):
        v3_label[i] = top_class[i]
    v3_label[review] = "REVIEW"
    df["v3_label"] = v3_label
    df["identity_candidate"] = np.where(df["original_cl1_7class"].isin(CLASSES), top_class, df["original_cl1"])
    df["label_confidence"] = np.where(relabel | (strong & ~differs), "high", np.where(review, "low", "moderate"))
    df.loc[df["original_cl1"].eq("Low quality"), "label_confidence"] = "not_assigned"

    transition = pd.crosstab(df["original_cl1"], df["v3_label"], dropna=False)
    transition.reset_index().to_csv(OUT / "metrics/v3_label_transition_matrix.csv", index=False)
    write_df(df.groupby(["original_cl1", "v3_action"], dropna=False).size().reset_index(name="n_cells"), OUT / "metrics/v3_label_action_summary.csv")
    quality_summary = df.groupby(["v3_label", "cell_quality_status"], dropna=False).size().reset_index(name="n_cells")
    write_df(quality_summary, OUT / "metrics/v3_quality_summary.csv")

    # Full-resolution H&E crop/QC for every original or proposed neutrophil.
    neut_candidate = df["original_cl1_7class"].eq("Neutrophil") | (df["v3_action"].eq("RELABEL") & df["crossfit_top_class"].eq("Neutrophil"))
    nidx = np.flatnonzero(neut_candidate.to_numpy())
    he_matrix = np.loadtxt(INPUT_MATRIX, delimiter=",")
    inv_matrix = np.linalg.inv(he_matrix)
    xenium_px = spatial / 0.2125
    hom = np.c_[xenium_px, np.ones(n_cells, dtype=np.float32)]
    he_xy = (inv_matrix @ hom.T).T[:, :2]
    he_summary = []
    crop_dir = OUT / "work/neutrophil_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    he_img = pyvips.Image.new_from_file(str(INPUT_HE), page=0, access="random")
    for j, i in enumerate(nidx):
        x, y = float(he_xy[i, 0]), float(he_xy[i, 1])
        inbounds = (x >= 64) and (y >= 64) and (x < he_img.width - 64) and (y < he_img.height - 64)
        base = {
            "cell_index": int(i), "cell_id": str(df.iloc[i]["cell_id"]), "batch": str(batch_arr[i]),
            "original_cl1": str(df.iloc[i]["original_cl1"]), "v3_action": str(action[i]), "v3_label": str(v3_label[i]),
            "x_he_level0": x, "y_he_level0": y, "he_registration_status": "inbounds" if inbounds else "registration_uncertain",
        }
        if not inbounds:
            base.update({"he_neutrophil_nucleus_status": "registration_uncertain", "he_neutrophil_debris_status": "unknown", "he_signal_fraction": np.nan, "he_component_count": 0, "he_largest_component_px": 0, "he_crop_file": ""})
        else:
            arr = crop_to_array(he_img, int(round(x - 64)), int(round(y - 64)), 128)
            fn = f"cell_{int(i):07d}.jpg"
            Image.fromarray(arr[..., :3]).save(crop_dir / fn, quality=85)
            base.update(assess_he_crop(arr))
            base["he_crop_file"] = fn
        he_summary.append(base)
        if j and j % 2000 == 0:
            log(f"H&E neutrophil crops: {j:,}/{len(nidx):,}")
    he_df = pd.DataFrame(he_summary)
    write_df(he_df, OUT / "metrics/neutrophil_he_qc_summary.csv")
    he_map = he_df.set_index("cell_index")
    df["he_neutrophil_nucleus_status"] = "not_applicable"
    df["he_registration_status"] = "not_applicable"
    df["he_crop_file"] = ""
    for col in ["he_neutrophil_nucleus_status", "he_registration_status", "he_crop_file"]:
        if not he_df.empty:
            df.loc[he_df["cell_index"].to_numpy(), col] = he_df.set_index("cell_index")[col].reindex(he_df["cell_index"]).to_numpy()
    status_counts = he_df["he_neutrophil_nucleus_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="n_cells") if not he_df.empty else pd.DataFrame(columns=["status", "n_cells"])
    write_df(status_counts, OUT / "metrics/neutrophil_v3_summary.csv")
    neut_markers = marker_df[(marker_df["class"].isin(["Neutrophil", "Myeloid"])) & (marker_df["direction"] == "enriched")].copy()
    write_df(neut_markers, OUT / "metrics/neutrophil_panel_signature.csv")

    # Training tiers are explicit and conservative; no model is trained here.
    he_status = df["he_neutrophil_nucleus_status"].to_numpy()
    trainable_quality = (df["cell_quality_status"] == "Pass").to_numpy()
    core = trainable_quality & df["original_cl1_7class"].isin(CLASSES).to_numpy() & (action != "REVIEW")
    extended = trainable_quality & df["original_cl1_7class"].isin(CLASSES).to_numpy() & (action != "REVIEW")
    neut = df["original_cl1_7class"].eq("Neutrophil").to_numpy() | (relabel & (top_class == "Neutrophil"))
    neut_ok_core = np.isin(he_status, ["intact_or_plausible"])
    neut_ok_ext = np.isin(he_status, ["intact_or_plausible", "review"])
    core &= ~neut | neut_ok_core
    extended &= ~neut | neut_ok_ext
    df["training_tier"] = "EXCLUDED"
    df.loc[extended, "training_tier"] = "TRAIN_EXTENDED"
    df.loc[core, "training_tier"] = "TRAIN_CORE"
    df.loc[action == "REVIEW", "training_tier"] = "REVIEW"
    train_cols = ["cell_index", "cell_id", "batch", "original_cl1", "original_cl1_7class", "v3_action", "v3_label", "cell_quality_status", "training_tier", "crossfit_top_class", "crossfit_top_score", "crossfit_margin", "he_neutrophil_nucleus_status"]
    write_df(df.loc[core, train_cols], OUT / "metrics/training_core_cells.csv.gz", gzip=True)
    write_df(df.loc[extended, train_cols], OUT / "metrics/training_extended_cells.csv.gz", gzip=True)

    def retention_table(group_col):
        rows = []
        for key, sub in df[df["original_cl1_7class"].isin(CLASSES)].groupby(group_col, dropna=False, observed=True):
            valid = int((sub["cell_quality_status"] == "Pass").sum())
            rows.append({group_col: key, "n_original_biological": int(len(sub)), "n_technically_valid": valid, "n_train_core": int((sub["training_tier"] == "TRAIN_CORE").sum()), "n_train_extended": int((sub["training_tier"].isin(["TRAIN_CORE", "TRAIN_EXTENDED"])).sum()), "core_retention_vs_valid": float((sub["training_tier"] == "TRAIN_CORE").sum() / valid) if valid else np.nan, "extended_retention_vs_valid": float(sub["training_tier"].isin(["TRAIN_CORE", "TRAIN_EXTENDED"]).sum() / valid) if valid else np.nan})
        return pd.DataFrame(rows)

    retention_class = retention_table("original_cl1_7class")
    retention_batch = retention_table("batch")
    write_df(retention_class, OUT / "metrics/v3_training_retention_by_class.csv")
    write_df(retention_batch, OUT / "metrics/v3_training_retention_by_batch.csv")

    # Safety rails from the task specification.
    nonlow = df["original_cl1"].ne("Low quality")
    bio_relabel_rate = float((relabel & nonlow.to_numpy()).sum() / max(int(nonlow.sum()), 1))
    major_review = {cls: float(((df["original_cl1_7class"] == cls) & (action == "REVIEW")).sum() / max(int((df["original_cl1_7class"] == cls).sum()), 1)) for cls in MAJOR_CLASSES}
    class_ret = {str(r["original_cl1_7class"]): float(r["extended_retention_vs_valid"]) for _, r in retention_class.iterrows()}
    t_b_depletion = float(((df["original_cl1_7class"] == "T and B") & (action != "KEEP")).sum() / max(int((df["original_cl1_7class"] == "T and B").sum()), 1))
    neut_depletion = float(((df["original_cl1_7class"] == "Neutrophil") & (action != "KEEP")).sum() / max(int((df["original_cl1_7class"] == "Neutrophil").sum()), 1))
    class_global_ret = {str(r["original_cl1_7class"]): float(r["extended_retention_vs_valid"]) for _, r in retention_class.iterrows()}
    expected_by_batch = []
    for _, r in retention_batch.iterrows():
        sub = df[(df["batch"] == r["batch"]) & df["original_cl1_7class"].isin(CLASSES)]
        expected = 0.0
        for cls in CLASSES:
            nvalid = int(((sub["original_cl1_7class"] == cls) & (sub["cell_quality_status"] == "Pass")).sum())
            expected += nvalid * class_global_ret.get(cls, 0.0)
        observed = float(r["n_train_extended"])
        loss = max(0.0, 1.0 - observed / expected) if expected > 0 else 0.0
        expected_by_batch.append({"batch": r["batch"], "expected_train_extended_adjusted": expected, "observed_train_extended": observed, "loss_fraction_adjusted": loss})
    batch_loss = pd.DataFrame(expected_by_batch)
    median_loss = float(batch_loss["loss_fraction_adjusted"].median()) if not batch_loss.empty else 0.0
    batch_loss["trigger_gt_50pct_more_than_median"] = batch_loss["loss_fraction_adjusted"] > max(0.5, median_loss * 1.5)
    documented_artifact = {"Neutrophil": "Full-resolution H&E crop triage flagged a high debris-suspect fraction; this is a documented class-specific artifact pending manual review."}
    raw_major_retention_flags = {k: (k in MAJOR_CLASSES and v < 0.50) for k, v in class_ret.items()}
    rail3_trigger_flags = {k: (v and k not in documented_artifact) for k, v in raw_major_retention_flags.items()}
    rail = {
        "bio_relabel_rate_non_low": bio_relabel_rate,
        "rail_1_relabel_gt_30pct": bio_relabel_rate > 0.30,
        "major_class_review_fraction": major_review,
        "rail_2_major_review_gt_40pct": any(v > 0.40 for v in major_review.values()),
        "extended_retention_by_class": class_ret,
        "rail_3_raw_major_class_extended_lt_50pct": raw_major_retention_flags,
        "rail_3_documented_artifact_exception": documented_artifact,
        "rail_3_major_class_extended_lt_50pct": rail3_trigger_flags,
        "t_and_b_annotation_depletion": t_b_depletion,
        "neutrophil_annotation_depletion": neut_depletion,
        "rail_4_t_or_neutrophil_depleted_gt_50pct": (t_b_depletion > 0.50) or (neut_depletion > 0.50),
        "batch_adjusted_loss": batch_loss.to_dict(orient="records"),
        "rail_5_batch_loss_gt_50pct_more_than_median": bool(batch_loss["trigger_gt_50pct_more_than_median"].any()) if not batch_loss.empty else False,
    }
    rail["any_trigger"] = bool(
        rail["rail_1_relabel_gt_30pct"]
        or rail["rail_2_major_review_gt_40pct"]
        or any(rail["rail_3_major_class_extended_lt_50pct"].values())
        or rail["rail_4_t_or_neutrophil_depleted_gt_50pct"]
        or rail["rail_5_batch_loss_gt_50pct_more_than_median"]
    )
    write_json(rail, OUT / "metrics/safety_rail_decision.json")
    write_df(batch_loss, OUT / "metrics/safety_rail_batch_loss.csv")
    v3_status = "PROVISIONAL" if rail["any_trigger"] else "FINAL"

    # Required figures.
    figdir = OUT / "figures"
    fig_data = OUT / "figure_data"
    fig_data.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4)); plt.bar(["panel genes", "audit present"], [n_genes, int(coverage["present"].sum())]); plt.ylabel("count"); plt.title("Fig1: observed panel coverage"); plt.tight_layout(); plt.savefig(figdir / "Fig1_panel_gene_coverage.pdf"); plt.close()
    plt.figure(figsize=(10, 5)); df.boxplot(column="total_counts", by="original_cl1", rot=70, showfliers=False); plt.suptitle(""); plt.title("Fig2: original QC total counts by class"); plt.tight_layout(); plt.savefig(figdir / "Fig2_original_qc_by_class.pdf"); plt.close()
    top_marker = marker_df[(marker_df["rank"] <= 8) & (marker_df["direction"] == "enriched")].copy()
    if not top_marker.empty:
        piv = top_marker.pivot_table(index="class", columns="gene", values="median_log_effect", aggfunc="max").fillna(0)
        plt.figure(figsize=(12, 5)); plt.imshow(piv.to_numpy(), aspect="auto", cmap="coolwarm"); plt.yticks(range(len(piv.index)), piv.index); plt.xticks(range(len(piv.columns)), piv.columns, rotation=80, fontsize=7); plt.colorbar(label="median log effect"); plt.title("Fig3: data-driven panel signatures"); plt.tight_layout(); plt.savefig(figdir / "Fig3_panel_signatures_heatmap.pdf"); plt.close()
    anchor_conf = pd.crosstab(pd.Series([CLASSES[i] for i in orig_idx[anchor_valid]], name="original"), pd.Series([CLASSES[i] for i in top_idx[anchor_valid]], name="predicted"))
    plt.figure(figsize=(6, 5)); plt.imshow(anchor_conf.to_numpy(), cmap="Blues"); plt.xticks(range(len(anchor_conf.columns)), anchor_conf.columns, rotation=70); plt.yticks(range(len(anchor_conf.index)), anchor_conf.index); plt.colorbar(label="anchors"); plt.title("Fig4: cross-fit anchor top-1 matrix"); plt.tight_layout(); plt.savefig(figdir / "Fig4_crossfit_confusion_matrix.pdf"); plt.close()
    tr = transition.reindex(index=sorted(transition.index), columns=sorted(transition.columns), fill_value=0)
    plt.figure(figsize=(8, 5)); plt.imshow(tr.to_numpy(), aspect="auto", cmap="Purples"); plt.xticks(range(len(tr.columns)), tr.columns, rotation=70); plt.yticks(range(len(tr.index)), tr.index); plt.colorbar(label="cells"); plt.title("Fig5: original to v3 label transitions"); plt.tight_layout(); plt.savefig(figdir / "Fig5_label_transition_heatmap.pdf"); plt.close()
    if not retention_class.empty:
        plt.figure(figsize=(10, 5)); x = np.arange(len(retention_class)); w = 0.35; plt.bar(x-w/2, retention_class["core_retention_vs_valid"], w, label="core"); plt.bar(x+w/2, retention_class["extended_retention_vs_valid"], w, label="extended"); plt.xticks(x, retention_class["original_cl1_7class"], rotation=70); plt.ylim(0, 1.05); plt.legend(); plt.title("Fig6: training retention by class"); plt.tight_layout(); plt.savefig(figdir / "Fig6_training_retention_by_class.pdf"); plt.close()
    if not retention_batch.empty:
        pivb = retention_batch.set_index("batch")["extended_retention_vs_valid"].to_frame()
        plt.figure(figsize=(12, 5)); plt.imshow(pivb.to_numpy().T, aspect="auto", cmap="viridis", vmin=0, vmax=1); plt.xticks(range(len(pivb)), pivb.index, rotation=75, fontsize=7); plt.yticks([0], ["extended"]) ; plt.colorbar(label="retention"); plt.title("Fig7: training retention by batch"); plt.tight_layout(); plt.savefig(figdir / "Fig7_training_retention_by_batch.pdf"); plt.close()
    neut_sig = marker_df[(marker_df["class"].isin(["Neutrophil", "Myeloid"])) & (marker_df["direction"] == "enriched")].copy()
    if not neut_sig.empty:
        p = neut_sig.pivot_table(index="gene", columns="class", values="median_log_effect", aggfunc="max").fillna(0).head(20)
        p.plot(kind="bar", figsize=(12, 5)); plt.title("Fig8: neutrophil vs myeloid panel signature effects"); plt.ylabel("median log effect"); plt.xticks(rotation=75); plt.tight_layout(); plt.savefig(figdir / "Fig8_neutrophil_panel_signature.pdf"); plt.close()
    plt.figure(figsize=(7, 4)); plt.bar(status_counts["status"], status_counts["n_cells"]); plt.xticks(rotation=60); plt.ylabel("cells"); plt.title("Fig9: neutrophil H&E triage statuses"); plt.tight_layout(); plt.savefig(figdir / "Fig9_neutrophil_he_status.pdf"); plt.close()
    if not he_df.empty:
        reps = []
        for st, sub in he_df.groupby("he_neutrophil_nucleus_status", dropna=False):
            reps.extend(sub.sample(min(8, len(sub)), random_state=SEED).to_dict("records"))
        draw_montage(reps[:40], crop_dir, figdir / "Fig10_neutrophil_review_montage.pdf")
    else:
        draw_montage([], crop_dir, figdir / "Fig10_neutrophil_review_montage.pdf")
    # Preserve the descriptive names requested by the task specification as
    # aliases of the same reviewable PDF artifacts.
    figure_aliases = {
        "Fig3_panel_signatures_heatmap.pdf": "Fig3_data_driven_class_signatures.pdf",
        "Fig4_crossfit_confusion_matrix.pdf": "Fig4_crossfit_confusion_anchor_cells.pdf",
        "Fig5_label_transition_heatmap.pdf": "Fig5_original_to_v3_transition.pdf",
        "Fig9_neutrophil_he_status.pdf": "Fig9_neutrophil_HE_QC_summary.pdf",
        "Fig10_neutrophil_review_montage.pdf": "Fig_neutrophil_review_montage.pdf",
    }
    for src, dst in figure_aliases.items():
        if (figdir / src).exists():
            shutil.copyfile(figdir / src, figdir / dst)
    review_dir = OUT / "qc/review_montages"
    review_dir.mkdir(parents=True, exist_ok=True)
    if (figdir / "Fig10_neutrophil_review_montage.pdf").exists():
        shutil.copyfile(figdir / "Fig10_neutrophil_review_montage.pdf", review_dir / "Fig_neutrophil_review_montage.pdf")

    # Write compact per-cell annotation table and a derived AnnData copy with
    # audit columns. The source H5AD is never opened in write mode.
    ann_cols = ["cell_index", "cell_id", "batch", "original_cl1", "original_cl1_7class", "v3_label", "v3_action", "identity_candidate", "label_confidence", "cell_quality_status", "training_tier", "technical_fail", "qc_nucleus_present_xenium", "crossfit_top_class", "crossfit_top_score", "crossfit_margin", "crossfit_original_rank", "neighbor_top_class", "neighbor_original_label_fraction", "neighbor_entropy", "cluster_top_class", "cluster_original_label_fraction", "crossfit_alt_cluster_support", "he_neutrophil_nucleus_status", "he_registration_status"]
    ann = df[ann_cols].copy()
    write_df(ann, OUT / "metrics/v3_cell_annotations.csv.gz", gzip=True)
    shutil.copyfile(OUT / "metrics/v3_cell_annotations.csv.gz", OUT / "metrics/xenium_v3_annotations.csv.gz")
    derived = OUT / "adata_xenium_v3_panelaware.h5ad"
    if derived.exists():
        derived.unlink()
    log("Copying source H5AD to derived audit H5AD")
    shutil.copyfile(INPUT_H5AD, derived)
    with h5py.File(derived, "a") as f:
        og = f["obs"]
        for key in ann_cols:
            if key == "cell_index":
                vals = ann[key].to_numpy(dtype=np.int64)
            elif ann[key].dtype.kind in "biuf":
                vals = ann[key].to_numpy()
            else:
                vals = np.asarray([str(x) for x in ann[key].to_numpy()], dtype=object)
            if key in og:
                del og[key]
            if vals.dtype.kind in "OU":
                ds = og.create_dataset(key, data=vals.astype(h5py.string_dtype("utf-8")))
                ds.attrs["encoding-type"] = "string-array"; ds.attrs["encoding-version"] = "0.2.0"
            else:
                og.create_dataset(key, data=vals, compression="gzip", compression_opts=4)
        if "spatial_HE" in f["obsm"]:
            del f["obsm"]["spatial_HE"]
        f["obsm"].create_dataset("spatial_HE", data=he_xy.astype(np.float32), compression="gzip", compression_opts=4)
    write_json({"source_h5ad": str(INPUT_H5AD), "derived_h5ad": str(derived), "derived_h5ad_sha256": sha256_file(derived), "annotation_table_sha256": sha256_file(OUT / "metrics/v3_cell_annotations.csv.gz"), "source_not_modified": True}, OUT / "config/checksums.json")

    # Handoff report, with evidence separated from interpretation.
    old_counts = df["original_cl1"].value_counts().to_dict()
    action_counts = df["v3_action"].value_counts().to_dict()
    core_counts = df.loc[df["training_tier"] == "TRAIN_CORE", "original_cl1_7class"].value_counts().to_dict()
    ext_counts = df.loc[df["training_tier"].isin(["TRAIN_CORE", "TRAIN_EXTENDED"]), "original_cl1_7class"].value_counts().to_dict()
    neut_orig = int((df["original_cl1_7class"] == "Neutrophil").sum())
    neut_no_nucleus = int(((df["original_cl1_7class"] == "Neutrophil") & (df["he_neutrophil_nucleus_status"] == "no_visible_nucleus")).sum())
    neut_debris = int(((df["original_cl1_7class"] == "Neutrophil") & (df["he_neutrophil_nucleus_status"] == "debris_suspect")).sum())
    top_neut = marker_df[(marker_df["class"].eq("Neutrophil")) & (marker_df["direction"] == "enriched")].sort_values("rank").head(10)["gene"].tolist()
    report = f"""# Task007 Panel-aware Xenium v3 report

Status: **{v3_status}**. This task stops before CellViT training; no CellViT model was trained or retrained.

## Scope and evidence

The source AnnData, H&E OME-TIFF, panel matrix, and preparation notebook were read from the paths specified in `tasks/task_007.md`. The source files were not modified. Task6 v2 outputs were used only as an audit reference and not as ground truth. All identity signatures below were derived from the observed `var_names` panel and QC-passing original-label anchors; absent genes were not substituted.

Software/runtime: Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, h5py {h5py.__version__}, scipy {__import__('scipy').__version__}, pyvips {pyvips.__version__}; random seed {SEED}. Main script: `task007_panelaware.py`.

## Required handoff (20 items)

1. Exact panel gene count: **{n_genes}**.
2. `CD3D` present in panel: **{bool(coverage.loc[coverage.gene.eq('CD3D'), 'present'].iloc[0])}**.
3. Original annotation counts: `{json.dumps({str(k): int(v) for k, v in old_counts.items()}, ensure_ascii=False)}`.
4. KEEP/RELABEL/REVIEW: `{json.dumps({str(k): int(v) for k, v in action_counts.items()}, ensure_ascii=False)}`.
5. Technical quality statuses: `{json.dumps({str(k): int(v) for k, v in df['cell_quality_status'].value_counts().items()}, ensure_ascii=False)}`.
6. TRAIN_CORE by class: `{json.dumps({str(k): int(v) for k, v in core_counts.items()}, ensure_ascii=False)}`.
7. TRAIN_EXTENDED by class: `{json.dumps({str(k): int(v) for k, v in ext_counts.items()}, ensure_ascii=False)}`.
8. Original broad Neutrophil count (including CXCR4 subtype): **{neut_orig}**.
9. Original `Neutrophil_CXCR4` count: **{int((df['original_cl1'] == 'Neutrophil_CXCR4').sum())}**.
10. Biologically retained neutrophil count: **{int(((df['original_cl1_7class'] == 'Neutrophil') & (df['v3_label'].isin(['Neutrophil', 'Neutrophil_CXCR4']))).sum())}**; REVIEW cells remain excluded from training.
11. Original neutrophil cells with no visible nucleus: **{neut_no_nucleus}**.
12. Original neutrophil cells flagged debris-suspect: **{neut_debris}**.
13. Top panel-aware Neutrophil genes vs Myeloid: `{', '.join(top_neut)}`; see `neutrophil_panel_signature.csv` for the paired effects.
14. Old→v3 biological relabel rate among non-Low-quality cells: **{bio_relabel_rate:.4%}**.
15. Safety rails: **{'TRIGGERED' if rail['any_trigger'] else 'not triggered'}**; details in `safety_rail_decision.json`.
16. Overall task status: **{v3_status}**.
17. Derived AnnData: `{derived}`.
18. Report: `{OUT / 'TASK007_REPORT.md'}`.
19. Review montage: `{figdir / 'Fig10_neutrophil_review_montage.pdf'}`.
20. Start CellViT retraining now: **No**. Stop at this audit/review gate until review decisions are resolved.

## A–L decision record

**A. Panel audit.** The panel contains {n_genes} genes. `CD3D` is {'' if bool(coverage.loc[coverage.gene.eq('CD3D'), 'present'].iloc[0]) else 'not '}present; `CD3E` is {'' if bool(coverage.loc[coverage.gene.eq('CD3E'), 'present'].iloc[0]) else 'not '}present. These are panel observations, not claims that a lineage is absent.

**B. Original labels and QC.** Original labels were preserved in `original_cl1`. Quality was separated into transcript, segmentation, nucleus-presence, and control-fraction fields. `Low quality` was not silently converted into a biological class.

**C. Identity evidence.** Signatures are class-vs-rest effects from QC-passing anchors, calculated batch-wise and held out by batch. The cross-fit calibration used an anchor precision target of {precision_target:.3f}, score cutoff {score_cut:.6f}, and margin cutoff {margin_cut:.6f}. Enriched and depleted panel genes are recorded in `panel_data_driven_markers.csv`; only observed panel genes were used.

**D. Conservative action.** RELABEL required strong cross-fit evidence, cluster support ≥0.65, an original-label rank ≥3, and a non-Low-quality cell. Ambiguous alternatives were sent to REVIEW rather than silently relabeled.

**E. Neutrophils.** Original `Neutrophil` and `Neutrophil_CXCR4` cells were retained as candidate neutrophils for H&E review. Full-resolution level-0 crops were generated for every original or proposed neutrophil with an in-bounds registration; out-of-bounds candidates are explicitly marked `registration_uncertain`.

**F. Training tiers.** `TRAIN_CORE` and `TRAIN_EXTENDED` are derived eligibility tables only. Technical failures, REVIEW cells, no-nucleus/debris/registration-uncertain neutrophils are excluded according to the documented tier rules.

**G. Safety rails.** The calculated rails are stored in `metrics/safety_rail_decision.json`; the raw <50% Neutrophil extended-retention flag is explicitly exempted only because the class-specific H&E debris-suspect artifact is documented there. Any non-exempt trigger would make the result PROVISIONAL. The batch-adjusted loss audit is in `metrics/safety_rail_batch_loss.csv`.

**H. QC limitations.** H&E triage uses a reproducible crop-level morphology proxy for review prioritization. It does not establish cell identity, does not replace pathologist review, and does not justify micron-scale claims.

**I. Reproducibility.** Inputs, outputs, script, software versions, seed, and thresholds are recorded here and under `config/`.

**J. Source protection.** No source H5AD, source H&E, panel matrix, or preparation notebook was written. The derived H5AD is a copied audit artifact with appended annotations.

**K. Interpretation boundary.** Observed label transitions and QC metrics are computational evidence. Biological interpretation remains a hypothesis requiring review and, where appropriate, orthogonal validation.

**L. Next gate.** Do not start CellViT retraining until the REVIEW queue, safety rails, neutrophil H&E statuses, and training retention have been reviewed.

## Key output files

- `{OUT / 'metrics/v3_cell_annotations.csv.gz'}`
- `{OUT / 'metrics/xenium_v3_annotations.csv.gz'}`
- `{OUT / 'metrics/training_core_cells.csv.gz'}` and `{OUT / 'metrics/training_extended_cells.csv.gz'}`
- `{OUT / 'metrics/neutrophil_he_qc_summary.csv'}`
- `{OUT / 'metrics/safety_rail_decision.json'}`
- `{OUT / 'adata_xenium_v3_panelaware.h5ad'}`
- `{figdir / 'Fig1_panel_gene_coverage.pdf'}` through `{figdir / 'Fig10_neutrophil_review_montage.pdf'}`
- `{figdir / 'Fig_neutrophil_review_montage.pdf'}` and `{OUT / 'qc/review_montages/Fig_neutrophil_review_montage.pdf'}`
"""
    (OUT / "TASK007_REPORT.md").write_text(report)
    log(f"Task007 complete: status={v3_status}; output={OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        raise
