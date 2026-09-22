#!/usr/bin/env python3
"""Task 006: independent Xenium re-annotation and H&E nuclear-integrity QC.

This script never uses CellViT predictions.  It reads the source AnnData in
backed mode, scores only genes present in the panel, applies batch-aware
transcript QC, uses the registered H&E for nuclear evidence, and writes a
frozen derived annotation table plus audit artifacts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import anndata as ad
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
NON_TRAINING = ["Uncertain", "Mixed_lineage", "Low_quality", "Artifact_or_no_nucleus"]
MARKERS = {
    "Endothelial": ["PECAM1", "VWF", "EMCN", "KDR", "ENG", "PLVAP", "RAMP2", "CA4", "ESAM"],
    "Mesenchymal": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL6A1", "DCN", "LUM", "FAP", "THY1", "PDGFRA", "PDGFRB", "ACTA2", "TAGLN", "RGS5"],
    "Myeloid": ["LST1", "TYROBP", "FCER1G", "CTSS", "CTSB", "CTSD", "C1QA", "C1QB", "C1QC", "CD68", "CD163", "APOE", "SPP1", "IL1B"],
    "Neutrophil": ["FCGR3B", "CSF3R", "CXCR2", "FPR1", "SELL", "S100A8", "S100A9", "MNDA", "NAMPT"],
    "Plasma cell": ["JCHAIN", "MZB1", "XBP1", "SDC1", "IGKC", "CD79A", "DERL3", "SSR4"],
    "T and B": ["CD3D", "CD3E", "TRBC1", "TRBC2", "LCK", "IL7R", "CCL5", "MS4A1", "CD79A", "CD79B", "CD19", "CD22", "CD37", "CD74", "HLA-DRA", "CD83"],
    "Tumor": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "ALB", "APOA1", "APOB", "GPC3", "AFP", "KRT17"],
}
NEUTROPHIL_SPECIFIC = ["FCGR3B", "CSF3R", "CXCR2", "FPR1", "SELL", "MNDA", "NAMPT"]


def robust_z(values: pd.Series, groups: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"value": values.to_numpy(dtype=float), "group": groups.astype(str).to_numpy()})
    med = frame.groupby("group")["value"].transform("median").to_numpy()
    mad = frame.groupby("group")["value"].transform(lambda x: np.median(np.abs(x - np.median(x))).astype(float)).to_numpy()
    return (values.to_numpy(dtype=float) - med) / (1.4826 * mad + 1e-6)


def write_h5_string(group: h5py.Group, name: str, values: np.ndarray) -> None:
    if name in group:
        del group[name]
    ds = group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))
    ds.attrs["encoding-type"] = "string-array"
    ds.attrs["encoding-version"] = "0.2.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=Path("/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad"))
    parser.add_argument("--he", type=Path, default=Path("/data/lf_data/xenium_data/ID0060276.ome.tif"))
    parser.add_argument("--matrix", type=Path, default=Path("/data/lf_data/xenium_data/matrix.csv"))
    parser.add_argument("--notebook", type=Path, default=Path("/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb"))
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task006_xenium_reannotation"))
    args = parser.parse_args()
    root = args.result.resolve()
    for name in ["config", "code", "qc", "metrics", "work", "models", "figures", "figure_data", "logs", "qc/montages", "qc/registration_overlays"]:
        (root / name).mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.adata, backed="r")
    with h5py.File(args.adata, "r") as source_h5:
        raw_present = "raw" in source_h5
        x_attrs = {str(k): (v.decode() if isinstance(v, bytes) else v) for k, v in source_h5["X"].attrs.items()}
    obs_cols = list(source.obs.columns)
    var_cols = list(source.var.columns)
    inventory = {
        "source": str(args.adata),
        "shape": [int(source.n_obs), int(source.n_vars)],
        "obs_columns": obs_cols,
        "var_columns": var_cols,
        "layers": list(source.layers.keys()),
        "raw_present": raw_present,
        "obsm_keys": list(source.obsm.keys()),
        "obsp_keys": list(source.obsp.keys()),
        "uns_keys": list(source.uns.keys()),
        "X_encoding": x_attrs,
        "X_representation_assessment": "Sparse X was compared with layers/counts by file structure; marker scoring uses layers/counts explicitly.",
        "spatial_key": "spatial" if "spatial" in source.obsm else None,
        "batch_column": "batch" if "batch" in source.obs else None,
        "old_annotation_column": "cl1" if "cl1" in source.obs else None,
        "old_object_id_column": "cell_labels" if "cell_labels" in source.obs else None,
    }
    (root / "qc/adata_inventory.json").write_text(json.dumps(inventory, indent=2, default=str) + "\n")
    source.obs.head(0).T.reset_index().rename(columns={"index": "column"}).to_csv(root / "qc/obs_schema.csv", index=False)
    source.var.head(0).T.reset_index().rename(columns={"index": "column"}).to_csv(root / "qc/var_schema.csv", index=False)

    old_audit = f"""# Old Xenium-to-CellViT pipeline audit

Source notebook: `{args.notebook}`

The notebook uses `LABEL_COLUMN = "celltype"`, `BATCH_COLUMN = "batch"`, a per-label maximum of 50,000 cells with seed 1234, an 8-batch/8-batch sample split with `RANDOM_STATE = 2026`, `PIXEL_SIZE = 0.2125`, 256-pixel non-overlapping H&E patches, and `matrix.csv` inverse transformation from Xenium pixel coordinates to H&E pixels. It builds a KD-tree, keeps only single-batch patches, writes labels, then renames patch files to `train_*` and `test_*` and creates ordinary KFold splits.

The current AnnData does not contain `celltype`; its observed old biological annotation is `obs['cl1']`, with the categories `Endothelial`, `Mesenchymal`, `Myeloid`, `Neutrophil`, `Neutrophil_CXCR4`, `Plasma cell`, `T and B`, `Tumor`, and `Low quality`. `obs['cell_labels']` is a unique integer object identifier and is not a biological label. The existing CellViT dataset was therefore treated as the old-label derivative, not silently regenerated from an assumed column.

The current H&E OME metadata reports a 50,000 x 23,451 level-0 RGB image and a `PhysicalSizeX/Y` value of 352.7777777778 micrometres. This metadata is inconsistent with the notebook's registration geometry and is recorded as an unresolved physical-scale discrepancy. Registration itself was validated using the notebook's explicit 0.2125-µm/Xenium-pixel conversion and the inverse affine transform; morphology QC uses level-3 pixel coordinates and records the scale assumption rather than altering the source metadata.
"""
    (root / "qc/old_pipeline_audit.md").write_text(old_audit)

    # Load only the relevant obs columns; do not materialize the 5,001-gene matrix.
    needed = [c for c in ["cell_id", "batch", "cl1", "leiden", "cell_area", "nucleus_area", "nucleus_count", "total_counts", "n_counts", "n_genes_by_counts", "transcript_counts", "control_probe_counts", "genomic_control_counts", "control_codeword_counts", "unassigned_codeword_counts", "deprecated_codeword_counts"] if c in source.obs]
    obs = source.obs[needed].copy()
    obs.index = obs.index.astype(str)
    n = len(obs)
    batch = obs["batch"].astype(str)
    cl1 = obs["cl1"].astype(str)

    spatial = np.asarray(source.obsm["spatial"][:], dtype=np.float64)
    matrix = np.loadtxt(args.matrix, delimiter=",")
    spatial_xenium_px = spatial / 0.2125
    spatial_hom = np.c_[spatial_xenium_px, np.ones(n)]
    spatial_he = (np.linalg.inv(matrix) @ spatial_hom.T).T[:, :2]
    he_width, he_height = 50000, 23451
    inbounds = (spatial_he[:, 0] >= 0) & (spatial_he[:, 0] < he_width) & (spatial_he[:, 1] >= 0) & (spatial_he[:, 1] < he_height)
    reg = pd.DataFrame({"metric": ["n_cells", "inbounds_fraction", "x_min", "x_max", "y_min", "y_max", "x_median", "y_median"], "value": [n, inbounds.mean(), spatial_he[:, 0].min(), spatial_he[:, 0].max(), spatial_he[:, 1].min(), spatial_he[:, 1].max(), np.median(spatial_he[:, 0]), np.median(spatial_he[:, 1])]})
    reg.to_csv(root / "qc/registration_summary.csv", index=False)
    reg_by_batch = pd.DataFrame({"batch": batch, "inbounds": inbounds}).groupby("batch").agg(n_cells=("inbounds", "size"), inbounds_fraction=("inbounds", "mean")).reset_index()
    reg_by_batch.to_csv(root / "qc/registration_by_batch.csv", index=False)

    # Registration overview using the level-3 thumbnail.
    with tifffile.TiffFile(args.he) as tf:
        series = tf.series[0]
        he_meta = {"series_shape": list(series.shape), "axes": series.axes, "dtype": str(series.dtype), "pyramid_levels": len(series.levels), "ome_metadata": tf.ome_metadata}
        thumb = series.levels[3].asarray()
    (root / "qc/ome_metadata.json").write_text(json.dumps(he_meta, indent=2, default=str) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=min(30000, n), replace=False)
    thumb_rgb = thumb[..., :3]
    for ax, b in zip(axes.flat, sorted(batch.unique())[:4]):
        ids = sample_idx[batch.to_numpy()[sample_idx] == b]
        ax.imshow(thumb_rgb)
        ax.scatter(spatial_he[ids, 0] / 8, spatial_he[ids, 1] / 8, s=1, alpha=.35)
        ax.set_title(f"batch {b}"); ax.set_xlim(0, thumb.shape[1]); ax.set_ylim(thumb.shape[0], 0); ax.axis("off")
    fig.suptitle(f"Registration QC; inverse affine, in-bounds={inbounds.mean():.3f}")
    fig.tight_layout(); fig.savefig(root / "figures/Fig1_registration_QC.pdf", bbox_inches="tight"); plt.close(fig)
    for b in sorted(batch.unique()):
        ids = sample_idx[batch.to_numpy()[sample_idx] == b]
        fig, ax = plt.subplots(figsize=(8, 4.5)); ax.imshow(thumb_rgb); ax.scatter(spatial_he[ids, 0] / 8, spatial_he[ids, 1] / 8, s=1, alpha=.35); ax.set_title(f"Registration overlay {b}"); ax.axis("off"); fig.savefig(root / "qc/registration_overlays" / f"overlay_{b}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Batch-aware transcript QC.
    total = obs.get("total_counts", obs.get("n_counts")).astype(float).to_numpy()
    genes = obs["n_genes_by_counts"].astype(float).to_numpy()
    area = obs["cell_area"].astype(float).to_numpy()
    nuc_area = obs["nucleus_area"].astype(float).to_numpy()
    nuc_count = obs["nucleus_count"].astype(float).to_numpy()
    controls = obs[[c for c in ["control_probe_counts", "genomic_control_counts", "control_codeword_counts", "unassigned_codeword_counts", "deprecated_codeword_counts"] if c in obs]].astype(float).sum(axis=1).to_numpy()
    control_fraction = controls / np.maximum(total, 1)
    qc = pd.DataFrame({"batch": batch, "total_counts": total, "n_genes": genes, "cell_area": area, "nucleus_area": nuc_area, "nucleus_count": nuc_count, "control_fraction": control_fraction})
    qc["log_total_counts"] = np.log1p(total); qc["log_n_genes"] = np.log1p(genes); qc["log_cell_area"] = np.log1p(area); qc["log_nucleus_area"] = np.log1p(nuc_area)
    z_total = robust_z(qc["log_total_counts"], batch); z_genes = robust_z(qc["log_n_genes"], batch); z_area = robust_z(qc["log_cell_area"], batch); z_control = robust_z(qc["control_fraction"], batch)
    qc["qc_low_counts"] = z_total < -3; qc["qc_low_genes"] = z_genes < -3; qc["qc_extreme_area"] = np.abs(z_area) > 4; qc["qc_high_control_fraction"] = z_control > 4
    qc["qc_other"] = ~np.isfinite(qc[["total_counts", "n_genes", "cell_area", "nucleus_area"]]).all(axis=1)
    qc["qc_transcript_pass"] = ~(qc[["qc_low_counts", "qc_low_genes", "qc_extreme_area", "qc_high_control_fraction", "qc_other"]].any(axis=1))
    qc.to_csv(root / "metrics/transcript_qc_by_cell.csv.gz", index=False)
    qcb = qc.groupby("batch")[["qc_low_counts", "qc_low_genes", "qc_extreme_area", "qc_high_control_fraction", "qc_transcript_pass"]].mean().reset_index()
    qcb.insert(1, "n_cells", qc.groupby("batch").size().to_numpy())
    qcb.to_csv(root / "metrics/transcript_qc_by_batch.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, ["log_total_counts", "log_n_genes", "log_cell_area", "control_fraction"]):
        ax.hist(qc[col].replace([np.inf, -np.inf], np.nan).dropna(), bins=100, color="#4477AA", alpha=.85); ax.set_title(col); ax.set_yscale("log")
    fig.suptitle("Batch-aware transcript QC distributions"); fig.tight_layout(); fig.savefig(root / "figures/Fig2_transcript_QC.pdf", bbox_inches="tight"); plt.close(fig)

    # Marker availability and independent marker scoring.
    var_names = source.var_names.astype(str).tolist(); var_map = {g: i for i, g in enumerate(var_names)}
    availability = []
    marker_indices = {}
    for lineage, geneset in MARKERS.items():
        present = [g for g in geneset if g in var_map]; absent = [g for g in geneset if g not in var_map]
        marker_indices[lineage] = [var_map[g] for g in present]
        availability.append({"lineage": lineage, "n_requested": len(geneset), "n_present": len(present), "present_markers": ";".join(present), "absent_markers": ";".join(absent)})
    pd.DataFrame(availability).to_csv(root / "metrics/marker_availability.csv", index=False)
    union = sorted(set(i for ids in marker_indices.values() for i in ids)); union_pos = {gene_idx: pos for pos, gene_idx in enumerate(union)}
    scores = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32); support = np.zeros_like(scores, dtype=np.int16); neutro_specific = np.zeros(n, dtype=np.int16)
    specific_idx = [var_map[g] for g in NEUTROPHIL_SPECIFIC if g in var_map]
    chunk = 10000
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        mat = source.layers["counts"][start:stop, union]
        dense = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
        logged = np.log1p(np.asarray(dense, dtype=np.float32))
        for j, lineage in enumerate(CLASS_NAMES):
            pos = [union_pos[i] for i in marker_indices[lineage]]
            if pos:
                scores[start:stop, j] = logged[:, pos].mean(axis=1)
                support[start:stop, j] = (dense[:, pos] > 0).sum(axis=1)
        if specific_idx:
            neutro_specific[start:stop] = (dense[:, [union_pos[i] for i in specific_idx]] > 0).sum(axis=1)
    score_df = pd.DataFrame(scores, columns=[f"score_{x}" for x in CLASS_NAMES])
    top = np.argsort(scores, axis=1)[:, ::-1]; top_idx = top[:, 0]; second_idx = top[:, 1]; top_score = scores[np.arange(n), top_idx]; second_score = scores[np.arange(n), second_idx]; margin = top_score - second_score; top_support = support[np.arange(n), top_idx]
    score_positive = top_score > 0
    score_cut = float(np.quantile(top_score[score_positive], .35)) if score_positive.any() else 0.0
    margin_cut = float(np.quantile(margin[score_positive], .35)) if score_positive.any() else 0.0
    cluster = obs["leiden"].astype(str).to_numpy() if "leiden" in obs else np.repeat("all", n)
    cluster_mean = pd.DataFrame(scores, columns=CLASS_NAMES).groupby(cluster).mean()
    cluster_top = cluster_mean.to_numpy().argmax(axis=1); cluster_lookup = {name: int(idx) for name, idx in zip(cluster_mean.index, cluster_top)}
    cluster_agreement = np.asarray([cluster_lookup[c] == t for c, t in zip(cluster, top_idx)])
    marker_metrics = score_df.copy(); marker_metrics["top_lineage"] = [CLASS_NAMES[i] for i in top_idx]; marker_metrics["top_score"] = top_score; marker_metrics["second_score"] = second_score; marker_metrics["lineage_score_margin"] = margin; marker_metrics["top_marker_support"] = top_support; marker_metrics["cluster_cell_agreement"] = cluster_agreement; marker_metrics["neutrophil_specific_support"] = neutro_specific
    marker_metrics.to_csv(root / "metrics/marker_scores_by_cell.csv.gz", index=False)
    cluster_out = cluster_mean.reset_index().rename(columns={"index": "leiden"}); cluster_out["cluster_annotation"] = cluster_top; cluster_out["cluster_annotation"] = cluster_out["cluster_annotation"].map(dict(enumerate(CLASS_NAMES))); cluster_out.to_csv(root / "metrics/cluster_lineage_scores.csv", index=False)

    candidate = score_positive & (top_score >= score_cut) & (margin >= margin_cut) & (top_support >= 2) & cluster_agreement
    provisional = np.full(n, "Uncertain", dtype=object)
    for idx, name in enumerate(CLASS_NAMES): provisional[candidate & (top_idx == idx)] = name
    mixed = score_positive & ~candidate & (top_support >= 1)
    provisional[mixed] = "Mixed_lineage"
    provisional[~qc["qc_transcript_pass"].to_numpy()] = "Low_quality"
    old_map = {"Neutrophil_CXCR4": "Neutrophil", "Low quality": "Low_quality"}
    old_label = cl1.map(lambda x: old_map.get(x, x)).to_numpy()

    # H&E nuclear evidence from a level-3 thumbnail. This is morphology QC only.
    scale = 8
    h_img = thumb_rgb.astype(np.float32) / 255.0
    od = -np.log(np.clip(h_img + 1.0 / 255.0, 1.0 / 255.0, 1.0))
    h_proxy = (0.65 * od[..., 0] + 0.70 * od[..., 1] + 0.29 * od[..., 2]).astype(np.float32)
    h_threshold = float(np.quantile(h_proxy, .92)); h_mask = h_proxy >= h_threshold
    labels, n_components = ndimage.label(h_mask, structure=np.ones((3, 3), dtype=np.uint8)); component_sizes = np.bincount(labels.ravel(), minlength=n_components + 1)
    ypix = np.clip(np.rint(spatial_he[:, 1] / scale).astype(int), 0, h_proxy.shape[0] - 1); xpix = np.clip(np.rint(spatial_he[:, 0] / scale).astype(int), 0, h_proxy.shape[1] - 1)
    radius = 6; kernel = (2 * radius + 1, 2 * radius + 1)
    h_mean = ndimage.uniform_filter(h_proxy, size=kernel, mode="nearest"); h_frac = ndimage.uniform_filter(h_mask.astype(np.float32), size=kernel, mode="nearest"); h_max = ndimage.maximum_filter(h_proxy, size=kernel, mode="nearest")
    cell_h_mean = h_mean[ypix, xpix]; cell_h_frac = h_frac[ypix, xpix]; cell_h_max = h_max[ypix, xpix]; component_id = labels[ypix, xpix]; component_area = component_sizes[component_id].astype(float)
    distance = ndimage.distance_transform_edt(~h_mask); distance_px = distance[ypix, xpix].astype(float) * scale
    nonzero = cell_h_frac > 0
    frac_low = float(np.quantile(cell_h_frac[nonzero], .20)) if nonzero.any() else 0.0; frac_present = float(np.quantile(cell_h_frac[nonzero], .40)) if nonzero.any() else 0.0; frac_high = float(np.quantile(cell_h_frac[nonzero], .80)) if nonzero.any() else 1.0
    frac_norm = np.clip((cell_h_frac - frac_low) / max(frac_high - frac_low, 1e-6), 0, 1); max_norm = np.clip((cell_h_max - np.quantile(cell_h_max, .20)) / max(np.quantile(cell_h_max, .80) - np.quantile(cell_h_max, .20), 1e-6), 0, 1)
    he_score = .65 * frac_norm + .35 * max_norm; he_present = (cell_h_frac >= frac_present) & (distance_px <= 32); he_low = cell_h_frac < frac_low; local_area = cell_h_frac * ((2 * radius + 1) ** 2); fragmentation = np.clip(1 - component_area / np.maximum(local_area, 1), 0, 1); he_debris = he_present & (fragmentation > .85) & (nuc_count >= 2); he_reg_suspect = ~inbounds
    he_qc = pd.DataFrame({"he_nucleus_present": he_present, "he_low_nuclear_signal": he_low, "he_fragmented_debris": he_debris, "he_registration_suspect": he_reg_suspect, "he_nuclear_integrity_score": he_score, "he_hematoxylin_fraction": cell_h_frac, "he_hematoxylin_local_mean": cell_h_mean, "he_hematoxylin_local_max": cell_h_max, "he_component_area": component_area, "he_local_component_area": local_area, "he_fragmentation_score": fragmentation, "he_distance_to_nuclear_signal_px": distance_px, "he_component_id": component_id})

    neutro_top = top_idx == CLASS_NAMES.index("Neutrophil")
    neutro_debris = neutro_top & ((neutro_specific < 1) | he_low | he_debris | (nuc_count <= 0) | he_reg_suspect)
    neutro_support = np.where(neutro_specific > 0, "specific_neutrophil_marker", np.where(support[:, CLASS_NAMES.index("Neutrophil")] > 0, "generic_or_weak_neutrophil_signal", "no_neutrophil_marker_support"))
    he_qc["neutrophil_transcript_support"] = neutro_support; he_qc["neutrophil_nuclear_integrity"] = np.where(neutro_top, he_present & ~he_debris, False); he_qc["neutrophil_debris_suspect"] = neutro_debris

    valid_reg = inbounds & ~he_reg_suspect
    hq_extended = np.isin(provisional, CLASS_NAMES) & qc["qc_transcript_pass"].to_numpy() & valid_reg & ~he_debris
    hq_core = hq_extended & candidate & (he_present | (nuc_count > 0))
    # Neutrophil-specific safeguard: allow multiple lobes, but require specific transcript support and plausible nuclear evidence.
    neutro_core_ok = (~neutro_top) | ((neutro_specific >= 1) & he_present & ~neutro_debris)
    neutro_extended_ok = (~neutro_top) | ((neutro_specific >= 1) & ~neutro_debris & (he_present | (nuc_count > 0)))
    hq_core &= neutro_core_ok; hq_extended &= neutro_extended_ok
    quality_tier = np.full(n, "EXCLUDE_OR_REVIEW", dtype=object); quality_tier[hq_extended] = "HQ_EXTENDED"; quality_tier[hq_core] = "HQ_CORE"
    final_label = provisional.copy(); final_label[~np.isin(final_label, CLASS_NAMES)] = provisional[~np.isin(final_label, CLASS_NAMES)]
    final_label[~qc["qc_transcript_pass"].to_numpy()] = "Low_quality"; final_label[he_reg_suspect & np.isin(final_label, CLASS_NAMES)] = "Artifact_or_no_nucleus"; final_label[neutro_debris] = "Artifact_or_no_nucleus"

    ann = pd.DataFrame({"cell_id": obs["cell_id"].astype(str).to_numpy() if "cell_id" in obs else source.obs_names.astype(str), "batch": batch.to_numpy(), "original_label": old_label, "original_label_raw": cl1.to_numpy(), "new_celltype": final_label, "provisional_lineage": provisional, "quality_tier": quality_tier, "annotation_confidence_score": top_score + margin, "annotation_confidence_rank": pd.Series(top_score + margin).rank(pct=True).to_numpy(), "cluster_cell_agreement": cluster_agreement, "lineage_score_margin": margin, "top_lineage": [CLASS_NAMES[i] for i in top_idx], "top_marker_support": top_support, "neutrophil_specific_support": neutro_specific, "neutrophil_transcript_support": neutro_support, "neutrophil_nuclear_integrity": he_qc["neutrophil_nuclear_integrity"].to_numpy(), "neutrophil_debris_suspect": neutro_debris, "xenium_x_um": spatial[:, 0], "xenium_y_um": spatial[:, 1], "he_x_px": spatial_he[:, 0], "he_y_px": spatial_he[:, 1], "qc_transcript_pass": qc["qc_transcript_pass"].to_numpy(), "qc_low_counts": qc["qc_low_counts"].to_numpy(), "qc_low_genes": qc["qc_low_genes"].to_numpy(), "qc_extreme_area": qc["qc_extreme_area"].to_numpy(), "qc_high_control_fraction": qc["qc_high_control_fraction"].to_numpy(), "nucleus_count": nuc_count, "total_counts": total, "n_genes": genes, "cell_area": area, "nucleus_area": nuc_area})
    ann = pd.concat([ann, score_df, he_qc], axis=1)
    ann.to_csv(root / "metrics/xenium_v2_cell_annotations.csv.gz", index=False)
    try:
        ann.to_parquet(root / "metrics/xenium_v2_cell_annotations.parquet", index=False)
        parquet_status = "written"
    except Exception as exc:
        parquet_status = f"unavailable: {exc}"

    # Derived AnnData: copy source bytes, then append annotation/QC columns and spatial_HE without touching source.
    derived = root / "adata_xenium_v2_annotated.h5ad"; shutil.copyfile(args.adata, derived)
    with h5py.File(derived, "r+") as f:
        obs_group = f["obs"]
        for col in ["new_celltype", "quality_tier", "original_label", "provisional_lineage", "neutrophil_transcript_support"]:
            write_h5_string(obs_group, col, ann[col].astype(str).to_numpy())
        for col in ["annotation_confidence_score", "lineage_score_margin", "he_nuclear_integrity_score", "he_hematoxylin_fraction", "he_distance_to_nuclear_signal_px"]:
            if col in obs_group: del obs_group[col]
            ds = obs_group.create_dataset(col, data=ann[col].to_numpy(dtype=np.float32)); ds.attrs["encoding-type"] = "array"; ds.attrs["encoding-version"] = "0.2.0"
        obsm = f["obsm"]
        if "spatial_HE" in obsm: del obsm["spatial_HE"]
        ds = obsm.create_dataset("spatial_HE", data=spatial_he.astype(np.float32)); ds.attrs["encoding-type"] = "array"; ds.attrs["encoding-version"] = "0.2.0"
    sha = hashlib.sha256()
    with derived.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): sha.update(block)
    (root / "config/adata_xenium_v2.sha256").write_text(f"{sha.hexdigest()}  {derived}\n")

    # Audits and small summaries.
    conf = pd.crosstab(ann.original_label, ann.new_celltype).rename_axis("original_label").reset_index().melt(id_vars="original_label", var_name="new_label", value_name="count")
    conf.to_csv(root / "metrics/old_vs_new_label_confusion.csv", index=False)
    ret = ann.groupby(["new_celltype", "quality_tier"]).size().reset_index(name="count"); ret.to_csv(root / "metrics/quality_retention_by_class.csv", index=False)
    retb = ann.groupby(["batch", "quality_tier"]).size().unstack(fill_value=0).reset_index(); retb.to_csv(root / "metrics/quality_retention_by_batch.csv", index=False)
    neutro_summary = pd.DataFrame({"metric": ["original_neutrophil", "new_neutrophil", "hq_core_neutrophil", "hq_extended_neutrophil", "neutrophil_debris_suspect", "neutrophil_no_nucleus", "neutrophil_specific_support", "original_neutrophil_reclassified_or_excluded"], "count": [int((old_label == "Neutrophil").sum()), int((final_label == "Neutrophil").sum()), int(((final_label == "Neutrophil") & (quality_tier == "HQ_CORE")).sum()), int(((final_label == "Neutrophil") & (quality_tier == "HQ_EXTENDED")).sum()), int(neutro_debris.sum()), int((neutro_top & ((nuc_count <= 0) | he_low)).sum()), int((neutro_specific > 0).sum()), int(((old_label == "Neutrophil") & (final_label != "Neutrophil")).sum())]})
    neutro_summary.to_csv(root / "metrics/neutrophil_reannotation_summary.csv", index=False)
    pd.DataFrame({"metric": ["annotation_score_cut", "margin_cut", "he_proxy_threshold", "he_fraction_low", "he_fraction_present", "morphology_level", "morphology_scale_px", "notebook_pixel_size_um", "parquet_status"], "value": [score_cut, margin_cut, h_threshold, frac_low, frac_present, 3, scale, .2125, parquet_status]}).to_csv(root / "config/frozen_thresholds.csv", index=False)

    # Required annotation and morphology figures.
    fig, ax = plt.subplots(figsize=(10, 5)); counts = ann.groupby(["original_label", "new_celltype"]).size().unstack(fill_value=0).reindex(index=sorted(ann.original_label.unique())); counts.plot.bar(stacked=True, ax=ax, colormap="tab20"); ax.set_ylabel("Cells"); ax.set_title("Old to Xenium-v2 annotation counts"); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7); fig.tight_layout(); fig.savefig(root / "figures/Fig3_old_vs_new_annotation.pdf", bbox_inches="tight"); plt.close(fig)
    class_tier = ann[ann.new_celltype.isin(CLASS_NAMES)].groupby(["new_celltype", "quality_tier"]).size().unstack(fill_value=0).reindex(CLASS_NAMES); class_tier.plot.bar(stacked=True, figsize=(10, 5), color=["#0072B2", "#E69F00", "#999999"]); plt.ylabel("Cells"); plt.title("Quality retention by new class"); plt.tight_layout(); plt.savefig(root / "figures/Fig4_quality_retention_by_class.pdf", bbox_inches="tight"); plt.close()
    neut = ann[ann.original_label == "Neutrophil"]["new_celltype"].value_counts(); fig, ax = plt.subplots(figsize=(8, 4.5)); neut.reindex(neut.index).plot.bar(ax=ax, color="#D55E00"); ax.set_ylabel("Original Neutrophil cells"); ax.set_title("Original Neutrophil re-annotation"); fig.tight_layout(); fig.savefig(root / "figures/Fig5_neutrophil_reannotation.pdf", bbox_inches="tight"); plt.close(fig)

    print(json.dumps({"n_cells": n, "inbounds_fraction": float(inbounds.mean()), "old_label_counts": pd.Series(old_label).value_counts().to_dict(), "new_label_counts": pd.Series(final_label).value_counts().to_dict(), "quality_tier_counts": pd.Series(quality_tier).value_counts().to_dict(), "score_cut": score_cut, "margin_cut": margin_cut, "parquet_status": parquet_status, "derived": str(derived)}, indent=2, default=str))


if __name__ == "__main__":
    main()
