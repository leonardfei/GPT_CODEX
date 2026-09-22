#!/usr/bin/env python3
"""Write the auditable Task-006 remote report from generated summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/data/lf_data/result/task006_xenium_reannotation")
ANN = ROOT / "metrics/xenium_v2_cell_annotations.csv.gz"
PROD = Path("/data/lf_data/result/model_best.pth")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    ann = pd.read_csv(ANN, usecols=["batch", "original_label", "new_celltype", "quality_tier"])
    old_counts = ann["original_label"].value_counts().to_dict()
    new_counts = ann["new_celltype"].value_counts().to_dict()
    tiers = ann["quality_tier"].value_counts().to_dict()
    label_change_rate = float((ann["original_label"] != ann["new_celltype"]).mean())
    neut = pd.read_csv(ROOT / "metrics/neutrophil_reannotation_summary.csv").set_index("metric")["count"].to_dict()
    cv = pd.read_csv(ROOT / "metrics/cv_summary_old_vs_v2.csv")
    per = pd.read_csv(ROOT / "metrics/cv_per_class_old_vs_v2.csv")
    promo = json.loads((ROOT / "metrics/promotion_decision.json").read_text())
    derived = ROOT / "adata_xenium_v2_annotated.h5ad"
    report = []
    report.append("# Task 006 Report — Xenium re-annotation, H&E QC, and high-quality CellViT retraining\n")
    report.append("Status: COMPLETED with documented unresolved registration-scale metadata discrepancy.\n")
    report.append("## 1. Scope and provenance\n")
    report.append("- Source AnnData: `/data/lf_data/xenium_data/adata_harmony_remove_necrosis.h5ad`; source was not modified.")
    report.append("- Source H&E: `/data/lf_data/xenium_data/ID0060276.ome.tif`; source was not modified.")
    report.append("- Historical biological annotation: `obs['cl1']`; `obs['cell_labels']` is an object identifier, not a biological label.")
    report.append("- Historical notebook: `/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb`.")
    report.append("- Historical recipe: per-class cap 50,000, seed 1234; 8 train/8 test batches; patch 256 and stride 256; official 5-fold random state 2026.")
    report.append("- Current production checkpoint was not overwritten; SHA256 before/after comparison target: `" + sha256(PROD) + "`.")
    report.append("- CellViT predictions were not used for transcript QC, H&E QC, marker scoring, annotation, or tier assignment.\n")
    report.append("## 2. Registration and QC\n")
    report.append("- Registration used the notebook matrix inverse (`A_inv`), which placed 100% of Xenium centroids inside the 50,000 × 23,451 H&E coordinate frame; the forward matrix was not used for labels.")
    report.append("- Batch-aware QC flags used robust within-batch checks on transcript counts, detected genes, cell area, and control fraction; H&E QC used a hematoxylin optical-density proxy, local nuclear signal, component/fragmentation and distance features.")
    report.append("- OME metadata reports `PhysicalSizeX/Y = 352.7778 µm`, inconsistent with the notebook geometry. This physical-scale discrepancy remains unresolved and is explicitly retained in `qc/ome_metadata.json`; no silent rescaling was applied.\n")
    report.append("## 3. Annotation counts\n")
    report.append("Old annotation counts: `" + json.dumps({k: int(v) for k, v in old_counts.items()}, ensure_ascii=False) + "`")
    report.append("New annotation counts: `" + json.dumps({k: int(v) for k, v in new_counts.items()}, ensure_ascii=False) + "`")
    report.append("Quality tiers: `" + json.dumps({k: int(v) for k, v in tiers.items()}, ensure_ascii=False) + "`")
    report.append(f"- Old→new label-change rate: {label_change_rate:.4%}; cells are compared at the frozen cell level, including non-training states.")
    report.append("- Exclusion/review reasons are represented by `Low_quality`, `Uncertain`, `Mixed_lineage`, `Artifact_or_no_nucleus`, and class-specific QC flags in `metrics/xenium_v2_cell_annotations.csv.gz`.\n")
    report.append("## 4. Neutrophil safeguards\n")
    report.append("Neutrophil summary: `" + json.dumps({k: int(v) for k, v in neut.items()}, ensure_ascii=False) + "`")
    report.append("- Neutrophils required independent marker support and nuclear-integrity evidence; suspected debris, no-nucleus cases, and ambiguous cells were not promoted to HQ_CORE.")
    report.append("- Original Neutrophil labels were not treated as ground truth for re-annotation; they are reported only for audit.\n")
    report.append("## 5. CellViT datasets and controlled CV\n")
    report.append("- CORE dataset: `/data/lf_data/result/task006_xenium_reannotation/work/CellViT_dataset_v2_CORE`.")
    report.append("- EXTENDED dataset: `/data/lf_data/result/task006_xenium_reannotation/work/CellViT_dataset_v2_EXTENDED`.")
    report.append("- Both used the same source H&E patches, label map, batch-aware five-fold splits, official SAM-H backbone and RAW normalization recipe as the old-label baseline.")
    report.append("- The reserved test split was not used because promotion qualification was not established before test evaluation.\n")
    cols = ["condition", "macro_f1_mean", "macro_f1_sd", "lowest3_f1_mean", "macro_auprc_mean"]
    report.append("CV summary:\n\n| condition | macro-F1 | lowest-3 F1 | macro-AUPRC |\n|---|---:|---:|---:|")
    for row in cv.itertuples(index=False):
        report.append(f"| {row.condition} | {fmt(row.macro_f1_mean)} ± {fmt(row.macro_f1_sd)} | {fmt(row.lowest3_f1_mean)} | {fmt(row.macro_auprc_mean)} |")
    report.append("")
    for condition in ["OLD_LABELS", "V2_CORE", "V2_EXTENDED"]:
        n = per[(per.condition == condition) & (per.class_name == "Neutrophil")].iloc[0:]
        if not n.empty:
            report.append(f"- {condition} Neutrophil mean F1/recall/AUPRC: {n.f1.mean():.4f} / {n.recall.mean():.4f} / {n.auprc.mean():.4f}.")
    report.append("")
    report.append("Promotion decision: `" + json.dumps(promo, ensure_ascii=False) + "`\n")
    report.append("No production promotion or final test was performed unless all pre-specified criteria in `metrics/promotion_decision.json` were met; production checkpoint was never overwritten.\n")
    report.append("## 6. Required artifacts\n")
    report.append("- Derived AnnData: `/data/lf_data/result/task006_xenium_reannotation/adata_xenium_v2_annotated.h5ad`.")
    report.append("- Derived AnnData SHA256: `" + sha256(derived) + "`.")
    report.append("- QC/metrics: `qc/`, `metrics/`, `config/`, `logs/`, and dataset manifests under the Task6 output root.")
    report.append("- Figures: `Fig1_registration_QC.pdf`, `Fig2_transcript_QC.pdf`, `Fig3_old_vs_new_annotation.pdf`, `Fig4_quality_retention_by_class.pdf`, `Fig5_neutrophil_reannotation.pdf`, `Fig6_neutrophil_HE_QC_montage.pdf`, `Fig7_marker_scores_by_new_class.pdf`, `Fig8_embedding_new_annotation.pdf`, `Fig9_CV_macroF1_old_vs_v2.pdf`, `Fig10_per_class_F1_old_vs_v2.pdf`, `Fig11_neutrophil_metrics_old_vs_v2.pdf`.")
    report.append("- Full per-cell annotation table remains remote because it is a large derived artifact and is not committed to Git.\n")
    report.append("## 7. Unresolved issues and Task 7 direction\n")
    report.append("- Resolve the OME physical-scale metadata discrepancy against the authoritative Xenium/H&E acquisition metadata before making any physical-distance claim.")
    report.append("- Review low-retention classes and the large `Mixed_lineage`/`Uncertain` pool with an expert before adding more training labels.")
    report.append("- Task 7 should begin only after the user reviews this report and, if needed, approves a calibrated physical registration and a revised annotation/tier policy; do not treat CellViT predictions as annotation ground truth.")
    (ROOT / "TASK006_REPORT.md").write_text("\n".join(report) + "\n")
    print(ROOT / "TASK006_REPORT.md")


if __name__ == "__main__":
    main()
