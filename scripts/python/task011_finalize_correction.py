#!/usr/bin/env python3
"""Finalize the corrected Task011 fine-tuning analysis.

This script:
1) uses one-vs-rest Neutrophil AUPRC for the prespecified promotion gate;
2) summarizes TRUE independently trained binary probes by five-fold mean;
3) keeps probability-ratio diagnostics explicitly separate;
4) writes corrected gate/decision/model-ladder artefacts without deleting old provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def one(df: pd.DataFrame, col: str, default=float("nan")) -> float:
    return float(df[col].iloc[0]) if col in df.columns and len(df) else default


def mean_col(df: pd.DataFrame, col: str, default=float("nan")) -> float:
    return float(df[col].mean()) if col in df.columns and len(df) else default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    metrics = root / "metrics"
    config = root / "config"

    ft = pd.read_csv(metrics / "fine_tune_summary_corrected.csv")
    ff = pd.read_csv(metrics / "fine_tune_fold_metrics_corrected.csv")
    fp = pd.read_csv(metrics / "fine_tune_per_class_corrected.csv")
    tb = pd.read_csv(metrics / "fine_tune_true_binary_corrected.csv")
    tbs = pd.read_csv(metrics / "fine_tune_true_binary_summary_corrected.csv")
    pdg = pd.read_csv(metrics / "fine_tune_pairwise_probability_diagnostics_corrected.csv")
    fg = pd.read_csv(metrics / "fine_tune_geometry_corrected.csv")
    ladder_old = pd.read_csv(metrics / "model_ladder.csv")

    candidate = "F1_FINAL_BLOCK"
    nclass = fp[fp["class_name"].astype(str).str.lower().eq("neutrophil")].copy()

    def binary_mean(comp: str, metric: str) -> float:
        x = tbs[tbs["comparison"].astype(str).eq(comp)]
        return mean_col(x, metric)

    def diag_mean(comp: str, metric: str) -> float:
        x = pdg[pdg["comparison"].astype(str).eq(comp)]
        return mean_col(x, metric)

    row = {
        "model": "Task011 F1 final block CORRECTED",
        "candidate": candidate,
        "macro_f1": one(ft, "macro_f1"),
        "macro_auprc": one(ft, "macro_auprc"),
        "lowest_three_f1": one(ft, "lowest_three_f1"),
        "accuracy": one(ft, "accuracy"),
        "balanced_accuracy": one(ft, "balanced_accuracy"),
        "neutrophil_f1": mean_col(nclass, "f1"),
        "neutrophil_auprc": mean_col(nclass, "auprc"),
        "true_n_vs_myeloid_auroc": binary_mean("neutrophil_vs_myeloid", "auroc"),
        "true_n_vs_myeloid_auprc": binary_mean("neutrophil_vs_myeloid", "auprc"),
        "true_n_vs_myeloid_f1": binary_mean("neutrophil_vs_myeloid", "f1"),
        "true_n_vs_tb_auroc": binary_mean("neutrophil_vs_tb", "auroc"),
        "true_n_vs_tb_auprc": binary_mean("neutrophil_vs_tb", "auprc"),
        "true_n_vs_tb_f1": binary_mean("neutrophil_vs_tb", "f1"),
        "pairwise_diag_n_vs_myeloid_auroc": diag_mean("neutrophil_vs_myeloid", "auroc"),
        "pairwise_diag_n_vs_myeloid_auprc": diag_mean("neutrophil_vs_myeloid", "auprc"),
        "pairwise_diag_n_vs_tb_auroc": diag_mean("neutrophil_vs_tb", "auroc"),
        "pairwise_diag_n_vs_tb_auprc": diag_mean("neutrophil_vs_tb", "auprc"),
        "knn_batch_purity": one(fg, "knn_batch_purity"),
        "spatial_knn_purity": one(fg, "spatial_knn_purity"),
        "class_knn_purity": one(fg, "knn_class_purity"),
    }

    frozen = ladder_old[ladder_old["model"].astype(str).eq("Task011 BEST_FOV")].copy()
    if len(frozen) != 1:
        raise RuntimeError("Task011 BEST_FOV row missing or duplicated")

    frozen_macro = one(frozen, "macro_f1")
    frozen_nf1 = one(frozen, "neutrophil_f1")
    frozen_n_auprc = one(frozen, "neutrophil_auprc")
    frozen_batch = one(frozen, "knn_batch_purity")
    frozen_spatial = one(frozen, "spatial_knn_purity")

    base_fold = pd.read_csv(metrics / "fov_sweep_fold_metrics.csv")
    base_fold = base_fold[base_fold["candidate"].astype(str).eq("FOV12")].sort_values("fold")
    ff2 = ff.sort_values("fold")
    if list(base_fold["fold"]) != list(ff2["fold"]):
        raise RuntimeError("Fold identities differ between FOV12 baseline and corrected fine-tuning")
    improved = int((ff2["macro_f1"].to_numpy() > base_fold["macro_f1"].to_numpy()).sum())

    promotion = {
        "macro_f1_gain": float(row["macro_f1"] - frozen_macro),
        "neutrophil_f1_gain": float(row["neutrophil_f1"] - frozen_nf1),
        "neutrophil_one_vs_rest_auprc_gain": float(row["neutrophil_auprc"] - frozen_n_auprc),
        "outer_folds_improved_macro_f1": improved,
        "purity_delta_batch": float(row["knn_batch_purity"] - frozen_batch),
        "purity_delta_spatial": float(row["spatial_knn_purity"] - frozen_spatial),
        "criteria": "macro-F1 >= +0.03 AND (Neutrophil F1 OR Neutrophil one-vs-rest AUPRC) >= +0.03, >=4/5 fold macro-F1 improvements, and no batch/spatial purity increase >0.05",
    }
    promotion["promoted_within_task011"] = bool(
        promotion["macro_f1_gain"] >= 0.03
        and (
            promotion["neutrophil_f1_gain"] >= 0.03
            or promotion["neutrophil_one_vs_rest_auprc_gain"] >= 0.03
        )
        and improved >= 4
        and promotion["purity_delta_batch"] <= 0.05
        and promotion["purity_delta_spatial"] <= 0.05
    )

    gate = {
        "status": "TASK011_FINE_TUNING_GATE_CORRECTED",
        "candidate": candidate,
        "f1_summary_corrected": row,
        "promotion_corrected": promotion,
        "previous_gate_superseded": True,
        "previous_error": "n_vs_myeloid_auprc was incorrectly substituted for Neutrophil one-vs-rest AUPRC",
        "true_binary_probes": True,
        "true_binary_summary_is_five_fold_mean": True,
        "pairwise_probability_diagnostics_separate": True,
        "production_model_modified": False,
        "ground_truth_modified": False,
    }
    (config / "fine_tuning_gate_corrected.json").write_text(json.dumps(gate, indent=2, allow_nan=True) + "\n")

    # Create a corrected ladder while preserving the original ladder as provenance.
    ladder = ladder_old[~ladder_old["model"].astype(str).isin(["Task011 F1 final block", "Task011 F1 final block CORRECTED"])].copy()
    # Preserve shared schema and add corrected true-binary fields as needed.
    for col in row:
        if col not in ladder.columns:
            ladder[col] = np.nan
    for col in ladder.columns:
        if col not in row:
            row[col] = np.nan
    ladder = pd.concat([ladder, pd.DataFrame([row], columns=ladder.columns)], ignore_index=True)
    ladder.to_csv(metrics / "model_ladder_corrected.csv", index=False)

    best_observed = "F1_FINAL_BLOCK" if row["macro_f1"] > frozen_macro else "FOV12"
    gate_preferred = "F1_FINAL_BLOCK" if promotion["promoted_within_task011"] else "FOV12"
    validation_candidates = ["FOV12", "F1_FINAL_BLOCK"] if not promotion["promoted_within_task011"] and best_observed == "F1_FINAL_BLOCK" else [gate_preferred]

    decision = {
        "status": "TASK011_CORRECTION_COMPLETE",
        "strict_promotion_gate_passed": promotion["promoted_within_task011"],
        "gate_preferred_model": gate_preferred,
        "best_observed_development_candidate": best_observed,
        "recommended_independent_validation_candidates": validation_candidates,
        "reason_for_dual_validation": (
            "F1_FINAL_BLOCK has the highest observed macro-F1 but narrowly misses the prespecified Neutrophil F1/AUPRC gain gate; validate both frozen FOV12 and F1_FINAL_BLOCK independently."
            if len(validation_candidates) == 2 else
            "Only the gate-preferred model needs primary validation."
        ),
        "true_binary_summary_corrected": {
            "n_vs_myeloid": {
                "auroc": row["true_n_vs_myeloid_auroc"],
                "auprc": row["true_n_vs_myeloid_auprc"],
                "f1": row["true_n_vs_myeloid_f1"],
            },
            "n_vs_tb": {
                "auroc": row["true_n_vs_tb_auroc"],
                "auprc": row["true_n_vs_tb_auprc"],
                "f1": row["true_n_vs_tb_f1"],
            },
        },
        "pairwise_probability_diagnostics_are_not_true_binary": True,
        "previous_task011_decision_superseded": True,
        "production_model_modified": False,
        "ground_truth_modified": False,
    }
    (metrics / "decision_summary_corrected.json").write_text(json.dumps(decision, indent=2, allow_nan=True) + "\n")

    audit = {
        "old_promotion_bug_fixed": True,
        "old_binary_labeling_bug_fixed": True,
        "old_fold0_binary_summary_bug_fixed": True,
        "old_outputs_preserved": True,
        "corrected_outputs": [
            "fine_tune_fold_metrics_corrected.csv",
            "fine_tune_summary_corrected.csv",
            "fine_tune_per_class_corrected.csv",
            "fine_tune_true_binary_corrected.csv",
            "fine_tune_true_binary_summary_corrected.csv",
            "fine_tune_pairwise_probability_diagnostics_corrected.csv",
            "fine_tune_geometry_corrected.csv",
            "model_ladder_corrected.csv",
            "decision_summary_corrected.json",
            "fine_tuning_gate_corrected.json",
        ],
    }
    (root / "qc/task011_correction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    print(json.dumps({"row": row, "promotion": promotion, "decision": decision}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
