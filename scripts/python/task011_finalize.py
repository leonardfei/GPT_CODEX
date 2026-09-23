#!/usr/bin/env python3
"""Finalize Task011 summaries after the optional F1 gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def mean_col(df: pd.DataFrame, name: str) -> float:
    return float(df[name].mean()) if name in df and len(df) else float("nan")


def one(df: pd.DataFrame, col: str, default=float("nan")) -> float:
    return float(df[col].iloc[0]) if col in df and len(df) else default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    metrics = root / "metrics"
    config = root / "config"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    ft = pd.read_csv(metrics / "fine_tune_summary.csv")
    ff = pd.read_csv(metrics / "fine_tune_fold_metrics.csv")
    fp = pd.read_csv(metrics / "fine_tune_per_class.csv")
    fb = pd.read_csv(metrics / "fine_tune_true_binary.csv")
    fg = pd.read_csv(metrics / "fine_tune_geometry.csv")
    candidate = "F1_FINAL_BLOCK"
    fclass = fp[fp["class_name"].astype(str).str.lower().eq("neutrophil")]
    fbin = fb.set_index("comparison")

    row = {
        "model": "Task011 F1 final block",
        "candidate": candidate,
        "macro_f1": one(ft, "macro_f1"),
        "macro_auprc": one(ft, "macro_auprc"),
        "lowest_three_f1": one(ft, "lowest_three_f1"),
        "accuracy": one(ft, "accuracy"),
        "balanced_accuracy": one(ft, "balanced_accuracy"),
        "neutrophil_f1": mean_col(fclass, "f1"),
        "neutrophil_auprc": mean_col(fclass, "auprc"),
        "n_vs_myeloid_auroc": one(fbin.loc[["neutrophil_vs_myeloid"]].reset_index(), "auroc") if "neutrophil_vs_myeloid" in fbin.index else float("nan"),
        "n_vs_myeloid_auprc": one(fbin.loc[["neutrophil_vs_myeloid"]].reset_index(), "auprc") if "neutrophil_vs_myeloid" in fbin.index else float("nan"),
        "n_vs_tb_auroc": one(fbin.loc[["neutrophil_vs_tb"]].reset_index(), "auroc") if "neutrophil_vs_tb" in fbin.index else float("nan"),
        "n_vs_tb_auprc": one(fbin.loc[["neutrophil_vs_tb"]].reset_index(), "auprc") if "neutrophil_vs_tb" in fbin.index else float("nan"),
        "knn_batch_purity": one(fg, "knn_batch_purity"),
        "spatial_knn_purity": one(fg, "spatial_knn_purity"),
        "class_knn_purity": one(fg, "knn_class_purity"),
    }
    ladder_path = metrics / "model_ladder.csv"
    ladder = pd.read_csv(ladder_path)
    ladder = ladder[ladder["model"].astype(str) != row["model"]].copy()
    for col in ladder.columns:
        if col not in row:
            row[col] = float("nan")
    for col in row:
        if col not in ladder.columns:
            ladder[col] = float("nan")
    ladder = pd.concat([ladder, pd.DataFrame([row], columns=ladder.columns)], ignore_index=True)
    ladder.to_csv(ladder_path, index=False)

    frozen = ladder[ladder["model"].astype(str) == "Task011 BEST_FOV"]
    frozen_macro = one(frozen, "macro_f1")
    frozen_nf1 = one(frozen, "neutrophil_f1")
    frozen_nap = one(frozen, "n_vs_myeloid_auprc")
    f_macro = float(row["macro_f1"])
    f_nf1 = float(row["neutrophil_f1"])
    f_nap = float(row["n_vs_myeloid_auprc"])
    fold_base = ladder[ladder["model"].astype(str) == "Task011 BEST_FOV"]
    # The fold-level comparison is computed against the FOV12 fold file when available.
    base_fold_path = metrics / "fov_sweep_fold_metrics.csv"
    base_fold = pd.read_csv(base_fold_path)
    base_fold = base_fold[base_fold["candidate"].astype(str).eq("FOV12")]
    improved = int((ff["macro_f1"].to_numpy() > base_fold["macro_f1"].to_numpy()).sum()) if len(base_fold) == len(ff) else -1
    promotion = {
        "macro_f1_gain": f_macro - frozen_macro,
        "neutrophil_f1_gain": f_nf1 - frozen_nf1,
        "n_vs_myeloid_auprc_gain": f_nap - frozen_nap,
        "outer_folds_improved_macro_f1": improved,
        "purity_delta_batch": float(row["knn_batch_purity"] - one(frozen, "knn_batch_purity")),
        "purity_delta_spatial": float(row["spatial_knn_purity"] - one(frozen, "spatial_knn_purity")),
        "criteria": "macro-F1 >= +0.03 AND (Neutrophil F1 or AUPRC) >= +0.03, >=4/5 fold macro-F1 improvements, and no purity increase >0.05",
    }
    promotion["promoted_within_task011"] = bool(
        promotion["macro_f1_gain"] >= 0.03
        and (promotion["neutrophil_f1_gain"] >= 0.03 or promotion["n_vs_myeloid_auprc_gain"] >= 0.03)
        and improved >= 4
        and promotion["purity_delta_batch"] <= 0.05
        and promotion["purity_delta_spatial"] <= 0.05
    )

    gate_path = config / "fine_tuning_gate.json"
    gate = json.loads(gate_path.read_text())
    gate.update({"status": "EXECUTED_F1_FINAL_BLOCK", "fine_tuning_executed": True, "f1_summary": row, "promotion": promotion, "production_model_modified": False})
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")

    decision_path = metrics / "decision_summary.json"
    decision = json.loads(decision_path.read_text())
    decision.update({"status": "TASK011_FROZEN_AND_F1_COMPLETE", "fine_tuning_gate": gate, "f1_candidate": candidate, "fine_tuned_promotion": promotion, "final_preferred_model": candidate if promotion["promoted_within_task011"] else "Task011 BEST_FOV", "production_model_modified": False, "ground_truth_modified": False})
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")

    robust = pd.DataFrame([
        {"model": "Task011 BEST_FOV", "macro_f1": frozen_macro, "neutrophil_f1": frozen_nf1, "knn_batch_purity": one(frozen, "knn_batch_purity"), "spatial_knn_purity": one(frozen, "spatial_knn_purity")},
        {"model": "Task011 F1 final block", "macro_f1": f_macro, "neutrophil_f1": f_nf1, "knn_batch_purity": row["knn_batch_purity"], "spatial_knn_purity": row["spatial_knn_purity"]},
    ])
    robust.to_csv(metrics / "spatial_robustness_summary.csv", index=False)

    plot_df = pd.DataFrame({"model": ["Frozen FOV12", "F1 final block"], "macro_f1": [frozen_macro, f_macro], "neutrophil_f1": [frozen_nf1, f_nf1]})
    ax = plot_df.set_index("model").plot(kind="bar", figsize=(7, 4), ylim=(0, 1), rot=0)
    ax.set_ylabel("Score"); ax.set_title("Task011 frozen versus F1 final-block tuning"); ax.legend(loc="best"); fig = ax.get_figure(); fig.tight_layout(); fig.savefig(figures / "Fig_fine_tune_vs_frozen.pdf"); plt.close(fig)
    print(json.dumps({"f1": row, "promotion": promotion, "final_preferred_model": decision["final_preferred_model"]}, indent=2))


if __name__ == "__main__":
    main()
