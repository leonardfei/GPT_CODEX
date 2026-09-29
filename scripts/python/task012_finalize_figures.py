#!/usr/bin/env python3
"""Task012: corrected metric package and editable vector publication figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path("/data/lf_data/result/final_model")
T10 = Path("/data/lf_data/result/task010_representation_benchmark")
T11 = Path("/data/lf_data/result/task011_midnight_local_optimization")
MUSK = Path("/data/lf_data/result/task011_musk_benchmark")
CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
BLUE, GREEN, RED, TEAL, GREY = "#527AA3", "#648F73", "#B76665", "#4E9297", "#797D83"
COLORS = [GREY, TEAL, BLUE, GREEN, RED]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42, "pdf.use14corefonts": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def read(path):
    if not Path(path).is_file(): raise FileNotFoundError(path)
    return pd.read_csv(path)


def finish(fig, name):
    fig.savefig(ROOT / "figures" / name, format="pdf", bbox_inches="tight", metadata={"Creator": "Task012 reproducible matplotlib figure script"})
    plt.close(fig)


def bar_figure(name, title, labels, values, ylabel, colors=None, errors=None):
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    x = np.arange(len(values))
    ax.bar(x, values, yerr=errors, capsize=2 if errors is not None else 0,
           color=colors or COLORS[:len(values)], edgecolor="none", width=0.65)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", weight="semibold")
    ax.set_ylim(0, max(values) * 1.24)
    ax.tick_params(axis="x", length=0)
    for i, val in enumerate(values): ax.text(i, val + max(values) * .028, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    finish(fig, name)


def save_metrics():
    oof = read(ROOT / "metrics/oof_predictions.csv.gz")
    canonical = read(T10 / "metrics/canonical_cell_order.csv.gz")
    assert len(oof) == len(canonical) == 96044
    assert np.array_equal(oof.cell_id.astype(str).to_numpy(), canonical.cell_id.astype(str).to_numpy())
    assert np.array_equal(oof.true_class_id.to_numpy(int), canonical.class_id.to_numpy(int))
    assert oof.fold.between(0, 4).all() and not oof.cell_id.duplicated().any()
    p = oof[[f"p_class{i}" for i in range(7)]].to_numpy(float)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=1e-4)
    assert np.array_equal(p.argmax(axis=1), oof.predicted_class_id.to_numpy(int))
    corrected = T11 / "metrics"
    mapping = {
        "fine_tune_summary_corrected.csv": "final_model_summary.csv",
        "fine_tune_fold_metrics_corrected.csv": "final_fold_metrics.csv",
        "fine_tune_per_class_corrected.csv": "final_per_class.csv",
        "fine_tune_true_binary_corrected.csv": "final_true_binary.csv",
    }
    for src, dst in mapping.items(): read(corrected / src).to_csv(ROOT / "metrics" / dst, index=False)
    per = read(ROOT / "metrics/final_per_class.csv")
    per[per.class_id.eq(3)].to_csv(ROOT / "metrics/final_neutrophil_metrics.csv", index=False)
    cm = confusion_matrix(oof.true_class_id.to_numpy(int), oof.predicted_class_id.to_numpy(int), labels=list(range(7)))
    proportions = cm / cm.sum(axis=1, keepdims=True)
    rows = []
    for i, true_name in enumerate(CLASSES):
        for j, predicted_name in enumerate(CLASSES):
            rows.append({"true_class_id": i, "true_class_name": true_name,
                         "predicted_class_id": j, "predicted_class_name": predicted_name,
                         "count": int(cm[i, j]), "row_percent": float(proportions[i, j] * 100)})
    pd.DataFrame(rows).to_csv(ROOT / "metrics/final_confusion_matrix.csv", index=False)
    exp = read(ROOT / "metrics/exported_fold_metrics.csv").sort_values("fold")
    ref = read(ROOT / "metrics/final_fold_metrics.csv").sort_values("fold")
    assert exp.fold.tolist() == ref.fold.tolist() == list(range(5))
    diff = {col: float(np.max(np.abs(exp[col].to_numpy(float) - ref[col].to_numpy(float))))
            for col in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "weighted_f1", "lowest_three_f1")}
    qc = {"canonical_n": len(oof), "one_oof_prediction_per_cell": True,
          "probabilities_finite_and_normalized": True, "fold_metric_max_abs_difference_vs_corrected_task011": diff,
          "fold_metrics_exact_within_1e-6": all(v <= 1e-6 for v in diff.values()),
          "confusion_counts_total": int(cm.sum()),
          "full_data_model_evaluated_on_training_cells": False}
    (ROOT / "qc/task012_export_qc.json").write_text(json.dumps(qc, indent=2) + "\n")
    return cm, proportions, qc


def model_ladder():
    t10 = read(T10 / "metrics/linear_probe_summary_corrected.csv")
    frozen = read(T11 / "metrics/fov_sweep_summary.csv")
    ft = read(T11 / "metrics/fine_tune_summary_corrected.csv")
    musk = read(MUSK / "metrics/musk_fov_summary.csv")
    names = ["CellViT aligned", "Phikon SMALL", "MUSK FOV8", "Midnight frozen FOV12", "Midnight final-block FT"]
    rows = [t10[t10.representation.eq("CELLVIT_TOKEN_ALIGNED")].iloc[0],
            t10[t10.representation.eq("PHIKON_V2_SMALL")].iloc[0],
            musk[musk.candidate.eq("MUSK_FOV8")].iloc[0],
            frozen[frozen.candidate.eq("FOV12")].iloc[0],
            ft[ft.candidate.eq("F1_FINAL_BLOCK")].iloc[0]]
    for field, ylabel, filename in [
        ("macro_f1", "Macro-F1", "Fig1_model_ladder_macroF1.pdf"),
        ("macro_auprc", "Macro-AUPRC", "Fig2_model_ladder_macroAUPRC.pdf")]:
        bar_figure(filename, ylabel + " across model strategies", names,
                   [float(r[field]) for r in rows], ylabel,
                   errors=[float(r[field + "_sd"]) for r in rows])
    return names, rows


def neutrophil_ladder(names):
    sources = [
        (T10 / "metrics/linear_probe_per_class_corrected.csv", "CELLVIT_TOKEN_ALIGNED", "representation"),
        (T10 / "metrics/linear_probe_per_class_corrected.csv", "PHIKON_V2_SMALL", "representation"),
        (MUSK / "metrics/musk_fov_neutrophil_metrics.csv", "MUSK_FOV8", "candidate"),
        (T11 / "metrics/fov_sweep_per_class.csv", "FOV12", "candidate"),
        (T11 / "metrics/fine_tune_per_class_corrected.csv", "F1_FINAL_BLOCK", "candidate"),
    ]
    vals = []
    for path, candidate, key in sources:
        df = read(path)
        df = df[df[key].eq(candidate)]
        if "class_id" in df: df = df[df.class_id.eq(3)]
        assert len(df) == 5, (path, candidate, len(df))
        vals.append(df)
    for field, ylabel, filename in [
        ("f1", "Neutrophil F1", "Fig3_neutrophil_F1.pdf"),
        ("auprc", "Neutrophil one-vs-rest AUPRC", "Fig4_neutrophil_AUPRC.pdf")]:
        bar_figure(filename, ylabel + " across model strategies", names,
                   [float(df[field].mean()) for df in vals], ylabel,
                   errors=[float(df[field].std(ddof=1)) for df in vals])


def fov_sweep(path, name, title):
    df = read(path).copy()
    df["fov_um"] = df.candidate.str.extract(r"(\d+)").astype(int)
    df = df.sort_values("fov_um")
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.errorbar(df.fov_um, df.macro_f1, yerr=df.macro_f1_sd, marker="o", lw=1.4,
                ms=5, capsize=2, color=BLUE, label="Macro-F1")
    ax.errorbar(df.fov_um, df.macro_auprc, yerr=df.macro_auprc_sd, marker="s", lw=1.4,
                ms=4, capsize=2, color=GREEN, label="Macro-AUPRC")
    ax.set_xticks(df.fov_um); ax.set_xlabel("Field of view (μm)"); ax.set_ylabel("Five-fold mean ± SD")
    ax.set_title(title, loc="left", weight="semibold"); ax.legend(frameon=False, fontsize=8)
    finish(fig, name)


def binary_figures():
    df = read(T11 / "metrics/fine_tune_true_binary_summary_corrected.csv")
    df = df.set_index("comparison").loc[["neutrophil_vs_myeloid", "neutrophil_vs_tb"]]
    names = ["N vs Myeloid", "N vs T/B"]
    for field, filename in (("auroc", "Fig7_true_binary_AUROC.pdf"), ("auprc", "Fig8_true_binary_AUPRC.pdf")):
        bar_figure(filename, "Independent true binary probes", names, df[field].tolist(), field.upper(),
                   colors=[BLUE, TEAL], errors=df[field + "_sd"].tolist())


def per_class():
    frozen = read(T11 / "metrics/fov_sweep_per_class.csv")
    final = read(T11 / "metrics/fine_tune_per_class_corrected.csv")
    a = frozen[frozen.candidate.eq("FOV12")].groupby("class_id").f1.mean().reindex(range(7)).to_numpy()
    b = final[final.candidate.eq("F1_FINAL_BLOCK")].groupby("class_id").f1.mean().reindex(range(7)).to_numpy()
    assert np.isfinite(a).all() and np.isfinite(b).all()
    x = np.arange(7); w = .38
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    ax.bar(x-w/2, a, w, label="Frozen Midnight FOV12", color=BLUE)
    ax.bar(x+w/2, b, w, label="Final-block fine-tuned", color=GREEN)
    ax.set_xticks(x, CLASSES, rotation=22, ha="right"); ax.set_ylabel("Five-fold mean F1")
    ax.set_title("Per-class F1", loc="left", weight="semibold"); ax.legend(frameon=False, fontsize=8)
    finish(fig, "Fig9_per_class_F1.pdf")


def confusion(proportions):
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("muted_blue", ["#ffffff", "#d6e1e9", BLUE])
    im = ax.imshow(proportions * 100, cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(range(7), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Corrected out-of-fold confusion", loc="left", weight="semibold")
    for i in range(7):
        for j in range(7):
            val = proportions[i, j] * 100
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if val > 55 else "black")
    fig.colorbar(im, ax=ax, label="Row percentage", fraction=.045, pad=.03)
    finish(fig, "Fig10_confusion_matrix.pdf")


def fold_pairing():
    frozen = read(T11 / "metrics/fov_sweep_fold_metrics.csv")
    frozen = frozen[frozen.candidate.eq("FOV12")].sort_values("fold")
    final = read(T11 / "metrics/fine_tune_fold_metrics_corrected.csv").sort_values("fold")
    assert frozen.fold.tolist() == final.fold.tolist() == list(range(5))
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    for fold in range(5):
        y = [float(frozen.iloc[fold].macro_f1), float(final.iloc[fold].macro_f1)]
        ax.plot([0, 1], y, color=GREY, lw=.85, alpha=.8)
        ax.scatter([0, 1], y, color=[BLUE, GREEN], s=23, zorder=3)
        ax.text(1.05, y[1], str(fold), fontsize=7, va="center", color=GREY)
    ax.set_xlim(-.2, 1.28); ax.set_xticks([0, 1], ["Frozen FOV12", "Final-block FT"])
    ax.set_ylabel("Macro-F1"); ax.set_title("Paired outer-fold performance", loc="left", weight="semibold")
    finish(fig, "Fig11_fold_pairing.pdf")


def strategy_schematic():
    labels = ["H&E slide", "CellViT nucleus\ncentroid", "12 μm crop\n57 native px", "Midnight-12k\n224 px input", "Final block\nfine-tuned", "7-class\nlinear head"]
    fig, ax = plt.subplots(figsize=(10.4, 2.5))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 2.5); ax.axis("off")
    for i, label in enumerate(labels):
        x = .2 + i * 1.72
        rect = FancyBboxPatch((x, .82), (1.39 if i < 5 else 1.55), .92,
                              boxstyle="round,pad=0.08,rounding_size=0.08",
                              facecolor=["#eef1f3", "#e8f0f2", "#edf3ee", "#e9eff4", "#edf3ee", "#f4eded"][i],
                              edgecolor=GREY, linewidth=.7)
        ax.add_patch(rect)
        ax.text(x + (.695 if i < 5 else .775), 1.28, label, ha="center", va="center", fontsize=8)
        if i < 5:
            ax.add_patch(FancyArrowPatch((x + 1.49, 1.28), (x + 1.68, 1.28),
                                         arrowstyle="-|>", mutation_scale=10, color=GREY, linewidth=.8))
    ax.set_title("Final H&E target-cell classification strategy", loc="left", weight="semibold", y=1.05)
    finish(fig, "Fig12_final_strategy_schematic.pdf")


def recipe_table():
    rows = [
        ("Backbone", "Midnight-12k"), ("Target crop", "12 μm / 57 native px at 0.2125 μm/px"),
        ("Input and feature", "224 × 224; CLS + mean patch tokens; 3,072 dim"),
        ("Trainable weights", "Final transformer block + 3,072 → 7 head"),
        ("Training", "3 epochs; AdamW; weighted cross entropy"),
        ("Learning rates", "Backbone 1e-5; head 1e-3; weight decay 1e-4"),
        ("Augmentation", "90° rotations, horizontal flip, mild brightness/contrast"),
        ("Seed", "20260923"), ("Performance source", "Corrected five-fold grouped cross-validation"),
        ("Deployment", "Independent slide/patient validation still required"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.8)); ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=["Parameter", "Selected recipe"],
                   colWidths=[.23, .75], cellLoc="left", colLoc="left", loc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.55)
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#c7cbd0"); cell.set_linewidth(.45)
        cell.set_facecolor("#e9eff4" if row == 0 else ("white" if row % 2 else "#f7f8f8"))
        if row == 0: cell.set_text_props(weight="bold")
    ax.set_title("Final model recipe", loc="left", weight="semibold", pad=18)
    finish(fig, "Fig13_final_recipe_table.pdf")


def main():
    (ROOT / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "qc").mkdir(parents=True, exist_ok=True)
    cm, prop, qc = save_metrics()
    names, _ = model_ladder()
    neutrophil_ladder(names)
    fov_sweep(T11 / "metrics/fov_sweep_summary.csv", "Fig5_midnight_FOV_sweep.pdf", "Midnight frozen field-of-view sweep")
    fov_sweep(MUSK / "metrics/musk_fov_summary.csv", "Fig6_musk_FOV_sweep.pdf", "MUSK frozen field-of-view sweep")
    binary_figures(); per_class(); confusion(prop); fold_pairing(); strategy_schematic(); recipe_table()
    paths = sorted((ROOT / "figures").glob("Fig*.pdf"))
    assert len(paths) == 13, [p.name for p in paths]
    assert all(p.stat().st_size > 1000 for p in paths)
    print(json.dumps({"figures": [p.name for p in paths], "qc": qc}, indent=2))


if __name__ == "__main__": main()
