#!/usr/bin/env python3
"""Audit and aggregate Task015 five-fold GHIST results without touching source data.

Run on the server with /data/lf_data/task010_env/bin/python.  Every result is
computed from the ten immutable fold-result JSON files and their OOF tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

ROOT = Path("/data/lf_data/result/task015_ghist_celltyping")
BASE = Path("/data/lf_data/result/task011_midnight_local_optimization/metrics")
CANON = Path("/data/lf_data/result/task010_representation_benchmark/metrics/canonical_cell_order.csv.gz")
FOLDS = Path("/data/lf_data/result/task010_representation_benchmark/metrics/fold_manifest.csv")
VARIANTS = ("GHIST_CT_OFFICIAL", "GHIST_CT_BALANCED")
CLASSES = ("Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "weighted_f1",
           "lowest_three_f1", "neutrophil_precision", "neutrophil_recall", "neutrophil_f1",
           "neutrophil_auroc", "neutrophil_auprc")
BIN_METRICS = ("auroc", "auprc", "f1", "sensitivity", "specificity", "precision")


def finite_table(df, columns, title):
    missing = set(columns) - set(df)
    if missing:
        raise RuntimeError(f"{title} missing columns: {sorted(missing)}")
    if not np.isfinite(df[list(columns)].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{title} contains nonfinite values")


def summary(df, group_cols, metric_cols):
    group = df.groupby(group_cols, sort=True)
    mean = group[list(metric_cols)].mean().reset_index()
    sd = group[list(metric_cols)].std(ddof=1).reset_index().rename(
        columns={c: f"{c}_sd" for c in metric_cols})
    return mean.merge(sd, on=group_cols, validate="one_to_one")


def plot_bar(df, field, title, path):
    labels = ["Midnight", "GHIST official", "GHIST balanced"]
    values = [float(df.loc[df.variant.eq(x), field].iloc[0]) for x in
              ("MIDNIGHT", *VARIANTS)]
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.bar(labels, values, color=["#777777", "#2878B5", "#D38934"])
    ax.set(ylabel=field.replace("_", " "), title=title, ylim=(0, max(values) * 1.22))
    for i, x in enumerate(values):
        ax.text(i, x, f"{x:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--baseline", type=Path, default=BASE)
    args = ap.parse_args()
    root, base = args.root, args.baseline
    out = root / "metrics"
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)
    canonical = pd.read_csv(CANON)
    folds = pd.read_csv(FOLDS)
    assert len(canonical) == 96044 and canonical.cell_id.is_unique
    assert set(folds.fold) == set(range(5))
    baseline = pd.read_csv(base / "fine_tune_fold_metrics_corrected.csv")
    baseline = baseline[baseline.candidate.eq("F1_FINAL_BLOCK")].sort_values("fold")
    base_class = pd.read_csv(base / "fine_tune_per_class_corrected.csv")
    base_class = base_class[base_class.candidate.eq("F1_FINAL_BLOCK")]
    base_bin = pd.read_csv(base / "fine_tune_true_binary_corrected.csv")
    base_bin = base_bin[base_bin.candidate.eq("F1_FINAL_BLOCK")]
    if list(baseline.fold) != list(range(5)) or len(base_class) != 35 or len(base_bin) != 10:
        raise RuntimeError("Incomplete corrected Midnight fold baseline")
    finite_table(baseline, ("macro_f1", "macro_auprc"), "Midnight folds")
    fold_rows, class_rows, binary_rows, qc_rows, all_pred = [], [], [], [], {}
    for variant in VARIANTS:
        oof = []
        for fold in range(5):
            path = out / f"{variant}_fold{fold}_result.json"
            record = json.loads(path.read_text())
            if (record.get("status"), record.get("variant"), record.get("fold")) != ("FOLD_COMPLETE", variant, fold):
                raise RuntimeError(f"Incomplete or mismatched result: {path}")
            val = folds[(folds.fold == fold) & folds.role.eq("val")]
            if sorted(record["outer_validation_batches"]) != sorted(val.batch.unique()):
                raise RuntimeError(f"Wrong validation batches: {path}")
            pred = pd.read_csv(out / f"{variant}_fold{fold}_oof_predictions.csv.gz")
            required = ["cell_id", "class_id"] + [f"prob_{k}" for k in range(7)]
            if not set(required) <= set(pred) or len(pred) != len(val) or pred.cell_id.duplicated().any():
                raise RuntimeError(f"Prediction schema, count or uniqueness failure: {path}")
            if set(pred.cell_id) != set(val.cell_id):
                raise RuntimeError(f"Wrong OOF cell IDs: {path}")
            matched = pred[["cell_id", "class_id"]].merge(val[["cell_id", "class_id"]], on="cell_id",
                                                               validate="one_to_one", suffixes=("_pred", "_fold"))
            if not matched.class_id_pred.eq(matched.class_id_fold).all():
                raise RuntimeError(f"Frozen labels differ: {path}")
            prob = pred[[f"prob_{k}" for k in range(7)]].to_numpy(dtype=float)
            if not np.isfinite(prob).all() or (prob < 0).any() or not np.allclose(prob.sum(axis=1), 1, atol=1e-5):
                raise RuntimeError(f"Invalid probabilities: {path}")
            if record["outer_eval_qc"]["n_cells"] != len(pred) or not record["outer_eval_qc"]["cell_ids_unique"]:
                raise RuntimeError(f"Fold QC failed: {path}")
            fm = record["outer_val_metrics"]
            if not set(METRICS) <= set(fm):
                raise RuntimeError(f"Missing fold metrics: {path}")
            y = pred.class_id.to_numpy(dtype=int)
            independent_f1 = f1_score(y, prob.argmax(axis=1), labels=list(range(7)),
                                      average="macro", zero_division=0)
            independent_ap = np.mean([average_precision_score(y == k, prob[:, k]) for k in range(7)])
            if abs(independent_f1 - fm["macro_f1"]) > 1e-8 or abs(independent_ap - fm["macro_auprc"]) > 1e-6:
                raise RuntimeError(f"Fold metrics do not reproduce from OOF probabilities: {path}")
            for binary in record["true_binary"]:
                negative = 2 if binary["comparison"] == "neutrophil_vs_myeloid" else 5
                expected_binary = int(np.isin(y, [3, negative]).sum())
                if binary["n_val"] != expected_binary:
                    raise RuntimeError(f"True-binary validation count mismatch: {path}")
            fold_rows.append({"variant": variant, "fold": fold, "selected_epochs": record["selected_epochs"],
                              "n_val": len(pred), **{key: fm[key] for key in METRICS}})
            class_rows.extend(record["per_class"])
            binary_rows.extend(record["true_binary"])
            qc_rows.append({"variant": variant, "fold": fold, "n_val": len(pred),
                            "unique_cell_ids": True, "finite_probabilities": True,
                            "probability_sum_max_abs_error": float(np.abs(prob.sum(axis=1) - 1).max())})
            oof.append(pred)
        joined = pd.concat(oof, ignore_index=True)
        if len(joined) != 96044 or joined.cell_id.duplicated().any() or set(joined.cell_id) != set(canonical.cell_id):
            raise RuntimeError(f"Five-fold canonical coverage failed: {variant}")
        all_pred[variant] = joined
    if not all_pred[VARIANTS[0]].set_index("cell_id").class_id.sort_index().equals(
            all_pred[VARIANTS[1]].set_index("cell_id").class_id.sort_index()):
        raise RuntimeError("Variant OOF labels differ")
    fd, pc, tb, qc = map(pd.DataFrame, (fold_rows, class_rows, binary_rows, qc_rows))
    if len(pc) != 70 or len(tb) != 20:
        raise RuntimeError("Per-class or binary fold metrics incomplete")
    finite_table(fd, METRICS, "GHIST fold metrics")
    finite_table(pc, ("precision", "recall", "f1", "auroc", "auprc"), "GHIST per-class")
    finite_table(tb, BIN_METRICS, "GHIST true binary")
    if not pc.merge(base_class[["fold", "class_id", "support"]], on=["fold", "class_id"],
                    suffixes=("_ghist", "_midnight"), validate="many_to_one").eval(
                        "support_ghist == support_midnight").all():
        raise RuntimeError("GHIST/Midnight fold class support differs")
    fs = summary(fd, ["variant"], METRICS)
    cs = summary(pc, ["variant", "class_id", "class_name"], ("precision", "recall", "f1", "auroc", "auprc"))
    ns = pc[pc.class_id.eq(3)].copy()
    bs = summary(tb, ["variant", "comparison"], BIN_METRICS)
    base_neut = base_class[base_class.class_id.eq(3)].set_index("fold")
    base_fold = baseline.set_index("fold")
    paired = []
    for row in fd.itertuples(index=False):
        b = base_fold.loc[row.fold]
        n = base_neut.loc[row.fold]
        paired.append({"variant": row.variant, "fold": row.fold,
                       "ghist_macro_f1": row.macro_f1, "midnight_macro_f1": b.macro_f1,
                       "delta_macro_f1": row.macro_f1 - b.macro_f1,
                       "ghist_macro_auprc": row.macro_auprc, "midnight_macro_auprc": b.macro_auprc,
                       "delta_macro_auprc": row.macro_auprc - b.macro_auprc,
                       "ghist_neutrophil_f1": row.neutrophil_f1, "midnight_neutrophil_f1": n.f1,
                       "delta_neutrophil_f1": row.neutrophil_f1 - n.f1,
                       "ghist_neutrophil_auprc": row.neutrophil_auprc, "midnight_neutrophil_auprc": n.auprc,
                       "delta_neutrophil_auprc": row.neutrophil_auprc - n.auprc})
    paired = pd.DataFrame(paired)
    for comparison, short in (("neutrophil_vs_myeloid", "n_vs_myeloid"),
                              ("neutrophil_vs_tb", "n_vs_tb")):
        gh = tb[tb.comparison.eq(comparison)].set_index(["variant", "fold"])
        mb = base_bin[base_bin.comparison.eq(comparison)].set_index("fold")
        if len(gh) != 10 or len(mb) != 5:
            raise RuntimeError("Binary baseline fold pairing failed")
        for metric in ("auroc", "auprc", "f1"):
            paired[f"delta_{short}_{metric}"] = [float(gh.loc[(r.variant, r.fold), metric]
                - mb.loc[r.fold, metric]) for r in paired.itertuples(index=False)]
    decisions = {}
    for variant in VARIANTS:
        q = paired[paired.variant.eq(variant)]
        gains = {field: float(q[f"delta_{field}"].mean()) for field in
                 ("macro_f1", "macro_auprc", "neutrophil_f1", "neutrophil_auprc")}
        positive = {field: int(q[f"delta_{field}"].gt(0).sum()) for field in gains}
        strong = (gains["macro_f1"] >= 0.03 and
                  max(gains["neutrophil_f1"], gains["neutrophil_auprc"]) >= 0.03 and
                  positive["macro_f1"] >= 4)
        moderate = ((gains["macro_f1"] >= 0.01 and positive["macro_f1"] >= 3) or
                    (gains["neutrophil_f1"] >= 0.02 and positive["neutrophil_f1"] >= 3) or
                    (gains["neutrophil_auprc"] >= 0.02 and positive["neutrophil_auprc"] >= 3))
        decisions[variant] = {"classification": "STRONG" if strong else ("MODERATE" if moderate else "NO_MEANINGFUL_GAIN"),
                              "mean_paired_delta": gains, "positive_fold_counts": positive,
                              "selected_epochs": fd[fd.variant.eq(variant)].selected_epochs.astype(int).tolist()}
    # Cell-level Macro-F1 is the primary endpoint; use it to identify the best
    # GHIST variant, then apply the explicit promotion gate independently.
    best = max(VARIANTS, key=lambda v: float(fs.loc[fs.variant.eq(v), "macro_f1"].iloc[0]))
    decision = {"status": "CV_AGGREGATED", "primary_endpoint": "mean outer-fold cell-level Macro-F1",
                "best_ghist_variant": best, "variants": decisions,
                "full_data_export_eligible": decisions[best]["classification"] in ("STRONG", "MODERATE"),
                "production_checkpoint_replacement_authorized": False,
                "optional_neighborhood_stage": "NOT_RUN_SECONDARY_OPTIONAL",
                "midnight_reference": str(base),
                "morphology_qc": "PENDING_SEPARATE_PIXEL_HEAD_EVALUATION"}
    fd.to_csv(out / "ghist_fold_metrics.csv", index=False)
    fs.to_csv(out / "ghist_summary.csv", index=False)
    pc.to_csv(out / "ghist_per_class.csv", index=False)
    ns.to_csv(out / "ghist_neutrophil.csv", index=False)
    tb.to_csv(out / "ghist_true_binary.csv", index=False)
    bs.to_csv(out / "ghist_true_binary_summary.csv", index=False)
    paired.to_csv(out / "ghist_vs_midnight_paired.csv", index=False)
    qc.to_csv(root / "qc" / "oof_coverage_qc.csv", index=False)
    (out / "decision_summary.json").write_text(json.dumps(decision, indent=2) + "\n")
    plot_df = pd.concat([fs, pd.DataFrame([{"variant": "MIDNIGHT",
                        "macro_f1": float(baseline.macro_f1.mean()),
                        "macro_auprc": float(baseline.macro_auprc.mean()),
                        "neutrophil_f1": float(base_neut.f1.mean()),
                        "neutrophil_auprc": float(base_neut.auprc.mean())}])], ignore_index=True)
    for field in ("macro_f1", "macro_auprc", "neutrophil_f1", "neutrophil_auprc"):
        plot_bar(plot_df, field, field.replace("_", " ").title(), figdir / f"{field}.pdf")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(7)
    for variant, label, color in [("F1_FINAL_BLOCK", "Midnight", "#777777"),
                                  (VARIANTS[0], "GHIST official", "#2878B5"),
                                  (VARIANTS[1], "GHIST balanced", "#D38934")]:
        z = base_class if variant == "F1_FINAL_BLOCK" else pc
        z = z[z.candidate.eq(variant)] if variant == "F1_FINAL_BLOCK" else z[z.variant.eq(variant)]
        y = z.groupby("class_id").f1.mean().reindex(range(7)).to_numpy()
        ax.plot(x, y, marker="o", label=label, color=color)
    ax.set_xticks(x, CLASSES, rotation=35, ha="right")
    ax.set(ylabel="Mean outer-fold F1", title="Per-class F1")
    ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "per_class_f1.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for variant, group in paired.groupby("variant"):
        ax.plot(group.fold, group.delta_macro_f1, marker="o", label=variant)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xticks=range(5), xlabel="Outer fold", ylabel="GHIST − Midnight Macro-F1", title="Paired fold deltas")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir / "paired_fold_deltas.pdf"); plt.close(fig)
    for metric in ("auroc", "auprc"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for variant, group in bs.groupby("variant"):
            ax.plot(group.comparison, group[metric], marker="o", label=variant)
        ax.plot(base_bin.groupby("comparison")[metric].mean().index,
                base_bin.groupby("comparison")[metric].mean().values, marker="o", label="Midnight")
        ax.set(ylabel=metric.upper(), title=f"Training-only binary probes: {metric.upper()}")
        ax.legend(); fig.tight_layout(); fig.savefig(figdir / f"true_binary_{metric}.pdf"); plt.close(fig)
    for variant in VARIANTS:
        pred = all_pred[variant]
        cm = confusion_matrix(pred.class_id.to_numpy(), pred[[f"prob_{k}" for k in range(7)]].to_numpy().argmax(axis=1), labels=range(7), normalize="true")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(7), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(7), CLASSES)
        ax.set(xlabel="Predicted", ylabel="True", title=f"Normalized confusion: {variant}")
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(figdir / f"confusion_{variant}.pdf"); plt.close(fig)
    print(json.dumps({"status": "AGGREGATED", "decision": decision, "oof_cells_per_variant": 96044}, indent=2))


if __name__ == "__main__":
    main()
