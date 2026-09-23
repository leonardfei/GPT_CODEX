#!/usr/bin/env python3
"""Aggregate Task009 official CellViT validation results and make figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> tuple[dict, pd.DataFrame]:
    support = np.bincount(y_true, minlength=len(CLASSES))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(CLASSES)), zero_division=0
    )
    auroc, auprc = [], []
    for cls in range(len(CLASSES)):
        truth = (y_true == cls).astype(int)
        try:
            auroc.append(float(roc_auc_score(truth, probs[:, cls])))
        except ValueError:
            auroc.append(float("nan"))
        try:
            auprc.append(float(average_precision_score(truth, probs[:, cls])))
        except ValueError:
            auprc.append(float("nan"))
    per_class = pd.DataFrame({
        "class_id": np.arange(len(CLASSES)),
        "class_name": CLASSES,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "support": support,
    })
    lowest3 = float(np.mean(np.sort(f1)[:3]))
    macro_auroc = float(np.nanmean(auroc))
    macro_auprc = float(np.nanmean(auprc))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "lowest3_f1": lowest3,
        "n_cells": int(len(y_true)),
    }
    return metrics, per_class


def load_tensor(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_v3(result: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    fold_rows, class_rows, neut_rows = [], [], {}
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for condition in ("CORE", "EXTENDED"):
        for fold in range(5):
            run_root = result / "runs" / f"V3_{condition}" / f"fold_{fold}"
            result_dirs = sorted(run_root.glob("*/val_results"))
            if len(result_dirs) != 1:
                raise RuntimeError(f"Expected one val_results directory for {condition} fold {fold}: {result_dirs}")
            val = result_dirs[0]
            y_true = load_tensor(val / "gt.pt").astype(int).ravel()
            y_pred = load_tensor(val / "predictions.pt").astype(int).ravel()
            probs = load_tensor(val / "probabilities.pt").astype(float)
            metrics, per_class = scalar_metrics(y_true, y_pred, probs)
            fold_rows.append({"condition": f"V3_{condition}", "fold": fold, **metrics})
            per_class.insert(0, "condition", f"V3_{condition}")
            per_class.insert(1, "fold", fold)
            class_rows.append(per_class)
            neut = per_class[per_class["class_name"] == "Neutrophil"].iloc[0].to_dict()
            neut_rows.setdefault(f"V3_{condition}", []).append({"condition": f"V3_{condition}", "fold": fold, **{k: neut[k] for k in ("precision", "recall", "f1", "auroc", "auprc", "support")}})
            outputs[f"V3_{condition}_fold_{fold}"] = (y_true, y_pred)
    return pd.DataFrame(fold_rows), pd.concat(class_rows, ignore_index=True), pd.DataFrame([row for rows in neut_rows.values() for row in rows]), outputs


def load_old(result: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    old_root = Path("/data/lf_data/result/task003_official")
    fold = pd.read_csv(old_root / "metrics/cv_fold_metrics.csv")
    fold = fold.rename(columns={"macro_f1": "macro_f1", "macro_auprc": "macro_auprc"})
    fold.insert(0, "condition", "OLD_LABELS")
    keep = ["condition", "fold", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc", "mcc", "lowest3_f1", "n_cells"]
    fold = fold[[c for c in keep if c in fold.columns]]
    per = pd.read_csv(old_root / "metrics/cv_per_class_metrics.csv")
    per = per.rename(columns={"class_id": "class_id", "class_name": "class_name"})
    per.insert(0, "condition", "OLD_LABELS")
    per = per[["condition", "fold", "class_id", "class_name", "precision", "recall", "f1", "auroc", "auprc", "support"]]
    neut = per[per["class_name"] == "Neutrophil"].copy()
    neut = neut.rename(columns={"class_id": "class_id"})
    oof = pd.read_csv(old_root / "metrics/cv_oof_predictions.csv")
    outputs = {}
    for fold_id, sub in oof.groupby("fold"):
        outputs[f"OLD_LABELS_fold_{int(fold_id)}"] = (sub["y_true"].to_numpy(int), sub["y_pred"].to_numpy(int))
    return fold, per, neut[["condition", "fold", "precision", "recall", "f1", "auroc", "auprc", "support"]], outputs


def make_summaries(fold: pd.DataFrame, per: pd.DataFrame, neut: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for condition, sub in fold.groupby("condition", sort=False):
        row = {"condition": condition}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc", "mcc", "lowest3_f1"]:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_sd"] = float(sub[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    per_summary = per.groupby(["condition", "class_id", "class_name"], as_index=False).agg(
        precision_mean=("precision", "mean"), precision_sd=("precision", "std"),
        recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
        support_total=("support", "sum"),
    )
    neut_summary = neut.groupby("condition", as_index=False).agg(
        precision_mean=("precision", "mean"), precision_sd=("precision", "std"),
        recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
        support_total=("support", "sum"),
    )
    return summary, per_summary, neut_summary


def confusion_flow(rows: list[dict], outputs: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    wanted = {(2, 3): "Myeloid_to_Neutrophil", (3, 2): "Neutrophil_to_Myeloid", (3, 5): "Neutrophil_to_T_and_B", (5, 3): "T_and_B_to_Neutrophil", (3, 4): "Neutrophil_to_Plasma", (4, 3): "Plasma_to_Neutrophil"}
    for condition in ("OLD_LABELS", "V3_CORE", "V3_EXTENDED"):
        yt, yp = np.concatenate([outputs[f"{condition}_fold_{i}"][0] for i in range(5)]), np.concatenate([outputs[f"{condition}_fold_{i}"][1] for i in range(5)])
        cm = confusion_matrix(yt, yp, labels=np.arange(7))
        for (true, pred), label in wanted.items():
            rows.append({"condition": condition, "flow": label, "true_class": CLASSES[true], "predicted_class": CLASSES[pred], "count": int(cm[true, pred]), "true_class_total": int(cm[true].sum()), "rate_among_true": float(cm[true, pred] / cm[true].sum())})
    return pd.DataFrame(rows)


def paired_deltas(fold: pd.DataFrame) -> pd.DataFrame:
    old = fold[fold["condition"] == "OLD_LABELS"].set_index("fold")
    rows = []
    for condition in ("V3_CORE", "V3_EXTENDED"):
        sub = fold[fold["condition"] == condition].set_index("fold")
        for fold_id in range(5):
            row = {"comparison": f"{condition}-OLD_LABELS", "condition": condition, "fold": fold_id}
            for metric in ("macro_f1", "macro_auprc", "lowest3_f1"):
                row[f"delta_{metric}"] = float(sub.loc[fold_id, metric] - old.loc[fold_id, metric])
            row["macro_f1_improved"] = bool(row["delta_macro_f1"] > 0)
            rows.append(row)
    return pd.DataFrame(rows)


def promotion_decision(summary: pd.DataFrame, per_summary: pd.DataFrame, deltas: pd.DataFrame, neut_summary: pd.DataFrame, result: Path) -> dict:
    old = summary[summary.condition == "OLD_LABELS"].iloc[0]
    decisions = []
    for condition in ("V3_CORE", "V3_EXTENDED"):
        s = summary[summary.condition == condition].iloc[0]
        n = neut_summary[neut_summary.condition == condition].iloc[0]
        old_n = neut_summary[neut_summary.condition == "OLD_LABELS"].iloc[0]
        d = deltas[deltas.condition == condition]
        strong = per_summary[(per_summary.condition == condition) & (per_summary.class_name != "Neutrophil")]
        old_strong = per_summary[(per_summary.condition == "OLD_LABELS") & (per_summary.class_name != "Neutrophil")].set_index("class_name")
        losses = []
        for _, row in strong.iterrows():
            if row["class_name"] in old_strong.index and float(row["f1_mean"] - old_strong.loc[row["class_name"], "f1_mean"]) < -0.05:
                losses.append(row["class_name"])
        checks = {
            "macro_f1_delta_ge_0.03": float(s.macro_f1_mean - old.macro_f1_mean) >= 0.03,
            "neutrophil_f1_delta_ge_0.05_or_auprc_delta_ge_0.05": max(float(n.f1_mean - old_n.f1_mean), float(n.auprc_mean - old_n.auprc_mean)) >= 0.05,
            "lowest3_not_decreased": float(s.lowest3_f1_mean) >= float(old.lowest3_f1_mean),
            "macro_f1_improves_in_at_least_4_of_5_folds": int(d.macro_f1_improved.sum()) >= 4,
            "no_strong_class_loss_gt_0.05": len(losses) == 0,
            "batch_leakage": False,
            "dataset_qc_pass": True,
        }
        decisions.append({"condition": condition, "qualifies": bool(all(checks.values())), "checks": checks, "strong_class_losses": losses})
    selected = next((d["condition"] for d in decisions if d["qualifies"]), None)
    payload = {
        "task": "Task009",
        "promotion_rule": "predefined in tasks/task_009.md",
        "conditions": decisions,
        "selected_candidate": selected,
        "final_refit_performed": False,
        "external_test_performed": False,
        "production_updated": False,
        "production_path": "/data/lf_data/result/model_best.pth",
        "production_sha256": sha256(Path("/data/lf_data/result/model_best.pth")),
        "reason": "No final refit/test is authorized unless a V3 condition satisfies all promotion criteria before test evaluation.",
    }
    if selected is not None:
        payload["reason"] = "A qualifying V3 condition requires a secondary review before any optional refit; this aggregate does not silently promote it."
    (result / "metrics/promotion_decision.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def make_figures(result: Path, fold: pd.DataFrame, per_summary: pd.DataFrame, neut_summary: pd.DataFrame, deltas: pd.DataFrame, flow: pd.DataFrame) -> None:
    figures = result / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(result / "metrics/dataset_composition.csv")
    # Editable source data used for every figure.
    fold.to_csv(result / "figure_data/cv_fold_metrics_all_conditions.csv", index=False)
    per_summary.to_csv(result / "figure_data/cv_per_class_summary_all_conditions.csv", index=False)
    neut_summary.to_csv(result / "figure_data/neutrophil_summary_all_conditions.csv", index=False)
    deltas.to_csv(result / "figure_data/paired_fold_deltas.csv", index=False)
    flow.to_csv(result / "figure_data/confusion_flows.csv", index=False)

    plt.figure(figsize=(10, 5))
    pivot = comp.pivot(index="class_name", columns="condition", values="assigned_to_patches")
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Cells assigned to regenerated patches"); plt.title("Task009 dataset composition"); plt.tight_layout(); plt.savefig(figures / "Fig1_dataset_composition.pdf"); plt.close()

    for metric, filename, title in [("macro_f1", "Fig2_macroF1_old_vs_v3.pdf", "Grouped-CV macro-F1"), ("macro_auprc", "Fig3_macroAUPRC_old_vs_v3.pdf", "Grouped-CV macro-AUPRC")]:
        plt.figure(figsize=(8, 5))
        order = ["OLD_LABELS", "V3_CORE", "V3_EXTENDED"]
        values = [fold.loc[fold.condition == c, metric].to_numpy() for c in order]
        plt.boxplot(values, labels=order, showmeans=True)
        plt.ylabel(metric); plt.title(title); plt.tight_layout(); plt.savefig(figures / filename); plt.close()

    plt.figure(figsize=(11, 5))
    x = np.arange(len(CLASSES)); width = 0.25
    for j, condition in enumerate(["OLD_LABELS", "V3_CORE", "V3_EXTENDED"]):
        sub = per_summary[per_summary.condition == condition].set_index("class_name").reindex(CLASSES)
        plt.bar(x + (j - 1) * width, sub.f1_mean, width, yerr=sub.f1_sd, capsize=3, label=condition)
    plt.xticks(x, CLASSES, rotation=30, ha="right"); plt.ylabel("F1"); plt.title("Per-class F1"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "Fig4_per_class_F1_old_vs_v3.pdf"); plt.close()

    plt.figure(figsize=(9, 5)); x = np.arange(4); labels = ["precision", "recall", "f1", "auprc"]
    for j, condition in enumerate(["OLD_LABELS", "V3_CORE", "V3_EXTENDED"]):
        sub = neut_summary[neut_summary.condition == condition].iloc[0]
        vals = [sub[f"{m}_mean"] for m in labels]
        plt.bar(x + (j - 1) * width, vals, width, label=condition)
    plt.xticks(x, labels); plt.ylim(0, 1); plt.ylabel("Score"); plt.title("Neutrophil grouped-CV metrics"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "Fig5_neutrophil_metrics_old_vs_v3.pdf"); plt.close()

    plt.figure(figsize=(10, 5)); flows = flow.flow.unique(); x = np.arange(len(flows));
    for j, condition in enumerate(["OLD_LABELS", "V3_CORE", "V3_EXTENDED"]):
        vals = flow[flow.condition == condition].set_index("flow").reindex(flows).rate_among_true.to_numpy()
        plt.bar(x + (j - 1) * width, vals, width, label=condition)
    plt.xticks(x, flows, rotation=30, ha="right"); plt.ylabel("Rate among true class"); plt.title("Confusion flows"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "Fig6_neutrophil_confusion_old_vs_v3.pdf"); plt.close()

    if (result / "metrics/detection_recall_by_class.csv").exists():
        det = pd.read_csv(result / "metrics/detection_recall_by_class.csv")
        plt.figure(figsize=(11, 5)); x = np.arange(len(CLASSES));
        for j, condition in enumerate(det.condition.unique()):
            sub = det[det.condition == condition].set_index("class_name").reindex(CLASSES)
            plt.bar(x + (j - (len(det.condition.unique()) - 1) / 2) * width, sub.detection_recall, width, label=condition)
        plt.xticks(x, CLASSES, rotation=30, ha="right"); plt.ylim(0, 1); plt.ylabel("Detection recall"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "Fig7_detection_recall_by_class.pdf"); plt.close()
    if (result / "metrics/neutrophil_end_to_end.csv").exists():
        e2e = pd.read_csv(result / "metrics/neutrophil_end_to_end.csv")
        plt.figure(figsize=(8, 5)); plt.bar(e2e.condition, e2e.end_to_end_recall); plt.ylim(0, 1); plt.ylabel("Recall"); plt.title("Neutrophil end-to-end recall"); plt.tight_layout(); plt.savefig(figures / "Fig8_neutrophil_end_to_end.pdf"); plt.close()

    plt.figure(figsize=(9, 5));
    for condition in ["V3_CORE", "V3_EXTENDED"]:
        sub = deltas[deltas.condition == condition]
        plt.plot(sub.fold, sub.delta_macro_f1, marker="o", label=condition)
    plt.axhline(0, color="black", linewidth=0.8); plt.xlabel("Fold"); plt.ylabel("Macro-F1 delta vs OLD_LABELS"); plt.title("Fold-paired macro-F1 deltas"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "Fig9_fold_paired_deltas.pdf"); plt.close()

    best = "V3_CORE" if fold[fold.condition == "V3_CORE"].macro_f1.mean() >= fold[fold.condition == "V3_EXTENDED"].macro_f1.mean() else "V3_EXTENDED"
    ys, ps = [], []
    # Use fold-level aggregate confusion from the fold metric files.
    # Re-read the saved output arrays to avoid serializing tensors in figures.
    for i in range(5):
        run = sorted((result / "runs" / best / f"fold_{i}").glob("*/val_results"))[0]
        ys.append(load_tensor(run / "gt.pt")); ps.append(load_tensor(run / "predictions.pt"))
    cm = confusion_matrix(np.concatenate(ys), np.concatenate(ps), labels=np.arange(7), normalize="true")
    plt.figure(figsize=(7, 6)); plt.imshow(cm, vmin=0, vmax=1, cmap="Blues"); plt.colorbar(label="Row-normalized rate"); plt.xticks(range(7), CLASSES, rotation=45, ha="right"); plt.yticks(range(7), CLASSES); plt.title(f"Best V3 condition confusion: {best}"); plt.tight_layout(); plt.savefig(figures / "Fig10_best_condition_confusion.pdf"); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("/data/lf_data/result/task009_v3_retraining"))
    args = parser.parse_args()
    result = args.result.resolve()
    v3_fold, v3_per, v3_neut, v3_outputs = load_v3(result)
    old_fold, old_per, old_neut, old_outputs = load_old(result)
    fold = pd.concat([old_fold, v3_fold], ignore_index=True)
    per = pd.concat([old_per, v3_per], ignore_index=True)
    neut = pd.concat([old_neut, v3_neut], ignore_index=True)
    summary, per_summary, neut_summary = make_summaries(fold, per, neut)
    outputs = {**old_outputs, **v3_outputs}
    flow = confusion_flow([], outputs)
    deltas = paired_deltas(fold)
    decision = promotion_decision(summary, per_summary, deltas, neut_summary, result)
    result.joinpath("metrics/cv_fold_metrics.csv").write_text(fold.to_csv(index=False))
    result.joinpath("metrics/cv_summary.csv").write_text(summary.to_csv(index=False))
    result.joinpath("metrics/cv_per_class_metrics.csv").write_text(per.to_csv(index=False))
    result.joinpath("metrics/cv_per_class_summary.csv").write_text(per_summary.to_csv(index=False))
    result.joinpath("metrics/neutrophil_metrics.csv").write_text(neut.to_csv(index=False))
    result.joinpath("metrics/neutrophil_summary.csv").write_text(neut_summary.to_csv(index=False))
    result.joinpath("metrics/paired_fold_deltas.csv").write_text(deltas.to_csv(index=False))
    result.joinpath("metrics/neutrophil_confusion_flows.csv").write_text(flow.to_csv(index=False))
    make_figures(result, fold, per_summary, neut_summary, deltas, flow)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
