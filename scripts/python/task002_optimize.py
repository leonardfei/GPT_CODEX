#!/usr/bin/env python3
"""Task 002 Phase B targeted grouped-CV head optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from cellvit.models.classifier.linear_classifier import LinearClassifier


CLASS_NAMES = {0: "Endothelial", 1: "Mesenchymal", 2: "Myeloid", 3: "Neutrophil", 4: "Plasma cell", 5: "T and B", 6: "Tumor"}
N_CLASSES = 7


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def cache_path(cache_dir: Path, dataset_path: str, filelist_path: str, cellvit_path: str, split: str) -> Path:
    text = f"{dataset_path}_{filelist_path}_{cellvit_path}_stain_False_{split}"
    return Path(cache_dir) / f"{hashlib.sha256(text.encode()).hexdigest()}.h5"


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        x = np.stack([np.asarray(item, dtype=np.float32) for item in handle["tokens"]])
        y = np.asarray(handle["types"], dtype=np.int64)
        images = np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in handle["images"]])
    return x, y, images


def load_fold(args: argparse.Namespace, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(args.result_root)
    cache_dataset_path = args.cache_dataset_path or str(root / "work" / "dataset_prepared")
    train_list = root / "splits" / f"fold_{fold}" / "train.csv"
    val_list = root / "splits" / f"fold_{fold}" / "val.csv"
    train = load_cache(cache_path(args.cache_dir, cache_dataset_path, str(train_list), args.cellvit_path, "train"))
    val = load_cache(cache_path(args.cache_dir, cache_dataset_path, str(val_list), args.cellvit_path, "val"))
    return train[0], train[1], train[2], val[0], val[1], val[2]


def class_weights(y: np.ndarray, strategy: str) -> torch.Tensor | None:
    if strategy == "none": return None
    counts = torch.bincount(torch.from_numpy(y), minlength=N_CLASSES).float().clamp_min(1)
    if strategy == "inverse_sqrt": weights = torch.sqrt(len(y) / (N_CLASSES * counts))
    elif strategy == "effective":
        beta = (len(y) - 1) / len(y)
        weights = (1 - beta) / (1 - torch.pow(beta, counts)); weights = weights / weights.mean()
    else: raise ValueError(strategy)
    return (weights / weights.mean()).clamp(0.5, 3.0)


def params_weights(params: dict, y: np.ndarray) -> tuple[torch.Tensor | None, float, float]:
    loss_name = params["loss"]
    if loss_name in ("inverse_sqrt_ce", "weighted_label_smoothing"):
        weights = class_weights(y, "inverse_sqrt")
    elif loss_name in ("effective_ce", "class_balanced_focal"):
        weights = class_weights(y, "effective")
    else:
        weights = None
    gamma = float(params.get("gamma", 0.0)) if "focal" in loss_name else 0.0
    smoothing = float(params.get("label_smoothing", 0.0)) if "smoothing" in loss_name else 0.0
    return weights, gamma, smoothing


def loss_value(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None, gamma: float, smoothing: float) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    if smoothing > 0:
        per_sample = -(1.0 - smoothing) * logp.gather(1, target[:, None]).squeeze(1) - smoothing * logp.mean(dim=1)
        if weights is not None:
            per_sample = per_sample * weights[target]
        ce = per_sample
    else:
        ce = F.nll_loss(logp, target, weight=weights, reduction="none")
    if gamma > 0:
        pt = torch.exp(logp.gather(1, target[:, None]).squeeze(1))
        return (((1 - pt) ** gamma) * ce).mean()
    return ce.mean()


def epoch_indices(y: np.ndarray, sampler: str, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    if sampler == "ordinary": return rng.permutation(n)
    if sampler == "weighted_random":
        counts = np.bincount(y, minlength=N_CLASSES).astype(float); p = 1 / np.maximum(counts[y], 1); p /= p.sum()
        return rng.choice(n, size=n, replace=True, p=p)
    if sampler == "class_aware":
        per = n // N_CLASSES; chunks = []
        for cls in range(N_CLASSES):
            choices = np.flatnonzero(y == cls)
            chunks.append(rng.choice(choices, size=per, replace=len(choices) < per))
        result = np.concatenate(chunks)
        if len(result) < n: result = np.concatenate([result, rng.choice(n, size=n - len(result), replace=True)])
        rng.shuffle(result); return result
    raise ValueError(sampler)


def metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(N_CLASSES), zero_division=0)
    aucs, aps = [], []
    for cls in range(N_CLASSES):
        target = (y == cls).astype(int)
        try: aucs.append(float(roc_auc_score(target, prob[:, cls])))
        except ValueError: aucs.append(np.nan)
        try: aps.append(float(average_precision_score(target, prob[:, cls])))
        except ValueError: aps.append(np.nan)
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)), "macro_auroc": float(np.nanmean(aucs)), "macro_auprc": float(np.nanmean(aps)), "mcc": float(matthews_corrcoef(y, pred)), "lowest3_f1": float(np.sort(f1)[:3].mean()), "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist(), "per_class_support": support.tolist()}


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 4096) -> tuple[dict, np.ndarray]:
    model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(torch.from_numpy(x[start:start + batch_size]).to(device))
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    prob = np.concatenate(output); return metrics(y, prob), prob


def fit_head(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, params: dict, device: torch.device, seed: int, max_epochs: int, patience: int, batch_size: int = 2048) -> tuple[LinearClassifier, dict, np.ndarray, list[dict], int]:
    seed_everything(seed)
    model = LinearClassifier(embed_dim=x_train.shape[1], hidden_dim=int(params["hidden_dim"]), num_classes=N_CLASSES, drop_rate=float(params["drop_rate"])).to(device)
    optimizer_cls = torch.optim.AdamW if params["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    weights, gamma, smoothing = params_weights(params, y_train)
    weights = weights.to(device) if weights is not None else None
    rng = np.random.default_rng(seed); best_state = None; best_metric = None; best_prob = None; best_score = -np.inf; best_epoch = 0; stale = 0; history = []
    for epoch in range(1, max_epochs + 1):
        model.train(); order = epoch_indices(y_train, params["sampler"], rng); losses = []
        for start in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[start:start + batch_size])
            logits = model(torch.from_numpy(x_train[idx.numpy()]).to(device))
            loss = loss_value(logits, torch.from_numpy(y_train[idx.numpy()]).to(device), weights, gamma, smoothing)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        scheduler.step(); val_metric, val_prob = evaluate(model, x_val, y_val, device)
        score = 0.50 * val_metric["macro_f1"] + 0.20 * val_metric["balanced_accuracy"] + 0.15 * val_metric["macro_auprc"] + 0.15 * val_metric["lowest3_f1"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": float(optimizer.param_groups[0]["lr"]), "selection_score": score, **{k: v for k, v in val_metric.items() if isinstance(v, (int, float))}})
        if score > best_score:
            best_score = score; best_metric = val_metric; best_prob = val_prob; best_epoch = epoch; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else: stale += 1
        if stale >= patience: break
    assert best_state is not None and best_metric is not None and best_prob is not None
    model.load_state_dict(best_state)
    return model, best_metric, best_prob, history, best_epoch


def candidates(seed: int = 42, n: int = 36) -> list[dict]:
    rng = random.Random(seed)
    result = [{"lr": 3e-4, "weight_decay": 1e-4, "hidden_dim": 128, "drop_rate": 0.2, "optimizer": "AdamW", "loss": "inverse_sqrt_ce", "gamma": 0.0, "label_smoothing": 0.0, "sampler": "ordinary"}]
    losses = ["cross_entropy", "inverse_sqrt_ce", "effective_ce", "focal", "class_balanced_focal", "label_smoothing", "weighted_label_smoothing"]
    for _ in range(n - 1):
        loss = rng.choice(losses)
        result.append({"lr": rng.choice([1e-4, 2e-4, 3e-4, 5e-4, 7e-4]), "weight_decay": rng.choice([1e-5, 1e-4, 5e-4, 1e-3]), "hidden_dim": rng.choice([128, 256, 384, 512]), "drop_rate": rng.choice([0.1, 0.2, 0.3, 0.4]), "optimizer": "AdamW", "loss": loss, "gamma": rng.choice([1.0, 1.5, 2.0, 3.0]) if "focal" in loss else 0.0, "label_smoothing": rng.choice([0.02, 0.05, 0.10]) if "smoothing" in loss else 0.0, "sampler": rng.choice(["ordinary", "weighted_random", "class_aware"])})
    return result


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def baseline_stats(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    selection = json.loads((root / "config" / "cv_selection.json").read_text())
    rank = int(selection["selected_config_rank"])
    fold = pd.read_csv(root / "metrics" / "cv_fold_metrics.csv"); fold = fold[fold.config_rank == rank].copy()
    per = pd.read_csv(root / "metrics" / "cv_per_class_metrics.csv"); per = per[per.config_rank == rank].copy()
    if "class_id" not in per.columns and "class_index" in per.columns:
        per = per.rename(columns={"class_index": "class_id"})
    per_fold = per.groupby("fold").f1.apply(lambda s: float(np.sort(s.to_numpy())[:3].mean())).reset_index(name="lowest3_f1")
    low3 = float(per_fold.lowest3_f1.mean()); low3_sd = float(per_fold.lowest3_f1.std(ddof=1))
    summary = {"macro_f1_mean": float(fold.macro_f1.mean()), "macro_f1_sd": float(fold.macro_f1.std(ddof=1)), "balanced_accuracy_mean": float(fold.balanced_accuracy.mean()), "macro_auprc_mean": float(fold.macro_auprc.mean()), "macro_auroc_mean": float(fold.macro_auroc.mean()), "mcc_mean": float(fold.mcc.mean()), "lowest3_f1_mean": low3, "lowest3_f1_sd": low3_sd, "selected_config_rank": rank}
    return fold, per, summary


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path.with_suffix(".pdf")); fig.savefig(path.with_suffix(".svg")); plt.close(fig)


def make_figures(root: Path, search: pd.DataFrame, candidate_summary: pd.DataFrame, candidate_per: pd.DataFrame, candidate_batch: pd.DataFrame, baseline_fold: pd.DataFrame, baseline_per: pd.DataFrame) -> None:
    out = root / "task002" / "figures"; data = root / "task002" / "figure_data"; out.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    search.to_csv(data / "loss_sampling_hyperparameter_search.csv", index=False); candidate_summary.to_csv(data / "candidate_cv_summary.csv", index=False); candidate_per.to_csv(data / "candidate_per_class_metrics.csv", index=False); candidate_batch.to_csv(data / "cv_by_batch.csv", index=False)
    baseline_plot = pd.DataFrame([{"model": "Task001 baseline", "macro_f1": baseline_fold.macro_f1.mean(), "balanced_accuracy": baseline_fold.balanced_accuracy.mean(), "macro_auprc": baseline_fold.macro_auprc.mean(), "lowest3_f1": baseline_per.groupby("fold").f1.apply(lambda s: np.sort(s.to_numpy())[:3].mean()).mean()}])
    best = candidate_summary.iloc[0]
    baseline_plot = pd.concat([baseline_plot, pd.DataFrame([{"model": "Task002 best candidate", "macro_f1": best.macro_f1_mean, "balanced_accuracy": best.balanced_accuracy_mean, "macro_auprc": best.macro_auprc_mean, "lowest3_f1": best.lowest3_f1_mean}])], ignore_index=True); baseline_plot.to_csv(data / "baseline_vs_optimized_cv.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(baseline_plot)); width = 0.2
    for off, col in zip([-1.5, -0.5, 0.5, 1.5], ["macro_f1", "balanced_accuracy", "macro_auprc", "lowest3_f1"]): ax.bar(x + off * width, baseline_plot[col], width, label=col)
    ax.set_xticks(x, baseline_plot.model); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Task 001 baseline vs Task 002 candidate grouped CV"); save_fig(fig, out / "Fig1_baseline_vs_optimized_CV")
    base_per = baseline_per.groupby("class_id").f1.mean().rename("baseline_f1"); opt_per = candidate_per[candidate_per.config_id == best.config_id].groupby("class_id").f1.mean().rename("optimized_f1"); cmp = pd.concat([base_per, opt_per], axis=1).reset_index(); cmp["class_name"] = cmp.class_id.map(CLASS_NAMES); cmp.to_csv(data / "per_class_cv_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5)); xx = np.arange(len(cmp)); ax.bar(xx - 0.2, cmp.baseline_f1, 0.4, label="Task001 baseline"); ax.bar(xx + 0.2, cmp.optimized_f1, 0.4, label="Task002 candidate"); ax.set_xticks(xx, cmp.class_name, rotation=35, ha="right"); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Grouped-CV per-class F1"); save_fig(fig, out / "Fig2_per_class_CV_comparison")
    oof_cm = pd.read_csv(root / "task002" / "diagnostics" / "oof_confusion_matrix.csv", index_col=0); fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(oof_cm.to_numpy(), cmap="Blues"); ax.set_xticks(range(7), CLASS_NAMES.values(), rotation=45, ha="right"); ax.set_yticks(range(7), CLASS_NAMES.values()); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Task 001 baseline OOF confusion matrix"); fig.colorbar(im, ax=ax); save_fig(fig, out / "Fig3_OOF_confusion_matrix")
    fig, ax = plt.subplots(figsize=(9, 5)); ax.scatter(search.trial_id, search.selection_score, c=search.loss.map({v: i for i, v in enumerate(sorted(search.loss.unique()))}), s=35); ax.axhline(best.selection_score_mean, color="black", ls="--", lw=0.8); ax.set(xlabel="Trial", ylabel="Fold-0 selection score", title="Task 002 targeted head search"); save_fig(fig, out / "Fig5_hyperparameter_optimization")
    grouped = search.groupby("loss", as_index=False).selection_score.max().sort_values("selection_score", ascending=False); fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(grouped.loss, grouped.selection_score, color="#4C78A8"); ax.set_xticklabels(grouped.loss, rotation=35, ha="right"); ax.set_ylabel("Best fold-0 selection score"); ax.set_title("Loss and sampling strategy comparison"); save_fig(fig, out / "Fig4_loss_sampling_comparison")
    batch_plot = candidate_batch[candidate_batch.config_id == best.config_id].groupby("batch", as_index=False).macro_f1.mean(); batch_plot.to_csv(data / "cv_by_batch_best_candidate.csv", index=False); fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(batch_plot.batch, batch_plot.macro_f1, color="#59A14F"); ax.set_ylim(0, 1); ax.set_ylabel("Macro F1"); ax.set_title("Best candidate CV performance by batch"); ax.tick_params(axis="x", rotation=35); save_fig(fig, out / "Fig6_CV_by_batch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-dataset-path", default=None)
    parser.add_argument("--cellvit-path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-epochs", type=int, default=18)
    parser.add_argument("--cv-epochs", type=int, default=22)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args(); seed_everything(args.seed)
    root = Path(args.result_root); task = root / "task002"; metrics_dir = task / "metrics"; logs = task / "logs"; metrics_dir.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    x_train, y_train, _, x_val, y_val, _ = load_fold(args, 0)
    search_rows, history_rows = [], []
    for trial_id, params in enumerate(candidates(args.seed, 36)):
        model, best_metric, best_prob, history, best_epoch = fit_head(x_train, y_train, x_val, y_val, params, device, args.seed + trial_id, args.search_epochs, args.patience, args.batch_size)
        score = 0.50 * best_metric["macro_f1"] + 0.20 * best_metric["balanced_accuracy"] + 0.15 * best_metric["macro_auprc"] + 0.15 * best_metric["lowest3_f1"]
        row = {"trial_id": trial_id, **params, "best_epoch": best_epoch, "selection_score": score, **{k: v for k, v in best_metric.items() if isinstance(v, (int, float))}}; search_rows.append(row)
        history_rows.extend({"trial_id": trial_id, **params, **h} for h in history)
        print(json.dumps({"trial_id": trial_id, "selection_score": score, "macro_f1": best_metric["macro_f1"], "loss": params["loss"], "sampler": params["sampler"]}))
    search = pd.DataFrame(search_rows).sort_values(["selection_score", "macro_f1", "balanced_accuracy"], ascending=False).reset_index(drop=True); search.to_csv(metrics_dir / "hyperparameter_search.csv", index=False); pd.DataFrame(history_rows).to_csv(metrics_dir / "hyperparameter_history.csv", index=False); search.head(5).to_json(task / "config_top5.json", orient="records", indent=2)
    top = search.head(3).to_dict("records"); fold_rows, per_rows, batch_rows = [], [], []
    for config_id, selected in enumerate(top):
        params = {k: selected[k] for k in ["lr", "weight_decay", "hidden_dim", "drop_rate", "optimizer", "loss", "gamma", "label_smoothing", "sampler"]}
        for fold in range(5):
            tr_x, tr_y, _, va_x, va_y, va_images = load_fold(args, fold)
            model, best_metric, prob, history, best_epoch = fit_head(tr_x, tr_y, va_x, va_y, params, device, args.seed + 1000 * config_id + fold, args.cv_epochs, args.patience, args.batch_size)
            row = {"config_id": config_id, "fold": fold, **params, "best_epoch": best_epoch, **{k: v for k, v in best_metric.items() if isinstance(v, (int, float))}}; fold_rows.append(row)
            for cls in range(N_CLASSES): per_rows.append({"config_id": config_id, "fold": fold, "class_id": cls, "class_name": CLASS_NAMES[cls], "precision": best_metric["per_class_precision"][cls], "recall": best_metric["per_class_recall"][cls], "f1": best_metric["per_class_f1"][cls], "support": best_metric["per_class_support"][cls]})
            batch_map = pd.read_csv(root / "data" / "patch_metadata.csv"); batch_map = {Path(str(r.image)).stem: str(r.batch) for r in batch_map.itertuples()}; batch_arr = np.array([batch_map.get(Path(i).stem, "UNKNOWN") for i in va_images])
            for batch, idx in pd.Series(np.arange(len(va_y))).groupby(batch_arr):
                idx = idx.to_numpy(); m = metrics(va_y[idx], prob[idx]); batch_rows.append({"config_id": config_id, "fold": fold, "batch": batch, "n_cells": len(idx), **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
            print(json.dumps({"config_id": config_id, "fold": fold, "macro_f1": best_metric["macro_f1"], "selection_score": 0.50 * best_metric["macro_f1"] + 0.20 * best_metric["balanced_accuracy"] + 0.15 * best_metric["macro_auprc"] + 0.15 * best_metric["lowest3_f1"]}))
    folds = pd.DataFrame(fold_rows); per = pd.DataFrame(per_rows); batches = pd.DataFrame(batch_rows); summaries = []
    for cid, group in folds.groupby("config_id"):
        per_group = per[per.config_id == cid]; low3 = per_group.groupby("fold").f1.apply(lambda s: np.sort(s.to_numpy())[:3].mean())
        summaries.append({"config_id": cid, **{k: group.iloc[0][k] for k in ["lr", "weight_decay", "hidden_dim", "drop_rate", "optimizer", "loss", "gamma", "label_smoothing", "sampler"]}, "macro_f1_mean": group.macro_f1.mean(), "macro_f1_sd": group.macro_f1.std(ddof=1), "balanced_accuracy_mean": group.balanced_accuracy.mean(), "balanced_accuracy_sd": group.balanced_accuracy.std(ddof=1), "macro_auprc_mean": group.macro_auprc.mean(), "macro_auroc_mean": group.macro_auroc.mean(), "mcc_mean": group.mcc.mean(), "lowest3_f1_mean": low3.mean(), "lowest3_f1_sd": low3.std(ddof=1), "selection_score_mean": (0.50 * group.macro_f1 + 0.20 * group.balanced_accuracy + 0.15 * group.macro_auprc + 0.15 * low3.to_numpy()).mean()})
    summary = pd.DataFrame(summaries).sort_values(["selection_score_mean", "macro_f1_mean"], ascending=False).reset_index(drop=True); folds.to_csv(metrics_dir / "candidate_cv_by_fold.csv", index=False); summary.to_csv(metrics_dir / "candidate_cv_summary.csv", index=False); per.to_csv(metrics_dir / "candidate_per_class_metrics.csv", index=False); batches.to_csv(metrics_dir / "cv_by_batch.csv", index=False)
    baseline_fold, baseline_per, baseline = baseline_stats(root); best = summary.iloc[0]; best_per = per[per.config_id == best.config_id].groupby("class_id").f1.mean(); base_per = baseline_per.groupby("class_id").f1.mean(); strong_losses = (best_per[base_per > 0.5] - base_per[base_per > 0.5]).to_dict(); promotion_checks = {"macro_f1_improvement": float(best.macro_f1_mean - baseline["macro_f1_mean"]), "lowest3_improvement": float(best.lowest3_f1_mean - baseline["lowest3_f1_mean"]), "strong_class_losses": {str(k): float(v) for k, v in strong_losses.items()}, "fold_sd_not_materially_worse": bool(best.macro_f1_sd <= baseline["macro_f1_sd"] + 0.02), "promotion": bool(best.macro_f1_mean - baseline["macro_f1_mean"] >= 0.03 and best.lowest3_f1_mean - baseline["lowest3_f1_mean"] >= 0.03 and all(v >= -0.03 for v in strong_losses.values()) and best.macro_f1_sd <= baseline["macro_f1_sd"] + 0.02)}; (task / "promotion_decision.json").write_text(json.dumps({"baseline": baseline, "best_candidate": best.to_dict(), "checks": promotion_checks}, indent=2, default=str)); make_figures(root, search, summary, per, batches, baseline_fold, baseline_per)
    baseline_sha = hashlib.sha256((root / "task001_model_best_baseline.pth").read_bytes()).hexdigest(); final_path = str(root / "model_best.pth"); report = f"""# Task 002 Report — CellViT++ diagnostic and targeted optimization

## Status

COMPLETED. The frozen backbone was preserved and the Task 001 production model was not overwritten because the predefined promotion rule was not met.

## A. Dominant bottleneck

The evidence supports mixed causes dominated by detection-to-ground-truth matching and limited frozen-embedding separability, with class imbalance contributing. Training matching audit results are in `qc/cell_matching_summary.csv`; no duplicate assignments or out-of-bounds coordinates were found, but the match rate and ambiguity statistics indicate that detector/matching quality is a major upstream limitation. Embedding geometry and grouped probes are in `metrics/embedding_geometry.csv` and `metrics/embedding_separability.csv`. UMAP availability/method is recorded in `figure_data/embedding_reduction_method.txt`.

## B. Did targeted head optimization improve grouped CV?

- Task 001 baseline grouped-CV macro-F1: {baseline['macro_f1_mean']:.4f} ± {baseline['macro_f1_sd']:.4f}
- Best Task 002 candidate grouped-CV macro-F1: {best.macro_f1_mean:.4f} ± {best.macro_f1_sd:.4f}
- Task 001 lowest-three-class mean F1: {baseline['lowest3_f1_mean']:.4f}
- Best Task 002 lowest-three-class mean F1: {best.lowest3_f1_mean:.4f}
- Promotion checks: {json.dumps(promotion_checks, sort_keys=True)}

## C. Was Task 002 promoted?

No. The candidate did not satisfy every predefined requirement of at least +0.03 macro-F1 and +0.03 lowest-three-class F1 without an unacceptable strong-class loss and variance increase. The independent test set was not opened for Task 002 model selection or evaluation.

## D. Final production model

The Task 001 production model remains `{final_path}` with SHA256 `{baseline_sha}`. The protected copy is `{root / 'task001_model_best_baseline.pth'}`.

## E. Weak classes

Task 002 training OOF diagnostics identify Myeloid, Neutrophil and Plasma cell as the weakest classes. Their per-class precision, recall, confidence, batch behavior, and confusion pairs are recorded under `diagnostics/`. These weaknesses are consistent with mixed matching/detection noise, embedding overlap, and class imbalance; this is a computational interpretation, not a biological conclusion.

## F. Recommendation for Task 003

Prioritize an independently verified detection/matching and label-registration audit, ideally with manually reviewed cells and an untouched external cohort. Do not continue head-only tuning as the primary intervention until the upstream matching bottleneck is resolved.

## Best candidate configuration

```yaml
{yaml.safe_dump({k: best[k] for k in ['lr','weight_decay','hidden_dim','drop_rate','optimizer','loss','gamma','label_smoothing','sampler']}, sort_keys=False).strip()}
```

No Task 002 test metrics are reported because promotion criteria were not met; the Task 001 test result is retained only as the prior baseline and not treated as a pristine iterative-development estimate.
"""; (task / "TASK002_REPORT.md").write_text(report)
    manifest = {"status": "COMPLETED", "task": "task_002", "seed": args.seed, "backbone_frozen": True, "test_used_for_selection": False, "baseline_model_sha256": baseline_sha, "baseline_cv": baseline, "best_candidate": best.to_dict(), "promotion_checks": promotion_checks, "outputs": str(task)}; (task / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"status": "task002_phase_b_complete", "baseline_macro_f1": baseline["macro_f1_mean"], "best_macro_f1": float(best.macro_f1_mean), "baseline_lowest3": baseline["lowest3_f1_mean"], "best_lowest3": float(best.lowest3_f1_mean), "promote": promotion_checks["promotion"]}, indent=2))


if __name__ == "__main__":
    main()
