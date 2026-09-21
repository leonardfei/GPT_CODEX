#!/usr/bin/env python3
"""Head-only optimization on CellViT++ cached cell tokens.

The CellViT++ native pipeline is used to extract/cache cell tokens. This
script trains only the repository's LinearClassifier on those immutable token
files, keeping the independent test set out of tuning and model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from cellvit.models.classifier.linear_classifier import LinearClassifier
from cellvit.utils.tools import flatten_dict


CLASS_NAMES = {
    0: "Endothelial",
    1: "Mesenchymal",
    2: "Myeloid",
    3: "Neutrophil",
    4: "Plasma cell",
    5: "T and B",
    6: "Tumor",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cache_path(cache_dir: Path, dataset_path: str, filelist_path: str, cellvit_path: str, split: str) -> Path:
    text = f"{dataset_path}_{filelist_path}_{cellvit_path}_stain_False_{split}"
    return cache_dir / f"{hashlib.sha256(text.encode()).hexdigest()}.h5"


def load_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        tokens = np.stack([np.asarray(item, dtype=np.float32) for item in handle["tokens"]])
        labels = np.asarray(handle["types"], dtype=np.int64)
    return torch.from_numpy(tokens), torch.from_numpy(labels)


def metrics(y_true: np.ndarray, probabilities: np.ndarray, num_classes: int = 7) -> dict[str, float | list[float]]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=np.arange(num_classes), zero_division=0
    )
    one_hot = np.eye(num_classes, dtype=np.float32)[y_true]
    try:
        macro_auroc = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auroc = float("nan")
    try:
        macro_auprc = float(average_precision_score(one_hot, probabilities, average="macro"))
    except ValueError:
        macro_auprc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "kappa": float(cohen_kappa_score(y_true, predictions)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
    }


def class_weights(labels: torch.Tensor, strategy: str, num_classes: int = 7) -> torch.Tensor | None:
    if strategy == "none":
        return None
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1)
    if strategy == "inverse_frequency":
        weights = labels.numel() / (num_classes * counts)
    elif strategy == "inverse_sqrt_frequency":
        weights = torch.sqrt(labels.numel() / (num_classes * counts))
    elif strategy == "effective_number":
        beta = (labels.numel() - 1) / labels.numel()
        weights = (1 - beta) / (1 - torch.pow(beta, counts))
        weights = weights / weights.mean()
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")
    weights = weights / weights.mean()
    return weights.clamp(0.5, 3.0)


def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, batch_size: int = 4096) -> dict:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start : start + batch_size].to(device, non_blocking=True))
            outputs.append(torch.softmax(logits, dim=1).cpu())
    probabilities = torch.cat(outputs).numpy()
    result = metrics(y.numpy(), probabilities)
    result["probabilities"] = probabilities
    result["predictions"] = probabilities.argmax(axis=1)
    return result


def train_head(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    params: dict,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int = 2048,
) -> tuple[LinearClassifier, dict, list[dict], int]:
    seed_everything(seed)
    model = LinearClassifier(
        embed_dim=x_train.shape[1],
        hidden_dim=int(params["hidden_dim"]),
        num_classes=7,
        drop_rate=float(params["drop_rate"]),
    ).to(device)
    optimizer_cls = torch.optim.AdamW if params["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    weight = class_weights(y_train, params["loss_strategy"])
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
    history: list[dict] = []
    best_state = None
    best_metrics = None
    best_epoch = 0
    best_score = -float("inf")
    stale = 0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train), generator=generator)
        losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            logits = model(x_train[indices].to(device, non_blocking=True))
            loss = loss_fn(logits, y_train[indices].to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_result = evaluate_model(model, x_val, y_val, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{k: v for k, v in val_result.items() if isinstance(v, (int, float))},
        }
        history.append(record)
        if val_result["macro_f1"] > best_score:
            best_score = val_result["macro_f1"]
            best_epoch = epoch
            best_metrics = val_result
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return model, best_metrics, history, best_epoch


def train_fixed(x_train: torch.Tensor, y_train: torch.Tensor, params: dict, device: torch.device, seed: int, epochs: int, batch_size: int = 2048) -> tuple[LinearClassifier, list[dict]]:
    seed_everything(seed)
    model = LinearClassifier(
        embed_dim=x_train.shape[1], hidden_dim=int(params["hidden_dim"]), num_classes=7, drop_rate=float(params["drop_rate"])
    ).to(device)
    optimizer_cls = torch.optim.AdamW if params["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    weight = class_weights(y_train, params["loss_strategy"])
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
    history = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train), generator=generator)
        losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            logits = model(x_train[indices].to(device, non_blocking=True))
            loss = loss_fn(logits, y_train[indices].to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": float(optimizer.param_groups[0]["lr"])})
    return model, history


def parameter_grid() -> list[dict]:
    candidates = [
        {"lr": 1e-4, "weight_decay": 1e-4, "hidden_dim": 128, "drop_rate": 0.1, "optimizer": "AdamW", "loss_strategy": "none"},
        {"lr": 3e-5, "weight_decay": 0.0, "hidden_dim": 64, "drop_rate": 0.0, "optimizer": "Adam", "loss_strategy": "none"},
        {"lr": 1e-4, "weight_decay": 0.0, "hidden_dim": 64, "drop_rate": 0.1, "optimizer": "Adam", "loss_strategy": "none"},
        {"lr": 3e-4, "weight_decay": 0.0, "hidden_dim": 128, "drop_rate": 0.1, "optimizer": "Adam", "loss_strategy": "none"},
        {"lr": 3e-4, "weight_decay": 1e-4, "hidden_dim": 256, "drop_rate": 0.1, "optimizer": "AdamW", "loss_strategy": "none"},
        {"lr": 1e-3, "weight_decay": 1e-3, "hidden_dim": 128, "drop_rate": 0.2, "optimizer": "AdamW", "loss_strategy": "none"},
        {"lr": 3e-5, "weight_decay": 1e-3, "hidden_dim": 256, "drop_rate": 0.3, "optimizer": "AdamW", "loss_strategy": "none"},
        {"lr": 1e-4, "weight_decay": 1e-3, "hidden_dim": 256, "drop_rate": 0.3, "optimizer": "AdamW", "loss_strategy": "inverse_sqrt_frequency"},
        {"lr": 3e-4, "weight_decay": 1e-4, "hidden_dim": 128, "drop_rate": 0.2, "optimizer": "AdamW", "loss_strategy": "inverse_sqrt_frequency"},
        {"lr": 1e-4, "weight_decay": 1e-4, "hidden_dim": 256, "drop_rate": 0.1, "optimizer": "AdamW", "loss_strategy": "inverse_frequency"},
        {"lr": 3e-4, "weight_decay": 1e-3, "hidden_dim": 64, "drop_rate": 0.0, "optimizer": "Adam", "loss_strategy": "inverse_sqrt_frequency"},
        {"lr": 1e-3, "weight_decay": 0.0, "hidden_dim": 64, "drop_rate": 0.1, "optimizer": "Adam", "loss_strategy": "inverse_sqrt_frequency"},
        {"lr": 1e-5, "weight_decay": 1e-4, "hidden_dim": 128, "drop_rate": 0.0, "optimizer": "AdamW", "loss_strategy": "none"},
        {"lr": 1e-3, "weight_decay": 1e-4, "hidden_dim": 256, "drop_rate": 0.1, "optimizer": "AdamW", "loss_strategy": "none"},
        {"lr": 3e-4, "weight_decay": 1e-5, "hidden_dim": 256, "drop_rate": 0.3, "optimizer": "AdamW", "loss_strategy": "effective_number"},
        {"lr": 1e-4, "weight_decay": 1e-5, "hidden_dim": 64, "drop_rate": 0.3, "optimizer": "AdamW", "loss_strategy": "effective_number"},
        {"lr": 3e-5, "weight_decay": 1e-4, "hidden_dim": 512, "drop_rate": 0.1, "optimizer": "AdamW", "loss_strategy": "none"},
        {"lr": 1e-4, "weight_decay": 1e-3, "hidden_dim": 512, "drop_rate": 0.2, "optimizer": "AdamW", "loss_strategy": "none"},
        {"lr": 3e-4, "weight_decay": 1e-4, "hidden_dim": 128, "drop_rate": 0.0, "optimizer": "Adam", "loss_strategy": "effective_number"},
        {"lr": 1e-3, "weight_decay": 1e-3, "hidden_dim": 64, "drop_rate": 0.3, "optimizer": "AdamW", "loss_strategy": "inverse_frequency"},
    ]
    return candidates


def model_config(params: dict, args: argparse.Namespace, logdir: str) -> dict:
    return {
        "gpu": args.gpu,
        "random_seed": args.seed,
        "cellvit_path": args.cellvit_path,
        "data": {
            "dataset": "DetectionDataset",
            "dataset_path": args.dataset_path,
            "num_classes": 7,
            "label_map": {int(k): v for k, v in CLASS_NAMES.items()},
            "input_shape": 256,
        },
        "model": {"hidden_dim": int(params["hidden_dim"])},
        "training": {
            "optimizer": params["optimizer"],
            "optimizer_hyperparameter": {"lr": float(params["lr"]), "weight_decay": float(params["weight_decay"])},
            "scheduler": {"scheduler_type": "cosine", "eta_min": 1e-6},
            "drop_rate": float(params["drop_rate"]),
            "batch_size": args.batch_size,
            "mixed_precision": True,
        },
        "transformations": {"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        "logging": {"log_dir": logdir, "wandb_dir": str(Path(args.result_root) / "logs" / "wandb"), "mode": "disabled", "project": "task001_cellvit_head", "level": "INFO"},
    }


def save_checkpoint(model: LinearClassifier, params: dict, args: argparse.Namespace, path: Path, epoch: int, logdir: str) -> None:
    config = model_config(params, args, logdir)
    state = {
        "arch": "LinearClassifier",
        "epoch": epoch,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "best_metric": None,
        "best_epoch": epoch,
        "config": flatten_dict(config),
        "wandb_id": "task001_custom_head",
        "logdir": logdir,
        "run_name": Path(logdir).name,
        "scaler_state_dict": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_fold(args: argparse.Namespace, fold: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_dir = Path(args.cache_dir)
    train_list = str(Path(args.result_root) / "splits" / f"fold_{fold}" / "train.csv")
    val_list = str(Path(args.result_root) / "splits" / f"fold_{fold}" / "val.csv")
    train_cache = cache_path(cache_dir, args.dataset_path, train_list, args.cellvit_path, "train")
    val_cache = cache_path(cache_dir, args.dataset_path, val_list, args.cellvit_path, "val")
    x_train, y_train = load_cache(train_cache)
    x_val, y_val = load_cache(val_cache)
    return x_train, y_train, x_val, y_val


def run_search(args: argparse.Namespace, device: torch.device) -> None:
    x_train, y_train, x_val, y_val = load_fold(args, 0)
    rows = []
    histories = []
    for trial_id, params in enumerate(parameter_grid()):
        model, result, history, best_epoch = train_head(x_train, y_train, x_val, y_val, params, device, args.seed + trial_id, args.max_epochs_search, args.patience, args.batch_size)
        row = {"trial_id": trial_id, **params, "best_epoch": best_epoch}
        row.update({key: value for key, value in result.items() if isinstance(value, (int, float))})
        rows.append(row)
        for record in history:
            histories.append({"trial_id": trial_id, **params, **record})
        print(json.dumps({"trial_id": trial_id, "macro_f1": row["macro_f1"], "balanced_accuracy": row["balanced_accuracy"], "best_epoch": best_epoch}))
    rows.sort(key=lambda row: (-row["macro_f1"], -row["balanced_accuracy"], -row["macro_auroc"]))
    top = rows[: args.top_k]
    result_root = Path(args.result_root)
    write_rows(result_root / "metrics" / "hyperparameter_trials.csv", rows)
    write_rows(result_root / "metrics" / "hyperparameter_history.csv", histories)
    (result_root / "config" / "top_configs.json").write_text(json.dumps(top, indent=2) + "\n")


def run_cv(args: argparse.Namespace, device: torch.device) -> None:
    result_root = Path(args.result_root)
    top = json.loads((result_root / "config" / "top_configs.json").read_text())
    fold_rows = []
    per_class_rows = []
    summary_rows = []
    for config_rank, selected in enumerate(top):
        params = {key: selected[key] for key in ["lr", "weight_decay", "hidden_dim", "drop_rate", "optimizer", "loss_strategy"]}
        all_results = []
        for fold in range(5):
            x_train, y_train, x_val, y_val = load_fold(args, fold)
            model, result, history, best_epoch = train_head(x_train, y_train, x_val, y_val, params, device, args.seed + 1000 * config_rank + fold, args.max_epochs_cv, args.patience, args.batch_size)
            row = {"config_rank": config_rank, "fold": fold, **params, "best_epoch": best_epoch}
            row.update({key: value for key, value in result.items() if isinstance(value, (int, float))})
            fold_rows.append(row)
            all_results.append(row)
            for class_id in range(7):
                per_class_rows.append({"config_rank": config_rank, "fold": fold, "class_index": class_id, "class_name": CLASS_NAMES[class_id], "precision": result["per_class_precision"][class_id], "recall": result["per_class_recall"][class_id], "f1": result["per_class_f1"][class_id], "support": result["per_class_support"][class_id]})
            checkpoint_path = result_root / "checkpoints" / f"cv_config_{config_rank}_fold_{fold}.pth"
            save_checkpoint(model, params, args, checkpoint_path, best_epoch, str(result_root / "checkpoints" / f"cv_config_{config_rank}_fold_{fold}"))
            print(json.dumps({"config_rank": config_rank, "fold": fold, "macro_f1": row["macro_f1"], "balanced_accuracy": row["balanced_accuracy"], "best_epoch": best_epoch}))
        aggregate = {"config_rank": config_rank, **params}
        for metric_name in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc", "mcc", "kappa"]:
            values = np.array([float(row[metric_name]) for row in all_results])
            aggregate[f"{metric_name}_mean"] = float(np.nanmean(values))
            aggregate[f"{metric_name}_sd"] = float(np.nanstd(values, ddof=1))
            aggregate[f"{metric_name}_median"] = float(np.nanmedian(values))
        summary_rows.append(aggregate)
    summary_rows.sort(key=lambda row: (-row["macro_f1_mean"], -row["balanced_accuracy_mean"], -row["macro_auroc_mean"], row["macro_f1_sd"]))
    best = summary_rows[0]
    best_params = {key: best[key] for key in ["lr", "weight_decay", "hidden_dim", "drop_rate", "optimizer", "loss_strategy"]}
    cv_best_epochs = [row["best_epoch"] for row in fold_rows if row["config_rank"] == best["config_rank"]]
    best["robust_epoch_guidance"] = int(round(float(np.median(cv_best_epochs))))
    (result_root / "config" / "best_hyperparameters.yaml").write_text(yaml.safe_dump({"selected": best_params, "selection": "highest mean CV macro-F1, then balanced accuracy, macro-AUROC, then lower macro-F1 SD", "cv_summary": best, "cv_best_epochs": cv_best_epochs}, sort_keys=False))
    (result_root / "config" / "cv_selection.json").write_text(json.dumps({"summary": summary_rows, "selected": best_params, "selected_config_rank": best["config_rank"], "cv_best_epochs": cv_best_epochs}, indent=2) + "\n")
    write_rows(result_root / "metrics" / "cv_fold_metrics.csv", fold_rows)
    write_rows(result_root / "metrics" / "cv_per_class_metrics.csv", per_class_rows)
    write_rows(result_root / "metrics" / "cv_summary.csv", summary_rows)


def run_final(args: argparse.Namespace, device: torch.device) -> None:
    result_root = Path(args.result_root)
    selection = yaml.safe_load((result_root / "config" / "best_hyperparameters.yaml").read_text())
    params = selection["selected"]
    epochs = max(1, int(selection["cv_summary"].get("robust_epoch_guidance", 1)))
    x_train, y_train, x_val, y_val = load_fold(args, 0)
    x_all = torch.cat([x_train, x_val], dim=0)
    y_all = torch.cat([y_train, y_val], dim=0)
    model, history = train_fixed(x_all, y_all, params, device, args.seed + 9000, epochs, args.batch_size)
    final_logdir = result_root / "logs" / "final_model"
    checkpoint_path = final_logdir / "checkpoints" / "model_best.pth"
    save_checkpoint(model, params, args, checkpoint_path, epochs, str(final_logdir))
    shutil.copy2(checkpoint_path, result_root / "model_best.pth")
    write_rows(result_root / "metrics" / "final_training_history.csv", history)
    final_config = model_config(params, args, str(final_logdir))
    final_config["final_epochs"] = epochs
    final_config["training_data"] = "fold_0 train + fold_0 val; together partition all 5002 training images"
    final_config["test_data_used"] = False
    (result_root / "config" / "final_training_config.yaml").write_text(yaml.safe_dump(final_config, sort_keys=False))
    print(json.dumps({"final_model": str(result_root / "model_best.pth"), "epochs": epochs, "params": params}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "cv", "final"], required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--cellvit-path", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs-search", type=int, default=15)
    parser.add_argument("--max-epochs-cv", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.mode == "search":
        run_search(args, device)
    elif args.mode == "cv":
        run_cv(args, device)
    else:
        run_final(args, device)


if __name__ == "__main__":
    main()
