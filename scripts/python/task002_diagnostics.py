#!/usr/bin/env python3
"""Task 002 Phase A diagnostics using the frozen CellViT++ backbone."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier
from torch.utils.data import DataLoader

from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.models.classifier.linear_classifier import LinearClassifier
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.training.evaluate.inference_cellvit_experiment_detection import (
    CellViTInfExpDetection,
)
from cellvit.training.utils.tools import pair_coordinates


CLASS_NAMES = {
    0: "Endothelial",
    1: "Mesenchymal",
    2: "Myeloid",
    3: "Neutrophil",
    4: "Plasma cell",
    5: "T and B",
    6: "Tumor",
}
N_CLASSES = 7


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cache_path(cache_dir: Path, dataset_path: str, filelist_path: str, cellvit_path: str, split: str) -> Path:
    cache_dir = Path(cache_dir)
    text = f"{dataset_path}_{filelist_path}_{cellvit_path}_stain_False_{split}"
    return cache_dir / f"{hashlib.sha256(text.encode()).hexdigest()}.h5"


def decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "tokens": np.stack([np.asarray(item, dtype=np.float32) for item in handle["tokens"]]),
            "types": np.asarray(handle["types"], dtype=np.int64),
            "coords": np.stack([np.asarray(item[0], dtype=np.float32) for item in handle["coords"]]),
            "images": np.asarray([decode(x) for x in handle["images"]]),
        }


def metric_dict(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(N_CLASSES), zero_division=0
    )
    aurocs, auprcs = [], []
    for cls in range(N_CLASSES):
        target = (y == cls).astype(int)
        try:
            aurocs.append(float(roc_auc_score(target, prob[:, cls])))
        except ValueError:
            aurocs.append(float("nan"))
        try:
            auprcs.append(float(average_precision_score(target, prob[:, cls])))
        except ValueError:
            auprcs.append(float("nan"))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_auroc": float(np.nanmean(aurocs)),
        "macro_auprc": float(np.nanmean(auprcs)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "lowest3_f1": float(np.sort(f1)[:3].mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
    }


def batch_map(result_root: Path) -> dict[str, str]:
    metadata = pd.read_csv(result_root / "data" / "patch_metadata.csv")
    return {Path(str(row.image)).stem: str(row.batch) for row in metadata.itertuples()}


class SplitDetectionExperiment(CellViTInfExpDetection):
    def __init__(self, split: str, *args, **kwargs):
        self.task_split = split
        super().__init__(*args, **kwargs)

    def _load_dataset(self, transforms, normalize_stains):
        dataset = DetectionDataset(
            dataset_path=self.dataset_path,
            split=self.task_split,
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()
        return dataset


def run_detection_audit(args: argparse.Namespace, result_root: Path, device_gpu: int, image_to_batch: dict[str, str]) -> pd.DataFrame:
    logdir = result_root / "logs" / "final_model"
    exp = SplitDetectionExperiment(
        "train",
        logdir=logdir,
        cellvit_path=args.cellvit_path,
        dataset_path=args.dataset_path,
        input_shape=[256, 256],
        gpu=device_gpu,
    )
    postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
    loader = DataLoader(
        exp.inference_dataset,
        batch_size=4,
        num_workers=8,
        shuffle=False,
        collate_fn=exp.inference_dataset.collate_batch,
    )
    cleaned = []
    image_rows = []
    pair_rows = []
    with torch.no_grad():
        for images, gt_batch, types_batch, image_names in loader:
            result = exp._get_cellvit_result(
                images=images,
                cell_gt_batch=gt_batch,
                types_batch=types_batch,
                image_names=image_names,
                postprocessor=postprocessor,
            )
            batch_cleaned, _, pred_dicts, _, _, _ = result
            cleaned.extend(batch_cleaned)
            for true_centroids, cell_types, image_name, pred_dict in zip(
                gt_batch, types_batch, image_names, pred_dicts.values()
            ):
                image_name = decode(image_name)
                true_centroids = np.asarray(true_centroids, dtype=float).reshape(-1, 2)
                cell_types = np.asarray(cell_types, dtype=int).reshape(-1)
                pred_centroids = np.asarray([v["centroid"] for v in pred_dict.values()], dtype=float).reshape(-1, 2)
                if pred_centroids.size == 0:
                    pred_centroids = np.empty((0, 2), dtype=float)
                if true_centroids.size == 0:
                    true_centroids = np.empty((0, 2), dtype=float)
                paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroids, pred_centroids, 15)
                distances = []
                if len(paired):
                    distances = np.linalg.norm(true_centroids[paired[:, 0]] - pred_centroids[paired[:, 1]], axis=1)
                dist_matrix = np.linalg.norm(true_centroids[:, None, :] - pred_centroids[None, :, :], axis=2) if len(true_centroids) and len(pred_centroids) else np.empty((len(true_centroids), len(pred_centroids)))
                amb_gt = int(np.sum((dist_matrix <= 15).sum(axis=1) > 1)) if dist_matrix.size else 0
                amb_det = int(np.sum((dist_matrix <= 15).sum(axis=0) > 1)) if dist_matrix.size else 0
                image_rows.append({
                    "image": image_name,
                    "batch": image_to_batch.get(Path(image_name).stem, "UNKNOWN"),
                    "gt_count": int(len(true_centroids)),
                    "detected_count": int(len(pred_centroids)),
                    "matched_count": int(len(paired)),
                    "unmatched_gt": int(len(unpaired_true)),
                    "unmatched_detections": int(len(unpaired_pred)),
                    "ambiguous_gt": amb_gt,
                    "ambiguous_detections": amb_det,
                    "duplicate_assignments": int(len(paired) - len(np.unique(paired[:, 0])) if len(paired) else 0),
                    "gt_out_of_bounds": int(np.sum((true_centroids < 0) | (true_centroids >= 256))) if len(true_centroids) else 0,
                    "det_out_of_bounds": int(np.sum((pred_centroids < 0) | (pred_centroids >= 256))) if len(pred_centroids) else 0,
                })
                for pair_idx, pair in enumerate(paired):
                    gt = true_centroids[pair[0]]
                    det = pred_centroids[pair[1]]
                    pair_rows.append({
                        "image": image_name,
                        "batch": image_to_batch.get(Path(image_name).stem, "UNKNOWN"),
                        "gt_x": float(gt[0]), "gt_y": float(gt[1]),
                        "det_x": float(det[0]), "det_y": float(det[1]),
                        "distance": float(distances[pair_idx]),
                        "true_id": int(cell_types[pair[0]]),
                        "true_name": CLASS_NAMES[int(cell_types[pair[0]])],
                    })
    pairs = pd.DataFrame(pair_rows)
    images = pd.DataFrame(image_rows)
    classifier = exp._get_classifier_result(cleaned)
    probs = classifier["probabilities"].numpy()
    pairs["pred_id"] = classifier["predictions"].numpy().astype(int)
    pairs["pred_name"] = pairs.pred_id.map(CLASS_NAMES)
    pairs["confidence"] = probs.max(axis=1)
    pairs["correct"] = pairs.true_id == pairs.pred_id
    for cls in range(N_CLASSES):
        pairs[f"prob_{cls}"] = probs[:, cls]
    qc = result_root / "task002" / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    images.to_csv(qc / "cell_matching_by_image.csv", index=False)
    pairs.to_csv(qc / "cell_matching_pairs.csv", index=False)
    summary = {
        "gt_cells": int(images.gt_count.sum()),
        "detected_cells": int(images.detected_count.sum()),
        "matched_cells": int(images.matched_count.sum()),
        "unmatched_gt": int(images.unmatched_gt.sum()),
        "unmatched_detections": int(images.unmatched_detections.sum()),
        "ambiguous_gt": int(images.ambiguous_gt.sum()),
        "ambiguous_detections": int(images.ambiguous_detections.sum()),
        "duplicate_assignments": int(images.duplicate_assignments.sum()),
        "out_of_bounds_gt": int(images.gt_out_of_bounds.sum()),
        "out_of_bounds_detections": int(images.det_out_of_bounds.sum()),
        "match_rate_gt": float(images.matched_count.sum() / max(images.gt_count.sum(), 1)),
        "median_match_distance": float(pairs.distance.median()),
        "p95_match_distance": float(pairs.distance.quantile(0.95)),
        "matching_threshold_px": 15,
        "source_split": "train",
    }
    pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()]).to_csv(qc / "cell_matching_summary.csv", index=False)
    class_rows = []
    for cls in range(N_CLASSES):
        gt_cls = sum(int((row == cls).sum()) for row in []) if False else int(sum((images.image == x).sum() for x in []))
        all_types = []
        # Class-level GT counts are recovered from the matched pairs plus native label files below.
        subset = pairs[pairs.true_id == cls]
        class_rows.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "matched_cells": int(len(subset)),
            "median_match_distance": float(subset.distance.median()) if len(subset) else np.nan,
            "p95_match_distance": float(subset.distance.quantile(0.95)) if len(subset) else np.nan,
            "correct_baseline_classification_rate": float(subset.correct.mean()) if len(subset) else np.nan,
        })
    # Add exact class GT counts from DetectionDataset labels without changing source files.
    class_gt = {cls: 0 for cls in range(N_CLASSES)}
    for label_file in sorted((Path(args.dataset_path) / "train" / "labels").glob("*.csv")):
        try:
            frame = pd.read_csv(label_file)
            col = "type" if "type" in frame.columns else "class_id" if "class_id" in frame.columns else frame.columns[-1]
            for value, count in frame[col].value_counts().items():
                if int(value) in class_gt:
                    class_gt[int(value)] += int(count)
        except Exception:
            pass
    class_df = pd.DataFrame(class_rows)
    class_df["gt_cells"] = class_df.class_id.map(class_gt)
    class_df["matched_rate_gt"] = class_df.matched_cells / class_df.gt_cells.replace(0, np.nan)
    class_df.to_csv(qc / "cell_matching_by_class.csv", index=False)
    batch_df = images.groupby("batch", dropna=False).agg(
        images=("image", "nunique"), gt_cells=("gt_count", "sum"), detected_cells=("detected_count", "sum"), matched_cells=("matched_count", "sum"), unmatched_gt=("unmatched_gt", "sum"), unmatched_detections=("unmatched_detections", "sum"), ambiguous_gt=("ambiguous_gt", "sum"), ambiguous_detections=("ambiguous_detections", "sum")
    ).reset_index()
    distance_by_batch = pairs.groupby("batch").distance.agg(["median", lambda x: x.quantile(0.95)]).reset_index().rename(columns={"median": "median_match_distance", "<lambda_0>": "p95_match_distance"})
    batch_df = batch_df.merge(distance_by_batch, on="batch", how="left")
    batch_df["matched_rate_gt"] = batch_df.matched_cells / batch_df.gt_cells.replace(0, np.nan)
    batch_df.to_csv(qc / "cell_matching_by_batch.csv", index=False)
    pd.DataFrame({"distance": pairs.distance}).to_csv(qc / "cell_matching_distance_distribution.csv", index=False)
    pairs.to_csv(result_root / "task002" / "diagnostics" / "baseline_train_matched_predictions.csv", index=False)
    return pairs


def load_baseline_model(checkpoint: Path, params: dict, device: torch.device, embed_dim: int) -> LinearClassifier:
    model = LinearClassifier(embed_dim=embed_dim, hidden_dim=int(params["hidden_dim"]), num_classes=N_CLASSES, drop_rate=float(params["drop_rate"])).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def predict_model(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(torch.from_numpy(x[start:start + batch_size]).to(device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


def run_oof(args: argparse.Namespace, result_root: Path, image_to_batch: dict[str, str], device: torch.device) -> pd.DataFrame:
    selection = json.loads((result_root / "config" / "cv_selection.json").read_text())
    rank = int(selection["selected_config_rank"])
    params = yaml_safe_load(result_root / "config" / "best_hyperparameters.yaml")["selected"]
    cache_dataset_path = args.cache_dataset_path or str(result_root / "work" / "dataset_prepared")
    rows = []
    for fold in range(5):
        filelist = result_root / "splits" / f"fold_{fold}" / "val.csv"
        path = cache_path(args.cache_dir, cache_dataset_path, str(filelist), args.cellvit_path, "val")
        data = load_cache(path)
        checkpoint = result_root / "checkpoints" / f"cv_config_{rank}_fold_{fold}.pth"
        model = load_baseline_model(checkpoint, params, device, data["tokens"].shape[1])
        prob = predict_model(model, data["tokens"], device)
        for i in range(len(prob)):
            image = data["images"][i]
            row = {"fold": fold, "image": image, "batch": image_to_batch.get(Path(image).stem, "UNKNOWN"), "x": float(data["coords"][i, 0]), "y": float(data["coords"][i, 1]), "true_id": int(data["types"][i]), "pred_id": int(prob[i].argmax()), "confidence": float(prob[i].max())}
            row.update({f"prob_{c}": float(prob[i, c]) for c in range(N_CLASSES)})
            rows.append(row)
    oof = pd.DataFrame(rows)
    diag = result_root / "task002" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    oof.to_csv(diag / "oof_predictions.csv", index=False)
    cm = confusion_matrix(oof.true_id, oof.pred_id, labels=list(range(N_CLASSES)))
    pd.DataFrame(cm, index=[CLASS_NAMES[i] for i in range(N_CLASSES)], columns=[CLASS_NAMES[i] for i in range(N_CLASSES)]).to_csv(diag / "oof_confusion_matrix.csv")
    precision, recall, f1, support = precision_recall_fscore_support(oof.true_id, oof.pred_id, labels=list(range(N_CLASSES)), zero_division=0)
    pd.DataFrame({"class_id": range(N_CLASSES), "class_name": [CLASS_NAMES[i] for i in range(N_CLASSES)], "precision": precision, "recall": recall, "f1": f1, "support": support}).to_csv(diag / "oof_per_class_metrics.csv", index=False)
    pairs = oof[oof.true_id != oof.pred_id].groupby(["true_id", "pred_id"]).size().reset_index(name="count").sort_values("count", ascending=False)
    pairs["true_name"] = pairs.true_id.map(CLASS_NAMES); pairs["pred_name"] = pairs.pred_id.map(CLASS_NAMES)
    pairs.to_csv(diag / "oof_confusion_pairs.csv", index=False)
    batch_rows = []
    for batch, subset in oof.groupby("batch"):
        prob = subset[[f"prob_{c}" for c in range(N_CLASSES)]].to_numpy()
        m = metric_dict(subset.true_id.to_numpy(), prob)
        batch_rows.append({"batch": batch, "n_cells": len(subset), **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
    pd.DataFrame(batch_rows).to_csv(diag / "oof_by_batch_metrics.csv", index=False)
    conf = oof.groupby("true_id").confidence.agg(["count", "mean", "median", "min", "max"]).reset_index(); conf["class_name"] = conf.true_id.map(CLASS_NAMES); conf.to_csv(diag / "oof_confidence_by_class.csv", index=False)
    return oof


def yaml_safe_load(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def embedding_diagnostics(args: argparse.Namespace, result_root: Path, image_to_batch: dict[str, str], oof: pd.DataFrame, seed: int = 42) -> None:
    train_list = result_root / "splits" / "fold_0" / "train.csv"
    val_list = result_root / "splits" / "fold_0" / "val.csv"
    cache_dataset_path = args.cache_dataset_path or str(result_root / "work" / "dataset_prepared")
    train = load_cache(cache_path(args.cache_dir, cache_dataset_path, str(train_list), args.cellvit_path, "train"))
    val = load_cache(cache_path(args.cache_dir, cache_dataset_path, str(val_list), args.cellvit_path, "val"))
    x = np.concatenate([train["tokens"], val["tokens"]])
    y = np.concatenate([train["types"], val["types"]])
    images = np.concatenate([train["images"], val["images"]])
    groups = np.array([image_to_batch.get(Path(name).stem, "UNKNOWN") for name in images])
    rng = np.random.default_rng(seed)
    n = len(y)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    model_rows = []
    for model_name in ("multinomial_logistic", "linear_probe", "knn"):
        fold_rows = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups)):
            if model_name == "multinomial_logistic":
                estimator = LogisticRegression(max_iter=80, solver="lbfgs", multi_class="multinomial", random_state=seed, n_jobs=1)
                estimator.fit(x[train_idx], y[train_idx])
                prob = estimator.predict_proba(x[val_idx])
            elif model_name == "linear_probe":
                estimator = SGDClassifier(loss="modified_huber", max_iter=100, early_stopping=True, random_state=seed, n_jobs=-1)
                estimator.fit(x[train_idx], y[train_idx])
                decision = estimator.decision_function(x[val_idx])
                if decision.ndim == 1:
                    decision = np.column_stack([-decision, decision])
                decision = decision - np.max(decision, axis=1, keepdims=True)
                prob = np.exp(decision)
                prob = prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
                expanded = np.zeros((len(val_idx), N_CLASSES), dtype=float)
                expanded[:, estimator.classes_.astype(int)] = prob
                prob = expanded
            else:
                sample_idx = train_idx if len(train_idx) <= 30000 else rng.choice(train_idx, size=30000, replace=False)
                pca = PCA(n_components=32, random_state=seed, svd_solver="randomized")
                train_p = pca.fit_transform(x[sample_idx]); val_p = pca.transform(x[val_idx])
                estimator = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
                estimator.fit(train_p, y[sample_idx])
                prob = estimator.predict_proba(val_p)
                expanded = np.zeros((len(val_idx), N_CLASSES), dtype=float)
                expanded[:, estimator.classes_.astype(int)] = prob
                prob = expanded
            metric = metric_dict(y[val_idx], prob)
            fold_rows.append({"model": model_name, "fold": fold, **{k: v for k, v in metric.items() if isinstance(v, (int, float))}})
        model_rows.extend(fold_rows)
    native = oof.copy()
    native_prob = native[[f"prob_{c}" for c in range(N_CLASSES)]].to_numpy()
    native_metric = metric_dict(native.true_id.to_numpy(), native_prob)
    model_rows.append({"model": "native_mlp_head", "fold": "OOF", **{k: v for k, v in native_metric.items() if isinstance(v, (int, float))}})
    embedding_root = result_root / "task002" / "embeddings"
    embedding_root.mkdir(parents=True, exist_ok=True)
    np.save(embedding_root / "matched_train_tokens.npy", x.astype(np.float32))
    np.save(embedding_root / "matched_train_labels.npy", y.astype(np.int64))
    pd.DataFrame(model_rows).to_csv(result_root / "task002" / "metrics" / "embedding_separability.csv", index=False)
    sample_n = min(6000, n)
    sample = rng.choice(n, size=sample_n, replace=False)
    pca50 = PCA(n_components=min(50, x.shape[1]), random_state=seed, svd_solver="randomized")
    reduced = pca50.fit_transform(x[sample])
    sil_class = float(silhouette_score(reduced, y[sample], sample_size=min(8000, len(sample)), random_state=seed))
    sil_batch = float(silhouette_score(reduced, groups[sample], sample_size=min(8000, len(sample)), random_state=seed))
    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(reduced)
    neigh = nn.kneighbors(return_distance=False)[:, 1]
    purity = float(np.mean(y[sample] == y[sample][neigh]))
    batch_purity = float(np.mean(groups[sample] == groups[sample][neigh]))
    pd.DataFrame([{"metric": "class_silhouette", "value": sil_class}, {"metric": "batch_silhouette", "value": sil_batch}, {"metric": "nearest_neighbor_class_purity", "value": purity}, {"metric": "nearest_neighbor_batch_purity", "value": batch_purity}, {"metric": "embedding_cells", "value": n}]).to_csv(result_root / "task002" / "metrics" / "embedding_geometry.csv", index=False)
    plot_df = pd.DataFrame({"class_id": y[sample], "class_name": [CLASS_NAMES[int(v)] for v in y[sample]], "batch": groups[sample], "pca_x": reduced[:, 0], "pca_y": reduced[:, 1]})
    plot_df.to_csv(result_root / "task002" / "figure_data" / "embedding_projection.csv", index=False)
    _plot_embedding(plot_df, "pca_x", "pca_y", "class_name", "FigB_embedding_PCA", result_root / "task002" / "figures", "Frozen embedding PCA by class")
    # Prefer UMAP when installed; otherwise produce an explicit deterministic t-SNE fallback and record it.
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.3)
        coords = reducer.fit_transform(reduced)
        method = "UMAP"
    except Exception as exc:
        coords = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca", learning_rate="auto").fit_transform(reduced)
        method = f"t-SNE fallback; UMAP unavailable: {type(exc).__name__}"
    plot_df["reduction_x"] = coords[:, 0]; plot_df["reduction_y"] = coords[:, 1]
    plot_df.to_csv(result_root / "task002" / "figure_data" / "embedding_umap_projection.csv", index=False)
    (result_root / "task002" / "figure_data" / "embedding_reduction_method.txt").write_text(method + "\n")
    _plot_embedding(plot_df, "reduction_x", "reduction_y", "class_name", "FigC_embedding_UMAP_class", result_root / "task002" / "figures", f"Frozen embedding {method} by class")
    _plot_embedding(plot_df, "reduction_x", "reduction_y", "batch", "FigD_embedding_UMAP_batch", result_root / "task002" / "figures", f"Frozen embedding {method} by batch")


def _plot_embedding(df: pd.DataFrame, xcol: str, ycol: str, color_col: str, stem: str, outdir: Path, title: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for value, sub in df.groupby(color_col):
        ax.scatter(sub[xcol], sub[ycol], s=4, alpha=0.45, label=str(value))
    ax.set_title(title); ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.legend(markerscale=3, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(outdir / f"{stem}.pdf"); fig.savefig(outdir / f"{stem}.svg"); plt.close(fig)


def registration_overlays(args: argparse.Namespace, result_root: Path, pairs: pd.DataFrame, seed: int = 42) -> None:
    outdir = result_root / "task002" / "qc" / "registration_overlays"
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in range(N_CLASSES):
        subset = pairs[(pairs.true_id == cls) & pairs.correct]
        if len(subset): chosen.append(subset.iloc[int(rng.integers(0, len(subset)))] )
        errors = pairs[(pairs.true_id == cls) & (~pairs.correct)].sort_values("confidence", ascending=False)
        if len(errors): chosen.append(errors.iloc[0])
    chosen_df = pd.DataFrame(chosen).drop_duplicates(subset=["image", "gt_x", "gt_y"]).head(20)
    with matplotlib.backends.backend_pdf.PdfPages(result_root / "task002" / "figures" / "FigA_registration_QC.pdf") as pdf:
        for idx, row in chosen_df.iterrows():
            image_path = Path(args.dataset_path) / "train" / "images" / str(row.image)
            if not image_path.exists():
                image_path = image_path.with_suffix(".png")
            if not image_path.exists():
                continue
            image = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.imshow(image)
            ax.scatter([row.gt_x], [row.gt_y], marker="x", s=70, c="yellow", linewidths=2, label="GT/Xenium")
            ax.scatter([row.det_x], [row.det_y], marker="o", s=55, facecolors="none", edgecolors="cyan", linewidths=2, label="CellViT detection")
            ax.set_xlim(max(0, row.det_x - 55), min(256, row.det_x + 55)); ax.set_ylim(min(256, row.det_y + 55), max(0, row.det_y - 55))
            ax.set_title(f"{row.true_name} → {row.pred_name}; d={row.distance:.1f}px; batch={row.batch}")
            ax.legend(fontsize=7, loc="upper right"); ax.axis("off"); fig.tight_layout(); pdf.savefig(fig); fig.savefig(outdir / f"overlay_{len(list(outdir.glob('overlay_*.png'))):03d}.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-dataset-path", default=None)
    parser.add_argument("--cellvit-path", required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    result_root = Path(args.result_root)
    image_to_batch = batch_map(result_root)
    pairs_path = result_root / "task002" / "diagnostics" / "baseline_train_matched_predictions.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
    else:
        pairs = run_detection_audit(args, result_root, args.gpu, image_to_batch)
    oof = run_oof(args, result_root, image_to_batch, torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"))
    embedding_diagnostics(args, result_root, image_to_batch, oof, args.seed)
    registration_overlays(args, result_root, pairs, args.seed)
    print(json.dumps({"status": "phase_a_complete", "matched_train_cells": int(len(pairs)), "oof_cells": int(len(oof))}, indent=2))


if __name__ == "__main__":
    main()
