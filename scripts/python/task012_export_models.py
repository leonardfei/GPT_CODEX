#!/usr/bin/env python3
"""Task012: reproduce corrected Task011 fold refits and export the final model.

Runs on the analysis server. The corrected Task011 module is imported unchanged so
crop loading, augmentation, fold construction, epoch selection and optimization
remain identical to the original corrected analysis.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/data/lf_data/result/final_model")
T11 = Path("/data/lf_data/result/task011_midnight_local_optimization")
sys.path.insert(0, str(T11 / "code"))
import task011_finetune_correction as src  # noqa: E402

CLASS_NAMES = src.CLASSES
PROBS = [f"p_class{i}" for i in range(7)]
PREPROCESS = {
    "fov_um": 12,
    "native_scale_um_per_px": 0.2125,
    "native_crop_px": 57,
    "input_size": 224,
    "resize": "torch.nn.functional.interpolate, bicubic, align_corners=False",
    "input_range": "uint8 RGB [0,255] to float32 [0,1]",
    "normalization_mean": [0.5, 0.5, 0.5],
    "normalization_std": [0.5, 0.5, 0.5],
    "feature": "CLS token concatenated with mean of all patch tokens",
    "feature_dim": 3072,
}
RECIPE = {
    "encoder_trainable": "final transformer block only",
    "classifier": "Linear(3072, 7)",
    "optimizer": "AdamW",
    "backbone_lr": 1e-5,
    "head_lr": 1e-3,
    "weight_decay": 1e-4,
    "loss": "class-balanced weighted cross entropy",
    "batch_size": 1024,
    "augmentation": "seeded batch-level 0/90/180/270-degree rotation, horizontal flip p=0.5, mild brightness/contrast p=0.5",
    "epoch_selection": "last lexicographic outer-training batch as inner validation; maximize inner macro-F1 over 3 epochs; refit from source checkpoint on all outer-training cells",
    "seed": src.SEED,
}


def dirs() -> None:
    for name in ("ensemble", "metrics", "figures", "config", "qc", "logs", "code"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path: Path, model, head, meta: dict) -> str:
    payload = {
        "encoder_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classifier_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "architecture": "Midnight-12k",
        "source_model_path": str(src.MIDNIGHT),
        "fov_um": 12,
        "native_crop_px": 57,
        "input_size": 224,
        "feature_dim": 3072,
        "class_names": CLASS_NAMES,
        "preprocessing_config": PREPROCESS,
        "training_recipe": RECIPE,
        "seed": src.SEED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, tmp)
    tmp.replace(path)
    return sha256(path)


def verified_inputs():
    c = src.load_canonical()
    required = {"cell_id", "class_id", "batch", "he_x", "he_y", "patch_id"}
    assert required.issubset(c.columns), sorted(required - set(c.columns))
    assert not c[list(required)].isna().any().any()
    assert c.class_id.astype(int).between(0, 6).all()
    folds = src.load_folds(c)
    assert sorted(folds) == list(range(5))
    assignment = np.zeros(len(c), dtype=np.int8)
    for tr, va in folds.values():
        assert len(np.intersect1d(tr, va)) == 0
        assert len(tr) + len(va) == len(c)
        assignment[va] += 1
    assert np.all(assignment == 1)
    crops = src.crop_memmap(c)
    assert crops.shape == (96044, 57, 57, 3) and crops.dtype == np.uint8
    assert src.MIDNIGHT.joinpath("model.safetensors").is_file()
    return c, folds, crops


def fold_export() -> None:
    src.seed_all()
    dirs()
    c, folds, crops = verified_inputs()
    y = c.class_id.to_numpy(int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Task012 model refits")
    fold_rows, per_rows, bin_rows, selected, manifests = [], [], [], [], []
    oof = np.full((len(c), 7), np.nan, dtype=np.float32)
    cv_ref = pd.read_csv(T11 / "metrics/fine_tune_fold_metrics_corrected.csv")
    for fold, (tr, va) in folds.items():
        inner_batch = sorted(set(c.batch.astype(str).to_numpy()[tr]))[-1]
        inner_va = tr[c.batch.astype(str).to_numpy()[tr] == inner_batch]
        inner_tr = tr[c.batch.astype(str).to_numpy()[tr] != inner_batch]
        m = src.load_model(device)
        h = torch.nn.Linear(3072, 7).to(device)
        best_epoch, inner_score = src.train_epochs(m, h, crops, inner_tr, inner_va, y, device, epochs=3)
        del m, h
        gc.collect(); torch.cuda.empty_cache()
        m = src.load_model(device)
        h = torch.nn.Linear(3072, 7).to(device)
        src.train_epochs(m, h, crops, tr, None, y, device, epochs=best_epoch)
        p, _ = src.evaluate(m, h, crops, va, y, device, keep_z=False)
        assert p.shape == (len(va), 7) and np.isfinite(p).all()
        assert np.allclose(p.sum(axis=1), 1, atol=1e-4)
        oof[va] = p
        row, per = src.metric_rows("F1_FINAL_BLOCK", fold, y[va], p)
        fold_rows.append(row); per_rows.extend(per)
        selected.append({"fold": fold, "inner_val_batch": inner_batch, "selected_epoch": best_epoch, "inner_macro_f1": inner_score})
        path = ROOT / "ensemble" / f"fold{fold}_midnight_fov12_finalblock_7class.pth"
        digest = save_checkpoint(path, m, h, {
            "fold": fold,
            "train_batch_ids": sorted(set(c.batch.astype(str).to_numpy()[tr])),
            "validation_batch_ids": sorted(set(c.batch.astype(str).to_numpy()[va])),
            "inner_validation_batch_id": inner_batch,
            "selected_epoch": best_epoch,
            "inner_macro_f1": inner_score,
            "training_n": len(tr),
            "validation_n": len(va),
        })
        manifests.append({"fold": fold, "checkpoint": str(path), "sha256": digest, "selected_epoch": best_epoch, "macro_f1": row["macro_f1"]})
        # Preserve the corrected Task011 RNG transition before the next fold.
        for comp, pos, neg in (("neutrophil_vs_myeloid", 3, 2), ("neutrophil_vs_tb", 3, 5)):
            bin_rows.append(src.true_binary_from_fold_encoder(m, h, crops, tr, va, y, device, fold, comp, pos, neg))
        ref = cv_ref[(cv_ref.candidate == "F1_FINAL_BLOCK") & (cv_ref.fold == fold)].iloc[0]
        print(f"fold={fold} selected_epoch={best_epoch} macro_f1={row['macro_f1']:.6f} corrected_reference={ref.macro_f1:.6f} checkpoint_sha256={digest}", flush=True)
        del m, h
        gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(oof).all()
    pred = oof.argmax(axis=1)
    out = pd.DataFrame({"cell_id": c.cell_id.astype(str), "fold": -1, "true_class_id": y, "predicted_class_id": pred})
    for fold, (_, va) in folds.items():
        out.loc[va, "fold"] = fold
    for k, col in enumerate(PROBS):
        out[col] = oof[:, k]
    assert not out.cell_id.duplicated().any() and out.fold.between(0, 4).all()
    out.to_csv(ROOT / "metrics/oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(ROOT / "metrics/exported_fold_metrics.csv", index=False)
    pd.DataFrame(per_rows).to_csv(ROOT / "metrics/exported_per_class.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(ROOT / "metrics/exported_true_binary.csv", index=False)
    pd.DataFrame(selected).to_csv(ROOT / "qc/selected_epochs.csv", index=False)
    meta = {"status": "FOLDS_COMPLETE", "canonical_n": len(c), "n_folds": 5, "folds": manifests,
            "source_corrected_metrics": str(T11 / "metrics/fine_tune_fold_metrics_corrected.csv"),
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    (ROOT / "ensemble/ensemble_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def full_export() -> None:
    dirs()
    metadata_path = ROOT / "ensemble/ensemble_metadata.json"
    if not metadata_path.is_file() or json.loads(metadata_path.read_text()).get("status") != "FOLDS_COMPLETE":
        raise RuntimeError("verified fold export required before full-data training")
    src.seed_all()
    c, _, crops = verified_inputs()
    y = c.class_id.to_numpy(int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Task012 full-data training")
    m = src.load_model(device)
    h = torch.nn.Linear(3072, 7).to(device)
    src.train_epochs(m, h, crops, np.arange(len(c)), None, y, device, epochs=3)
    path = ROOT / "midnight_fov12_finalblock_7class.pth"
    digest = save_checkpoint(path, m, h, {
        "training_n": len(c), "epochs": 3, "backbone_lr": 1e-5, "head_lr": 1e-3,
        "optimizer": "AdamW", "weight_decay": 1e-4, "loss": "class-balanced weighted cross entropy",
        "augmentation": RECIPE["augmentation"], "git_commit_hash": os.environ.get("TASK012_GIT_COMMIT", ""),
    })
    (ROOT / "checkpoint_sha256.txt").write_text(f"{digest}  {path.name}\n")
    (ROOT / "preprocessing_config.json").write_text(json.dumps(PREPROCESS, indent=2) + "\n")
    (ROOT / "class_mapping.json").write_text(json.dumps({str(i): n for i, n in enumerate(CLASS_NAMES)}, indent=2) + "\n")
    meta = {"checkpoint_path": str(path), "sha256": digest, "training_n": len(c), "epochs": 3,
            "architecture": "Midnight-12k", "recipe": RECIPE, "preprocessing": PREPROCESS,
            "class_names": CLASS_NAMES, "performance_source": "corrected Task011 five-fold grouped CV",
            "full_data_training_performance_estimated": False,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    (ROOT / "model_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"full checkpoint={path} sha256={digest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("folds", "full"), required=True)
    args = ap.parse_args()
    if args.stage == "folds": fold_export()
    else: full_export()


if __name__ == "__main__":
    main()
