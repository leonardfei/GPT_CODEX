#!/usr/bin/env python3
"""Independent Task012 fold refit for parallel GPU execution and exact QC."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import task012_export_models as export

ROOT = export.ROOT
src = export.src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--start-seed", type=int)
    ap.add_argument("--eval-checkpoint", type=Path)
    args = ap.parse_args()
    if args.fold not in range(5): raise ValueError("fold must be 0..4")
    if args.start_seed is None and args.eval_checkpoint is None:
        raise ValueError("provide start-seed for training or eval-checkpoint")
    export.dirs()
    parallel = ROOT / "parallel"
    parallel.mkdir(exist_ok=True)
    src.seed_all(args.start_seed if args.start_seed is not None else src.SEED)
    c, folds, crops = export.verified_inputs()
    tr, va = folds[args.fold]
    y = c.class_id.to_numpy(int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA required")
    if args.eval_checkpoint is None:
        inner_batch = sorted(set(c.batch.astype(str).to_numpy()[tr]))[-1]
        inner_va = tr[c.batch.astype(str).to_numpy()[tr] == inner_batch]
        inner_tr = tr[c.batch.astype(str).to_numpy()[tr] != inner_batch]
        m = src.load_model(device); h = torch.nn.Linear(3072, 7).to(device)
        best_epoch, inner_score = src.train_epochs(m, h, crops, inner_tr, inner_va, y, device, epochs=3)
        del m, h; gc.collect(); torch.cuda.empty_cache()
        m = src.load_model(device); h = torch.nn.Linear(3072, 7).to(device)
        src.train_epochs(m, h, crops, tr, None, y, device, epochs=best_epoch)
        path = parallel / f"fold{args.fold}_midnight_fov12_finalblock_7class.pth"
        digest = export.save_checkpoint(path, m, h, {
            "fold": args.fold, "train_batch_ids": sorted(set(c.batch.astype(str).to_numpy()[tr])),
            "validation_batch_ids": sorted(set(c.batch.astype(str).to_numpy()[va])),
            "inner_validation_batch_id": inner_batch, "selected_epoch": best_epoch,
            "inner_macro_f1": inner_score, "training_n": len(tr), "validation_n": len(va),
        })
    else:
        from transformers import AutoModel
        checkpoint = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=False)
        m = AutoModel.from_pretrained(str(src.MIDNIGHT), local_files_only=True)
        m.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        h = torch.nn.Linear(3072, 7)
        h.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
        m.to(device); h.to(device)
        path = args.eval_checkpoint
        digest = export.sha256(path)
        best_epoch = checkpoint["selected_epoch"]
        inner_score = checkpoint["inner_macro_f1"]
        inner_batch = checkpoint["inner_validation_batch_id"]
    p, _ = src.evaluate(m, h, crops, va, y, device, keep_z=False)
    assert p.shape == (len(va), 7) and np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1, atol=2e-3, rtol=0)
    row, per = src.metric_rows("F1_FINAL_BLOCK", args.fold, y[va], p)
    ref = pd.read_csv(src.ROOT / "metrics/fine_tune_fold_metrics_corrected.csv")
    ref = ref[(ref.candidate == "F1_FINAL_BLOCK") & (ref.fold == args.fold)].iloc[0]
    delta = float(row["macro_f1"] - ref.macro_f1)
    output = pd.DataFrame({"cell_id": c.cell_id.astype(str).iloc[va].to_numpy(),
                           "fold": args.fold, "true_class_id": y[va],
                           "predicted_class_id": p.argmax(axis=1)})
    for k in range(7): output[f"p_class{k}"] = p[:, k]
    output.to_csv(parallel / f"oof_fold{args.fold}.csv.gz", index=False, compression="gzip")
    pd.DataFrame([row]).to_csv(parallel / f"fold{args.fold}_metrics.csv", index=False)
    pd.DataFrame(per).to_csv(parallel / f"fold{args.fold}_per_class.csv", index=False)
    manifest = {"fold": args.fold, "checkpoint": str(path), "sha256": digest,
                "selected_epoch": best_epoch, "inner_val_batch": inner_batch,
                "inner_macro_f1": inner_score, "macro_f1": row["macro_f1"],
                "corrected_reference_macro_f1": float(ref.macro_f1), "macro_f1_delta": delta,
                "start_seed": args.start_seed, "eval_only": args.eval_checkpoint is not None}
    (parallel / f"fold{args.fold}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
