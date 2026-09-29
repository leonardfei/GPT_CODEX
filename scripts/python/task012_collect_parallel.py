#!/usr/bin/env python3
"""Assemble exact-QC Task012 fold workers into canonical OOF and ensemble."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import task012_export_models as export

ROOT = export.ROOT
PARALLEL = ROOT / "parallel"
T11 = export.T11


def main():
    export.dirs()
    c, folds, _ = export.verified_inputs()
    ref = pd.read_csv(T11 / "metrics/fine_tune_fold_metrics_corrected.csv").sort_values("fold")
    ref_epochs = pd.read_csv(T11 / "qc/fine_tune_inner_validation_corrected.csv").sort_values("fold")
    frames, rows, per_rows, manifests, qc_rows = [], [], [], [], []
    for fold in range(5):
        mf = PARALLEL / f"fold{fold}_manifest.json"
        manifest = json.loads(mf.read_text())
        path = Path(manifest["checkpoint"])
        assert path.is_file() and export.sha256(path) == manifest["sha256"]
        dest = ROOT / "ensemble" / f"fold{fold}_midnight_fov12_finalblock_7class.pth"
        if path != dest:
            if dest.exists():
                raise RuntimeError(f"Refusing to replace existing ensemble checkpoint: {dest}")
            path.rename(dest)
        manifest["checkpoint"] = str(dest)
        mf.write_text(json.dumps(manifest, indent=2) + "\n")
        frame = pd.read_csv(PARALLEL / f"oof_fold{fold}.csv.gz")
        _, va = folds[fold]
        assert len(frame) == len(va)
        assert np.array_equal(frame.cell_id.astype(str).to_numpy(), c.cell_id.astype(str).iloc[va].to_numpy())
        assert np.array_equal(frame.true_class_id.to_numpy(int), c.class_id.iloc[va].to_numpy(int))
        assert frame.fold.eq(fold).all()
        frames.append(frame)
        row = pd.read_csv(PARALLEL / f"fold{fold}_metrics.csv").iloc[0]
        rows.append(row.to_dict())
        per_rows.append(pd.read_csv(PARALLEL / f"fold{fold}_per_class.csv"))
        baseline = ref[ref.fold.eq(fold)].iloc[0]
        differences = {col: float(row[col] - baseline[col]) for col in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "weighted_f1", "lowest_three_f1")}
        expected_epoch = int(ref_epochs[ref_epochs.fold.eq(fold)].selected_epoch.iloc[0])
        if int(manifest["selected_epoch"]) != expected_epoch:
            raise RuntimeError(f"fold {fold} selected epoch differs from corrected Task011")
        qc_rows.append({"fold": fold, "selected_epoch": expected_epoch, **differences})
        manifests.append({k: manifest[k] for k in ("fold", "checkpoint", "sha256", "selected_epoch", "macro_f1")})
    qcd = pd.DataFrame(qc_rows)
    cols = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "weighted_f1", "lowest_three_f1"]
    if qcd[cols].abs().to_numpy().max() > 1e-3:
        raise RuntimeError(f"Parallel fold rerun did not reproduce corrected Task011 within 1e-3: {qcd.to_string(index=False)}")
    oof = pd.concat(frames, ignore_index=True).set_index("cell_id").loc[c.cell_id.astype(str)].reset_index()
    assert len(oof) == 96044 and not oof.cell_id.duplicated().any()
    oof.to_csv(ROOT / "metrics/oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(rows).sort_values("fold").to_csv(ROOT / "metrics/exported_fold_metrics.csv", index=False)
    pd.concat(per_rows, ignore_index=True).sort_values(["fold", "class_id"]).to_csv(ROOT / "metrics/exported_per_class.csv", index=False)
    qcd.to_csv(ROOT / "qc/exported_fold_reproduction_deltas.csv", index=False)
    pd.DataFrame([{"fold": fold, "inner_val_batch": json.loads((PARALLEL / f"fold{fold}_manifest.json").read_text())["inner_val_batch"],
                   "selected_epoch": int(manifests[fold]["selected_epoch"])} for fold in range(5)]).to_csv(ROOT / "qc/selected_epochs.csv", index=False)
    meta = {"status": "FOLDS_COMPLETE", "canonical_n": len(c), "n_folds": 5, "folds": manifests,
            "source_corrected_metrics": str(T11 / "metrics/fine_tune_fold_metrics_corrected.csv"),
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            "max_abs_corrected_metric_delta": float(qcd[cols].abs().to_numpy().max())}
    (ROOT / "ensemble/ensemble_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__": main()
