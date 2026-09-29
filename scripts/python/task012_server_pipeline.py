#!/usr/bin/env python3
"""Continue Task012 after the already-running fold export completes."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("/data/lf_data/result/final_model")
PYTHON = "/data/lf_data/task010_env/bin/python"
SOURCE_CROPS = Path("/data/lf_data/result/task011_midnight_local_optimization/work/midnight_fov12_crops_uint8.npy")
PRODUCTION = Path("/data/lf_data/result/model_best.pth")
EXPECTED_PRODUCTION_SHA = "f161afbb90f42ccfbfe9c6843cae6eafd7a12a2bc25620d5b4489e7e3faf6164"
FOLD_PID = 3494371


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run(label, args):
    path = ROOT / "logs" / (label + ".log")
    with path.open("w") as log:
        p = subprocess.run(args, cwd=ROOT / "code", stdout=log, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {p.returncode}; see {path}")


def smoke():
    crop_dir = ROOT / "qc/smoke_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crops = np.load(SOURCE_CROPS, mmap_mode="r")
    assert crops.shape == (96044, 57, 57, 3)
    for idx in (0, 1):
        Image.fromarray(np.asarray(crops[idx])).save(crop_dir / f"canonical_{idx}.png")
    output_single = ROOT / "qc/smoke_single.csv"
    output_ensemble = ROOT / "qc/smoke_ensemble.csv"
    run("smoke_single", [PYTHON, "inference_single_model.py", "--input", str(crop_dir), "--output", str(output_single)])
    run("smoke_ensemble", [PYTHON, "inference_ensemble.py", "--input", str(crop_dir), "--output", str(output_ensemble)])
    expected = ["sample_id", "predicted_class_id", "predicted_class_name", "p_Endothelial", "p_Mesenchymal", "p_Myeloid", "p_Neutrophil", "p_Plasma", "p_T_and_B", "p_Tumor"]
    qc = {}
    for label, path in (("single", output_single), ("ensemble", output_ensemble)):
        df = pd.read_csv(path)
        p = df[expected[3:]].to_numpy(float)
        assert df.columns.tolist() == expected and len(df) == 2
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=2e-3, rtol=0)
        assert df.sample_id.tolist() == ["canonical_0", "canonical_1"]
        qc[label] = {"rows": len(df), "probabilities_finite": True, "row_sums_one": True}
    (ROOT / "qc/inference_smoke_test.json").write_text(json.dumps(qc, indent=2) + "\n")
    return qc


def main():
    while alive(FOLD_PID):
        time.sleep(30)
    fold_meta_path = ROOT / "ensemble/ensemble_metadata.json"
    if not fold_meta_path.is_file():
        raise RuntimeError("fold exporter stopped without completion metadata")
    fold_meta = json.loads(fold_meta_path.read_text())
    if fold_meta.get("status") != "FOLDS_COMPLETE" or len(fold_meta.get("folds", [])) != 5:
        raise RuntimeError("fold export metadata incomplete")
    for item in fold_meta["folds"]:
        if sha256(Path(item["checkpoint"])) != item["sha256"]:
            raise RuntimeError(f"fold {item['fold']} checkpoint hash mismatch")
    print("folds verified; starting full-data export", flush=True)
    if sha256(PRODUCTION) != EXPECTED_PRODUCTION_SHA:
        raise RuntimeError("production checkpoint baseline SHA256 mismatch")
    run("full", [PYTHON, "task012_export_models.py", "--stage", "full"])
    print("full-data checkpoint complete; generating metrics and figures", flush=True)
    run("finalize", [PYTHON, "task012_finalize_figures.py"])
    print("figure package complete; testing inference", flush=True)
    smoke_result = smoke()
    if sha256(PRODUCTION) != EXPECTED_PRODUCTION_SHA:
        raise RuntimeError("production checkpoint changed")
    final_path = ROOT / "midnight_fov12_finalblock_7class.pth"
    recorded = (ROOT / "checkpoint_sha256.txt").read_text().split()[0]
    actual = sha256(final_path)
    if actual != recorded: raise RuntimeError("final checkpoint hash mismatch")
    completion = {"status": "COMPLETE", "final_checkpoint": str(final_path),
                  "final_sha256": actual, "fold_checkpoint_count": 5,
                  "figure_count": len(list((ROOT / "figures").glob("Fig*.pdf"))),
                  "smoke_test": smoke_result, "production_sha256_unchanged": True}
    (ROOT / "config/task012_completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TASK012_PIPELINE_ERROR: {exc}", file=sys.stderr, flush=True)
        raise
