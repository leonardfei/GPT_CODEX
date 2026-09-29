#!/usr/bin/env python3
"""Finish Task012 from two already-running fold workers, with QC gates."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import task012_server_pipeline as pipe

ROOT = pipe.ROOT
PYTHON = pipe.PYTHON
CODE = ROOT / "code"
FOLD0_PID = 3508143
FOLD1_PID = 3502428
GIT_COMMIT = "bd9bc69"


def wait_pid(pid):
    while pipe.alive(pid): time.sleep(30)


def run(label, arguments, gpu=None):
    env = os.environ.copy()
    if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TASK012_GIT_COMMIT"] = GIT_COMMIT
    path = ROOT / "logs" / (label + ".log")
    with path.open("w") as log:
        process = subprocess.run([PYTHON, *arguments], cwd=CODE, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed; see {path}")


def gpu1_chain():
    wait_pid(FOLD1_PID)
    checkpoint = ROOT / "parallel/fold1_midnight_fov12_finalblock_7class.pth"
    if not checkpoint.is_file():
        raise RuntimeError("fold1 checkpoint missing after first worker exited")
    # The first fold1 worker used an overly strict fp16 row-sum assertion after
    # checkpoint save; evaluate the completed checkpoint with the corrected QC.
    run("parallel_fold1_eval", ["task012_fold_worker.py", "--fold", "1", "--eval-checkpoint", str(checkpoint)], gpu=1)
    run("parallel_fold3", ["task012_fold_worker.py", "--fold", "3", "--start-seed", "20265951"], gpu=1)


def gpu0_chain():
    wait_pid(FOLD0_PID)
    if not (ROOT / "parallel/fold0_manifest.json").is_file():
        raise RuntimeError("fold0 worker exited without complete manifest")
    run("parallel_fold2", ["task012_fold_worker.py", "--fold", "2", "--start-seed", "20265941"], gpu=0)
    run("parallel_fold4", ["task012_fold_worker.py", "--fold", "4", "--start-seed", "20265961"], gpu=0)


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        future0 = pool.submit(gpu0_chain)
        future1 = pool.submit(gpu1_chain)
        future0.result(); future1.result()
    print("All five fold checkpoints generated; collecting and verifying", flush=True)
    run("collect_parallel", ["task012_collect_parallel.py"])
    if pipe.sha256(pipe.PRODUCTION) != pipe.EXPECTED_PRODUCTION_SHA:
        raise RuntimeError("production checkpoint baseline SHA256 mismatch")
    print("Five folds verified; training full-data final model", flush=True)
    run("full", ["task012_export_models.py", "--stage", "full"], gpu=0)
    print("Final checkpoint saved; generating metric and figure package", flush=True)
    run("finalize", ["task012_finalize_figures.py"])
    print("Figures complete; running inference smoke tests", flush=True)
    smoke = pipe.smoke()
    final_path = ROOT / "midnight_fov12_finalblock_7class.pth"
    final_digest = pipe.sha256(final_path)
    recorded = (ROOT / "checkpoint_sha256.txt").read_text().split()[0]
    if recorded != final_digest: raise RuntimeError("final checkpoint hash mismatch")
    if pipe.sha256(pipe.PRODUCTION) != pipe.EXPECTED_PRODUCTION_SHA:
        raise RuntimeError("production checkpoint changed")
    info = {"status": "COMPLETE", "final_checkpoint": str(final_path),
            "final_sha256": final_digest, "fold_checkpoint_count": 5,
            "figure_count": len(list((ROOT / "figures").glob("Fig*.pdf"))),
            "smoke_test": smoke, "production_sha256_unchanged": True}
    (ROOT / "config/task012_completion.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"TASK012_ORCHESTRATOR_ERROR: {exc}", file=sys.stderr, flush=True)
        raise
