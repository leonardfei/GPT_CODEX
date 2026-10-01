#!/usr/bin/env bash
# Launch gated Task015 outer-CV jobs in GPU-disjoint pairs. No Git mutations.
set -euo pipefail

ROOT=/data/lf_data/result/task015_ghist_celltyping
PYTHON=/data/lf_data/task010_env/bin/python
TRAINER=$ROOT/code/task015_train_ghist_celltype.py
export PYTHONPATH=$ROOT/code

# Root filesystem /tmp filled during the first fold-0 selection run. Keep all
# Python/PyTorch multiprocessing sockets and temporary files on /data.
TASK015_TMPDIR=$ROOT/work/tmp
mkdir -p "$TASK015_TMPDIR"
export TMPDIR=$TASK015_TMPDIR
export TMP=$TASK015_TMPDIR
export TEMP=$TASK015_TMPDIR
"$PYTHON" -c 'import pathlib, shutil, tempfile; p=pathlib.Path("/data/lf_data/result/task015_ghist_celltyping/work/tmp").resolve(); q=pathlib.Path(tempfile.gettempdir()).resolve(); assert q == p, (p,q); assert shutil.disk_usage(p).free > 50*1024**3, "Less than 50 GiB free on Task015 data volume"; print("Task015 temporary directory:", q, flush=True)'

"$PYTHON" -c 'import json, pathlib; p=pathlib.Path("/data/lf_data/result/task015_ghist_celltyping/qc/pilot_summary.json"); assert p.is_file() and json.loads(p.read_text())["status"] == "PILOT_GATE_PASS", "Task015 pilot gate has not passed"'

run_pair() {
  local variant0=$1 fold0=$2 variant1=$3 fold1=$4
  local log0="$ROOT/logs/cv_${variant0}_fold${fold0}.log"
  local log1="$ROOT/logs/cv_${variant1}_fold${fold1}.log"
  echo "Starting $variant0 fold$fold0 on GPU0 and $variant1 fold$fold1 on GPU1"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TRAINER" --cv --variant "$variant0" --fold "$fold0" --gpu 0 > "$log0" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$TRAINER" --cv --variant "$variant1" --fold "$fold1" --gpu 0 > "$log1" 2>&1 &
  local pid1=$!
  local status0=0 status1=0
  wait "$pid0" || status0=$?
  wait "$pid1" || status1=$?
  echo "Finished $variant0 fold$fold0 status=$status0; $variant1 fold$fold1 status=$status1"
  if [[ $status0 -ne 0 || $status1 -ne 0 ]]; then
    echo "Stopping CV launch sequence after a failed fold; inspect individual logs before continuing" >&2
    return 1
  fi
}

run_pair GHIST_CT_OFFICIAL 0 GHIST_CT_BALANCED 0
run_pair GHIST_CT_OFFICIAL 1 GHIST_CT_BALANCED 1
run_pair GHIST_CT_OFFICIAL 2 GHIST_CT_BALANCED 2
run_pair GHIST_CT_OFFICIAL 3 GHIST_CT_BALANCED 3
run_pair GHIST_CT_OFFICIAL 4 GHIST_CT_BALANCED 4
echo "All ten Task015 CV runs finished successfully; final metric aggregation still required."
