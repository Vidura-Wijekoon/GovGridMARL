#!/usr/bin/env bash
# Phase-2 Month-4 runs on the laptop (tasks m4-t5, m4-t6, m4-t8, m4-t9):
#   (a) weighted-sum fairness sweep: MAPPO x lambda in {0, 0.5, 1, 2, 4, 8} x 3 seeds
#   (b) fairness-form ablation at lambda = 2: gap/fraction and jain/fraction x 3 seeds
#   (c) MOPPO (preference-conditioned) x 3 seeds
# 120k env-steps per run (~3 min each at ~1,400 steps/s); PAR runs in parallel.
# Usage: bash scripts/run_phase2_month4.sh [PAR]   (from the repo root, venv active)
set -uo pipefail
cd "$(dirname "$0")/.."
PAR="${1:-8}"
PY="${PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p results/logs results/runs_m4
RUN_ROOT=results/runs_m4

jobs_running() { jobs -rp | wc -l; }
launch() {  # tag, args...
  local tag="$1"; shift
  while [ "$(jobs_running)" -ge "$PAR" ]; do sleep 5; done
  echo "[$(date -u +%FT%TZ)] start $tag"
  "$PY" experiments/train.py "$@" run_root="$RUN_ROOT" > "results/logs/m4_${tag}.log" 2>&1 &
}

for seed in 0 1 2; do
  for wf in 0.0 0.5 1.0 2.0 4.0 8.0; do
    launch "mappo_wf${wf}_s${seed}" +experiment=fairness_sweep reward.weights.fairness="$wf" \
      seed="$seed" run_tag="wf${wf}"
  done
  launch "mappo_gapfrac_s${seed}" +experiment=fairness_sweep reward.weights.fairness=2.0 \
    env.fairness_form=gap env.fairness_basis=fraction seed="$seed" run_tag="gapfrac2.0"
  launch "mappo_jainfrac_s${seed}" +experiment=fairness_sweep reward.weights.fairness=2.0 \
    env.fairness_basis=fraction seed="$seed" run_tag="jainfrac2.0"
  launch "moppo_s${seed}" +experiment=moppo_ieee14 seed="$seed"
done
wait
echo "[$(date -u +%FT%TZ)] ALL DONE" | tee -a results/logs/m4_all.log
