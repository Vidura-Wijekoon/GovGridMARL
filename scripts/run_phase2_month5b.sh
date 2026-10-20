#!/usr/bin/env bash
# Phase-2 Month-5, second batch (m5-t5 tuning outcome -> m5-t7/t8/t10):
# the same learners with the *indicator* cost signal (violating hours per episode),
# which puts the constraint error on a 0..24 scale and lets the multipliers act.
#   lag_ind    MAPPO-Lagrangian (dual ascent)              x 3 seeds
#   pid_ind    MAPPO-PID-Lagrangian                        x 3 seeds   (m5-t6/t7/t8)
#   joint_ind  MAPPO-PID-Lagrangian + fairness constraint  x 3 seeds   (m5-t10)
#   ppolag_ind PPO-Lagrangian reference                    x 3 seeds
#   joint      magnitude-cost joint rerun on the amended fairness basis x 3 seeds
# Usage: PYTHON=.venv/Scripts/python bash scripts/run_phase2_month5b.sh [PAR]
set -uo pipefail
cd "$(dirname "$0")/.."
PAR="${1:-8}"
PY="${PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p results/logs results/runs_m5
RUN_ROOT=results/runs_m5

jobs_running() { jobs -rp | wc -l; }
launch() {  # tag, args...
  local tag="$1"; shift
  while [ "$(jobs_running)" -ge "$PAR" ]; do sleep 5; done
  echo "[$(date -u +%FT%TZ)] start $tag"
  "$PY" experiments/train.py "$@" run_root="$RUN_ROOT" > "results/logs/m5_${tag}.log" 2>&1 &
}

for seed in 0 1 2; do
  launch "lag_ind_s${seed}"   +experiment=fairness_safety agent.method=lagrangian seed="$seed" run_tag=lag_ind
  launch "pid_ind_s${seed}"   +experiment=fairness_safety seed="$seed" run_tag=pid_ind
  launch "joint_ind_s${seed}" +experiment=fairness_safety \
      'agent.constraints=[voltage,loading,comfort,fairness]' seed="$seed" run_tag=joint_ind
  launch "ppolag_ind_s${seed}" +experiment=fairness_safety agent=ppo_lagrangian seed="$seed" run_tag=lag_ind
  launch "joint_s${seed}" +experiment=fairness_safety agent.cost_signal=magnitude \
      'agent.constraints=[voltage,loading,comfort,fairness]' seed="$seed" run_tag=joint
done
wait
echo "[$(date -u +%FT%TZ)] ALL DONE" | tee -a results/logs/m5b_all.log
