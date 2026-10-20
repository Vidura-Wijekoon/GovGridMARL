#!/usr/bin/env bash
# Phase-2 Month-5 runs on the laptop (tasks m5-t3 .. m5-t10):
#   unc    MAPPO, same reward (fairness weight 2), no constraints      x 3 seeds   (ablation reference)
#   lag    MAPPO-Lagrangian, dual ascent, safety set                   x 3 seeds   (m5-t3/t4)
#   pid    MAPPO-PID-Lagrangian, safety set                            x 3 seeds   (m5-t6/t7/t8)
#   joint  MAPPO-PID-Lagrangian, safety + fairness constraints         x 3 seeds   (m5-t10)
#   ppolag PPO-Lagrangian (centralised controller) reference           x 3 seeds   (m5-t3)
#   lr     multiplier learning-rate sweep for dual ascent, seed 0       x 3         (m5-t5)
# 120k env-steps per run; PAR runs in parallel.
# Usage: PYTHON=.venv/Scripts/python bash scripts/run_phase2_month5.sh [PAR]
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
  launch "unc_s${seed}"   +experiment=fairness_safety agent=mappo seed="$seed" run_tag=unc
  launch "lag_s${seed}"   +experiment=fairness_safety agent.method=lagrangian seed="$seed" run_tag=lag
  launch "pid_s${seed}"   +experiment=fairness_safety seed="$seed" run_tag=pid
  launch "joint_s${seed}" +experiment=fairness_safety \
      'agent.constraints=[voltage,loading,comfort,fairness]' seed="$seed" run_tag=joint
  launch "ppolag_s${seed}" +experiment=fairness_safety agent=ppo_lagrangian seed="$seed" run_tag=lag
done
for lr in 0.01 0.2 1.0; do
  launch "lag_lr${lr}_s0" +experiment=fairness_safety agent.method=lagrangian \
      agent.multiplier_lr="$lr" seed=0 run_tag="lag_lr${lr}"
done
wait
echo "[$(date -u +%FT%TZ)] ALL DONE" | tee -a results/logs/m5_all.log
