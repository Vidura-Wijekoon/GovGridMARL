#!/usr/bin/env bash
# Cloud CPU convergence-trend runs (m3-t5): 3 learners x 3 seeds x 120k steps,
# two processes at a time (2 vCPU box). Full 5-seed / 500k runs: run_baselines.ps1
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p results/logs
run() {  # agent seed
  python experiments/train.py +experiment=baseline_cloud agent="$1" seed="$2" \
    > "results/logs/train_${1}_seed${2}.log" 2>&1
}
for seed in 0 1 2; do
  run mappo $seed & run ippo $seed & wait
  run ppo $seed &
  wait
done
echo "ALL DONE $(date -u +%FT%TZ)" >> results/logs/train_all.log
