#!/usr/bin/env bash
# Submit five L3 full-replay smoke tasks without releasing L4 formal work.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$ROOT"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "five L3 smoke tasks: 25k steps, full 1M/2M mmap replay allocation, array %5"
  echo "task map: 1M seeds 1/3; 2M seeds 1/2/3; L4 excluded"
  exit 0
fi
: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

art="$ROOT/experiments/rl/atari/amarel/artifacts/atari_breakout_l3_gawf_lr_buffer_smoke"
parent="$AIM3_RESULTS_PATH/train_data/diagnostics/breakout_gawf_lrbuf_smoke25k"
mkdir -p "$art"
job="$(sbatch --parsable --array=0-4%5 --time=01:00:00 --chdir="$ROOT" \
  --output="$art/%A_%a.out" --error="$art/%A_%a.err" \
  --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,HIDDEN_L3=605,HIDDEN_L4=527,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1,SWEEP_PARENT=$parent,RUN_TAG=lrbufsmoke25k,TOTAL_TIMESTEPS=25000,CHECKPOINT_INTERVAL_STEPS=5000" \
  experiments/rl/atari/amarel/run_atari_breakout_gawf_lr_buffer_sweep_array.sh)"
echo "JOB_ID=${job%%;*}"
echo "RESULT_PARENT=$parent"
