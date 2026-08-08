#!/usr/bin/env bash
# Submit the unified 11-run L3/L4 GaWF LR-decay and replay-buffer sweep.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$ROOT"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "11 tasks: L3 (1M buffer seeds 1/3; 2M seeds 1/2/3) and L4 (1M/2M seeds 1/2/3)"
  echo "all tasks: LR 1e-4 through 1M, then 1e-5; common diagnostic result parent; array throttle %4"
  exit 0
fi
: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

art="$ROOT/experiments/rl/atari/amarel/artifacts/atari_breakout_gawf_lr_buffer_sweep"
mkdir -p "$art"
job="$(sbatch --parsable --array=0-10%4 --chdir="$ROOT" \
  --output="$art/%A_%a.out" --error="$art/%A_%a.err" \
  --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,HIDDEN_L3=605,HIDDEN_L4=527,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1" \
  experiments/rl/atari/amarel/run_atari_breakout_gawf_lr_buffer_sweep_array.sh)"
echo "JOB_ID=${job%%;*}"
echo "RESULT_PARENT=$AIM3_RESULTS_PATH/train_data/diagnostics/breakout_gawf_lrdecay_buffer_sweep"
