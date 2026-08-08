#!/usr/bin/env bash
# Submit compute-node rendering of the combined completed-ten Breakout sweep figure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
cd "$ROOT"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "render: one mean±SD figure for completed L3/L4 GaWF LR-decay conditions"
  echo "output: \$AIM3_RESULTS_PATH/figs/rl/atari/breakout_4action/fs4_stack4_l3_3seed/"
  echo "lrdecay_l3_l4_mean_std.png"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_breakout_gawf_lr_buffer_completed10_curves.sh"
ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_breakout_gawf_lr_buffer_completed10_curves"
mkdir -p "$ART"

EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
EXPORTS+=",AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
JOB_RAW="$(sbatch --parsable --chdir="$ROOT" --output="$ART/%j.out" --error="$ART/%j.err" \
  --export="$EXPORTS" \
  "$RUNNER")"
echo "JOB_ID=${JOB_RAW%%;*}"
echo "OUTPUT_DIR=$AIM3_RESULTS_PATH/figs/rl/atari/breakout_4action/fs4_stack4_l3_3seed"
echo "OUTPUT_FILE=lrdecay_l3_l4_mean_std.png"
