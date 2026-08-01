#!/usr/bin/env bash
set -euo pipefail
ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"; cd "$ROOT"
if [[ "${1:-}" == --dry-run ]]; then
  echo "six L3 GaWF seed2 diagnostics plus dependent fixed-seed evaluations"
  exit 0
fi
: "${AIM3_RESULTS_PATH:?}"
# The completed L3 GaWF seed2 metrics record hidden_size=605 (the L1
# LSTM(512)-matched configuration); retain that exact already-verified size.
train_job="$(sbatch --parsable --array=0-5%6 --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,HIDDEN_SIZE=605,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1" experiments/rl/atari/amarel/run_atari_breakout_l3_gawf_diagnostic_array.sh)"
train_job="${train_job%%;*}"
eval_job="$(sbatch --parsable --dependency="afterok:$train_job" --array=0-5%6 --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH" experiments/rl/atari/amarel/run_atari_breakout_l3_gawf_diagnostic_eval_array.sh)"
echo "TRAIN_JOB_ID=$train_job"
echo "EVAL_JOB_ID=${eval_job%%;*}"
