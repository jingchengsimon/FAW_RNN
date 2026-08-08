#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-l34-lrbuf-curves
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

# Render the cross-seed aggregate figure for all completed Breakout LR-decay cells.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
PLOTTER="$ROOT/utils/analysis/rl/atari/atari_breakout_gawf_lr_buffer_completed10.py"
if [[ -z "$ROOT" || ! -f "$PLOTTER" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

DATA_ROOT="$AIM3_RESULTS_PATH/train_data/diagnostics/breakout_gawf_lrdecay_buffer_sweep"
OUT_DIR="$AIM3_RESULTS_PATH/figs/rl/atari/breakout_4action/fs4_stack4_l3_3seed"
python -m utils.analysis.rl.atari.atari_breakout_gawf_lr_buffer_completed11_mean_std \
  --data-root "$DATA_ROOT" --diagnostic-data-root "$AIM3_RESULTS_PATH/train_data" \
  --output "$OUT_DIR/lrdecay_l3_l4_mean_std.png" \
  --smooth 10 --y-min 0 --y-max 190
