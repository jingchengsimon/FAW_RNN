#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-l5-curves
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

# Render strict plain Breakout L5 curves from the completed 5-model, 3-seed result set.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/utils/analysis/rl/atari/atari_breakout_depth_curves.py" ]]; then
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

OUT_DIR="$AIM3_RESULTS_PATH/figs/rl/atari/breakout_4action/fs4_stack4_l5_3seed"
python -m utils.analysis.rl.atari.atari_breakout_depth_curves \
  --data-root "$AIM3_RESULTS_PATH/train_data" --num-layers 5 --expected-steps 3000000 \
  --seeds 1 2 3 --smooth 10 --y-min 0 --y-max 190 --output-dir "$OUT_DIR"
