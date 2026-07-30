#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-depth-curves
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

# Render reproducible completed/partial strict Breakout depth learning curves on a compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/utils/analysis/rl/atari/atari_breakout_depth_curves.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

OUT_DIR="$AIM3_RESULTS_PATH/figs/rl/atari/breakout_4action"
mkdir -p "$OUT_DIR"
COMMON=(--data-root "$AIM3_RESULTS_PATH/train_data" --smooth 10)

python -m utils.analysis.rl.atari.atari_breakout_depth_curves "${COMMON[@]}" \
  --num-layers 3 --expected-steps 3000000 --seeds 1 2 3 \
  --output-dir "$OUT_DIR/fs4_stack4_l3_3seed"
python -m utils.analysis.rl.atari.atari_breakout_depth_curves "${COMMON[@]}" \
  --num-layers 4 --expected-steps 3000000 --seeds 1 2 3 --partial gawf:1 \
  --output-dir "$OUT_DIR/fs4_stack4_l4_3seed"
