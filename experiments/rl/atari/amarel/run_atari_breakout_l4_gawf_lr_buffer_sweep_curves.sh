#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-l4-lrbuf-curves
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

# Render L4 GaWF learning curves for completed cells of the Breakout LR/replay sweep.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/utils/analysis/rl/atari/atari_learning_curves.py" ]]; then
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
OUT_DIR="$AIM3_RESULTS_PATH/figs/rl/atari/breakout_4action/fs4_stack4_l4_3seed"
PREFIX="atari_dqn_breakout_fs4_stack4_l4_lrdecay1m"
mkdir -p "$OUT_DIR"

for buffer in 1m 2m; do
  for seed in 1 2 3; do
    python -m utils.analysis.rl.atari.atari_learning_curves \
      --data_root "$DATA_ROOT" --prefix "${PREFIX}_buf${buffer}" --models gawf --setting plain \
      --seed "$seed" --smooth 10 --y-min 0 --y-max 190 \
      --output "$OUT_DIR/lrdecay_buf${buffer}_seed${seed}.png"
  done
  python -m utils.analysis.rl.atari.atari_learning_curves \
    --data_root "$DATA_ROOT" --prefix "${PREFIX}_buf${buffer}" --models gawf --setting plain \
    --smooth 10 --band std --y-min 0 --y-max 190 \
    --output "$OUT_DIR/lrdecay_buf${buffer}_mean_std.png"
done
