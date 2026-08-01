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
Y_LIMITS=(--y-min 0 --y-max 190)
SEEDS_5=(42 1 2 3 4)

for layer in 1 2; do
  if [[ "$layer" == "1" ]]; then
    prefix="atari_dqn_breakout_fs4_stack4_l1"
  else
    prefix="atari_dqn_breakout_fs4_stack4_l2match"
  fi
  layer_dir="$OUT_DIR/fs4_stack4_l${layer}_5seed"
  for seed in "${SEEDS_5[@]}"; do
    python -m utils.analysis.rl.atari.atari_learning_curves \
      --data_root "$AIM3_RESULTS_PATH/train_data" --prefix "$prefix" --setting both \
      --seed "$seed" --smooth 10 "${Y_LIMITS[@]}" --output "$layer_dir/seed${seed}.png"
  done
  python -m utils.analysis.rl.atari.atari_learning_curves \
    --data_root "$AIM3_RESULTS_PATH/train_data" --prefix "$prefix" --setting both \
    --smooth 10 "${Y_LIMITS[@]}" --output "$layer_dir/mean_std.png"
done

python -m utils.analysis.rl.atari.atari_breakout_depth_curves "${COMMON[@]}" \
  --num-layers 3 --expected-steps 3000000 --seeds 1 2 3 "${Y_LIMITS[@]}" \
  --output-dir "$OUT_DIR/fs4_stack4_l3_3seed"
python -m utils.analysis.rl.atari.atari_breakout_depth_curves "${COMMON[@]}" \
  --num-layers 4 --expected-steps 3000000 --seeds 1 2 3 --partial gawf:1 "${Y_LIMITS[@]}" \
  --output-dir "$OUT_DIR/fs4_stack4_l4_3seed"
