#!/usr/bin/env bash
# Build reset-excluded, matching-versus-other input-current summaries for Digit and Sector.

set -eo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:-$ROOT/results}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:-$ROOT/source/clutter/stimuli}"
RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
SELECTIVITY_ROOT="${INPUT_CURRENT_SELECTIVITY_ROOT:?INPUT_CURRENT_SELECTIVITY_ROOT is required}"
OUT="${INPUT_CURRENT_OUT:?INPUT_CURRENT_OUT is required}"
CONDA_SH="${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"

source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTHONDONTWRITEBYTECODE=1

[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite output: $OUT" >&2; exit 1; }

run_seed() {
  local seed="$1"
  local seed_num="${seed#0}"
  local checkpoint="$RUN_ROOT/gawf-seed$seed/gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
  local selectivity="$SELECTIVITY_ROOT/seed$seed/part1_selectivity.npz"
  [[ -f "$checkpoint" && -f "$selectivity" ]] || { echo "Missing seed input $seed" >&2; return 1; }
  python -m utils.analysis.clutter.fig6_net_input_current collect \
    --ckpt "$checkpoint" --data_dir "$DATA_DIR" --output_dir "$OUT/sector/seed$seed" \
    --seed "$seed_num" --condition sector --device cuda --batch_size 16 --num_workers 2
  python -m utils.analysis.clutter.fig6_net_input_current collect \
    --ckpt "$checkpoint" --data_dir "$DATA_DIR" --output_dir "$OUT/digit/seed$seed" \
    --seed "$seed_num" --condition digit --selectivity "$selectivity" \
    --device cuda --batch_size 16 --num_workers 2
}

(
  export CUDA_VISIBLE_DEVICES=0
  for seed in 01 03 05 07 09; do run_seed "$seed"; done
) &
lane_zero=$!
(
  export CUDA_VISIBLE_DEVICES=1
  for seed in 02 04 06 08 10; do run_seed "$seed"; done
) &
lane_one=$!
wait "$lane_zero"
wait "$lane_one"

for condition in sector digit; do
  final="$OUT/$condition/final"
  python -m utils.analysis.clutter.fig6_net_input_current summarize \
    --data_root "$OUT/$condition" --output_dir "$final" --condition "$condition"
  python -m utils.analysis.clutter.fig6_net_input_current plot \
    --summary "$final/net_input_current_10seed_summary.npz" --figure_dir "$final" \
    --condition "$condition"
done
touch "$OUT/.complete"
