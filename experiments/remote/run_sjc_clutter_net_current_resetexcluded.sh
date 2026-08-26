#!/usr/bin/env bash
# Rebuild digit and sector net recurrent-current summaries from registered compact caches.

set -eo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:-$ROOT/results}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:-$ROOT/source/clutter/stimuli}"
RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
OUT="${NET_CURRENT_OUT:?NET_CURRENT_OUT is required}"
COMPACT_ROOT="${NET_CURRENT_COMPACT_ROOT:?NET_CURRENT_COMPACT_ROOT is required}"
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
  local compact="$COMPACT_ROOT/seed$seed/compact/recurrent_gate_condition_means.npz"
  [[ -f "$checkpoint" && -f "$compact" ]] || { echo "Missing seed input $seed" >&2; return 1; }
  for condition in digit sector; do
    python -m utils.analysis.clutter.fig6_net_recurrent_current collect \
      --ckpt "$checkpoint" --data_dir "$DATA_DIR" --compact "$compact" \
      --output_dir "$OUT/$condition/seed$seed" --seed "$seed_num" --condition "$condition" \
      --device cuda --batch_size 16 --num_workers 2
  done
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

for condition in digit sector; do
  final="$OUT/$condition/final"
  python -m utils.analysis.clutter.fig6_net_recurrent_current summarize \
    --data_root "$OUT/$condition" --output_dir "$final" --condition "$condition"
  python -m utils.analysis.clutter.fig6_net_recurrent_current plot \
    --summary "$final/net_recurrent_current_10seed_summary.npz" \
    --figure_dir "$final" --condition "$condition"
done

cp "$OUT/digit/final/Supple4_net_recurrent_current_3x4_10seed.png" "$RESULTS/save/"
cp "$OUT/sector/final/Supple4_net_recurrent_current_3x4_10seed_sector.png" "$RESULTS/save/"
touch "$OUT/.complete"
