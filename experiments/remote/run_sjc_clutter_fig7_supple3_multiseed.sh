#!/usr/bin/env bash
# Build compact ten-seed recurrent-gate summaries and render Figure 7/Supplementary 3.

set -eo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:-$ROOT/results}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:-$ROOT/source/clutter/stimuli}"
RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
TRAJECTORIES="$RESULTS/data/analysis/fig6_sector_gate_weight_sign_10seed/trajectories"
OUT="${FIG7_OUT:-$RESULTS/data/analysis/fig7_recurrent_gate_10seed}"
SELECTIVITY_ROOT="${FIG7_SELECTIVITY_ROOT:-$OUT}"
CONDA_SH="${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"

source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTHONDONTWRITEBYTECODE=1

[[ ! -e "$OUT/final" ]] || { echo "Refusing to overwrite final output: $OUT/final" >&2; exit 1; }
mkdir -p "$OUT"

run_seed() {
  local seed="$1"
  local seed_num="${seed#0}"
  local checkpoint="$RUN_ROOT/gawf-seed$seed/gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
  local trajectory="$TRAJECTORIES/seed$seed/gawf_gate_trajectory.npz"
  local seed_root="$OUT/seed$seed"
  local selectivity="$SELECTIVITY_ROOT/seed$seed/part1_selectivity.npz"
  local compact="$seed_root/compact/recurrent_gate_condition_means.npz"
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; return 1; }
  [[ -f "$trajectory" ]] || { echo "Missing trajectory: $trajectory" >&2; return 1; }
  if [[ "$SELECTIVITY_ROOT" == "$OUT" && ! -f "$selectivity" ]]; then
    [[ ! -e "$seed_root/selectivity" ]] || {
      echo "Incomplete selectivity output: $seed_root/selectivity" >&2; return 1;
    }
    python -m utils.analysis.clutter.fig7_hidden_selectivity_collect \
      --ckpt "$checkpoint" --data_dir "$DATA_DIR" --output_dir "$seed_root/selectivity" \
      --device cuda --batch_size 16 --num_workers 2 --seed "$seed_num"
  elif [[ ! -f "$selectivity" ]]; then
    echo "Missing registered selectivity: $selectivity" >&2; return 1
  fi
  if [[ ! -f "$compact" ]]; then
    [[ ! -e "$seed_root/compact" ]] || {
      echo "Incomplete compact output: $seed_root/compact" >&2; return 1;
    }
    python -m utils.analysis.clutter.fig7_recurrent_gate_multiseed collect \
      --trajectory "$trajectory" --selectivity "$selectivity" \
      --output_dir "$seed_root/compact" --seed "$seed_num" --device cuda
  fi
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

mkdir -p "$OUT/final"
python -m utils.analysis.clutter.fig7_recurrent_gate_multiseed plot \
  --data_root "$OUT" --figure_dir "$OUT/final" --summary_dir "$OUT/final"
touch "$OUT/final/.complete"

for figure in \
  Fig7_recurrent_gate_disinhibition_poster_delta_10seed.pdf \
  Fig7_recurrent_gate_disinhibition_poster_delta_vertical_10seed.pdf \
  Supple3_rec_gate_sign_vs_mag_disinh_digit_zoom_10seed.png \
  Supple3_rec_gate_sign_vs_mag_disinh_digit_delta_zoom_10seed.png \
  Supple3_rec_gate_sign_vs_mag_disinh_sector_zoom_10seed.png \
  Supple3_rec_gate_sign_vs_mag_disinh_sector_delta_zoom_10seed.png; do
  cp "$OUT/final/$figure" "$RESULTS/save/$figure"
done
