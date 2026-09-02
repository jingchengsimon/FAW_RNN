#!/usr/bin/env bash
# Collect and plot descriptive GaWF dynamics on the joint-balanced CM-MNIST test set.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
CONDA_SH="${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"
RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
DATA_SUFFIX="40h-float32-jointswitch-balanced-10digit-unique"
MODE="${1:-pilot}"

source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
export PYTHONDONTWRITEBYTECODE=1

case "$MODE" in
  pilot)
    BASE="$RESULTS/data/analysis/F_timing/gawf_dynamics_jointbalanced_pilot_v1"
    SEEDS=(1)
    EVENTS_PER_CELL=1
    MINIMUM_EVENTS_PER_CELL=1
    WINDOW_CANDIDATES=(10)
    ;;
  formal)
    BASE="$RESULTS/data/analysis/F_timing/gawf_dynamics_jointbalanced_10seed_v1"
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
    EVENTS_PER_CELL=10
    MINIMUM_EVENTS_PER_CELL=10
    WINDOW_CANDIDATES=(10 20 32 50)
    ;;
  *)
    echo "Usage: $0 [pilot|formal]" >&2
    exit 2
    ;;
esac

[[ ! -e "$BASE" ]] || {
  echo "Refusing to overwrite analysis output: $BASE" >&2
  exit 1
}
mkdir -p "$BASE"

checkpoint_for() {
  local seed="$1" matches
  printf -v seed '%02d' "$seed"
  shopt -s nullglob
  matches=("$RUN_ROOT/gawf-seed$seed/"*_model.pth)
  shopt -u nullglob
  (( ${#matches[@]} == 1 )) || {
    echo "Expected one GaWF checkpoint for seed $seed, found ${#matches[@]}" >&2
    return 1
  }
  printf '%s\n' "${matches[0]}"
}

run_seed() {
  local seed="$1" gpu="$2" seed_label checkpoint output
  printf -v seed_label '%02d' "$seed"
  checkpoint="$(checkpoint_for "$seed")"
  output="$BASE/gawf-seed$seed_label"
  CUDA_VISIBLE_DEVICES="$gpu" python -m utils.analysis.clutter.gawf_dynamics collect \
    --ckpt "$checkpoint" \
    --seed "$seed" \
    --data_dir "$DATA_DIR" \
    --data_suffix "$DATA_SUFFIX" \
    --save_dir "$output" \
    --device cuda \
    --sequence_length 512 \
    --chan_num 2 \
    --events_per_cell "$EVENTS_PER_CELL" \
    --minimum_events_per_cell "$MINIMUM_EVENTS_PER_CELL" \
    --spectrum_events_per_cell 1 \
    --window_candidates "${WINDOW_CANDIDATES[@]}"
}

cd "$ROOT"
if [[ "$MODE" == "pilot" ]]; then
  start_time="$(date +%s)"
  run_seed 1 0
  elapsed="$(( $(date +%s) - start_time ))"
  printf '%s\n' "$elapsed" > "$BASE/elapsed_seconds.txt"
  [[ -f "$BASE/gawf-seed01/.complete" ]]
  echo "Pilot complete in ${elapsed}s: $BASE"
  exit 0
fi

worker() {
  local gpu="$1" seed
  for seed in "${SEEDS[@]}"; do
    if (( (seed - 1) % 2 == gpu )); then
      run_seed "$seed" "$gpu"
    fi
  done
}

worker 0 &
pid0=$!
worker 1 &
pid1=$!
wait "$pid0"
wait "$pid1"

FIGURES="$BASE/figures"
python -m utils.analysis.clutter.gawf_dynamics plot \
  --input_root "$BASE" \
  --figure_dir "$FIGURES" \
  --expected_seeds 10

mkdir -p "$RESULTS/save"
for source in "$FIGURES"/*.pdf; do
  destination="$RESULTS/save/$(basename "$source")"
  [[ ! -e "$destination" ]] || {
    echo "Refusing to overwrite saved figure: $destination" >&2
    exit 1
  }
  cp "$source" "$destination"
done
touch "$BASE/.complete"
