#!/usr/bin/env bash
# Run the ten-seed Figure 4 activation/gate ANOVA under Figure 2 feedback shuffles on SJC.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-$(pwd)}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
BASE="$RESULTS/data/analysis/fig4_shuffle_activation_anova_10seed"
FINAL="$BASE/final"

[[ ! -e "$BASE" ]] || { echo "Refusing to overwrite analysis output: $BASE" >&2; exit 1; }
mkdir -p "$BASE"

checkpoint_for() {
  local seed="$1" matches
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
  local seed="$1" gpu="$2" checkpoint output
  printf -v seed '%02d' "$seed"
  checkpoint="$(checkpoint_for "$seed")"
  output="$BASE/seed$seed"
  CUDA_VISIBLE_DEVICES="$gpu" python -m utils.analysis.clutter.fig4_shuffle_activation_anova collect \
    --ckpt "$checkpoint" --data_dir "$DATA_DIR" --output_dir "$output" --seed "${seed#0}" \
    --device cuda --batch_size 16 --num_workers 2 --data_suffix 40h-uint8 --sequence_length 512
}

worker() {
  local gpu="$1" seed
  for ((seed=gpu + 1; seed<=10; seed+=2)); do
    run_seed "$seed" "$gpu"
  done
}

cd "$ROOT"
worker 0 &
pid0=$!
worker 1 &
pid1=$!
wait "$pid0"
wait "$pid1"

mkdir "$FINAL"
python -m utils.analysis.clutter.fig4_shuffle_activation_anova aggregate \
  --data_root "$BASE" --figure_dir "$FINAL" --expected_seeds 10
cp "$FINAL/Fig4_shuffle_activation_anova_1x3_10seed.pdf" \
  "$RESULTS/save/Fig4_shuffle_activation_anova_1x3_10seed.pdf"
touch "$FINAL/.complete"
