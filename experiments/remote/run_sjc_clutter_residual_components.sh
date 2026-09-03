#!/usr/bin/env bash
# Recompute reset-excluded compact ANOVA residuals and render the three requested extra panels.

set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-$(pwd)}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
FIG4_BASE="$RESULTS/data/analysis/fig4_activation_anova_6model_10seed_residual_resetexcluded"
FIG5_BASE="$RESULTS/data/analysis/fig5_unit_gate_context_residual_resetexcluded_10seed"

[[ ! -e "$FIG4_BASE" && ! -e "$FIG5_BASE" ]] || {
  echo "Refusing to overwrite existing residual analysis output" >&2
  exit 1
}

cd "$ROOT"
source /G/anaconda3/etc/profile.d/conda.sh
conda activate aim3_rnn
set -u
shopt -s nullglob
mkdir -p "$FIG4_BASE"

MODELS=(gawf rnn lstm gru mamba s5)

run_fig4_worker() {
  local gpu="$1" task model seed seed_tag checkpoint
  for ((task=gpu; task<60; task+=2)); do
    model="${MODELS[task / 10]}"
    seed=$((task % 10 + 1))
    printf -v seed_tag '%02d' "$seed"
    checkpoint=("$RUN_ROOT/$model-seed$seed_tag/"*_model.pth)
    (( ${#checkpoint[@]} == 1 )) || {
      echo "Expected one checkpoint for $model seed $seed_tag" >&2
      return 1
    }
    CUDA_VISIBLE_DEVICES="$gpu" python -B -m utils.analysis.clutter.fig4_activation_anova collect \
      --ckpt "${checkpoint[0]}" --data_dir "$DATA_DIR" \
      --output_dir "$FIG4_BASE/$model-seed$seed_tag" --seed "$seed" --device cuda \
      --batch_size 16 --num_workers 2 --data_suffix 40h-uint8
  done
}

run_fig4_worker 0 &
pid0=$!
run_fig4_worker 1 &
pid1=$!
wait "$pid0"
wait "$pid1"

mkdir "$FIG4_BASE/final"
python -B -m utils.analysis.clutter.fig4_activation_anova plot \
  --data_root "$FIG4_BASE" --figure_dir "$FIG4_BASE/final" --with-residual

mkdir -p "$FIG5_BASE"
CUDA_VISIBLE_DEVICES=0 python -B -m utils.analysis.clutter.fig5_unit_gate_context_multiseed \
  --checkpoint_root "$RUN_ROOT" --data_dir "$DATA_DIR" --data_suffix 40h-uint8 \
  --save_json "$FIG5_BASE/unit_gate_context_variance_multiseed.json" --device cuda
mkdir "$FIG5_BASE/final"
python -B -m utils.analysis.clutter.fig5_unit_gate_context_plot \
  --report "$FIG5_BASE/unit_gate_context_variance_multiseed.json" --fig_dir "$FIG5_BASE/final" \
  --with-residual --publication_fig_dir "$RESULTS/save"

cp "$FIG4_BASE/final/activation_anova_1x2_6model_10seed_with_residual.pdf" \
  "$RESULTS/save/Fig4_activation_anova_1x2_6model_10seed_with_residual.pdf"
cp "$FIG4_BASE/final/activation_anova_1x2_6model_10seed_with_residual.png" \
  "$RESULTS/save/Fig4_activation_anova_1x2_6model_10seed_with_residual.png"
cp "$FIG5_BASE/final/03_unit_gate_marginalization_1x3_with_residual.pdf" \
  "$RESULTS/save/Fig5_unit_gate_marginalization_1x3_with_residual.pdf"
cp "$FIG5_BASE/final/03_unit_gate_marginalization_1x3_with_residual.png" \
  "$RESULTS/save/Fig5_unit_gate_marginalization_1x3_with_residual.png"
touch "$FIG4_BASE/final/.complete" "$FIG5_BASE/final/.complete"
