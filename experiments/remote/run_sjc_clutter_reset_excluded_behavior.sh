#!/usr/bin/env bash
# Recompute formal Clutter behavioral accuracy after excluding every rollout's t=0 frame.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-$(pwd)}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
TEST_BASE="$RESULTS/data/analysis/fig1_reset_excluded_test_accuracy_6model_10seed_v3"
RECOVERY_BASE="$RESULTS/data/analysis/fig1_target_switch_recovery_resetexcluded_6model_10seed_v4"
SUPPLE1_BASE="$RESULTS/data/analysis/supple1_feedback_ablation_resetexcluded_10seed_v5"
FINAL="$RESULTS/data/analysis/fig1_reset_excluded_behavior_6model_10seed_v8/final"
MODELS=(gawf rnn lstm gru mamba s5)

(( $(find "$TEST_BASE" -name reset_excluded_test_accuracy.json | wc -l) == 60 )) || {
  echo "Expected the complete v3 reset-excluded test accuracy input" >&2; exit 1;
}
(( $(find "$RECOVERY_BASE" -name 'fg_switch_offset_acc_*.npz' | wc -l) == 60 )) || {
  echo "Expected the complete v4 reset-excluded recovery input" >&2; exit 1;
}
(( $(find "$SUPPLE1_BASE" -name ablation_metrics.json | wc -l) == 10 )) || {
  echo "Expected the complete v5 reset-excluded Supplementary 1 input" >&2; exit 1;
}
[[ ! -e "$FINAL" ]] || {
  echo "Formal reset-excluded final output leaf already exists" >&2; exit 1;
}
mkdir -p "$RECOVERY_BASE" "$SUPPLE1_BASE" "$FINAL" "$RESULTS/save"

checkpoint_for() {
  local model="$1" seed="$2" matches
  printf -v seed '%02d' "$seed"
  shopt -s nullglob
  matches=("$RUN_ROOT/$model-seed$seed/"*_model.pth)
  shopt -u nullglob
  (( ${#matches[@]} == 1 )) || { echo "Expected one $model seed $seed checkpoint" >&2; return 1; }
  printf '%s\n' "${matches[0]}"
}

run_task() {
  local task="$1" gpu="$2" model seed ckpt
  if (( task < 130 )); then
    return
  elif (( task < 120 )); then
    task=$((task - 60)); model="${MODELS[task / 10]}"; seed=$((task % 10 + 1)); ckpt="$(checkpoint_for "$model" "$seed")"
    printf -v seed '%02d' "$seed"
    CUDA_VISIBLE_DEVICES="$gpu" python -m utils.analysis.clutter.fig1_target_switch_recovery \
      --ckpts "$ckpt" --save_dir "$RECOVERY_BASE/$model-seed$seed" --data_dir "$DATA_DIR" \
      --data_suffix 40h-float32-jointswitch-balanced-10digit-unique \
      --window_radius 10 --batch_size 64 --device cuda --seed "${seed#0}" --exclude_window_initial_frame
  else
    seed=$((task - 120 + 1)); ckpt="$(checkpoint_for gawf "$seed")"; printf -v seed '%02d' "$seed"
    CUDA_VISIBLE_DEVICES="$gpu" python -m utils.analysis.clutter.fig2_feedback_ablation \
      --ckpt "$ckpt" --save_dir "$SUPPLE1_BASE/gawf-seed$seed" --data_dir "$DATA_DIR" \
      --data_suffix 40h-float32-jointswitch-balanced-10digit-unique --conditions baseline clear_digit clear_sector clear_all \
      --K 10 --pre_K 10 --sequence_length 512 --batch_size 16 --device cuda --seed "${seed#0}" --exclude_window_initial_frame
  fi
}

worker() { local gpu="$1" task; for ((task=gpu; task<130; task+=2)); do run_task "$task" "$gpu"; done; }
cd "$ROOT"
worker 0 & pid0=$!
worker 1 & pid1=$!
wait "$pid0"; wait "$pid1"

python -m utils.analysis.clutter.fig1_reset_excluded_test_accuracy aggregate \
  --data_root "$TEST_BASE" --output_csv "$FINAL/reset_excluded_test_accuracy_10seed.csv"
python -m utils.analysis.clutter.fig1_multiseed_summary \
  --test_csv "$FINAL/reset_excluded_test_accuracy_10seed.csv" --train_data_dir "$RESULTS/save_data/fig1/validation_loss_histories" \
  --recovery_dir "$RECOVERY_BASE" --shuffle_anova_long_csv "$RESULTS/data/analysis/fig4_shuffle_activation_anova_10seed/final/Fig4_shuffle_activation_anova_long_10seed.csv" \
  --ablation_baseline_source ablation --output_png "$FINAL/Fig1_best6_multiseed_shuffle_2x4_seq512.png" \
  --output_pdf "$FINAL/Fig1_best6_multiseed_shuffle_2x4_seq512.pdf"
python -m utils.analysis.clutter.fig2_feedback_ablation_plot \
  --shuffle_anova_long_csv "$RESULTS/data/analysis/fig4_shuffle_activation_anova_10seed/final/Fig4_shuffle_activation_anova_long_10seed.csv" \
  --save_dir "$FINAL" --out_name Fig2_ablation_shuffle_standalone_seq512_yticks.pdf
python -m utils.analysis.clutter.supple1_feedback_ablation \
  --data_dir "$SUPPLE1_BASE" --save_dir "$FINAL" --conditions baseline clear_digit clear_sector clear_all
cp "$FINAL/Fig1_best6_multiseed_shuffle_2x4_seq512.pdf" "$RESULTS/save/Fig1_best6_multiseed_shuffle_2x4_seq512.pdf"
cp "$FINAL/Fig2_ablation_shuffle_standalone_seq512_yticks.pdf" "$RESULTS/save/Fig2_ablation_shuffle_standalone_seq512_yticks.pdf"
cp "$FINAL/fig_ablation_switch_recovery.pdf" "$RESULTS/save/Supple1_jointswitch_balanced_10digit_unique_sector_covered_prepost10_fig_ablation_switch_recovery.pdf"
touch "$FINAL/.complete"
