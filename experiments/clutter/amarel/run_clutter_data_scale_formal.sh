#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-scale
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue

# Train one formal Clutter data-scale model/seed unit on an Amarel compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export DISABLE_TQDM=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
SCALE="${AIM3_DATA_SCALE:?AIM3_DATA_SCALE is required}"
STATUS_DIR="${AIM3_STATUS_DIR:?AIM3_STATUS_DIR is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Slurm array task id is required}"

case "$SCALE" in
  4h|10h|20h|40h) ;;
  *) echo "AIM3_DATA_SCALE must be one of 4h, 10h, 20h, or 40h" >&2; exit 2 ;;
esac
(( TASK_ID >= 0 && TASK_ID < 60 )) || {
  echo "Task id must be in [0, 59]" >&2
  exit 2
}

MODELS=(rnn lstm gru gawf mamba s5)
WIDTHS=(275 80 105 256 170 256)
LRS=(0.001 0.001 0.005 0.005 0.001 0.001)
WDS=(0.00001 0.001 0.001 0.001 0.001 0.0)
MODEL_INDEX=$(( TASK_ID / 10 ))
SEED=$(( TASK_ID % 10 + 1 ))
MODEL="${MODELS[MODEL_INDEX]}"
WIDTH="${WIDTHS[MODEL_INDEX]}"
LR="${LRS[MODEL_INDEX]}"
WD="${WDS[MODEL_INDEX]}"
printf -v SEED_TAG '%02d' "$SEED"

RESULT_SUFFIX="data_scale/clutter_formal_4scale_ep150/$SCALE/$MODEL-seed$SEED_TAG"
RESULT_DIR="$RESULTS/data/clutter/runs/$RESULT_SUFFIX"
mkdir -p "$STATUS_DIR"
DONE_FILE="$STATUS_DIR/task_${TASK_ID}.done"
FAIL_FILE="$STATUS_DIR/task_${TASK_ID}.fail"
RUNNING_FILE="$STATUS_DIR/task_${TASK_ID}.running"

if compgen -G "$RESULT_DIR/*_metrics.json" >/dev/null \
  || compgen -G "$RESULT_DIR/*_model.pth" >/dev/null \
  || compgen -G "$RESULT_DIR/*.pkl" >/dev/null; then
  echo "Refusing to overwrite completed artifacts in $RESULT_DIR" >&2
  exit 1
fi

on_error() {
  status=$?
  trap - ERR
  printf 'status=failed task=%s scale=%s model=%s seed=%s exit=%s timestamp=%s\n' \
    "$TASK_ID" "$SCALE" "$MODEL" "$SEED" "$status" "$(date -Is)" > "$FAIL_FILE"
  exit "$status"
}
trap on_error ERR
printf 'status=running task=%s scale=%s model=%s seed=%s timestamp=%s\n' \
  "$TASK_ID" "$SCALE" "$MODEL" "$SEED" "$(date -Is)" > "$RUNNING_FILE"

WIDTH_ARGS=(--hidden_sizes "$WIDTH")
if [[ "$MODEL" == "mamba" ]]; then
  WIDTH_ARGS=(--mamba_d_models "$WIDTH")
elif [[ "$MODEL" == "s5" ]]; then
  WIDTH_ARGS=(--s5_d_models "$WIDTH" --s5_state_sizes 128)
fi

cd "$ROOT"
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

python -B run_task.py clutter \
  --model_types "$MODEL" "${WIDTH_ARGS[@]}" \
  --num_layers 1 --num_epochs 150 --patience 0 \
  --lrs "$LR" --wds "$WD" --optim adamw \
  --cnn_dropout 0.0 --rnn_dropout 0.5 \
  --gawf_feedback_lr_scale 1.0 \
  --s5_num_layers 1 --s5_dropout 0.0 --s5_ssm_lr_scale 0.1 \
  --seed "$SEED" --use_acceleration --use_sector_mode --use_mmap --chan_num 2 \
  --data_dir "$DATA_DIR" --results_dir "$RESULTS" \
  --data_suffix "$SCALE-uint8" --eval_data_suffix 40h-uint8 \
  --input_cast_mode device --frame_layout compact --shuffle_block_size -1 \
  --checkpoint_interval_epochs 5 --auto_resume \
  --result_suffix "$RESULT_SUFFIX"

printf 'status=done task=%s scale=%s model=%s seed=%s timestamp=%s\n' \
  "$TASK_ID" "$SCALE" "$MODEL" "$SEED" "$(date -Is)" > "$DONE_FILE"
trap - ERR

