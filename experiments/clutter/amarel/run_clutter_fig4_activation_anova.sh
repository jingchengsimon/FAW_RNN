#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-fig4-act-anova
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Compute one model/seed activation ANOVA without retaining raw activation arrays.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
BASE="${AIM3_FIG4_ACTIVATION_ANOVA_BASE:?AIM3_FIG4_ACTIVATION_ANOVA_BASE is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Slurm array task id is required}"
MODELS=(gawf rnn lstm gru mamba s5)
(( TASK_ID >= 0 && TASK_ID < 60 )) || { echo "Task id must be in [0, 59]" >&2; exit 2; }
MODEL="${MODELS[TASK_ID / 10]}"
SEED=$(( TASK_ID % 10 + 1 ))
printf -v SEED_TAG '%02d' "$SEED"
RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
shopt -s nullglob
CHECKPOINTS=("$RUN_ROOT/$MODEL-seed$SEED_TAG/"*_model.pth)
(( ${#CHECKPOINTS[@]} == 1 )) || {
  echo "Expected exactly one checkpoint for $MODEL seed $SEED_TAG, found ${#CHECKPOINTS[@]}" >&2
  exit 1
}

cd "$ROOT"
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

mkdir -p "$BASE"
python -m utils.analysis.clutter.fig4_activation_anova collect \
  --ckpt "${CHECKPOINTS[0]}" --data_dir "$DATA_DIR" --output_dir "$BASE/$MODEL-seed$SEED_TAG" \
  --seed "$SEED" --device cuda --batch_size 16 --num_workers 2 --data_suffix 40h-uint8
