#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-l3-diag-eval
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"; cd "$ROOT"
: "${AIM3_RESULTS_PATH:?}"
variants=(baseline double_dqn lr_decay buffer_500k buffer_2m no_feedback)
variant="${variants[${SLURM_ARRAY_TASK_ID:?}]}"
suffix="atari_dqn_breakout_fs4_stack4_l3diag_${variant}_gawf_seed2"
run_dir="$AIM3_RESULTS_PATH/train_data/$suffix"
snapshot_dir="$run_dir/diagnostic_checkpoints"
output_dir="$AIM3_RESULTS_PATH/train_data/diagnostics/$suffix"
source "${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"; conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
for step in 1400000 1600000 1800000 2000000 2500000 3000000; do
  [[ -s "$snapshot_dir/model_step${step}.pth" ]] || { echo "Missing snapshot: $step" >&2; exit 1; }
done
python -m utils.analysis.rl.atari.evaluate_dqn_checkpoints \
  --metrics_path "$run_dir/metrics.json" \
  --checkpoints "$snapshot_dir"/model_step*.pth \
  --eval_seeds 20260730 20260731 20260732 \
  --output_json "$output_dir/fixed_seed_greedy_returns.json" --device cuda --amp_dtype bfloat16
