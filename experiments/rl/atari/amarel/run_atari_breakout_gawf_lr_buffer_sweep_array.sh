#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-gawf-lrbuf
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# Run one recovery-enabled cell of the L3/L4 GaWF LR-decay and replay sweep.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
cd "$ROOT"
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
: "${HIDDEN_L3:?HIDDEN_L3 is required}"
: "${HIDDEN_L4:?HIDDEN_L4 is required}"

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) layer=3; buffer=1000000; seed=1 ;;
  1) layer=3; buffer=1000000; seed=3 ;;
  2) layer=3; buffer=2000000; seed=1 ;;
  3) layer=3; buffer=2000000; seed=2 ;;
  4) layer=3; buffer=2000000; seed=3 ;;
  5) layer=4; buffer=1000000; seed=1 ;;
  6) layer=4; buffer=1000000; seed=2 ;;
  7) layer=4; buffer=1000000; seed=3 ;;
  8) layer=4; buffer=2000000; seed=1 ;;
  9) layer=4; buffer=2000000; seed=2 ;;
  10) layer=4; buffer=2000000; seed=3 ;;
  *) echo "Unknown sweep task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

hidden="$HIDDEN_L3"
[[ "$layer" == 4 ]] && hidden="$HIDDEN_L4"
buffer_tag="$((buffer / 1000000))m"
parent="${SWEEP_PARENT:-$AIM3_RESULTS_PATH/train_data/diagnostics/breakout_gawf_lrdecay_buffer_sweep}"
run_tag="${RUN_TAG:-lrdecay1m}"
total_timesteps="${TOTAL_TIMESTEPS:-3000000}"
checkpoint_interval_steps="${CHECKPOINT_INTERVAL_STEPS:-50000}"
suffix="atari_dqn_breakout_fs4_stack4_l${layer}_${run_tag}_buf${buffer_tag}_gawf_seed${seed}"
run_dir="$parent/$suffix"
required_gib=27
[[ "$buffer" == 2000000 ]] && required_gib=54

source "${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
python -m experiments.rl.atari.amarel.scratch_quota_guard \
  --user "${QUOTA_USER:-js3269}" --filesystem scratch --required_gib "$required_gib" \
  --headroom_factor 2

resume=()
if [[ -f "$run_dir/checkpoint.pth" ]]; then
  resume=(--resume_from "$run_dir/checkpoint.pth")
fi

DISABLE_TQDM=1 python run_task.py atari-dqn \
  --env_id ALE/Breakout-v5 --action_space_mode minimal --model_type gawf \
  --num_layers "$layer" --hidden_size "$hidden" --frame_skip 4 --frame_stack 4 \
  --flicker_prob 0 --total_timesteps "$total_timesteps" --seq_len 16 --seed "$seed" --device cuda \
  --result_suffix "$suffix" --save_dir "$run_dir" --replay_backing mmap \
  --buffer_size "$buffer" --checkpoint_interval_steps "$checkpoint_interval_steps" \
  --learning_rate_decay_step 1000000 --learning_rate_decay_scale 0.1 \
  --amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer "${resume[@]}"
