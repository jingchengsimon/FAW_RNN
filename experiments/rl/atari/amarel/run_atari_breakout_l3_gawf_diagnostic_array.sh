#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-l3-gawf-diag
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"; cd "$ROOT"
: "${AIM3_RESULTS_PATH:?}"; : "${HIDDEN_SIZE:?}"
variants=(baseline double_dqn lr_decay buffer_500k buffer_2m no_feedback)
variant="${variants[${SLURM_ARRAY_TASK_ID:?}]}"
extra=()
case "$variant" in
 double_dqn) extra+=(--double_dqn);;
 lr_decay) extra+=(--learning_rate_decay_step 1000000 --learning_rate_decay_scale 0.1);;
 buffer_500k) extra+=(--buffer_size 500000);;
 buffer_2m) extra+=(--buffer_size 2000000);;
 no_feedback) extra+=(--feedback_mode none);;
esac
required_gib=27
[[ "$variant" == buffer_500k ]] && required_gib=14
[[ "$variant" == buffer_2m ]] && required_gib=54
suffix="atari_dqn_breakout_fs4_stack4_l3diag_${variant}_gawf_seed2"
dir="$AIM3_RESULTS_PATH/train_data/$suffix"
source "${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"; conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
python -m experiments.rl.atari.amarel.scratch_quota_guard --user "${QUOTA_USER:-js3269}" --filesystem scratch --required_gib "$required_gib" --headroom_factor 2
resume=()
if [[ -f "$dir/checkpoint.pth" ]]; then
  resume=(--resume_from "$dir/checkpoint.pth")
fi
DISABLE_TQDM=1 python run_task.py atari-dqn --env_id ALE/Breakout-v5 --action_space_mode minimal --model_type gawf --num_layers 3 --hidden_size "$HIDDEN_SIZE" --frame_skip 4 --frame_stack 4 --flicker_prob 0 --total_timesteps 3000000 --seq_len 16 --seed 2 --device cuda --result_suffix "$suffix" --save_dir "$dir" --replay_backing mmap --checkpoint_interval_steps 50000 --diagnostic_checkpoint_steps 1400000 1600000 1800000 2000000 2500000 3000000 --amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer "${extra[@]}" "${resume[@]}"
