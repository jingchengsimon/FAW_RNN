#!/usr/bin/env bash
#SBATCH --job-name=aim3-gawf-single-video
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/rl/atari/amarel/artifacts/atari_gawf_single_episode_videos/%A_%a.out
#SBATCH --error=experiments/rl/atari/amarel/artifacts/atari_gawf_single_episode_videos/%A_%a.err

# Render one greedy episode for each selected, checkpoint-aligned GaWF run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_atari_dqn.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must point to persistent Amarel storage}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
case "$TASK_ID" in
  0)
    ENV_SLUG="pong_6action"; ENV_TITLE="Pong 6-action"; LAYERS=1; TRAINING_SEED=1
    RUN_SUFFIX="atari_dqn_pong_fs4_stack4_l1_gawf_seed${TRAINING_SEED}"
    ;;
  1)
    ENV_SLUG="pong_6action"; ENV_TITLE="Pong 6-action"; LAYERS=2; TRAINING_SEED=1
    RUN_SUFFIX="atari_dqn_pong_fs4_stack4_l2match_gawf_seed${TRAINING_SEED}"
    ;;
  2)
    ENV_SLUG="breakout_4action"; ENV_TITLE="Breakout 4-action"; LAYERS=1; TRAINING_SEED=1
    RUN_SUFFIX="atari_dqn_breakout_fs4_stack4_l1_gawf_seed${TRAINING_SEED}"
    ;;
  3)
    ENV_SLUG="breakout_4action"; ENV_TITLE="Breakout 4-action"; LAYERS=2; TRAINING_SEED=2
    RUN_SUFFIX="atari_dqn_breakout_fs4_stack4_l2match_gawf_seed${TRAINING_SEED}"
    ;;
  *)
    echo "Expected array task 0 through 3, got $TASK_ID" >&2
    exit 2
    ;;
esac

SOURCE_DIR="${AIM3_RESULTS_PATH}/train_data/${RUN_SUFFIX}"
METRICS_PATH="${SOURCE_DIR}/metrics.json"
OUTPUT_DIR="${AIM3_RESULTS_PATH}/train_figs/rl/atari/${ENV_SLUG}/videos/"
OUTPUT_DIR+="fs4_stack4_gawf_l${LAYERS}_seed${TRAINING_SEED}_single_episode_eval${EVAL_SEED:-20260727}"
OUTPUT_VIDEO="${OUTPUT_DIR}/${ENV_SLUG}_fs4_stack4_gawf_l${LAYERS}_seed${TRAINING_SEED}_episode.mp4"
OUTPUT_META="${OUTPUT_DIR}/metadata.json"
VIDEO_TITLE="${ENV_TITLE} | GaWF L${LAYERS} | seed ${TRAINING_SEED}"

[[ -f "$METRICS_PATH" ]] || { echo "Missing metrics: $METRICS_PATH" >&2; exit 2; }
[[ ! -e "$OUTPUT_VIDEO" && ! -e "$OUTPUT_META" ]] || {
  echo "Refusing to overwrite existing video artifacts in $OUTPUT_DIR" >&2
  exit 3
}
mkdir -p "$OUTPUT_DIR"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

python utils_anal/evaluate_atari_dqn_video.py \
  --metrics_path "$METRICS_PATH" \
  --output_path "$OUTPUT_VIDEO" \
  --metadata_path "$OUTPUT_META" \
  --num_episodes "${NUM_EVAL_EPISODES:-1}" \
  --eval_seed "${EVAL_SEED:-20260727}" \
  --fps "${VIDEO_FPS:-15}" \
  --video_title "$VIDEO_TITLE" \
  --device cuda \
  --amp_dtype bfloat16

[[ -s "$OUTPUT_VIDEO" ]] || { echo "Missing final video: $OUTPUT_VIDEO" >&2; exit 4; }
[[ -s "$OUTPUT_META" ]] || { echo "Missing video metadata: $OUTPUT_META" >&2; exit 4; }
echo "layers=$LAYERS training_seed=$TRAINING_SEED video=$OUTPUT_VIDEO"
