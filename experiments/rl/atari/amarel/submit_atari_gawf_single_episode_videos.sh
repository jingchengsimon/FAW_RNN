#!/usr/bin/env bash
# Submit one greedy evaluation episode for the best strict fs4/stack4 GaWF runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNNER="$SCRIPT_DIR/run_atari_gawf_single_episode_video_array.sh"
RESULTS_ROOT="${AIM3_RESULTS_PATH:-/scratch/js3269/results}"
DRY_RUN=false
TASKS="0-3"
EVAL_SEED=20260727
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    *) echo "Usage: $0 [--tasks <0|1|2|3|0-3>] [--eval-seed <int>] [--dry-run]" >&2; exit 2 ;;
  esac
done

[[ "$TASKS" =~ ^[0-3](-[0-3])?$ ]] || {
  echo "--tasks must select one contiguous subset of task IDs 0 through 3" >&2
  exit 2
}

[[ -f "$RUNNER" ]] || { echo "Missing runner: $RUNNER" >&2; exit 2; }
mkdir -p "$ROOT/experiments/rl/atari/amarel/artifacts/atari_gawf_single_episode_videos"

EXPORT_VARS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$RESULTS_ROOT"
EXPORT_VARS+=",AIM3_CONDA_ENV=aim3_rnn"
EXPORT_VARS+=",AIM3_CONDA_SH=/home/js3269/enter/etc/profile.d/conda.sh"
EXPORT_VARS+=",NUM_EVAL_EPISODES=1,EVAL_SEED=$EVAL_SEED,VIDEO_FPS=15"

SBATCH_ARGS=(
  --array="$TASKS"
  --export="$EXPORT_VARS"
  "$RUNNER"
)

if [[ "$DRY_RUN" == true ]]; then
  echo "sbatch ${SBATCH_ARGS[*]}"
  exit 0
fi

job_id="$(sbatch --parsable "${SBATCH_ARGS[@]}")"
echo "job_id=$job_id"
echo "tasks=0:Pong-L1-seed1,1:Pong-L2-seed1,2:Breakout-L1-seed1,3:Breakout-L2-seed2"
echo "results=$RESULTS_ROOT/train_figs/rl/atari/{pong_6action,breakout_4action}/videos"
