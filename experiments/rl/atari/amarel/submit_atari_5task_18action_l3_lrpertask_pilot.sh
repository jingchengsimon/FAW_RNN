#!/usr/bin/env bash
# Submit the five-task per-task-replay pilot without a smoke gate.

set -euo pipefail
ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

BASE="$AIM3_RESULTS_PATH/data/rl/atari/5task_18action/per_task_buf500k"
MATCH_JSON="${AIM3_MATCH_JSON:-$AIM3_RESULTS_PATH/data/rl/atari/5task_18action/parameter_match/l3_full18/atari_param_match.json}"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_5task_18action_l3_array.sh"
ARTIFACT_ROOT="$ROOT/experiments/rl/atari/amarel/artifacts/atari_5task_18action_l3_per_task_buf500k"
if (( DRY_RUN )); then
  echo "protocol: five-task full18 L3; 5M global (=about 1M/task); per-task replay=500k"
  echo "LR: decay when min_task_steps reaches 1M"
  echo "GaWF: array ordinals 0-2 map to GaWF seeds; max concurrency 5; no smoke gate"
  echo "results: $BASE"
  exit 0
fi
[[ -f "$MATCH_JSON" ]] || { echo "Missing parameter match: $MATCH_JSON" >&2; exit 2; }
mkdir -p "$ARTIFACT_ROOT/pilot"
COMMON="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,MATCH_JSON=$MATCH_JSON"
COMMON="$COMMON,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
PILOT_EXPORTS="$COMMON,RESULT_PARENT=$BASE/pilot,SEED_COUNT=3,RUN_PHASE=pilot"
PILOT_EXPORTS="$PILOT_EXPORTS,TOTAL_TIMESTEPS=5000000,CHECKPOINT_INTERVAL_STEPS=50000"
PILOT_EXPORTS="$PILOT_EXPORTS,LR_DECAY_STEP=0,LR_DECAY_PER_TASK_STEPS=1000000"
PILOT_EXPORTS="$PILOT_EXPORTS,LEARNING_STARTS_PER_TASK=20000,REQUIRED_GIB=66"
PILOT_EXPORTS="$PILOT_EXPORTS,GAWF_FIRST_SCHEDULING=1"
# Array ordinals 0-2 deterministically map to the three GaWF seeds; with a
# five-task throttle, the first active wave is GaWF x3 plus two non-GaWF units.
PILOT_RAW="$(sbatch --parsable --job-name=aim3-atari-5task-per-task-replay --array=0-14%5 \
  --time=30:00:00 --chdir="$ROOT" --output="$ARTIFACT_ROOT/pilot/%A_%a.out" \
  --error="$ARTIFACT_ROOT/pilot/%A_%a.err" \
  --export="ALL,$PILOT_EXPORTS" "$RUNNER")"
echo "PILOT_JOB_ID=${PILOT_RAW%%;*}"
echo "RESULT_ROOT=$BASE"
