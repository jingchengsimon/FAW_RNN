#!/usr/bin/env bash
# Submit full18 parameter matching, five-model smoke, then a gated 3-seed pilot.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$ROOT"
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if (( DRY_RUN )); then
  echo "protocol: Pong+Breakout+Assault+Seaquest+Skiing; full18; fs4/stack4; L3"
  echo "balance: transition-balanced collection; rotating replay remainder; batch=32"
  echo "warm-up: every task >=20k valid environment steps before the first update"
  echo "smoke: 5 models x seed1; 150k total steps; 1M mmap replay"
  echo "pilot: afterok(smoke); 5 models x 3 seeds; 5M total steps; array=0-14%4"
  echo "LR: 1e-4 -> 1e-5 at 1M global steps; recurrent seq_len=16"
  echo "result root: \$AIM3_RESULTS_PATH/data/rl/atari/5task_18action"
  echo "figure root: \$AIM3_RESULTS_PATH/data/rl/atari/5task_18action/figs"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

BASE="$AIM3_RESULTS_PATH/data/rl/atari/5task_18action"
MATCH_DIR="$BASE/parameter_match/l3_full18"
MATCH_JSON="$MATCH_DIR/atari_param_match.json"
SMOKE_PARENT="$BASE/smoke"
PILOT_PARENT="$BASE/pilot"
FIGURE_ROOT="$BASE/figs"
MATCH_ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_5task_18action_l3_match"
SMOKE_ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_5task_18action_l3_smoke"
PILOT_ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_5task_18action_l3_pilot"
mkdir -p "$MATCH_ART" "$SMOKE_ART" "$PILOT_ART" "$FIGURE_ROOT"

MATCH_RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_param_match.sh"
ARRAY_RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_5task_18action_l3_array.sh"
COMMON="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
MATCH_EXPORTS="PARAM_MATCH_NUM_LAYERS=3,PARAM_MATCH_NUM_ACTIONS=18"
MATCH_EXPORTS="$MATCH_EXPORTS,PARAM_MATCH_MODELS=rnn:gru:lstm:gawf"
MATCH_EXPORTS="$MATCH_EXPORTS,PARAM_MATCH_REQUIRED=ann:rnn:gru:lstm:gawf"
MATCH_EXPORTS="$MATCH_EXPORTS,PARAM_MATCH_OUT_DIR=$MATCH_DIR"
MATCH_EXPORTS="$MATCH_EXPORTS,ARTIFACT_TAG=atari_5task_18action_l3_match"

MATCH_RAW="$(sbatch --parsable --job-name=aim3-atari-5task-l3-match --chdir="$ROOT" \
  --output="$MATCH_ART/%j.out" --error="$MATCH_ART/%j.err" \
  --export="ALL,$COMMON,$MATCH_EXPORTS" \
  "$MATCH_RUNNER")"
MATCH_JOB_ID="${MATCH_RAW%%;*}"

SMOKE_COMMON="$COMMON,MATCH_JSON=$MATCH_JSON,RESULT_PARENT=$SMOKE_PARENT"
SMOKE_COMMON="$SMOKE_COMMON,SEED_COUNT=1,RUN_PHASE=smoke,TOTAL_TIMESTEPS=150000"
SMOKE_COMMON="$SMOKE_COMMON,CHECKPOINT_INTERVAL_STEPS=25000,LR_DECAY_STEP=1000000"
SMOKE_COMMON="$SMOKE_COMMON,LEARNING_STARTS_PER_TASK=20000,REQUIRED_GIB=27"
SMOKE_COMMON="$SMOKE_COMMON,ARTIFACT_TAG=atari_5task_18action_l3_smoke"
SMOKE_RAW="$(sbatch --parsable --job-name=aim3-atari-5task-l3-smoke --array=0-4%5 \
  --time=06:00:00 --chdir="$ROOT" --output="$SMOKE_ART/%A_%a.out" \
  --error="$SMOKE_ART/%A_%a.err" --dependency="afterok:$MATCH_JOB_ID" \
  --export="ALL,$SMOKE_COMMON" "$ARRAY_RUNNER")"
SMOKE_JOB_ID="${SMOKE_RAW%%;*}"

PILOT_COMMON="$COMMON,MATCH_JSON=$MATCH_JSON,RESULT_PARENT=$PILOT_PARENT"
PILOT_COMMON="$PILOT_COMMON,SEED_COUNT=3,RUN_PHASE=pilot,TOTAL_TIMESTEPS=5000000"
PILOT_COMMON="$PILOT_COMMON,CHECKPOINT_INTERVAL_STEPS=50000,LR_DECAY_STEP=1000000"
PILOT_COMMON="$PILOT_COMMON,LEARNING_STARTS_PER_TASK=20000,REQUIRED_GIB=27"
PILOT_COMMON="$PILOT_COMMON,ARTIFACT_TAG=atari_5task_18action_l3_pilot"
PILOT_RAW="$(sbatch --parsable --job-name=aim3-atari-5task-l3-pilot --array=0-14%4 \
  --time=72:00:00 --chdir="$ROOT" --output="$PILOT_ART/%A_%a.out" \
  --error="$PILOT_ART/%A_%a.err" --dependency="afterok:$SMOKE_JOB_ID" \
  --export="ALL,$PILOT_COMMON" "$ARRAY_RUNNER")"

echo "MATCH_JOB_ID=$MATCH_JOB_ID"
echo "SMOKE_JOB_ID=$SMOKE_JOB_ID"
echo "PILOT_JOB_ID=${PILOT_RAW%%;*}"
echo "RESULT_ROOT=$BASE"
echo "FIGURE_ROOT=$FIGURE_ROOT"
