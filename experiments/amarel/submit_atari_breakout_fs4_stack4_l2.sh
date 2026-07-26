#!/usr/bin/env bash
# Submit strict 4-action, fs4/stack4, two-layer Breakout: 5 models x 2 settings x 5 seeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT"

DRY_RUN=0
SKIP_SMOKE=0
SMOKE_STEPS=25000
CONCURRENCY=8
CHECKPOINT_INTERVAL_STEPS=50000
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --smoke-steps) SMOKE_STEPS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --checkpoint-interval-steps) CHECKPOINT_INTERVAL_STEPS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

MODELS=(ann rnn gru lstm gawf)
SEEDS=(42 1 2 3 4)
TASKS=$((${#MODELS[@]} * ${#SEEDS[@]} * 2))
if (( CONCURRENCY > 10 )); then
  echo "Refusing concurrency=$CONCURRENCY: 27 GiB replay per task exceeds the quota plan" >&2
  exit 2
fi
if (( DRY_RUN )); then
  echo "protocol: ALE/Breakout-v5, minimal 4-action, frame_skip=4, frame_stack=4, layers=2"
  echo "models: ${MODELS[*]}; settings: plain and flicker=0.5; seeds: ${SEEDS[*]}"
  echo "tasks=$TASKS array=0-$((TASKS - 1))%$CONCURRENCY checkpoint_interval_steps=$CHECKPOINT_INTERVAL_STEPS"
  echo "recovery: mmap replay, checkpoint, --requeue; parameter match and GaWF smoke gate formal array"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

MATCH_DIR="$AIM3_RESULTS_PATH/atari_param_match_breakout_fs4_stack4_l2"
MATCH_JSON="$MATCH_DIR/atari_param_match.json"
MATCH_RUNNER="$ROOT/experiments/amarel/prepare_atari_breakout_fs4_stack4_l2_match.sh"
RUNNER="$ROOT/experiments/amarel/run_atari_breakout_fs4_stack4_l2_array.sh"
ART="$ROOT/experiments/amarel/artifacts/atari_breakout_fs4_stack4_l2"
MATCH_ART="$ROOT/experiments/amarel/artifacts/atari_breakout_fs4_stack4_l2_match"
SMOKE_ART="$ROOT/experiments/amarel/artifacts/atari_breakout_fs4_stack4_l2_smoke"
mkdir -p "$ART" "$MATCH_ART" "$SMOKE_ART"

COMMON_EXPORT="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
MATCH_RAW="$(sbatch --parsable --chdir="$ROOT" --output="$MATCH_ART/%j.out" --error="$MATCH_ART/%j.err" \
  --export="ALL,$COMMON_EXPORT,MATCH_DIR=$MATCH_DIR,NUM_ACTIONS=4" "$MATCH_RUNNER")"
MATCH_JOB_ID="${MATCH_RAW%%;*}"
DEPENDENCY=(--dependency="afterok:$MATCH_JOB_ID")
SMOKE_JOB_ID=""
if (( ! SKIP_SMOKE )); then
  SMOKE_RAW="$(sbatch --parsable --job-name=aim3-breakout-fs4s4-l2-smoke --array=4 --time=01:00:00 \
    --chdir="$ROOT" --output="$SMOKE_ART/%A_%a.out" --error="$SMOKE_ART/%A_%a.err" \
    "${DEPENDENCY[@]}" --export="ALL,$COMMON_EXPORT,MATCH_JSON=$MATCH_JSON,TOTAL_TIMESTEPS=$SMOKE_STEPS,CHECKPOINT_INTERVAL_STEPS=5000,REQUIRED_GIB=1,RUN_TAG=breakout_fs4_stack4_l2match_smoke,ARTIFACT_TAG=atari_breakout_fs4_stack4_l2_smoke" "$RUNNER")"
  SMOKE_JOB_ID="${SMOKE_RAW%%;*}"
  DEPENDENCY=(--dependency="afterok:$SMOKE_JOB_ID")
fi
FORMAL_RAW="$(sbatch --parsable --job-name=aim3-breakout-fs4s4-l2 --array="0-$((TASKS - 1))%$CONCURRENCY" \
  --chdir="$ROOT" --output="$ART/%A_%a.out" --error="$ART/%A_%a.err" "${DEPENDENCY[@]}" \
  --export="ALL,$COMMON_EXPORT,MATCH_JSON=$MATCH_JSON,TOTAL_TIMESTEPS=1000000,CHECKPOINT_INTERVAL_STEPS=$CHECKPOINT_INTERVAL_STEPS,RUN_TAG=breakout_fs4_stack4_l2match,ARTIFACT_TAG=atari_breakout_fs4_stack4_l2" "$RUNNER")"
FORMAL_JOB_ID="${FORMAL_RAW%%;*}"
echo "MATCH_JOB_ID=$MATCH_JOB_ID"
echo "SMOKE_JOB_ID=$SMOKE_JOB_ID"
echo "FORMAL_JOB_ID=$FORMAL_JOB_ID tasks=$TASKS concurrency=$CONCURRENCY"
