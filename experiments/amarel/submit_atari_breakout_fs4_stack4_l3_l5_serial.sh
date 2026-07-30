#!/usr/bin/env bash
# Submit recoverable L3 -> L4 -> L5 plain Breakout sweeps, strictly serial by depth.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT"

DRY_RUN=0
SMOKE_STEPS=25000
CONCURRENCY=8
CHECKPOINT_INTERVAL_STEPS=50000
SMOKE_RUN_ID=""
L3_SMOKE_JOB_ID=""
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --smoke-steps) SMOKE_STEPS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --checkpoint-interval-steps) CHECKPOINT_INTERVAL_STEPS="$2"; shift 2 ;;
    --smoke-run-id) SMOKE_RUN_ID="$2"; shift 2 ;;
    --l3-smoke-job-id) L3_SMOKE_JOB_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

(( CONCURRENCY >= 1 && CONCURRENCY <= 8 )) || { echo "concurrency must be 1..8" >&2; exit 2; }
if [[ -n "$L3_SMOKE_JOB_ID" && ! "$L3_SMOKE_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "--l3-smoke-job-id must be a numeric Slurm job ID" >&2
  exit 2
fi
LAYERS=(3 4 5)
MODELS=(ann rnn gru lstm gawf)
SEEDS=(1 2 3)
TASKS=$((${#MODELS[@]} * ${#SEEDS[@]}))
if (( DRY_RUN )); then
  echo "protocol: ALE/Breakout-v5, minimal 4-action, fs4/stack4, plain only, 3M steps"
  echo "layers: ${LAYERS[*]}, strictly serial; models: ${MODELS[*]}; seeds: ${SEEDS[*]}"
  echo "tasks per layer=$TASKS; formal array=0-$((TASKS - 1))%$CONCURRENCY"
  echo "matching: every candidate depth matches the L1 LSTM(hidden_size=512) core"
  echo "recovery: mmap replay, 50k-step checkpoint, --requeue; every newly submitted smoke has a unique result suffix"
  [[ -n "$L3_SMOKE_JOB_ID" ]] && echo "L3 formal will also require passed smoke job $L3_SMOKE_JOB_ID"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

MATCH_RUNNER="$ROOT/experiments/amarel/prepare_atari_breakout_fs4_stack4_depth_match.sh"
RUNNER="$ROOT/experiments/amarel/run_atari_breakout_fs4_stack4_depth_array.sh"
COMMON_EXPORT="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
PREVIOUS_JOB_ID=""
SMOKE_RUN_ID="${SMOKE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

for LAYER in "${LAYERS[@]}"; do
  TAG="atari_breakout_fs4_stack4_l${LAYER}"
  MATCH_DIR="$AIM3_RESULTS_PATH/atari_param_match_breakout_fs4_stack4_l${LAYER}"
  MATCH_JSON="$MATCH_DIR/atari_param_match.json"
  ART="$ROOT/experiments/amarel/artifacts/$TAG"
  MATCH_ART="$ROOT/experiments/amarel/artifacts/${TAG}_match"
  SMOKE_ART="$ROOT/experiments/amarel/artifacts/${TAG}_smoke"
  mkdir -p "$ART" "$MATCH_ART" "$SMOKE_ART"

  if [[ -n "$PREVIOUS_JOB_ID" ]]; then
    MATCH_RAW="$(sbatch --parsable --job-name="aim3-breakout-fs4s4-l${LAYER}-match" --chdir="$ROOT" \
      --output="$MATCH_ART/%j.out" --error="$MATCH_ART/%j.err" \
      --dependency="afterok:$PREVIOUS_JOB_ID" \
      --export="ALL,$COMMON_EXPORT,MATCH_DIR=$MATCH_DIR,NUM_ACTIONS=4,NUM_LAYERS=$LAYER" "$MATCH_RUNNER")"
  else
    MATCH_RAW="$(sbatch --parsable --job-name="aim3-breakout-fs4s4-l${LAYER}-match" --chdir="$ROOT" \
      --output="$MATCH_ART/%j.out" --error="$MATCH_ART/%j.err" \
      --export="ALL,$COMMON_EXPORT,MATCH_DIR=$MATCH_DIR,NUM_ACTIONS=4,NUM_LAYERS=$LAYER" "$MATCH_RUNNER")"
  fi
  MATCH_JOB_ID="${MATCH_RAW%%;*}"
  FORMAL_DEPENDENCY=""
  if (( LAYER == 3 )) && [[ -n "$L3_SMOKE_JOB_ID" ]]; then
    SMOKE_JOB_ID="$L3_SMOKE_JOB_ID"
    FORMAL_DEPENDENCY="afterok:$MATCH_JOB_ID:$SMOKE_JOB_ID"
  else
    SMOKE_RAW="$(sbatch --parsable --job-name="aim3-breakout-fs4s4-l${LAYER}-smoke" --array=4 --time=01:00:00 \
      --chdir="$ROOT" --output="$SMOKE_ART/%A_%a.out" --error="$SMOKE_ART/%A_%a.err" \
      --dependency="afterok:$MATCH_JOB_ID" --export="ALL,$COMMON_EXPORT,MATCH_JSON=$MATCH_JSON,NUM_LAYERS=$LAYER,TOTAL_TIMESTEPS=$SMOKE_STEPS,CHECKPOINT_INTERVAL_STEPS=5000,REQUIRED_GIB=1,RUN_TAG=breakout_fs4_stack4_l${LAYER}match_smoke_${SMOKE_RUN_ID},ARTIFACT_TAG=${TAG}_smoke" "$RUNNER")"
    SMOKE_JOB_ID="${SMOKE_RAW%%;*}"
    FORMAL_DEPENDENCY="afterok:$SMOKE_JOB_ID"
  fi
  FORMAL_RAW="$(sbatch --parsable --job-name="aim3-breakout-fs4s4-l${LAYER}" --array="0-$((TASKS - 1))%$CONCURRENCY" \
    --chdir="$ROOT" --output="$ART/%A_%a.out" --error="$ART/%A_%a.err" \
    --dependency="$FORMAL_DEPENDENCY" --export="ALL,$COMMON_EXPORT,MATCH_JSON=$MATCH_JSON,NUM_LAYERS=$LAYER,TOTAL_TIMESTEPS=3000000,CHECKPOINT_INTERVAL_STEPS=$CHECKPOINT_INTERVAL_STEPS,RUN_TAG=breakout_fs4_stack4_l${LAYER}match,ARTIFACT_TAG=$TAG" "$RUNNER")"
  PREVIOUS_JOB_ID="${FORMAL_RAW%%;*}"
  echo "L${LAYER}_MATCH_JOB_ID=$MATCH_JOB_ID L${LAYER}_SMOKE_JOB_ID=$SMOKE_JOB_ID L${LAYER}_FORMAL_JOB_ID=$PREVIOUS_JOB_ID"
done

echo "tasks_per_layer=$TASKS concurrency=$CONCURRENCY serial_chain=L3->L4->L5"
