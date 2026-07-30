#!/usr/bin/env bash
# Submit recoverable L4/L5 plain Breakout sweeps with a shared replay budget.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT"

DRY_RUN=0
L3_FORMAL_JOB_ID=""
L4_CONCURRENCY=8
L5_CONCURRENCY=4
SMOKE_STEPS=25000
CHECKPOINT_INTERVAL_STEPS=50000
RUN_ID=""
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --l3-formal-job-id) L3_FORMAL_JOB_ID="$2"; shift 2 ;;
    --l4-concurrency) L4_CONCURRENCY="$2"; shift 2 ;;
    --l5-concurrency) L5_CONCURRENCY="$2"; shift 2 ;;
    --smoke-steps) SMOKE_STEPS="$2"; shift 2 ;;
    --checkpoint-interval-steps) CHECKPOINT_INTERVAL_STEPS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$L3_FORMAL_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "--l3-formal-job-id must be a numeric Slurm job ID" >&2
  exit 2
}
(( L4_CONCURRENCY >= 1 && L4_CONCURRENCY <= 8 )) || {
  echo "L4 concurrency must be 1..8 while L3 may still be running" >&2
  exit 2
}
(( L5_CONCURRENCY >= 1 && L4_CONCURRENCY + L5_CONCURRENCY <= 12 )) || {
  echo "L4 + L5 concurrency must be 1..12" >&2
  exit 2
}

MODELS=(ann rnn gru lstm gawf)
SEEDS=(1 2 3)
TASKS=$((${#MODELS[@]} * ${#SEEDS[@]}))
if (( DRY_RUN )); then
  echo "protocol: plain 4-action Breakout fs4/stack4, 3M steps, L4/L5 overlap"
  echo "L3 formal dependency=$L3_FORMAL_JOB_ID; L4=%$L4_CONCURRENCY now; L5=%$L5_CONCURRENCY after L3"
  echo "shared replay budget: max 12 formal tasks; tasks per layer=$TASKS"
  echo "each match/smoke is compute-node gated; smoke result suffixes are unique per run"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MATCH_RUNNER="$ROOT/experiments/rl/atari/amarel/prepare_atari_breakout_fs4_stack4_depth_match.sh"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_breakout_fs4_stack4_depth_array.sh"
COMMON_EXPORT="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"

submit_match() {
  local layer="$1"
  local dependency="$2"
  local match_dir="$3"
  local match_art="$4"
  if [[ -n "$dependency" ]]; then
    sbatch --parsable --job-name="aim3-breakout-fs4s4-l${layer}-match" --chdir="$ROOT" \
      --output="$match_art/%j.out" --error="$match_art/%j.err" "$dependency" \
      --export="ALL,$COMMON_EXPORT,MATCH_DIR=$match_dir,NUM_ACTIONS=4,NUM_LAYERS=$layer" \
      "$MATCH_RUNNER"
  else
    sbatch --parsable --job-name="aim3-breakout-fs4s4-l${layer}-match" --chdir="$ROOT" \
      --output="$match_art/%j.out" --error="$match_art/%j.err" \
      --export="ALL,$COMMON_EXPORT,MATCH_DIR=$match_dir,NUM_ACTIONS=4,NUM_LAYERS=$layer" \
      "$MATCH_RUNNER"
  fi
}

submit_smoke() {
  local layer="$1"
  local match_job_id="$2"
  local match_json="$3"
  local smoke_art="$4"
  local tag="$5"
  sbatch --parsable --job-name="aim3-breakout-fs4s4-l${layer}-smoke" --array=4 --time=01:00:00 \
    --chdir="$ROOT" --output="$smoke_art/%A_%a.out" --error="$smoke_art/%A_%a.err" \
    --dependency="afterok:$match_job_id" \
    --export="ALL,$COMMON_EXPORT,MATCH_JSON=$match_json,NUM_LAYERS=$layer,TOTAL_TIMESTEPS=$SMOKE_STEPS,CHECKPOINT_INTERVAL_STEPS=5000,REQUIRED_GIB=1,RUN_TAG=$tag,ARTIFACT_TAG=atari_breakout_fs4_stack4_l${layer}_smoke" \
    "$RUNNER"
}

submit_formal() {
  local layer="$1"
  local smoke_job_id="$2"
  local match_json="$3"
  local art="$4"
  local concurrency="$5"
  sbatch --parsable --job-name="aim3-breakout-fs4s4-l${layer}" --array="0-$((TASKS - 1))%$concurrency" \
    --chdir="$ROOT" --output="$art/%A_%a.out" --error="$art/%A_%a.err" \
    --dependency="afterok:$smoke_job_id" \
    --export="ALL,$COMMON_EXPORT,MATCH_JSON=$match_json,NUM_LAYERS=$layer,TOTAL_TIMESTEPS=3000000,CHECKPOINT_INTERVAL_STEPS=$CHECKPOINT_INTERVAL_STEPS,RUN_TAG=breakout_fs4_stack4_l${layer}match,ARTIFACT_TAG=atari_breakout_fs4_stack4_l${layer}" \
    "$RUNNER"
}

for LAYER in 4 5; do
  TAG="atari_breakout_fs4_stack4_l${LAYER}"
  MATCH_DIR="$AIM3_RESULTS_PATH/data/rl/atari/parameter_match/breakout_fs4_stack4_l${LAYER}"
  MATCH_JSON="$MATCH_DIR/atari_param_match.json"
  ART="$ROOT/experiments/rl/atari/amarel/artifacts/$TAG"
  MATCH_ART="$ROOT/experiments/rl/atari/amarel/artifacts/${TAG}_match"
  SMOKE_ART="$ROOT/experiments/rl/atari/amarel/artifacts/${TAG}_smoke"
  mkdir -p "$ART" "$MATCH_ART" "$SMOKE_ART"

  MATCH_DEPENDENCY=""
  CONCURRENCY="$L4_CONCURRENCY"
  if (( LAYER == 5 )); then
    MATCH_DEPENDENCY="--dependency=afterok:$L3_FORMAL_JOB_ID"
    CONCURRENCY="$L5_CONCURRENCY"
  fi
  MATCH_RAW="$(submit_match "$LAYER" "$MATCH_DEPENDENCY" "$MATCH_DIR" "$MATCH_ART")"
  MATCH_JOB_ID="${MATCH_RAW%%;*}"
  SMOKE_RAW="$(submit_smoke "$LAYER" "$MATCH_JOB_ID" "$MATCH_JSON" "$SMOKE_ART" "breakout_fs4_stack4_l${LAYER}match_smoke_${RUN_ID}")"
  SMOKE_JOB_ID="${SMOKE_RAW%%;*}"
  FORMAL_RAW="$(submit_formal "$LAYER" "$SMOKE_JOB_ID" "$MATCH_JSON" "$ART" "$CONCURRENCY")"
  FORMAL_JOB_ID="${FORMAL_RAW%%;*}"
  echo "L${LAYER}_MATCH_JOB_ID=$MATCH_JOB_ID L${LAYER}_SMOKE_JOB_ID=$SMOKE_JOB_ID L${LAYER}_FORMAL_JOB_ID=$FORMAL_JOB_ID concurrency=$CONCURRENCY"
done

echo "shared_formal_replay_budget=12 L4=$L4_CONCURRENCY L5=$L5_CONCURRENCY run_id=$RUN_ID"
