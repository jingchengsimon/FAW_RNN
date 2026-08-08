#!/usr/bin/env bash
# Submit a smoke-gated, recoverable L3 Breakout LR-decay array for five models and ten seeds.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$ROOT"
SKIP_SMOKE=0
FORMAL_ARRAY="0-49%10"
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --array)
      [[ $# -ge 2 ]] || { echo "--array requires START-END%MAX_CONCURRENT" >&2; exit 2; }
      FORMAL_ARRAY="$2"
      shift 2
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
DRY_RUN="${DRY_RUN:-0}"
[[ "$FORMAL_ARRAY" =~ ^[0-9]+-[0-9]+%[1-9][0-9]*$ ]] || {
  echo "Invalid --array value: $FORMAL_ARRAY" >&2
  exit 2
}
if (( DRY_RUN )); then
  echo "protocol: plain 4-action Breakout fs4/stack4; L3; 1M mmap replay; LR 1e-4 -> 1e-5 at 1M"
  echo "formal: 5 models × 10 seeds = 50 tasks, array=$FORMAL_ARRAY, 3M steps, 50k checkpoints"
  if (( SKIP_SMOKE )); then
    echo "recovery: existing checkpoint resumes; formal array uses a previously accepted smoke gate"
  else
    echo "recovery: existing checkpoint resumes; smoke: one 25k task per model before formal array"
  fi
  echo "result parent: \$AIM3_RESULTS_PATH/train_data/fs4_stack4_l3_10seed_lrdecay"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

MATCH_JSON="$AIM3_RESULTS_PATH/data/rl/atari/breakout_4action/parameter_match/atari_param_match_breakout_fs4_stack4_l3/atari_param_match.json"
[[ -f "$MATCH_JSON" ]] || { echo "Missing existing L3 match table: $MATCH_JSON" >&2; exit 2; }
RESULT_PARENT="$AIM3_RESULTS_PATH/train_data/fs4_stack4_l3_10seed_lrdecay"
SMOKE_PARENT="$AIM3_RESULTS_PATH/train_data/fs4_stack4_l3_10seed_lrdecay_smoke"
ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_breakout_fs4_stack4_l3_10seed_lrdecay"
SMOKE_ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_breakout_fs4_stack4_l3_10seed_lrdecay_smoke"
mkdir -p "$ART" "$SMOKE_ART"

COMMON="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,MATCH_JSON=$MATCH_JSON,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_breakout_l3_10seed_lrdecay_array.sh"
SMOKE_JOB_ID="accepted-existing-smoke"
if (( ! SKIP_SMOKE )); then
  SMOKE_RAW="$(sbatch --parsable --job-name=aim3-breakout-l3-10seed-lrdecay-smoke --array=0-4%5 \
    --chdir="$ROOT" --output="$SMOKE_ART/%A_%a.out" --error="$SMOKE_ART/%A_%a.err" \
    --export="ALL,$COMMON,RESULT_PARENT=$SMOKE_PARENT,SEED_COUNT=1,TOTAL_TIMESTEPS=25000,CHECKPOINT_INTERVAL_STEPS=5000,REQUIRED_GIB=27,RUN_TAG=breakout_fs4_stack4_l3_10seed_lrdecay_smoke,ARTIFACT_TAG=atari_breakout_fs4_stack4_l3_10seed_lrdecay_smoke" \
    "$RUNNER")"
  SMOKE_JOB_ID="${SMOKE_RAW%%;*}"
fi
if (( SKIP_SMOKE )); then
  FORMAL_RAW="$(sbatch --parsable --job-name=aim3-breakout-l3-10seed-lrdecay --array="$FORMAL_ARRAY" \
    --chdir="$ROOT" --output="$ART/%A_%a.out" --error="$ART/%A_%a.err" \
    --export="ALL,$COMMON,RESULT_PARENT=$RESULT_PARENT,SEED_COUNT=10,TOTAL_TIMESTEPS=3000000,CHECKPOINT_INTERVAL_STEPS=50000,REQUIRED_GIB=27,RUN_TAG=breakout_fs4_stack4_l3_10seed_lrdecay,ARTIFACT_TAG=atari_breakout_fs4_stack4_l3_10seed_lrdecay" \
    "$RUNNER")"
else
  FORMAL_RAW="$(sbatch --parsable --job-name=aim3-breakout-l3-10seed-lrdecay --array="$FORMAL_ARRAY" \
    --chdir="$ROOT" --output="$ART/%A_%a.out" --error="$ART/%A_%a.err" \
    --dependency="afterok:$SMOKE_JOB_ID" \
    --export="ALL,$COMMON,RESULT_PARENT=$RESULT_PARENT,SEED_COUNT=10,TOTAL_TIMESTEPS=3000000,CHECKPOINT_INTERVAL_STEPS=50000,REQUIRED_GIB=27,RUN_TAG=breakout_fs4_stack4_l3_10seed_lrdecay,ARTIFACT_TAG=atari_breakout_fs4_stack4_l3_10seed_lrdecay" \
    "$RUNNER")"
fi
echo "SMOKE_JOB_ID=$SMOKE_JOB_ID"
echo "FORMAL_JOB_ID=${FORMAL_RAW%%;*}"
echo "RESULT_PARENT=$RESULT_PARENT"
