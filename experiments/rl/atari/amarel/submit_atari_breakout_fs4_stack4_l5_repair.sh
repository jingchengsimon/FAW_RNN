#!/usr/bin/env bash
# Submit only the repaired L5 Breakout dependency chain.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$ROOT"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "L5 only: parameter match -> GaWF smoke -> 15 recoverable 3M plain Breakout tasks (%4)"
  exit 0
fi
: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

layer=5
tag="atari_breakout_fs4_stack4_l5_repair"
match_dir="$AIM3_RESULTS_PATH/train_data/parameter_match/breakout_fs4_stack4_l${layer}"
match_json="$match_dir/atari_param_match.json"
art="$ROOT/experiments/rl/atari/amarel/artifacts/$tag"
match_art="$ROOT/experiments/rl/atari/amarel/artifacts/${tag}_match"
smoke_art="$ROOT/experiments/rl/atari/amarel/artifacts/${tag}_smoke"
mkdir -p "$art" "$match_art" "$smoke_art"

common="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
match_job="$(sbatch --parsable --job-name=aim3-breakout-fs4s4-l5-match-repair --chdir="$ROOT" \
  --output="$match_art/%j.out" --error="$match_art/%j.err" \
  --export="ALL,$common,MATCH_DIR=$match_dir,NUM_ACTIONS=4,NUM_LAYERS=$layer" \
  experiments/rl/atari/amarel/prepare_atari_breakout_fs4_stack4_depth_match.sh)"
match_job="${match_job%%;*}"
smoke_job="$(sbatch --parsable --job-name=aim3-breakout-fs4s4-l5-smoke-repair --array=4 --time=01:00:00 \
  --chdir="$ROOT" --output="$smoke_art/%A_%a.out" --error="$smoke_art/%A_%a.err" \
  --dependency="afterok:$match_job" \
  --export="ALL,$common,MATCH_JSON=$match_json,NUM_LAYERS=$layer,TOTAL_TIMESTEPS=25000,CHECKPOINT_INTERVAL_STEPS=5000,REQUIRED_GIB=1,RUN_TAG=breakout_fs4_stack4_l5match_smoke_repair,ARTIFACT_TAG=$tag" \
  experiments/rl/atari/amarel/run_atari_breakout_fs4_stack4_depth_array.sh)"
smoke_job="${smoke_job%%;*}"
formal_job="$(sbatch --parsable --job-name=aim3-breakout-fs4s4-l5-repair --array=0-14%4 \
  --chdir="$ROOT" --output="$art/%A_%a.out" --error="$art/%A_%a.err" \
  --dependency="afterok:$smoke_job" \
  --export="ALL,$common,MATCH_JSON=$match_json,NUM_LAYERS=$layer,TOTAL_TIMESTEPS=3000000,CHECKPOINT_INTERVAL_STEPS=50000,REQUIRED_GIB=27,RUN_TAG=breakout_fs4_stack4_l5match,ARTIFACT_TAG=$tag" \
  experiments/rl/atari/amarel/run_atari_breakout_fs4_stack4_depth_array.sh)"
echo "L5_MATCH_JOB_ID=$match_job"
echo "L5_SMOKE_JOB_ID=$smoke_job"
echo "L5_FORMAL_JOB_ID=${formal_job%%;*}"
