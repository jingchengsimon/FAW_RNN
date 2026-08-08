#!/usr/bin/env bash
# Submit the isolated ten-seed Clutter Figure 3/4/6/7 analysis and its afterok aggregator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DRY_RUN=0
ARRAY_SPEC="1-10%1"
OUTPUT_NAME="multiseed_figs_1_10_quota_safe"
SUBMIT_AGGREGATE=1
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --array) ARRAY_SPEC="${2:?--array requires a Slurm array specification}"; shift 2 ;;
    --output-name) OUTPUT_NAME="${2:?--output-name requires a leaf name}"; shift 2 ;;
    --no-aggregate) SUBMIT_AGGREGATE=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$OUTPUT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "--output-name must be a simple output leaf name" >&2
  exit 2
fi
if [[ ! "$ARRAY_SPEC" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*(%[1-9][0-9]*)?$ ]]; then
  echo "--array must be a numeric Slurm array specification, optionally with a concurrency cap" >&2
  exit 2
fi
if (( DRY_RUN )); then
  echo "submit: array=${ARRAY_SPEC}, output=${OUTPUT_NAME}, aggregate=${SUBMIT_AGGREGATE}"
  exit 0
fi
: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
ART="$ROOT/experiments/clutter/amarel/artifacts/multiseed_figs_1_10"
mkdir -p "$ART"
BASE="$AIM3_RESULTS_PATH/data/analysis/$OUTPUT_NAME"
if [[ -e "$BASE" ]]; then
  echo "Refusing to overwrite existing analysis output: $BASE" >&2
  exit 1
fi
ARRAY_RUNNER="$ROOT/experiments/clutter/amarel/run_clutter_multiseed_figs_1_10.sh"
AGG_RUNNER="$ROOT/experiments/clutter/amarel/run_clutter_multiseed_figs_aggregate.sh"
ARRAY_RAW="$(sbatch --parsable --chdir="$ROOT" --array="$ARRAY_SPEC" --output="$ART/%A_%a.out" --error="$ART/%A_%a.err" \
  --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_MULTISEED_FIGS_BASE=$BASE,AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR,AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1" \
  "$ARRAY_RUNNER")"
ARRAY_ID="${ARRAY_RAW%%;*}"
echo "ARRAY_JOB_ID=$ARRAY_ID"
echo "OUTPUT_BASE=$BASE"
if (( SUBMIT_AGGREGATE )); then
  AGG_RAW="$(sbatch --parsable --chdir="$ROOT" --dependency="afterok:${ARRAY_ID}" --output="$ART/%j.aggregate.out" --error="$ART/%j.aggregate.err" \
    --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_MULTISEED_FIGS_BASE=$BASE,AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1" \
    "$AGG_RUNNER")"
  echo "AGGREGATE_JOB_ID=${AGG_RAW%%;*}"
fi
