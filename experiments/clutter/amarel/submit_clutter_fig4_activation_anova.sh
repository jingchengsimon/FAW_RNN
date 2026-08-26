#!/usr/bin/env bash
# Submit the six-model, ten-seed activation ANOVA array and its afterok Figure 4 renderer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ARRAY_SPEC="0-59%4"
OUTPUT_NAME="fig4_activation_anova_6model_10seed"
DRY_RUN=0
SUBMIT_AGGREGATE=1
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --array) ARRAY_SPEC="${2:?--array requires an array specification}"; shift 2 ;;
    --output-name) OUTPUT_NAME="${2:?--output-name requires a leaf}"; shift 2 ;;
    --no-aggregate) SUBMIT_AGGREGATE=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  echo "--output-name must be a simple output leaf name" >&2; exit 2;
}
[[ "$ARRAY_SPEC" =~ ^[0-9]+-[0-9]+%[1-9][0-9]*$ ]] || {
  echo "--array must be a bounded numeric range with a positive concurrency cap" >&2; exit 2;
}
if (( DRY_RUN )); then
  printf 'submit: array=%s, output=%s, aggregate=%s\n' \
    "$ARRAY_SPEC" "$OUTPUT_NAME" "$SUBMIT_AGGREGATE"
  exit 0
fi
: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
BASE="$AIM3_RESULTS_PATH/data/analysis/$OUTPUT_NAME"
[[ ! -e "$BASE" ]] || { echo "Refusing to overwrite existing analysis output: $BASE" >&2; exit 1; }
ART="$ROOT/experiments/clutter/amarel/artifacts/$OUTPUT_NAME"
mkdir -p "$ART"
ARRAY_RUNNER="$ROOT/experiments/clutter/amarel/run_clutter_fig4_activation_anova.sh"
AGG_RUNNER="$ROOT/experiments/clutter/amarel/run_clutter_fig4_activation_anova_aggregate.sh"
ARRAY_EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
ARRAY_EXPORTS+=",AIM3_FIG4_ACTIVATION_ANOVA_BASE=$BASE,AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR"
ARRAY_EXPORTS+=",AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1"
ARRAY_RAW="$(sbatch --parsable --chdir="$ROOT" --array="$ARRAY_SPEC" \
  --output="$ART/%A_%a.out" --error="$ART/%A_%a.err" --export="$ARRAY_EXPORTS" \
  "$ARRAY_RUNNER")"
ARRAY_ID="${ARRAY_RAW%%;*}"
printf 'ARRAY_JOB_ID=%s\nOUTPUT_BASE=%s\n' "$ARRAY_ID" "$BASE"
if (( SUBMIT_AGGREGATE )); then
  AGG_EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
  AGG_EXPORTS+=",AIM3_FIG4_ACTIVATION_ANOVA_BASE=$BASE,AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1"
  AGG_RAW="$(sbatch --parsable --chdir="$ROOT" --dependency="afterok:${ARRAY_ID}" \
    --output="$ART/%j.aggregate.out" --error="$ART/%j.aggregate.err" --export="$AGG_EXPORTS" \
    "$AGG_RUNNER")"
  printf 'AGGREGATE_JOB_ID=%s\n' "${AGG_RAW%%;*}"
fi
