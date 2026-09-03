#!/usr/bin/env bash
# Submit MNIST preflight plus 4h/10h/20h uint8 Clutter generation jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if (( DRY_RUN )); then
  echo "submit: preflight=MNIST array=0-2%3 scales=4h,10h,20h dtype=uint8 seed=42"
  exit 0
fi

: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
for hour in 4 10 20; do
  for target in \
    "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-${hour}h-uint8.npy" \
    "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-${hour}h-uint8.tsv" \
    "$AIM3_CLUTTER_DATA_DIR/generation-${hour}h-uint8.json"; do
    [[ ! -e "$target" ]] || { echo "Refusing to overwrite existing target: $target" >&2; exit 1; }
  done
done

ARTIFACT_ROOT="$ROOT/experiments/clutter/amarel/artifacts/clutter_data_scale_generate_uint8"
STATUS_DIR="$ARTIFACT_ROOT/status"
MNIST_ROOT="$AIM3_CLUTTER_DATA_DIR/mnist"
mkdir -p "$STATUS_DIR"
COMMON_EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR"
COMMON_EXPORTS+=",AIM3_MNIST_ROOT=$MNIST_ROOT,AIM3_STATUS_DIR=$STATUS_DIR"

PREFLIGHT_RAW="$(sbatch --parsable --chdir="$ROOT" \
  --output="$ARTIFACT_ROOT/%j.mnist.out" --error="$ARTIFACT_ROOT/%j.mnist.err" \
  --export="$COMMON_EXPORTS" \
  "$ROOT/experiments/clutter/amarel/run_clutter_data_scale_mnist_preflight.sh")"
PREFLIGHT_ID="${PREFLIGHT_RAW%%;*}"
ARRAY_RAW="$(sbatch --parsable --chdir="$ROOT" --dependency="afterok:$PREFLIGHT_ID" \
  --array="0-2%3" --output="$ARTIFACT_ROOT/%A_%a.out" \
  --error="$ARTIFACT_ROOT/%A_%a.err" --export="$COMMON_EXPORTS" \
  "$ROOT/experiments/clutter/amarel/run_clutter_data_scale_generate_uint8.sh")"

printf 'PREFLIGHT_JOB_ID=%s\n' "$PREFLIGHT_ID"
printf 'ARRAY_JOB_ID=%s\n' "${ARRAY_RAW%%;*}"
printf 'DATA_DIR=%s\n' "$AIM3_CLUTTER_DATA_DIR"

