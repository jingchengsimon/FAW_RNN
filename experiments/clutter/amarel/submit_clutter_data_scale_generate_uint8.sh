#!/usr/bin/env bash
# Submit MNIST preflight plus 4h/10h/20h uint8 Clutter generation jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DRY_RUN=0
RECOVER_STAGING_ARRAY=""
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --recover-staging-array)
      RECOVER_STAGING_ARRAY="${2:?--recover-staging-array requires a Slurm array job ID}"
      shift 2
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$RECOVER_STAGING_ARRAY" || "$RECOVER_STAGING_ARRAY" =~ ^[1-9][0-9]*$ ]] || {
  echo "--recover-staging-array must be a positive integer" >&2
  exit 2
}

if (( DRY_RUN )); then
  printf 'submit: preflight=%s array=0-2%%3 scales=4h,10h,20h dtype=uint8 seed=42' \
    "$([[ -n "$RECOVER_STAGING_ARRAY" ]] && echo skipped || echo MNIST)"
  [[ -z "$RECOVER_STAGING_ARRAY" ]] || printf ' recover_staging=%s' "$RECOVER_STAGING_ARRAY"
  printf '\n'
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
if [[ -n "$RECOVER_STAGING_ARRAY" ]]; then
  for task_id in 0 1 2; do
    case "$task_id" in
      0) hour=4 ;;
      1) hour=10 ;;
      2) hour=20 ;;
    esac
    stage="$AIM3_CLUTTER_DATA_DIR/.clutter_scale_generation/$RECOVER_STAGING_ARRAY/task_$task_id"
    [[ -s "$stage/stimulus_reg-train-${hour}h-uint8.npy" ]] || {
      echo "Missing recovery staging NPY: $stage" >&2
      exit 1
    }
    [[ -s "$stage/stimulus_reg-train-${hour}h-uint8.tsv" ]] || {
      echo "Missing recovery staging TSV: $stage" >&2
      exit 1
    }
  done
fi

ARTIFACT_ROOT="$ROOT/experiments/clutter/amarel/artifacts/clutter_data_scale_generate_uint8"
STATUS_DIR="$ARTIFACT_ROOT/status"
MNIST_ROOT="$AIM3_CLUTTER_DATA_DIR/mnist"
mkdir -p "$STATUS_DIR"
COMMON_EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR"
COMMON_EXPORTS+=",AIM3_MNIST_ROOT=$MNIST_ROOT,AIM3_STATUS_DIR=$STATUS_DIR"
COMMON_EXPORTS+=",AIM3_SOURCE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)"
[[ -z "$RECOVER_STAGING_ARRAY" ]] || {
  COMMON_EXPORTS+=",AIM3_STAGING_JOB_ID=$RECOVER_STAGING_ARRAY"
}

DEPENDENCY_ARGS=()
PREFLIGHT_ID="skipped"
if [[ -z "$RECOVER_STAGING_ARRAY" ]]; then
  PREFLIGHT_RAW="$(sbatch --parsable --chdir="$ROOT" \
    --output="$ARTIFACT_ROOT/%j.mnist.out" --error="$ARTIFACT_ROOT/%j.mnist.err" \
    --export="$COMMON_EXPORTS" \
    "$ROOT/experiments/clutter/amarel/run_clutter_data_scale_mnist_preflight.sh")"
  PREFLIGHT_ID="${PREFLIGHT_RAW%%;*}"
  DEPENDENCY_ARGS=(--dependency="afterok:$PREFLIGHT_ID")
fi
ARRAY_RAW="$(sbatch --parsable --chdir="$ROOT" "${DEPENDENCY_ARGS[@]}" \
  --array="0-2%3" --output="$ARTIFACT_ROOT/%A_%a.out" \
  --error="$ARTIFACT_ROOT/%A_%a.err" --export="$COMMON_EXPORTS" \
  "$ROOT/experiments/clutter/amarel/run_clutter_data_scale_generate_uint8.sh")"

printf 'PREFLIGHT_JOB_ID=%s\n' "$PREFLIGHT_ID"
printf 'ARRAY_JOB_ID=%s\n' "${ARRAY_RAW%%;*}"
printf 'DATA_DIR=%s\n' "$AIM3_CLUTTER_DATA_DIR"
