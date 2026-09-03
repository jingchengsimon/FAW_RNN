#!/usr/bin/env bash
# Submit selected model/seed units from the formal Clutter data-scale campaign.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SCALE="40h"
ARRAY_TASKS="0,10,20,30,40,50"
MAX_CONCURRENT=6
DRY_RUN=0

while (( $# )); do
  case "$1" in
    --scale) SCALE="${2:?--scale requires 4h, 10h, 20h, or 40h}"; shift 2 ;;
    --array-tasks) ARRAY_TASKS="${2:?--array-tasks requires a task specification}"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="${2:?--max-concurrent requires an integer}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$SCALE" in
  4h|10h|20h|40h) ;;
  *) echo "--scale must be one of 4h, 10h, 20h, or 40h" >&2; exit 2 ;;
esac
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-concurrent must be a positive integer" >&2
  exit 2
}
[[ "$ARRAY_TASKS" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || {
  echo "Invalid --array-tasks specification" >&2
  exit 2
}

SEEN_IDS=","
NORMALIZED_IDS=()
IFS=',' read -r -a CHUNKS <<< "$ARRAY_TASKS"
for chunk in "${CHUNKS[@]}"; do
  if [[ "$chunk" == *-* ]]; then
    start="${chunk%-*}"
    end="${chunk#*-}"
  else
    start="$chunk"
    end="$chunk"
  fi
  (( start <= end )) || { echo "Invalid descending task range: $chunk" >&2; exit 2; }
  (( start >= 0 && end < 60 )) || {
    echo "--array-tasks indices must be within 0-59" >&2
    exit 2
  }
  for (( task_id=start; task_id<=end; task_id++ )); do
    [[ "$SEEN_IDS" != *",$task_id,"* ]] || {
      echo "Duplicate --array-tasks index: $task_id" >&2
      exit 2
    }
    SEEN_IDS+="$task_id,"
    NORMALIZED_IDS+=("$task_id")
  done
done
NORMALIZED_ARRAY_TASKS="$(IFS=,; echo "${NORMALIZED_IDS[*]}")"

MODELS=(rnn lstm gru gawf mamba s5)
if (( DRY_RUN )); then
  printf 'submit: scale=%s array=%s%%%s\n' \
    "$SCALE" "$NORMALIZED_ARRAY_TASKS" "$MAX_CONCURRENT"
  for task_id in "${NORMALIZED_IDS[@]}"; do
    model="${MODELS[task_id / 10]}"
    seed=$(( task_id % 10 + 1 ))
    printf 'task=%s scale=%s model=%s seed=%s\n' "$task_id" "$SCALE" "$model" "$seed"
  done
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
for required in \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-$SCALE-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-$SCALE-uint8.tsv" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.tsv"; do
  [[ -s "$required" ]] || { echo "Missing required dataset file: $required" >&2; exit 1; }
done

RESULT_BASE="$AIM3_RESULTS_PATH/data/clutter/runs/data_scale/clutter_formal_4scale_ep150/$SCALE"
for task_id in "${NORMALIZED_IDS[@]}"; do
  model="${MODELS[task_id / 10]}"
  seed=$(( task_id % 10 + 1 ))
  printf -v seed_tag '%02d' "$seed"
  target="$RESULT_BASE/$model-seed$seed_tag"
  [[ ! -e "$target" ]] || { echo "Refusing to overwrite existing result leaf: $target" >&2; exit 1; }
done

ARTIFACT_ROOT="$ROOT/experiments/clutter/amarel/artifacts/clutter_data_scale_formal_4scale_ep150"
STATUS_DIR="$ARTIFACT_ROOT/$SCALE/status"
mkdir -p "$STATUS_DIR"
RUNNER="$ROOT/experiments/clutter/amarel/run_clutter_data_scale_formal.sh"
EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
EXPORTS+=",AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR,AIM3_DATA_SCALE=$SCALE"
EXPORTS+=",AIM3_STATUS_DIR=$STATUS_DIR,AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1"
RAW_JOB_ID="$(sbatch --parsable --chdir="$ROOT" \
  --array="${NORMALIZED_ARRAY_TASKS}%${MAX_CONCURRENT}" \
  --output="$ARTIFACT_ROOT/$SCALE/%A_%a.out" \
  --error="$ARTIFACT_ROOT/$SCALE/%A_%a.err" \
  --export="$EXPORTS" "$RUNNER")"

printf 'ARRAY_JOB_ID=%s\n' "${RAW_JOB_ID%%;*}"
printf 'ARRAY_TASKS=%s\n' "$NORMALIZED_ARRAY_TASKS"
printf 'RESULT_BASE=%s\n' "$RESULT_BASE"
printf 'STATUS_DIR=%s\n' "$STATUS_DIR"
