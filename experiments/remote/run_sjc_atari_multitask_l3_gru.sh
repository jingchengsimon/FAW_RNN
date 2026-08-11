#!/usr/bin/env bash
# Run one recoverable sjc-remote Atari GRU experiment; activation is supplied by run.sh.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
Usage: run_sjc_atari_multitask_l3_gru.sh --mode two-task|breakout|pong --cuda-device ID [options]

Options:
  --smoke                       Run the fixed 50,000-step smoke in an isolated result leaf.
  --phase formal|smoke           Result phase (default: formal).
  --total-timesteps N            Override the formal budget (two-task: 4M; single task: 2M).
  --run-tag TAG                  Explicit unique result-leaf name.
  --results-root PATH            Root containing data/rl/atari.
                                 Default: AIM3_RESULTS_PATH or ROOT/results.
  --dry-run                      Print the resolved command without creating files or training.
  -h, --help                     Show this help.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE=""
CUDA_DEVICE=""
PHASE="formal"
TOTAL_TIMESTEPS=""
RUN_TAG=""
RESULTS_ROOT="${AIM3_RESULTS_PATH:-${ROOT}/results}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --cuda-device) CUDA_DEVICE="${2:-}"; shift 2 ;;
    --smoke) PHASE="smoke"; shift ;;
    --phase) PHASE="${2:-}"; shift 2 ;;
    --total-timesteps) TOTAL_TIMESTEPS="${2:-}"; shift 2 ;;
    --run-tag) RUN_TAG="${2:-}"; shift 2 ;;
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in
  two-task|breakout|pong) ;;
  *) echo "--mode is required" >&2; exit 2 ;;
esac
[[ "$CUDA_DEVICE" =~ ^[0-9]+$ ]] || {
  echo "--cuda-device must be a non-negative integer" >&2
  exit 2
}
case "$PHASE" in
  formal|smoke) ;;
  *) echo "--phase must be formal or smoke" >&2; exit 2 ;;
esac
[[ -n "$RESULTS_ROOT" ]] || { echo "--results-root must not be empty" >&2; exit 2; }

if [[ "$PHASE" == "smoke" ]]; then
  if [[ -n "$TOTAL_TIMESTEPS" && "$TOTAL_TIMESTEPS" != "50000" ]]; then
    echo "A smoke run is fixed at 50000 steps" >&2
    exit 2
  fi
  TOTAL_TIMESTEPS=50000
elif [[ -z "$TOTAL_TIMESTEPS" ]]; then
  if [[ "$MODE" == "two-task" ]]; then
    TOTAL_TIMESTEPS=4000000
  else
    TOTAL_TIMESTEPS=2000000
  fi
fi
[[ "$TOTAL_TIMESTEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--total-timesteps must be positive" >&2
  exit 2
}

case "$MODE" in
  two-task)
    MODE_TAG="pong_breakout"
    ENV_ARGS=(--env_ids ALE/Pong-v5 ALE/Breakout-v5)
    ;;
  breakout) MODE_TAG="breakout_only"; ENV_ARGS=(--env_id ALE/Breakout-v5) ;;
  pong) MODE_TAG="pong_only"; ENV_ARGS=(--env_id ALE/Pong-v5) ;;
esac
BASE_TAG="fs4_stack4_l3_h458_gru_${MODE_TAG}_full18_per_task_buf1m_lrpertask1m_seed42_${PHASE}"
RUN_TAG="${RUN_TAG:-$BASE_TAG}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "--run-tag contains unsafe characters" >&2
  exit 2
}

RESULT_DIR="$RESULTS_ROOT/data/rl/atari/multitask_18action/$RUN_TAG"
ARTIFACT_DIR="$ROOT/experiments/remote/artifacts/$RUN_TAG"
STATUS_DIR="$ARTIFACT_DIR/status"
CHECKPOINT="$RESULT_DIR/checkpoint.pth"
HISTORY="$RESULT_DIR/metrics_history.jsonl"
METRICS="$RESULT_DIR/metrics.json"
DONE_FILE="$STATUS_DIR/done"
FAIL_FILE="$STATUS_DIR/fail"

RESUME_ARGS=()
if [[ -f "$CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume_from "$CHECKPOINT")
elif [[ -f "$HISTORY" || -f "$METRICS" ]]; then
  echo "Refusing existing history/metrics without resumable checkpoint: $RESULT_DIR" >&2
  exit 3
fi

COMMON_ARGS=(
  --action_space_mode full18 --model_type gru --num_layers 3 --hidden_size 458
  --frame_skip 4 --frame_stack 4 --flicker_prob 0.0 --total_timesteps "$TOTAL_TIMESTEPS"
  --learning_rate 1e-4 --learning_rate_decay_scale 0.1 --learning_starts 20000
  --replay_backing mmap --buffer_size 1000000
  --checkpoint_interval_steps 50000
  --amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer
  --batch_size 32 --seq_len 16 --sequences_per_batch 8 --seed 42 --device cuda
  --result_suffix "$RUN_TAG" --save_dir "$RESULT_DIR"
)
if [[ "$MODE" == "two-task" ]]; then
  COMMON_ARGS+=(--learning_rate_decay_step 0 --learning_rate_decay_per_task_steps 1000000
    --learning_starts_per_task 20000 --replay_layout per_task --replay_sampling task_balanced
    --task_schedule transition_balanced)
else
  COMMON_ARGS+=(--learning_rate_decay_step 1000000 --learning_rate_decay_per_task_steps 0
    --learning_starts_per_task 0)
fi
COMMAND=(python run_task.py atari-dqn "${ENV_ARGS[@]}" "${COMMON_ARGS[@]}")
if (( ${#RESUME_ARGS[@]} > 0 )); then
  COMMAND+=("${RESUME_ARGS[@]}")
fi

if [[ "$DRY_RUN" == true ]]; then
  printf 'RESULT_DIR=%q\n' "$RESULT_DIR"
  printf 'ARTIFACT_DIR=%q\n' "$ARTIFACT_DIR"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_DEVICE"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$STATUS_DIR" "$RESULT_DIR"
printf 'status=started phase=%s mode=%s total_timesteps=%s result_dir=%s timestamp=%s\n' \
  "$PHASE" "$MODE" "$TOTAL_TIMESTEPS" "$RESULT_DIR" "$(date -Is)" > "$STATUS_DIR/started"

set +e
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" DISABLE_TQDM=1 "${COMMAND[@]}"
TRAIN_RC=$?
set -e
if (( TRAIN_RC != 0 )); then
  printf 'status=train_failed mode=%s exit_code=%s timestamp=%s\n' \
    "$MODE" "$TRAIN_RC" "$(date -Is)" > "$FAIL_FILE"
  exit "$TRAIN_RC"
fi

if [[ ! -f "$METRICS" ]]; then
  printf 'status=paused checkpoint=%s timestamp=%s\n' "$CHECKPOINT" "$(date -Is)" \
    > "$STATUS_DIR/paused"
  exit 0
fi

python - "$RESULT_DIR" "$TOTAL_TIMESTEPS" <<'PY'
import glob
import json
import math
import os
import sys

result_dir, total_steps = sys.argv[1:]
with open(os.path.join(result_dir, "metrics.json"), encoding="utf-8") as handle:
    metrics = json.load(handle)
if int(metrics.get("global_step", -1)) < int(total_steps):
    raise RuntimeError("Training did not reach the requested total steps")
if not math.isfinite(float(metrics.get("loss", float("nan")))):
    raise RuntimeError("Final loss is not finite")
if not os.path.isfile(os.path.join(result_dir, "metrics_history.jsonl")):
    raise RuntimeError("Missing metrics history")
if not glob.glob(os.path.join(result_dir, "*.pth")):
    raise RuntimeError("Missing final or resumable checkpoint")
PY

printf 'status=done phase=%s mode=%s result_dir=%s timestamp=%s\n' \
  "$PHASE" "$MODE" "$RESULT_DIR" "$(date -Is)" > "$DONE_FILE"
rm -f "$FAIL_FILE" "$STATUS_DIR/paused"
