#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-5task-formal-10m
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# Run one recoverable formal five-task/full-18 GRU or LSTM unit.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
cd "$ROOT"
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
: "${FORMAL_BASE:?FORMAL_BASE is required}"
: "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"
: "${RUN_PHASE:?RUN_PHASE is required}"
case "$RUN_PHASE" in
  smoke|formal) ;;
  *) echo "RUN_PHASE must be smoke or formal" >&2; exit 2 ;;
esac

MODELS=(gru lstm)
HIDDEN_SIZES=(458 373)
ENV_IDS=(ALE/Pong-v5 ALE/Breakout-v5 ALE/Assault-v5 ALE/Seaquest-v5 ALE/Skiing-v5)
if [[ "$RUN_PHASE" == "smoke" ]]; then
  TASK_ID=0
else
  TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  (( TASK_ID >= 0 && TASK_ID < 6 )) || { echo "invalid formal task id: $TASK_ID" >&2; exit 2; }
fi
MODEL_INDEX=$((TASK_ID / 3))
MODEL="${MODELS[$MODEL_INDEX]}"
HIDDEN_SIZE="${HIDDEN_SIZES[$MODEL_INDEX]}"
SEED=$((TASK_ID % 3 + 1))
FORMAL_SUFFIX="atari_dqn_5task_fs4_stack4_l3_buf1m_lrdecay1m_10m_${MODEL}_seed${SEED}"

if [[ "$RUN_PHASE" == "smoke" ]]; then
  TOTAL_TIMESTEPS=500
  CHECKPOINT_INTERVAL=250
  RESULT_DIR="$FORMAL_BASE/smoke/${FORMAL_SUFFIX}_smoke"
  RESULT_LABEL="${FORMAL_SUFFIX}_smoke"
else
  TOTAL_TIMESTEPS=10000000
  CHECKPOINT_INTERVAL=50000
  RESULT_DIR="$FORMAL_BASE/$FORMAL_SUFFIX"
  RESULT_LABEL="$FORMAL_SUFFIX"
fi
CHECKPOINT="$RESULT_DIR/checkpoint.pth"
HISTORY="$RESULT_DIR/metrics_history.jsonl"
METRICS="$RESULT_DIR/metrics.json"
STATUS_DIR="$ARTIFACT_ROOT/status"
DONE_FILE="$STATUS_DIR/${RESULT_LABEL}.done"
FAIL_FILE="$STATUS_DIR/${RESULT_LABEL}.fail"
SMOKE_INTERRUPTED="$STATUS_DIR/${RESULT_LABEL}.controlled_sigusr1"
SMOKE_CHECKPOINT_OK="$STATUS_DIR/${RESULT_LABEL}.checkpoint_ok"
SMOKE_PASS_FILE="$STATUS_DIR/${RESULT_LABEL}.smoke_pass"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"
mkdir -p "$STATUS_DIR" "$RESULT_DIR"

python -m experiments.rl.atari.amarel.scratch_quota_guard \
  --user "${QUOTA_USER:-js3269}" --filesystem scratch --required_gib 140 \
  --headroom_factor 1 --marker_path "$STATUS_DIR/${RESULT_LABEL}.quota"

RESUME_ARGS=()
if [[ -f "$CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume_from "$CHECKPOINT")
  echo "[$(date -Is)] resuming from $CHECKPOINT"
elif [[ -f "$HISTORY" || -f "$METRICS" ]]; then
  echo "Refusing history or metrics without a resumable checkpoint: $RESULT_DIR" >&2
  exit 3
fi

# A prior allocation may have checkpointed and been requeued before this
# marker was persisted. Validate that checkpoint before any resumed training.
if [[ "$RUN_PHASE" == "smoke" && -f "$SMOKE_INTERRUPTED" && ! -f "$SMOKE_CHECKPOINT_OK" ]]; then
  [[ -f "$CHECKPOINT" ]] || {
    echo "controlled smoke interruption is missing its resumable checkpoint" >&2
    exit 4
  }
  python - "$CHECKPOINT" "$RESULT_DIR/replay" <<'PY'
import os
import sys

import torch

checkpoint_path, replay_dir = sys.argv[1:]
payload = torch.load(checkpoint_path, map_location="cpu")
replay = payload["replay"]
expected = {
    "replay_layout": "per_task",
    "buffer_size_per_task": 1_000_000,
    "num_envs": 1,
    "num_tasks": 5,
    "sampling_mode": "task_balanced",
}
for key, value in expected.items():
    if replay.get(key) != value:
        raise RuntimeError(f"checkpoint replay {key}={replay.get(key)!r}, expected {value!r}")
if len(replay.get("task_states", [])) != 5:
    raise RuntimeError("checkpoint is missing one or more replay partitions")
for task_id in range(5):
    task_dir = os.path.join(replay_dir, f"task_{task_id}")
    if not os.path.isfile(os.path.join(task_dir, "meta.json")):
        raise RuntimeError(f"missing mmap partition metadata: {task_dir}")
PY
  printf 'checkpoint=%s replay_layout=per_task partitions=5\n' "$CHECKPOINT" \
    > "$SMOKE_CHECKPOINT_OK"
fi

TRAIN_ARGS=(
  --env_ids "${ENV_IDS[@]}" --action_space_mode full18 --model_type "$MODEL"
  --num_layers 3 --hidden_size "$HIDDEN_SIZE" --frame_skip 4 --frame_stack 4
  --flicker_prob 0 --num_envs 1 --total_timesteps "$TOTAL_TIMESTEPS"
  --task_schedule transition_balanced --replay_sampling task_balanced --replay_layout per_task
  --buffer_size 1000000 --learning_starts 20000 --learning_starts_per_task 20000
  --batch_size 32 --seq_len 16 --sequences_per_batch 8 --learning_rate 1e-4
  --learning_rate_decay_step 0 --learning_rate_decay_per_task_steps 1000000
  --learning_rate_decay_scale 0.1 --start_epsilon 1.0 --end_epsilon 0.01
  --exploration_steps 500000 --seed "$SEED" --device cuda --result_suffix "$RESULT_LABEL"
  --save_dir "$RESULT_DIR" --replay_backing mmap --checkpoint_interval_steps "$CHECKPOINT_INTERVAL"
  --amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer
)
if [[ "$RUN_PHASE" == "smoke" ]]; then
  TRAIN_ARGS+=(--keep_replay_on_success --log_interval 100)
fi

TRAIN_RC=0
if [[ "$RUN_PHASE" == "smoke" && ! -f "$SMOKE_INTERRUPTED" && ! -f "$CHECKPOINT" ]]; then
  set +e
  DISABLE_TQDM=1 python run_task.py atari-dqn "${TRAIN_ARGS[@]}" "${RESUME_ARGS[@]}" &
  TRAIN_PID=$!
  for ((attempt=0; attempt<600; attempt++)); do
    if [[ -f "$CHECKPOINT" ]]; then
      kill -USR1 "$TRAIN_PID"
      printf 'sent_at=%s checkpoint=%s\n' "$(date -Is)" "$CHECKPOINT" > "$SMOKE_INTERRUPTED"
      break
    fi
    kill -0 "$TRAIN_PID" 2>/dev/null || break
    sleep 0.25
  done
  wait "$TRAIN_PID"
  TRAIN_RC=$?
  set -e
  [[ -f "$SMOKE_INTERRUPTED" ]] || {
    echo "500-step smoke finished before controlled SIGUSR1 checkpoint" >&2
    exit 4
  }
else
  set +e
  DISABLE_TQDM=1 python run_task.py atari-dqn "${TRAIN_ARGS[@]}" "${RESUME_ARGS[@]}"
  TRAIN_RC=$?
  set -e
fi

if (( TRAIN_RC != 0 )); then
  printf 'status=train_failed task_id=%s model=%s seed=%s exit_code=%s\n' \
    "$TASK_ID" "$MODEL" "$SEED" "$TRAIN_RC" > "$FAIL_FILE"
  exit "$TRAIN_RC"
fi

if [[ ! -f "$METRICS" ]]; then
  printf 'status=paused checkpoint=%s timestamp=%s\n' "$CHECKPOINT" "$(date -Is)" \
    > "$STATUS_DIR/${RESULT_LABEL}.paused"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi

python - "$RESULT_DIR" "$TOTAL_TIMESTEPS" "$RUN_PHASE" <<'PY'
import glob
import json
import math
import os
import sys

result_dir, total_steps, phase = sys.argv[1:]
with open(os.path.join(result_dir, "metrics.json"), encoding="utf-8") as handle:
    metrics = json.load(handle)
if int(metrics.get("global_step", -1)) != int(total_steps):
    raise RuntimeError("training did not reach exactly the requested total steps")
if not os.path.isfile(os.path.join(result_dir, "metrics_history.jsonl")):
    raise RuntimeError("missing metrics history")
if not glob.glob(os.path.join(result_dir, "*.pth")):
    raise RuntimeError("missing final model checkpoint")
if phase == "formal" and not math.isfinite(float(metrics.get("loss", float("nan")))):
    raise RuntimeError("formal final loss is not finite")
if phase == "smoke":
    if metrics.get("resume_count", 0) < 1 or not metrics.get("resumed_at_steps"):
        raise RuntimeError("smoke did not resume after its controlled SIGUSR1")
PY

if [[ "$RUN_PHASE" == "smoke" ]]; then
  [[ -f "$SMOKE_CHECKPOINT_OK" ]] || {
    echo "smoke completed without checkpoint/replay validation" >&2
    exit 4
  }
  printf 'status=accepted checkpoint=%s result_dir=%s timestamp=%s\n' \
    "$CHECKPOINT" "$RESULT_DIR" "$(date -Is)" > "$SMOKE_PASS_FILE"
else
  printf 'status=done task_id=%s model=%s seed=%s result_dir=%s timestamp=%s\n' \
    "$TASK_ID" "$MODEL" "$SEED" "$RESULT_DIR" "$(date -Is)" > "$DONE_FILE"
fi
rm -f "$FAIL_FILE" "$STATUS_DIR/${RESULT_LABEL}.quota" "$STATUS_DIR/${RESULT_LABEL}.paused"
