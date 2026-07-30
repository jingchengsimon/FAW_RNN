#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-fs4s4-l2
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# One recoverable unit of the strict 4-action, two-layer Breakout sweep.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/run_task.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must point to persistent Amarel storage}"
: "${MATCH_JSON:?MATCH_JSON must point to the Breakout L2 parameter-match JSON}"
[[ -f "$MATCH_JSON" ]] || { echo "Missing parameter match JSON: $MATCH_JSON" >&2; exit 2; }

FRAME_SKIP=4
FRAME_STACK=4
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-50000}"
RUN_TAG="${RUN_TAG:-breakout_fs4_stack4_l2match}"
ARTIFACT_TAG="${ARTIFACT_TAG:-atari_breakout_fs4_stack4_l2}"
[[ "$RUN_TAG" == *"fs4_stack4"* ]] || { echo "RUN_TAG must include fs4_stack4" >&2; exit 2; }

ART_ROOT="$ROOT/experiments/rl/atari/amarel/artifacts/$ARTIFACT_TAG"
STATUS_DIR="$ART_ROOT/status"
mkdir -p "$ART_ROOT" "$STATUS_DIR"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

MODELS=(ann rnn gru lstm gawf)
SEEDS=(42 1 2 3 4)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
N_TASKS=$((${#MODELS[@]} * ${#SEEDS[@]} * 2))
if (( TASK_ID < 0 || TASK_ID >= N_TASKS )); then
  echo "task $TASK_ID outside valid range 0..$((N_TASKS - 1))" >&2
  exit 2
fi
MODEL="${MODELS[$((TASK_ID % ${#MODELS[@]}))]}"
REST=$((TASK_ID / ${#MODELS[@]}))
SETTING=$((REST / ${#SEEDS[@]}))
SEED="${SEEDS[$((REST % ${#SEEDS[@]}))]}"
if (( SETTING == 0 )); then
  FLICKER_PROB=0.0
  SUFFIX="atari_dqn_${RUN_TAG}_${MODEL}_seed${SEED}"
else
  FLICKER_PROB=0.5
  SUFFIX="atari_dqn_${RUN_TAG}_flicker_${MODEL}_seed${SEED}"
fi

SIZE_ARGS=()
if [[ "$MODEL" != "ann" ]]; then
  HIDDEN="$(python - "$MATCH_JSON" "$MODEL" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    entry = json.load(handle)["matched"][sys.argv[2]]
print(entry["hidden_size"])
PY
)"
  SIZE_ARGS=(--hidden_size "$HIDDEN")
fi

ACCEL_ARGS=(--amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer)
FUSED_EXPECTED=true
COMPILE_EXPECTED=false
if [[ "$MODEL" == "ann" || "$MODEL" == "gawf" ]]; then
  ACCEL_ARGS+=(--compile_model)
  COMPILE_EXPECTED=true
fi

RESULT_DIR="$AIM3_RESULTS_PATH/data/rl/atari/runs/$SUFFIX"
DONE_FILE="$STATUS_DIR/${SUFFIX}.done"
FAIL_FILE="$STATUS_DIR/${SUFFIX}.fail"
CHECKPOINT="$RESULT_DIR/checkpoint.pth"

set +e
python -m experiments.amarel.scratch_quota_guard \
  --user "${QUOTA_USER:-js3269}" --filesystem scratch \
  --required_gib "${REQUIRED_GIB:-27}" --headroom_factor "${QUOTA_HEADROOM_FACTOR:-2}" \
  --marker_path "$STATUS_DIR/${SUFFIX}.quota"
QUOTA_RC=$?
set -e
(( QUOTA_RC == 0 )) || exit "$QUOTA_RC"

RESUME_ARGS=()
if [[ -f "$CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume_from "$CHECKPOINT")
  echo "[$(date -Is)] resuming from $CHECKPOINT"
elif [[ ! -f "$DONE_FILE" && ( -f "$RESULT_DIR/metrics_history.jsonl" || -f "$RESULT_DIR/metrics.json" ) ]]; then
  echo "Refusing partial results without a resumable checkpoint: $RESULT_DIR" >&2
  exit 3
fi

echo "[$(date -Is)] task=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
echo "protocol=4-action-minimal frame_skip=4 frame_stack=4 layers=2 flicker=$FLICKER_PROB"
set +e
DISABLE_TQDM=1 python run_task.py atari-dqn \
  --env_id ALE/Breakout-v5 --action_space_mode minimal --model_type "$MODEL" \
  --num_layers 2 --gawf_feedback_lr_scale 1.0 --frame_skip "$FRAME_SKIP" \
  --frame_stack "$FRAME_STACK" --flicker_prob "$FLICKER_PROB" \
  --total_timesteps "$TOTAL_TIMESTEPS" --seq_len 16 --seed "$SEED" --device cuda \
  --result_suffix "$SUFFIX" --save_dir "$RESULT_DIR" --replay_backing mmap \
  --checkpoint_interval_steps "$CHECKPOINT_INTERVAL_STEPS" "${RESUME_ARGS[@]}" \
  "${ACCEL_ARGS[@]}" "${SIZE_ARGS[@]}"
TRAIN_RC=$?
set -e
if (( TRAIN_RC != 0 )); then
  echo "status=train_failed task_id=$TASK_ID exit_code=$TRAIN_RC timestamp=$(date -Is)" > "$FAIL_FILE"
  exit "$TRAIN_RC"
fi

if [[ ! -f "$RESULT_DIR/metrics.json" ]]; then
  echo "status=paused task_id=$TASK_ID checkpoint=$CHECKPOINT timestamp=$(date -Is)" > "$STATUS_DIR/${SUFFIX}.paused"
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60
  fi
  exit 0
fi

python - "$RESULT_DIR" "$MODEL" "$TOTAL_TIMESTEPS" "$FUSED_EXPECTED" "$COMPILE_EXPECTED" <<'PY'
import glob
import json
import math
import os
import sys

result_dir, model_type, total_steps, fused, compiled = sys.argv[1:]
with open(os.path.join(result_dir, "metrics.json"), encoding="utf-8") as handle:
    metrics = json.load(handle)
expected = {
    "global_step": int(total_steps), "frame_skip": 4, "frame_stack": 4, "num_layers": 2,
    "model_type": model_type, "action_space_mode": "minimal", "num_actions": 4,
    "optimizer": "adam", "fused_optimizer": fused == "true", "compile_model": compiled == "true",
    "replay_backing": "mmap",
}
actual = {key: metrics.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(f"Invalid metrics: expected={expected}, actual={actual}")
history_path = os.path.join(result_dir, "metrics_history.jsonl")
if not os.path.isfile(history_path):
    raise RuntimeError(f"Missing history: {history_path}")
if len(glob.glob(os.path.join(result_dir, "*.pth"))) != 1:
    raise RuntimeError(f"Expected exactly one checkpoint in {result_dir}")
if os.path.isdir(os.path.join(result_dir, "replay")):
    raise RuntimeError(f"Replay storage was not reclaimed: {result_dir}/replay")
for key, value in metrics.items():
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite metric {key}={value}")
PY

{
  echo "status=done task_id=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
  echo "frame_skip=4 frame_stack=4 layers=2 action_space=minimal num_actions=4"
  echo "result_dir=$RESULT_DIR timestamp=$(date -Is)"
} > "$DONE_FILE"
rm -f "$FAIL_FILE" "$STATUS_DIR/${SUFFIX}.blocked" "$STATUS_DIR/${SUFFIX}.quota" "$STATUS_DIR/${SUFFIX}.paused"
echo "[$(date -Is)] done -> $RESULT_DIR"
