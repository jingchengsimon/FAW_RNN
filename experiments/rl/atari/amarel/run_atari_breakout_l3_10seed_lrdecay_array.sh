#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-l3-10seed-lrdecay
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# Run one recoverable L3 Breakout LR-decay/replay cell for a model and seed.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
cd "$ROOT"
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
: "${MATCH_JSON:?MATCH_JSON is required}"
: "${RESULT_PARENT:?RESULT_PARENT is required}"
: "${SEED_COUNT:?SEED_COUNT is required}"
[[ -f "$MATCH_JSON" ]] || { echo "Missing L3 match table: $MATCH_JSON" >&2; exit 2; }
(( SEED_COUNT >= 1 && SEED_COUNT <= 10 )) || { echo "SEED_COUNT must be 1..10" >&2; exit 2; }

MODELS=(ann rnn gru lstm gawf)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
N_TASKS=$((${#MODELS[@]} * SEED_COUNT))
if (( TASK_ID < 0 || TASK_ID >= N_TASKS )); then
  echo "task $TASK_ID outside valid range 0..$((N_TASKS - 1))" >&2
  exit 2
fi
MODEL="${MODELS[$((TASK_ID / SEED_COUNT))]}"
SEED=$((TASK_ID % SEED_COUNT + 1))
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-50000}"
RUN_TAG="${RUN_TAG:-breakout_fs4_stack4_l3_lrdecay1m_buf1m}"
ARTIFACT_TAG="${ARTIFACT_TAG:-atari_breakout_fs4_stack4_l3_10seed_lrdecay}"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

ART_ROOT="$ROOT/experiments/rl/atari/amarel/artifacts/$ARTIFACT_TAG"
STATUS_DIR="$ART_ROOT/status"
mkdir -p "$STATUS_DIR"
SUFFIX="atari_dqn_${RUN_TAG}_${MODEL}_seed${SEED}"
RESULT_DIR="$RESULT_PARENT/$SUFFIX"
CHECKPOINT="$RESULT_DIR/checkpoint.pth"
DONE_FILE="$STATUS_DIR/${SUFFIX}.done"
FAIL_FILE="$STATUS_DIR/${SUFFIX}.fail"

python -m experiments.rl.atari.amarel.scratch_quota_guard \
  --user "${QUOTA_USER:-js3269}" --filesystem scratch \
  --required_gib "${REQUIRED_GIB:-27}" --headroom_factor "${QUOTA_HEADROOM_FACTOR:-2}" \
  --marker_path "$STATUS_DIR/${SUFFIX}.quota"

HIDDEN="$(python - "$MATCH_JSON" "$MODEL" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    match = json.load(handle)
model = sys.argv[2]
entry = match["matched"][model]
if match.get("anchor") != "lstm" or match.get("anchor_num_layers") != 1:
    raise RuntimeError("Parameter-match anchor is not L1 LSTM")
if match.get("hidden_size") != 512 or match.get("candidate_num_layers") != 3:
    raise RuntimeError("Parameter-match table is not the L3/L1-LSTM-512 protocol")
if entry.get("num_layers") != 3:
    raise RuntimeError(f"Unexpected layer count for {model}")
print(entry["hidden_size"])
PY
)"

RESUME_ARGS=()
if [[ -f "$CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume_from "$CHECKPOINT")
  echo "[$(date -Is)] resuming from $CHECKPOINT"
elif [[ ! -f "$DONE_FILE" && ( -f "$RESULT_DIR/metrics_history.jsonl" || -f "$RESULT_DIR/metrics.json" ) ]]; then
  echo "Refusing partial results without a resumable checkpoint: $RESULT_DIR" >&2
  exit 3
fi

ACCEL_ARGS=(--amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer)
if [[ "$MODEL" == "ann" ]]; then
  ACCEL_ARGS+=(--compile_model)
fi

set +e
DISABLE_TQDM=1 python run_task.py atari-dqn \
  --env_id ALE/Breakout-v5 --action_space_mode minimal --model_type "$MODEL" \
  --num_layers 3 --hidden_size "$HIDDEN" --gawf_feedback_lr_scale 1.0 \
  --frame_skip 4 --frame_stack 4 --flicker_prob 0.0 --total_timesteps "$TOTAL_TIMESTEPS" \
  --seq_len 16 --seed "$SEED" --device cuda --result_suffix "$SUFFIX" --save_dir "$RESULT_DIR" \
  --replay_backing mmap --buffer_size 1000000 --checkpoint_interval_steps "$CHECKPOINT_INTERVAL_STEPS" \
  --learning_rate_decay_step 1000000 --learning_rate_decay_scale 0.1 "${RESUME_ARGS[@]}" \
  "${ACCEL_ARGS[@]}"
TRAIN_RC=$?
set -e
if (( TRAIN_RC != 0 )); then
  echo "status=train_failed task_id=$TASK_ID model=$MODEL seed=$SEED exit_code=$TRAIN_RC" > "$FAIL_FILE"
  exit "$TRAIN_RC"
fi

if [[ ! -f "$RESULT_DIR/metrics.json" ]]; then
  echo "status=paused task_id=$TASK_ID checkpoint=$CHECKPOINT" > "$STATUS_DIR/${SUFFIX}.paused"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi

python - "$RESULT_DIR" "$MODEL" "$TOTAL_TIMESTEPS" <<'PY'
import glob
import json
import os
import sys

result_dir, model, total_steps = sys.argv[1:]
with open(os.path.join(result_dir, "metrics.json"), encoding="utf-8") as handle:
    metrics = json.load(handle)
expected = {
    "global_step": int(total_steps), "model_type": model, "num_layers": 3,
    "frame_skip": 4, "frame_stack": 4, "flicker_prob": 0.0,
    "action_space_mode": "minimal", "num_actions": 4, "replay_backing": "mmap",
}
actual = {key: metrics.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(f"Invalid metrics: expected={expected}, actual={actual}")
if not os.path.isfile(os.path.join(result_dir, "metrics_history.jsonl")):
    raise RuntimeError("Missing metrics history")
if len(glob.glob.glob(os.path.join(result_dir, "*.pth"))) != 1:
    raise RuntimeError("Expected exactly one final model checkpoint")
PY

{
  echo "status=done task_id=$TASK_ID model=$MODEL seed=$SEED"
  echo "result_dir=$RESULT_DIR timestamp=$(date -Is)"
} > "$DONE_FILE"
rm -f "$FAIL_FILE" "$STATUS_DIR/${SUFFIX}.quota" "$STATUS_DIR/${SUFFIX}.paused"
