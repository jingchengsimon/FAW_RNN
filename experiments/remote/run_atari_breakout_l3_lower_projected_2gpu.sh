#!/usr/bin/env bash
# Run dz={32,64} x seeds 1..3 for the 1M-step L3 Breakout projected-feedback ablation.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS_ROOT="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
RESULT_PARENT="${RESULT_PARENT:-$RESULTS_ROOT/train_data/breakout_l3_lower_projected_1m}"
DEFAULT_MATCH_JSON="$RESULTS_ROOT/data/rl/atari/breakout_4action/parameter_match"
DEFAULT_MATCH_JSON="$DEFAULT_MATCH_JSON/atari_param_match_breakout_fs4_stack4_l3"
MATCH_JSON="${MATCH_JSON:-$DEFAULT_MATCH_JSON/atari_param_match.json}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-50000}"
GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"

[[ -d "$ROOT/.git" || -f "$ROOT/.git" ]] || { echo "Invalid AIM3_ROOT: $ROOT" >&2; exit 2; }
[[ -f "$MATCH_JSON" ]] || { echo "Missing match table: $MATCH_JSON" >&2; exit 2; }
[[ "$TOTAL_TIMESTEPS" == "1000000" ]] || {
  echo "This ablation is fixed to TOTAL_TIMESTEPS=1000000" >&2
  exit 2
}
[[ "${CONDA_DEFAULT_ENV:-}" == "aim3_rnn" ]] || {
  echo "Activate the aim3_rnn Conda environment before launching" >&2
  exit 2
}

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

LOG_DIR="$RESULT_PARENT/_logs"
STATUS_DIR="$RESULT_PARENT/_status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

HIDDEN="$(python -B - "$MATCH_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    match = json.load(handle)
entry = match["matched"]["gawf"]
if match.get("anchor") != "lstm" or match.get("anchor_num_layers") != 1:
    raise RuntimeError("Parameter-match anchor is not L1 LSTM")
if match.get("hidden_size") != 512 or match.get("candidate_num_layers") != 3:
    raise RuntimeError("Parameter-match table is not the L3/L1-LSTM-512 protocol")
if entry.get("num_layers") != 3:
    raise RuntimeError("GaWF entry is not L3")
print(entry["hidden_size"])
PY
)"

run_one() {
  local dz="$1"
  local seed="$2"
  local gpu="$3"
  local suffix="atari_dqn_breakout_fs4_stack4_l3_lrdecay1m_buf1m_lowerproj_dz${dz}_gawf_seed${seed}"
  local result_dir="$RESULT_PARENT/$suffix"
  local checkpoint="$result_dir/checkpoint.pth"
  local done_file="$STATUS_DIR/${suffix}.done"
  local fail_file="$STATUS_DIR/${suffix}.fail"
  local log_file="$LOG_DIR/${suffix}.log"
  local resume_args=()

  if [[ -f "$result_dir/metrics.json" ]]; then
    echo "[$(date -Is)] validating existing result $result_dir" >> "$log_file"
  else
    if [[ -f "$checkpoint" ]]; then
      resume_args=(--resume_from "$checkpoint")
    elif [[ -f "$result_dir/metrics_history.jsonl" ]]; then
      echo "Refusing partial result without checkpoint: $result_dir" | tee -a "$log_file" >&2
      return 3
    fi

    set +e
    CUDA_VISIBLE_DEVICES="$gpu" DISABLE_TQDM=1 python -B run_task.py atari-dqn \
      --env_id ALE/Breakout-v5 --action_space_mode minimal --model_type gawf \
      --num_layers 3 --hidden_size "$HIDDEN" --feedback_mode qvalues --dz "$dz" \
      --gawf_feedback_lr_scale 1.0 --frame_skip 4 --frame_stack 4 --flicker_prob 0.0 \
      --total_timesteps "$TOTAL_TIMESTEPS" --seq_len 16 --seed "$seed" --device cuda \
      --result_suffix "$suffix" --save_dir "$result_dir" --replay_backing mmap \
      --buffer_size 1000000 --checkpoint_interval_steps "$CHECKPOINT_INTERVAL_STEPS" \
      --learning_rate_decay_step 1000000 --learning_rate_decay_scale 0.1 \
      --amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer \
      "${resume_args[@]}" >> "$log_file" 2>&1
    local train_rc=$?
    set -e
    if (( train_rc != 0 )); then
      printf 'status=train_failed dz=%s seed=%s exit_code=%s\n' \
        "$dz" "$seed" "$train_rc" > "$fail_file"
      return "$train_rc"
    fi
  fi

  python -B - "$result_dir" "$dz" "$TOTAL_TIMESTEPS" <<'PY'
import glob
import json
import math
import os
import sys

result_dir, dz, total_steps = sys.argv[1:]
with open(os.path.join(result_dir, "metrics.json"), encoding="utf-8") as handle:
    metrics = json.load(handle)
expected = {
    "global_step": int(total_steps),
    "model_type": "gawf",
    "num_layers": 3,
    "feedback_mode": "qvalues",
    "feedback_dim": int(dz),
    "lower_feedback_projected": True,
    "frame_skip": 4,
    "frame_stack": 4,
    "flicker_prob": 0.0,
    "action_space_mode": "minimal",
    "num_actions": 4,
    "replay_backing": "mmap",
    "learning_rate_decay_step": 1_000_000,
    "learning_rate_decay_scale": 0.1,
}
actual = {key: metrics.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(f"Invalid metrics: expected={expected}, actual={actual}")
if not math.isfinite(float(metrics["loss"])):
    raise RuntimeError("Non-finite final loss")
if not os.path.isfile(os.path.join(result_dir, "metrics_history.jsonl")):
    raise RuntimeError("Missing metrics history")
if len(glob.glob(os.path.join(result_dir, "*.pth"))) != 1:
    raise RuntimeError("Expected exactly one final model checkpoint")
PY

  printf 'status=done dz=%s seed=%s result_dir=%s timestamp=%s\n' \
    "$dz" "$seed" "$result_dir" "$(date -Is)" > "$done_file"
  rm -f "$fail_file"
}

run_group() {
  local dz="$1"
  local gpu="$2"
  local seed
  for seed in 1 2 3; do
    run_one "$dz" "$seed" "$gpu" || return $?
  done
}

run_group 32 "$GPU_0" &
PID_32=$!
run_group 64 "$GPU_1" &
PID_64=$!

set +e
wait "$PID_32"
RC_32=$?
wait "$PID_64"
RC_64=$?
set -e
if (( RC_32 != 0 || RC_64 != 0 )); then
  echo "Projected-feedback sweep failed: dz32=$RC_32 dz64=$RC_64" >&2
  exit 1
fi
echo "All six projected-feedback runs completed successfully."
