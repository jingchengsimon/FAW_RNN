#!/usr/bin/env bash
# Submit the ten independent GaWF long-context feedback-ablation jobs after a successful smoke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_feedback_ablation_seq512.sh"
ARTIFACT_TAG="${AIM3_ARTIFACT_TAG:-feedback_ablation_seq512_10seed}"
ARTIFACT_DIR="$ROOT/experiments/amarel/artifacts/$ARTIFACT_TAG"
AIM3_CONDA_INIT="${AIM3_CONDA_INIT:-/home/${USER}/enter/etc/profile.d/conda.sh}"
AIM3_DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
AIM3_RESULTS_PATH="${AIM3_RESULTS_PATH:-}"
AIM3_CHECKPOINT_ROOT="${AIM3_CHECKPOINT_ROOT:-${AIM3_RESULTS_PATH:+$AIM3_RESULTS_PATH/train_data/clutter_best6_multiseed_40h_ep150}}"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'root=%s\nrun_script=%s\nresults_root=%s\ncheckpoint_root=%s\nseeds=1..10\nsequence_length=512\nbatch_size=16\n' \
    "$ROOT" "$RUN_SCRIPT" "$AIM3_RESULTS_PATH" "$AIM3_CHECKPOINT_ROOT"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ -z "$AIM3_RESULTS_PATH" ]]; then
  echo "AIM3_RESULTS_PATH is required." >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this launcher on an Amarel login node." >&2
  exit 1
fi
if [[ ! -f "$AIM3_CONDA_INIT" ]]; then
  echo "Conda initialization script not found: $AIM3_CONDA_INIT" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_DIR"
submission_log="$ARTIFACT_DIR/submission_$(date +%Y%m%d_%H%M%S).log"
job_ids=()
for seed in $(seq 1 10); do
  job_id="$(sbatch --parsable \
    --job-name="aim3-fbabl-s512-s$(printf '%02d' "$seed")" \
    --output="$ARTIFACT_DIR/%j.out" \
    --error="$ARTIFACT_DIR/%j.err" \
    --export=ALL,AIM3_ROOT="$ROOT",AIM3_CONDA_INIT="$AIM3_CONDA_INIT",AIM3_CONDA_ENV=aim3_rnn,AIM3_DATA_DIR="$AIM3_DATA_DIR",AIM3_RESULTS_PATH="$AIM3_RESULTS_PATH",AIM3_CHECKPOINT_ROOT="$AIM3_CHECKPOINT_ROOT",AIM3_SEED="$seed",AIM3_SEQUENCE_LENGTH=512,AIM3_BATCH_SIZE=16 \
    "$RUN_SCRIPT")"
  job_id="${job_id%%;*}"
  job_ids+=("$job_id")
  echo "seed=$seed job_id=$job_id" | tee -a "$submission_log"
done
job_ids_csv="$(IFS=,; echo "${job_ids[*]}")"
{
  echo "timestamp=$(date -Is)"
  echo "job_ids=$job_ids_csv"
  echo "protocol=feedback_ablation_sequence_length_512_batch_size_16"
  echo "result_root=$AIM3_RESULTS_PATH/anal_data/G_behaviour/feedback_ablation_seq512_10seed"
  echo "status_command=squeue -j $job_ids_csv"
} | tee -a "$submission_log"
printf '%s\n' "$job_ids_csv"
