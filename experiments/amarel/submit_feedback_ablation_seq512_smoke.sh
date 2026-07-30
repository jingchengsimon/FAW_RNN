#!/usr/bin/env bash
# Submit the compute-node smoke test for the 512-frame GaWF feedback-ablation protocol.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_feedback_ablation_seq512_smoke.sh"
ARTIFACT_DIR="$ROOT/experiments/amarel/artifacts/feedback_ablation_seq512_smoke"
AIM3_CONDA_INIT="${AIM3_CONDA_INIT:-/home/${USER}/enter/etc/profile.d/conda.sh}"
AIM3_DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
AIM3_RESULTS_PATH="${AIM3_RESULTS_PATH:-}"
AIM3_CHECKPOINT_ROOT="${AIM3_CHECKPOINT_ROOT:-${AIM3_RESULTS_PATH:+$AIM3_RESULTS_PATH/train_data/clutter_best6_multiseed_40h_ep150}}"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'root=%s\nrun_script=%s\nresults_root=%s\ncheckpoint_root=%s\nsequence_length=512\nbatch_size=16\nmax_batches=1\n' \
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
sbatch --parsable \
  --output="$ARTIFACT_DIR/%j.out" \
  --error="$ARTIFACT_DIR/%j.err" \
  --export=ALL,AIM3_ROOT="$ROOT",AIM3_CONDA_INIT="$AIM3_CONDA_INIT",AIM3_CONDA_ENV=aim3_rnn,AIM3_DATA_DIR="$AIM3_DATA_DIR",AIM3_RESULTS_PATH="$AIM3_RESULTS_PATH",AIM3_CHECKPOINT_ROOT="$AIM3_CHECKPOINT_ROOT" \
  "$RUN_SCRIPT"
