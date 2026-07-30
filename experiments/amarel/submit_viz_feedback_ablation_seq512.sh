#!/usr/bin/env bash
# Submit the standalone long-context feedback-ablation figure rendering job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_viz_feedback_ablation_seq512.sh"
ARTIFACT_DIR="$ROOT/experiments/amarel/artifacts/viz_feedback_ablation_seq512"
AIM3_CONDA_INIT="${AIM3_CONDA_INIT:-/home/${USER}/enter/etc/profile.d/conda.sh}"
AIM3_RESULTS_PATH="${AIM3_RESULTS_PATH:-}"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'root=%s\nrun_script=%s\nresults_root=%s\noutput=anal_figs/G_behaviour/fig_ablation_shuffle_standalone_seq512_yticks.{png,pdf}\n' \
    "$ROOT" "$RUN_SCRIPT" "$AIM3_RESULTS_PATH"
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
  --export=ALL,AIM3_ROOT="$ROOT",AIM3_CONDA_INIT="$AIM3_CONDA_INIT",AIM3_CONDA_ENV=aim3_rnn,AIM3_RESULTS_PATH="$AIM3_RESULTS_PATH" \
  "$RUN_SCRIPT"
