#!/usr/bin/env bash
# Submit compute-node rendering of strict Breakout depth learning curves.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT"

DRY_RUN=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if (( DRY_RUN )); then
  echo "render: seed<N>.png plus mean_std.png under separate L3/L4 directories"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
RUNNER="$ROOT/experiments/amarel/run_atari_breakout_depth_curves.sh"
ART="$ROOT/experiments/amarel/artifacts/atari_breakout_depth_curves"
mkdir -p "$ART"
JOB_RAW="$(sbatch --parsable --chdir="$ROOT" --output="$ART/%j.out" --error="$ART/%j.err" \
  --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1" \
  "$RUNNER")"
echo "JOB_ID=${JOB_RAW%%;*}"
