#!/usr/bin/env bash
# Submit representative full-18 GaWF/LSTM Atari videos after fixed-suite seed selection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
RUNNER="$SCRIPT_DIR/run_atari_5task_18action_videos.sh"

usage() {
  echo "Usage: $0 [--dry-run]" >&2
}

dry_run=0
if [[ $# -gt 0 ]]; then
  [[ $# -eq 1 && $1 == "--dry-run" ]] || { usage; exit 2; }
  dry_run=1
fi

[[ -f "$ROOT/run_task.py" ]] || { echo "Missing project root: $ROOT" >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo "Missing runner: $RUNNER" >&2; exit 2; }
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must point to persistent Amarel storage}"

if [[ $dry_run -eq 1 ]]; then
  printf 'sbatch --chdir=%q --export=AIM3_RESULTS_PATH=%q,AIM3_ROOT=%q %q\n' \
    "$ROOT" "$AIM3_RESULTS_PATH" "$ROOT" "$RUNNER"
  exit 0
fi

mkdir -p "$ROOT/experiments/rl/atari/amarel/artifacts/atari_5task_18action_videos"
sbatch --chdir="$ROOT" --export="AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_ROOT=$ROOT" "$RUNNER"
