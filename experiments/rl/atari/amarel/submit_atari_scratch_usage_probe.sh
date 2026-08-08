#!/usr/bin/env bash
# Submit a read-only DSSK-side scratch usage probe.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_scratch_usage_probe.sh"
ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_scratch_usage_probe"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "probe: DSSK mmlsquota plus read-only largest-directory summaries for /scratch/js3269"
  exit 0
fi
(( $# == 0 )) || { echo "Unknown argument: $1" >&2; exit 2; }
mkdir -p "$ART"
job="$(sbatch --parsable --chdir="$ROOT" --output="$ART/%j.out" --error="$ART/%j.err" "$RUNNER")"
echo "JOB_ID=${job%%;*}"
