#!/usr/bin/env bash
# Submit a read-only DSSK quota-versus-visible-inode comparison.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_dssk_quota_gap_probe.sh"
ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_dssk_quota_gap_probe"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "read-only probe: compare DSSK quota with user-owned inodes below /scratch/js3269"
  exit 0
fi
if (( $# )); then
  echo "Unknown argument: $1" >&2
  exit 2
fi

mkdir -p "$ART"
job="$(sbatch --parsable --chdir="$ROOT" \
  --output="$ART/%j.out" --error="$ART/%j.err" "$RUNNER")"
echo "JOB_ID=${job%%;*}"
