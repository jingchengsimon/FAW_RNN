#!/usr/bin/env bash
# Submit a read-only GPU-node probe of mounts and GPFS filesets.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_storage_fileset_probe.sh"
ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_storage_fileset_probe"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "read-only probe: GPU-node mounts, GPFS mount table, and DSSK scratch quota"
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
