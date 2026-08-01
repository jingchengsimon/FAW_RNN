#!/usr/bin/env bash
# Submit a read-only GPU-node scratch quota probe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_scratch_quota_probe.sh"
ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_scratch_quota_probe"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "probe: mmlsquota and df on one Ada Lovelace GPU compute node"
  exit 0
fi
if (( $# )); then
  echo "Unknown argument: $1" >&2
  exit 2
fi

mkdir -p "$ART"
JOB_RAW="$(sbatch --parsable --chdir="$ROOT" --output="$ART/%j.out" --error="$ART/%j.err" "$RUNNER")"
echo "JOB_ID=${JOB_RAW%%;*}"
