#!/usr/bin/env bash
# Submit the long read-only DSSK scratch attribution probe to a named gpuk node.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_dssk_scratch_attribution_probe.sh"
ART="$ROOT/experiments/rl/atari/amarel/artifacts/atari_dssk_scratch_attribution_probe"
NODE="${1:-gpuk013}"
if [[ "$NODE" == "--dry-run" ]]; then
  echo "read-only four-hour DSSK attribution probe; default node gpuk013"
  exit 0
fi
[[ "$NODE" =~ ^gpuk[0-9]{3}$ ]] || { echo "Node must be a gpuk### name" >&2; exit 2; }
mkdir -p "$ART"
job="$(sbatch --parsable --nodelist="$NODE" --chdir="$ROOT" \
  --output="$ART/%j-${NODE}.out" --error="$ART/%j-${NODE}.err" "$RUNNER")"
echo "JOB_ID=${job%%;*} NODE=$NODE"
