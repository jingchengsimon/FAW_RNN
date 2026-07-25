#!/usr/bin/env bash
# Submit the Figure-03 cross-seed unit-gate marginalization analysis to an Amarel compute node.
# All computation lives in the sbatch-launched run_*.sh; this launcher only validates and submits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$PROJECT_ROOT/experiments/amarel/run_unit_gate_marginalization_multiseed.sh"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

if [[ ! -f "$RUNNER" ]]; then
  echo "Missing runner: $RUNNER" >&2
  exit 1
fi
mkdir -p "$PROJECT_ROOT/experiments/amarel/artifacts/unit_gate_marginalization_multiseed"

EXPORTS="ALL,AIM3_CONDA_INIT=/home/js3269/enter/etc/profile.d/conda.sh"
COMMAND=(
  sbatch
  --export="$EXPORTS"
  "$RUNNER"
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
"${COMMAND[@]}"
