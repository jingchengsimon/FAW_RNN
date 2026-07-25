#!/usr/bin/env bash
# Login-node-safe submitter for the GaWF efferent/afferent gate-distribution
# analysis pipeline.  All computational work is delegated to the sbatch
# runner: run_gawf_relevance_alignment_distributions.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$PROJECT_ROOT/experiments/amarel/run_gawf_relevance_alignment_distributions.sh"
RESULTS_ROOT="/scratch/js3269/results"

DRY_RUN=0
PLOT_ONLY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "${1:-}" == "--plots-only" ]]; then
  PLOT_ONLY=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run|--plots-only]" >&2
  exit 2
fi

if [[ ! -f "$RUNNER" ]]; then
  echo "Missing runner: $RUNNER" >&2
  exit 1
fi
mkdir -p "$PROJECT_ROOT/experiments/amarel/artifacts/gawf_relevance_alignment_distributions"

SELECTIVITY_NPZ="$RESULTS_ROOT/anal_data/gawf_symmetric_relevance_timing/part1_selectivity.npz"
SPLIT_REPORT_JSON="$RESULTS_ROOT/anal_data/gawf_symmetric_relevance_timing/part0_splits.json"

EXPORTS="ALL,AIM3_RESULTS_PATH=$RESULTS_ROOT,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
EXPORTS+=",AIM3_PLOT_ONLY=$PLOT_ONLY"
EXPORTS+=",SELECTIVITY_NPZ=$SELECTIVITY_NPZ,SPLIT_REPORT_JSON=$SPLIT_REPORT_JSON"

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
