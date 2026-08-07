#!/usr/bin/env bash
# Submit isolated Amarel L3 GaWF full-scan compile benchmarks.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
: "${AIM3_SOURCE_SNAPSHOT:?Set an isolated, clean source snapshot path}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }
[[ "$AIM3_SOURCE_SNAPSHOT" == /* ]] || { echo "AIM3_SOURCE_SNAPSHOT must be absolute" >&2; exit 2; }

RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_gawf_fullscan_compile_benchmark.sh"
ARTIFACT_DIR="$AIM3_SOURCE_SNAPSHOT/experiments/rl/atari/amarel/artifacts/gawf_fullscan_compile_benchmark"
if (( DRY_RUN )); then
  echo "source snapshot: $AIM3_SOURCE_SNAPSHOT"
  echo "results: $AIM3_RESULTS_PATH/data/rl/atari/5task_18action/gawf_fullscan_compile_benchmark"
  echo "array: task0=B4,T8,warmup2,iters5; task1=B8,T16,warmup5,iters20"
  echo "runner: $RUNNER"
  exit 0
fi

[[ -x "$RUNNER" ]] || { echo "Runner is not executable: $RUNNER" >&2; exit 2; }
[[ -d "$AIM3_SOURCE_SNAPSHOT" ]] || { echo "Missing source snapshot: $AIM3_SOURCE_SNAPSHOT" >&2; exit 2; }
mkdir -p "$ARTIFACT_DIR"

RAW="$(sbatch --parsable --job-name=aim3-gawf-fullscan-bench --array=0-1 \
  --chdir="$AIM3_SOURCE_SNAPSHOT" --output="$ARTIFACT_DIR/%A_%a.out" \
  --error="$ARTIFACT_DIR/%A_%a.err" \
  --export="ALL,AIM3_SOURCE_SNAPSHOT=$AIM3_SOURCE_SNAPSHOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH" \
  "$RUNNER")"
echo "BENCHMARK_JOB_ID=${RAW%%;*}"
echo "ARTIFACT_DIR=$ARTIFACT_DIR"
