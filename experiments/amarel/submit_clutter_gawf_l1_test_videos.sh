#!/usr/bin/env bash
# Submit annotated test-dataset Clutter videos for the formal single-layer GaWF checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNNER="$SCRIPT_DIR/run_clutter_gawf_l1_test_videos.sh"
RESULTS_ROOT="${AIM3_RESULTS_PATH:-/scratch/js3269/results}"
DATA_ROOT="${AIM3_DATA_DIR:-/scratch/js3269/stimuli}"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

[[ -f "$RUNNER" ]] || { echo "Missing runner: $RUNNER" >&2; exit 2; }
mkdir -p "$ROOT/experiments/amarel/artifacts/clutter_gawf_l1_test_videos"
EXPORT_VARS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$RESULTS_ROOT,AIM3_DATA_DIR=$DATA_ROOT"
EXPORT_VARS+=",AIM3_CONDA_ENV=aim3_rnn,AIM3_CONDA_SH=/home/js3269/enter/etc/profile.d/conda.sh"

if [[ "$DRY_RUN" == true ]]; then
  echo "sbatch --export=$EXPORT_VARS $RUNNER"
  exit 0
fi

CHECKPOINT_RELATIVE="train_data/sector_40h_adamw/"
CHECKPOINT_RELATIVE+="gawf_sector_acc_h256_lr0.0005_wd0.0001_cdo0.0_rdo0.5_model.pth"
job_id="$(sbatch --parsable --export="$EXPORT_VARS" "$RUNNER")"
echo "job_id=$job_id"
echo "checkpoint=${RESULTS_ROOT}/${CHECKPOINT_RELATIVE}"
echo "results=${RESULTS_ROOT}/videos/clutter_gawf_l1_test"
