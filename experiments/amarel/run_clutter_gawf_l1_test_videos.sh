#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-gawf-l1-video
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/amarel/artifacts/clutter_gawf_l1_test_videos/%j.out
#SBATCH --error=experiments/amarel/artifacts/clutter_gawf_l1_test_videos/%j.err

# Render four annotated test-set videos from the formal single-layer Clutter GaWF checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must point to persistent Amarel storage}"
: "${AIM3_DATA_DIR:?AIM3_DATA_DIR must point to the Clutter stimulus root}"
CHECKPOINT="${AIM3_RESULTS_PATH}/train_data/sector_40h_adamw/"
CHECKPOINT+="gawf_sector_acc_h256_lr0.0005_wd0.0001_cdo0.0_rdo0.5_model.pth"
OUTPUT_DIR="${AIM3_RESULTS_PATH}/videos/clutter_gawf_l1_test"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing to overwrite: $OUTPUT_DIR" >&2; exit 3; }

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

python utils_viz/clutter_gawf_test_videos.py \
  --checkpoint "$CHECKPOINT" \
  --data_dir "$AIM3_DATA_DIR" \
  --data_suffix 40h-uint8 \
  --output_dir "$OUTPUT_DIR" \
  --sample_indices 0 1 2 3 \
  --device cuda \
  --fps 2 \
  --scale 4

for sample_index in 0 1 2 3; do
  [[ -s "$OUTPUT_DIR/clutter_gawf_l1_test_sample${sample_index}.mp4" ]] || exit 4
done
[[ -s "$OUTPUT_DIR/manifest.json" ]] || exit 4
