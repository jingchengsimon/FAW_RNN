#!/usr/bin/env bash
#SBATCH --job-name=aim3-viz-fbabl-s512
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Render the standalone long-context feedback-shuffle figure on an Amarel compute node.

set -euo pipefail

echo "[$(date -Is)] starting long-context feedback-ablation figure render"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

if [[ -z "${AIM3_CONDA_INIT:-}" || ! -f "$AIM3_CONDA_INIT" ]]; then
  echo "AIM3_CONDA_INIT must identify the Amarel Conda initialization script." >&2
  exit 2
fi
source "$AIM3_CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
python -c 'import matplotlib; print(f"matplotlib={matplotlib.__version__}")'

RESULTS_ROOT="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
ABLATION_DIR="$RESULTS_ROOT/anal_data/G_behaviour/feedback_ablation_seq512_10seed"
SAVE_DIR="$RESULTS_ROOT/anal_figs/G_behaviour"
OUT_NAME="fig_ablation_shuffle_standalone_seq512_yticks.png"
PNG_PATH="$SAVE_DIR/$OUT_NAME"
PDF_PATH="$SAVE_DIR/fig_ablation_shuffle_standalone_seq512_yticks.pdf"

for seed in $(seq -w 1 10); do
  if [[ ! -f "$ABLATION_DIR/gawf-seed$seed/ablation_metrics.json" ]]; then
    echo "Missing ablation metrics for gawf-seed$seed." >&2
    exit 2
  fi
done
if [[ -e "$PNG_PATH" || -e "$PDF_PATH" ]]; then
  echo "Refusing to overwrite existing figure: $PNG_PATH or $PDF_PATH" >&2
  exit 3
fi

mkdir -p "$SAVE_DIR"
python utils_viz/viz_feedback_shuffle_standalone.py \
  --ablation_dir "$ABLATION_DIR" \
  --baseline_source ablation \
  --save_dir "$SAVE_DIR" \
  --out_name "$OUT_NAME" \
  --ymin 50 \
  --ymax 95 \
  --yticks 50 65 80 95

test -s "$PNG_PATH"
test -s "$PDF_PATH"
echo "status=complete"
echo "png=$PNG_PATH"
echo "pdf=$PDF_PATH"
