#!/usr/bin/env bash
# Render 10-seed, nine-sector input sign-vs-magnitude Supplementary 2 figures on SJC.

set -eo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:-$ROOT/results}"
OUT_DEFAULT="$RESULTS/data/analysis/supple2_input_gate_sign_magnitude_9sector_10seed"
OUT="${AIM3_SUPPLE2_INPUT_GATE_SIGN_MAG_OUT:-$OUT_DEFAULT}"
TRAJECTORIES_DEFAULT="$RESULTS/data/analysis/fig6_sector_gate_weight_sign_10seed/trajectories"
TRAJECTORIES="${AIM3_SUPPLE2_TRAJECTORIES:-$TRAJECTORIES_DEFAULT}"
SAVE_DIR="$RESULTS/save"
CONDA_SH="${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"

source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$OUT"
for seed in $(seq -w 1 10); do
  TRAJECTORY="$TRAJECTORIES/seed$seed/gawf_gate_trajectory.npz"
  SEED_OUT="$OUT/seed$seed"
  [[ -f "$TRAJECTORY" ]] || { echo "Missing trajectory: $TRAJECTORY" >&2; exit 1; }
  if [[ -e "$SEED_OUT" ]]; then
    [[ -f "$SEED_OUT/input_gate_sign_magnitude_9sector.npz" ]] || {
      echo "Refusing to reuse incomplete output: $SEED_OUT" >&2; exit 1;
    }
    continue
  fi
  python -m utils.analysis.clutter.supple2_input_gate_sign_magnitude_sector collect \
    --trajectory "$TRAJECTORY" --output_dir "$SEED_OUT" --seed "${seed#0}" --device cuda \
    --all_sectors
done

FINAL="$OUT/final"
[[ ! -e "$FINAL" ]] || { echo "Refusing to overwrite existing final output: $FINAL" >&2; exit 1; }
python -m utils.analysis.clutter.supple2_input_gate_sign_magnitude_sector plot \
  --data_root "$OUT" --figure_dir "$FINAL" --all_sectors
touch "$FINAL/.complete"
mkdir -p "$SAVE_DIR"
cp "$FINAL/Supple2_input_gate_sign_vs_mag_sector_delta_zoom_10seed_9sector.png" \
  "$SAVE_DIR/Supple2_input_gate_sign_vs_mag_sector_delta_zoom.png"
