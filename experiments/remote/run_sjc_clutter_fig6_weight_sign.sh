#!/usr/bin/env bash
# Render the ten-seed Figure 6 point-excluded input-gate maps split by input-weight sign on SJC.

set -eo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:-$ROOT/results}"
OUT_DEFAULT="$RESULTS/data/analysis/fig6_sector_gate_weight_sign_10seed"
OUT="${AIM3_FIG6_WEIGHT_SIGN_OUT:-$OUT_DEFAULT}"
TRAJECTORIES="$OUT/trajectories"
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
    [[ -f "$SEED_OUT/sector_gate_mean_sequential_equal_n_weight_sign.npz" ]] || {
      echo "Refusing to reuse incomplete output: $SEED_OUT" >&2; exit 1;
    }
    continue
  fi
  python -m utils.analysis.clutter.fig6_sector_gate_weight_sign collect \
    --trajectory "$TRAJECTORY" --output_dir "$SEED_OUT" --seed "${seed#0}" --device cuda
done

FINAL="$OUT/final"
[[ ! -e "$FINAL" ]] || { echo "Refusing to overwrite existing final output: $FINAL" >&2; exit 1; }
python -m utils.analysis.clutter.fig6_sector_gate_weight_sign plot \
  --data_root "$OUT" --figure_dir "$FINAL"
touch "$FINAL/.complete"
