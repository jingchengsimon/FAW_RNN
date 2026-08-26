#!/usr/bin/env bash
# Compute the ten-seed sector-conditioned encoder activation patterns for Figure 6 on SJC.

set -eo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:-$ROOT/results}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:-$ROOT/source/clutter/stimuli}"
OUT_DEFAULT="$RESULTS/data/analysis/fig6_encoder_sector_patterns_gawf_10seed"
OUT="${AIM3_FIG6_ENCODER_SECTOR_PATTERNS_OUT:-$OUT_DEFAULT}"
CONDA_SH="${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"

source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTHONDONTWRITEBYTECODE=1

RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
mkdir -p "$OUT"
for seed in $(seq -w 1 10); do
  CKPT="$RUN_ROOT/gawf-seed$seed/gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
  SEED_OUT="$OUT/gawf-seed$seed"
  [[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
  if [[ -e "$SEED_OUT" ]]; then
    [[ -f "$SEED_OUT/encoder_sector_patterns.npz" && -f "$SEED_OUT/manifest.json" ]] || {
      echo "Refusing to reuse incomplete output: $SEED_OUT" >&2; exit 1;
    }
    continue
  fi
  python -m utils.analysis.clutter.fig6_encoder_sector_patterns collect \
    --ckpt "$CKPT" --data_dir "$DATA_DIR" --output_dir "$SEED_OUT" \
    --device cuda --batch_size 16 --num_workers 2 --data_suffix 40h-uint8 --condition sector
done

FINAL="$OUT/final"
[[ ! -e "$FINAL" ]] || { echo "Refusing to overwrite existing final output: $FINAL" >&2; exit 1; }
python -m utils.analysis.clutter.fig6_encoder_sector_patterns plot \
  --data_root "$OUT" --figure_dir "$FINAL" --condition sector
touch "$FINAL/.complete"
