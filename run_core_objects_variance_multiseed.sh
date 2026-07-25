#!/usr/bin/env bash
# Run the unified GaWF variance decomposition across best-model seeds, then aggregate the four
# core-object condition-mean fractions into cross-seed (mean +/- sd) figures.
#
# For each seed it (1) exports the exact trial-level sources from that seed's checkpoint, (2) runs
# the unified decomposition, (3) copies only the compact {object}_per_unit_distributions.npz into
# OUT_ROOT/<seed>/, then (4) deletes the ~78 GiB intermediate representation arrays before moving
# to the next seed, so peak extra disk stays ~one seed's worth. Afterwards it builds the 2x2 and
# 1x4 core-object figures with cross-seed error bars (utils_viz/core_objects_aggregate.py
# --seed_dirs), matching the best-model-accuracy figure's mean +/- sd convention.
#
# Run this where the seed checkpoints, stimuli, and a CUDA GPU live. `python` must already be the
# right environment (activate your conda/venv first). Override any default below via environment.
#
# sjc-remote example (checkpoints live under the pre-rename aim3_RNN project):
#   CKPT_ROOT=/G/MIMOlab/Codes/aim3_RNN/results/train_data/clutter/best6_multiseed_40h_ep150 \
#   DATA_DIR=/G/MIMOlab/Codes/aim3_gawf_rnn/stimuli \
#   bash run_core_objects_variance_multiseed.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# utils_viz/core_objects_aggregate.py has no sys.path bootstrap (unlike the utils_anal scripts
# below, which self-insert their project root), so it needs the project root on PYTHONPATH.
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# --- configuration (override via environment) --------------------------------------------------
# Root holding one subdir per seed, each with the GaWF checkpoint.
CKPT_ROOT="${CKPT_ROOT:-/scratch/js3269/results/train_data/clutter_best6_multiseed_40h_ep150}"
SEEDS="${SEEDS:-gawf-seed01 gawf-seed02 gawf-seed03 gawf-seed04 gawf-seed05 gawf-seed06 gawf-seed07 gawf-seed08 gawf-seed09 gawf-seed10}"
CKPT_NAME="${CKPT_NAME:-gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth}"
DATA_DIR="${DATA_DIR:-/scratch/js3269/stimuli}"
DATA_SUFFIX="${DATA_SUFFIX:-40h-uint8}"
DEVICE="${DEVICE:-cuda}"
CHAN_NUM="${CHAN_NUM:-2}"
EXPORT_BATCH="${EXPORT_BATCH:-16}"          # Above the export default of 8; raise to 32 on a 48GB GPU.
# Fixed output dirs the two analysis scripts write to (utils_anal.anal_paths.output_dir layout).
EXPORT_DIR="${EXPORT_DIR:-results/anal_data/D_variance_decomposition/export_unified_variance_sources}"
UNIFIED_DIR="${UNIFIED_DIR:-results/anal_data/D_variance_decomposition/unified}"
# Where per-seed compact NPZ summaries are collected, and where the final figures land.
OUT_ROOT="${OUT_ROOT:-results/anal_data/D_variance_decomposition/unified_multiseed}"
FIG_DIR="${FIG_DIR:-results/anal_figs/D_variance_decomposition}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "CKPT_ROOT   = $CKPT_ROOT"
echo "SEEDS       = $SEEDS"
echo "DEVICE      = $DEVICE  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "DATA_DIR    = $DATA_DIR ($DATA_SUFFIX)"
echo "OUT_ROOT    = $OUT_ROOT"
echo "FIG_DIR     = $FIG_DIR"
echo ""

# Validate every checkpoint up front so a missing seed fails before any hours-long export.
for seed in $SEEDS; do
  ckpt="$CKPT_ROOT/$seed/$CKPT_NAME"
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: checkpoint not found: $ckpt" >&2
    exit 1
  fi
done

seed_dirs=()
for seed in $SEEDS; do
  ckpt="$CKPT_ROOT/$seed/$CKPT_NAME"
  out="$OUT_ROOT/$seed"
  echo "=== $seed ==="

  # The exporter refuses to overwrite existing arrays, so clear both fixed dirs from any prior
  # seed (their contents are regenerated below; only the copied NPZ under OUT_ROOT is kept).
  rm -rf "$EXPORT_DIR" "$UNIFIED_DIR"

  python utils_anal/export_unified_variance_sources.py \
    --ckpt "$ckpt" \
    --data_dir "$DATA_DIR" \
    --data_suffix "$DATA_SUFFIX" \
    --device "$DEVICE" \
    --batch_size "$EXPORT_BATCH" \
    --chan_num "$CHAN_NUM"

  python utils_anal/run_unified_variance_decomposition.py \
    --input_manifest "$EXPORT_DIR/input_manifest.json" \
    --skip_published_regression

  mkdir -p "$out"
  cp "$UNIFIED_DIR"/*_per_unit_distributions.npz "$out"/
  seed_dirs+=("$out")

  # Free the giant intermediate representation arrays before the next seed.
  rm -f "$EXPORT_DIR"/*.npy
  echo "  saved per-seed summaries -> $out"
  echo ""
done

echo "=== aggregating ${#seed_dirs[@]} seeds into cross-seed figures ==="
mkdir -p "$FIG_DIR"
python utils_viz/core_objects_aggregate.py \
  --seed_dirs "${seed_dirs[@]}" \
  --figure_dir "$FIG_DIR"

echo "Done. Cross-seed core_objects_aggregate_2x2 / _1x4 written under $FIG_DIR"
