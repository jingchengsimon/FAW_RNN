#!/usr/bin/env bash
# Run GaWF feedback-component ablation (zero + shuffle) across best-model seeds, then aggregate.
#
# For each seed checkpoint it runs utils_anal/feedback_ablation.py with --shuffle, so a single
# pass per seed produces all six conditions (baseline, clear_digit/sector/all, shuffle_digit/
# sector). Afterwards it builds the two-row multi-seed test-accuracy figure. PNG only.
#
# Run this where the seed checkpoints and stimuli live (e.g. the amarel cluster). Override the
# defaults below with environment variables or edit them in place.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- configuration (override via environment) --------------------------------------------------
# Root holding one subdir per seed, each with the GaWF checkpoint.
CKPT_ROOT="${CKPT_ROOT:-/scratch/js3269/results/train_data/clutter_best6_multiseed_40h_ep150}"
# Seed subdir names and the checkpoint filename inside each.
SEEDS="${SEEDS:-gawf-seed01 gawf-seed02 gawf-seed03 gawf-seed04 gawf-seed05 gawf-seed06 gawf-seed07 gawf-seed08 gawf-seed09 gawf-seed10}"
CKPT_NAME="${CKPT_NAME:-gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth}"
DATA_DIR="${DATA_DIR:-/scratch/js3269/stimuli}"
DATA_SUFFIX="${DATA_SUFFIX:-40h-uint8}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
# Output root for per-seed ablation metrics.
OUT_ROOT="${OUT_ROOT:-results/data/anal_data/G_behaviour/feedback_ablation_multiseed}"
FIG_DIR="${FIG_DIR:-results/anal_figs/G_behaviour}"
# Preserve a preset CUDA_VISIBLE_DEVICES if the caller set one.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "CKPT_ROOT   = $CKPT_ROOT"
echo "SEEDS       = $SEEDS"
echo "DEVICE      = $DEVICE  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "OUT_ROOT    = $OUT_ROOT"
echo ""

seed_dirs=()
for seed in $SEEDS; do
  ckpt="$CKPT_ROOT/$seed/$CKPT_NAME"
  out="$OUT_ROOT/$seed"
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: checkpoint not found: $ckpt" >&2
    exit 1
  fi
  echo "=== $seed ==="
  python utils_anal/feedback_ablation.py \
    --ckpt "$ckpt" \
    --shuffle \
    --data_dir "$DATA_DIR" \
    --data_suffix "$DATA_SUFFIX" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --save_dir "$out"
  seed_dirs+=("$out")
done

echo ""
echo "=== aggregating ${#seed_dirs[@]} seeds into two-row figure ==="
python utils_viz/viz_feedback_ablation_multiseed.py \
  --seed_dirs "${seed_dirs[@]}" \
  --save_dir "$FIG_DIR"

echo "Done."
