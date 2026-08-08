#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-figs-1-10
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Compute-node worker for one GaWF Clutter seed in the Figure 3/4/6/7 campaign.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
BASE="${AIM3_MULTISEED_FIGS_BASE:?AIM3_MULTISEED_FIGS_BASE is required}"
SEED="${SLURM_ARRAY_TASK_ID:?Slurm array task id is required}"
printf -v SEED_TAG '%02d' "$SEED"
cd "$ROOT"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

CKPT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150/gawf-seed${SEED_TAG}/gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
OUT="$BASE/seed${SEED_TAG}"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing seed output: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"/{fig3,fig3_digit,fig4_sources,fig4_data,fig4_figs,fig6,fig7/relevance,fig7/cache}

python -m utils.analysis.clutter.fig3_gate_distribution \
  --ckpt "$CKPT" --data_dir "$DATA_DIR" --data_suffix 40h-uint8 --device cuda \
  --batch_size 16 --gate_chunk_size 32 --save_dir "$OUT/fig3"
python -m utils.analysis.clutter.fig3_gate_digit_distribution \
  --trajectory "$OUT/fig3/gawf_gate_trajectory.npz" --save_dir "$OUT/fig3_digit"
python -m utils.analysis.clutter.fig4_variance_sources \
  --ckpt "$CKPT" --data_dir "$DATA_DIR" --device cuda --batch_size 8 --output_dir "$OUT/fig4_sources"
python -m utils.analysis.clutter.fig4_variance_decomposition \
  --input_manifest "$OUT/fig4_sources/input_manifest.json" --seed "$SEED" \
  --skip_published_regression --output_dir "$OUT/fig4_data" --figure_dir "$OUT/fig4_figs"
SOURCE_DIR="$OUT/fig4_sources"
for provenance_file in input_manifest.json source_provenance.json; do
  cp "$SOURCE_DIR/$provenance_file" "$OUT/fig4_data/fig4_source_$provenance_file"
done
RAW_FIG4_SOURCES=(
  "$SOURCE_DIR/encoder_activation.npy"
  "$SOURCE_DIR/input_gate.npy"
  "$SOURCE_DIR/hidden_state.npy"
  "$SOURCE_DIR/recurrent_gate.npy"
)
for raw_source in "${RAW_FIG4_SOURCES[@]}"; do
  [[ -f "$raw_source" ]] || { echo "Missing expected Fig4 raw source: $raw_source" >&2; exit 1; }
done
printf 'Fig4 raw source cleanup completed after successful decomposition.\n' \
  > "$OUT/fig4_data/fig4_raw_sources_cleanup.txt"
rm -f -- "${RAW_FIG4_SOURCES[@]}"
python -m utils.analysis.clutter.fig6_sector_gate_sequential \
  --trajectory "$OUT/fig3/gawf_gate_trajectory.npz" --save_dir "$OUT/fig6" --device cuda \
  --seed "$SEED"
python -m utils.analysis.clutter.fig7_relevance_timing \
  --ckpt "$CKPT" --data_dir "$DATA_DIR" --device cuda --batch_size 16 --seed "$SEED" \
  --save_dir "$OUT/fig7/relevance"
python -m utils.analysis.clutter.fig7_recurrent_gate_digit_collect \
  --ckpt "$CKPT" --selectivity "$OUT/fig7/relevance/part1_selectivity.npz" \
  --cache_dir "$OUT/fig7/cache" --data_dir "$DATA_DIR" --device cuda
python -m utils.analysis.clutter.fig7_recurrent_gate_sector_collect \
  --ckpt "$CKPT" --selectivity "$OUT/fig7/relevance/part1_selectivity.npz" \
  --cache_dir "$OUT/fig7/cache" --data_dir "$DATA_DIR" --device cuda

touch "$OUT/.complete"
