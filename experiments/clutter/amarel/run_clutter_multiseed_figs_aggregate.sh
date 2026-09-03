#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-figs-aggregate
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Compute-node aggregation of all ten completed GaWF Clutter seed analyses.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
BASE="${AIM3_MULTISEED_FIGS_BASE:?AIM3_MULTISEED_FIGS_BASE is required}"
cd "$ROOT"
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

FINAL="${AIM3_MULTISEED_FIGS_FINAL_DIR:-$BASE/final}"
if [[ -e "$FINAL" ]]; then
  echo "Refusing to overwrite existing final output: $FINAL" >&2
  exit 1
fi
SEED_DIRS=()
FIG3_DIRS=()
FIG6_DATA=()
CACHE_DIRS=()
for seed in $(seq -w 1 10); do
  SEED_ROOT="$BASE/seed${seed}"
  [[ -f "$SEED_ROOT/.complete" ]] || { echo "Incomplete seed: $SEED_ROOT" >&2; exit 1; }
  FIG3_DIRS+=("$SEED_ROOT/fig3")
  SEED_DIRS+=("$SEED_ROOT/fig4_data")
  FIG6_DATA+=("$SEED_ROOT/fig6/sector_gate_mean_sequential_equal_n.npz")
  CACHE_DIRS+=("$SEED_ROOT/fig7/cache")
done
mkdir -p "$FINAL"/{fig3,fig4,fig6,fig7}
python -m utils.analysis.clutter.fig3_gate_distribution_plot \
  --seed_dirs "${FIG3_DIRS[@]}" \
  --raw_dir "$FINAL/fig3" --only_gate_weight_2x2 \
  --gate_weight_stem gate_and_weight_distributions_2x2_10seed \
  --metadata_path "$FINAL/fig3/gate_and_weight_distributions_2x2_10seed_metadata.json"
python -m utils.analysis.clutter.fig4_variance_decomposition_plot \
  --seed_dirs "${SEED_DIRS[@]}" --figure_dir "$FINAL/fig4"
cp "$FINAL/fig4/core_objects_aggregate_1x4.pdf" "$FINAL/fig4/core_objects_aggregate_1x4_10seed.pdf"
python -m utils.analysis.clutter.fig6_sector_gate_sequential_plot \
  --seed_data "${FIG6_DATA[@]}" --fig_dir "$FINAL/fig6" \
  --stem sector_gate_mean_sequential_equal_n_10seed
cp "$FINAL/fig6/sector_gate_mean_sequential_equal_n_10seed_point_excluded.pdf" \
  "$FINAL/fig6/sector_gate_mean_sequential_equal_n_point_excluded_10seed.pdf"
python -m utils.analysis.clutter.fig7_recurrent_gate_disinhibition \
  --cache_dirs "${CACHE_DIRS[@]}" --fig_dir "$FINAL/fig7" \
  --output_stem recurrent_gate_disinhibition_poster_10seed
python -m utils.analysis.clutter.fig7_recurrent_gate_disinhibition_delta \
  --cache_dirs "${CACHE_DIRS[@]}" --fig_dir "$FINAL/fig7" \
  --output_stem recurrent_gate_disinhibition_poster_delta_10seed
touch "$FINAL/.complete"
