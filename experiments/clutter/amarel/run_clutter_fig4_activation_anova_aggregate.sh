#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-fig4-act-anova-aggregate
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Validate the 60 compact summaries and render the preliminary Figure 4 activation panel.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
BASE="${AIM3_FIG4_ACTIVATION_ANOVA_BASE:?AIM3_FIG4_ACTIVATION_ANOVA_BASE is required}"
cd "$ROOT"
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

for model in gawf rnn lstm gru mamba s5; do
  for seed in $(seq -w 1 10); do
    [[ -f "$BASE/$model-seed$seed/activation_anova.npz" ]] || {
      echo "Missing compact summary: $BASE/$model-seed$seed/activation_anova.npz" >&2
      exit 1
    }
  done
done
FINAL="$BASE/final"
[[ ! -e "$FINAL" ]] || { echo "Refusing to overwrite existing final output: $FINAL" >&2; exit 1; }
mkdir -p "$FINAL"
python -m utils.analysis.clutter.fig4_activation_anova plot \
  --data_root "$BASE" --figure_dir "$FINAL"
touch "$FINAL/.complete"
