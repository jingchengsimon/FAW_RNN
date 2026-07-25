#!/usr/bin/env bash
#SBATCH --job-name=aim3-gawf-align-dist
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/amarel/artifacts/gawf_relevance_alignment_distributions/slurm-%j.out
#SBATCH --error=experiments/amarel/artifacts/gawf_relevance_alignment_distributions/slurm-%j.err

# Compute-node runner for GaWF efferent/afferent 2-group and 4-group
# gate-distribution analyses (single-seed checkpoint).  Login-node safe
# submitter is submit_gawf_relevance_alignment_distributions.sh.

set -euo pipefail

PROJECT_ROOT="/cache/home/js3269/projects/aim3_gawf_rnn"
CONDA_INIT="/home/js3269/enter/etc/profile.d/conda.sh"
STIMULI_ROOT="/scratch/js3269/stimuli"

# --- prerequisite inputs (may be overridden via --export at submission) ---
: "${AIM3_RESULTS_PATH:=/scratch/js3269/results}"
: "${AIM3_NUM_WORKERS:=12}"
: "${AIM3_PIN_MEMORY:=1}"
: "${CHECKPOINT:=$AIM3_RESULTS_PATH/train_data/gen_hparam_full_grid/task_1007/gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth}"
: "${SELECTIVITY_NPZ:=$PROJECT_ROOT/results/anal_data/D_variance_decomposition/gawf_symmetric_relevance_timing/data/part1_selectivity.npz}"
: "${SPLIT_REPORT_JSON:=$PROJECT_ROOT/results/anal_data/H_controls/gawf_symmetric_relevance_timing/data/part0_splits.json}"

source "$CONDA_INIT"
conda activate aim3_rnn
cd "$PROJECT_ROOT"

# --- ensure `git` is on PATH -------------------------------------------------
# Some Amarel compute nodes (observed on gpu039) launch Slurm jobs with a minimal
# PATH that drops /usr/bin, so subprocess calls to ``git`` (matplotlib atexit hook,
# huggingface fetches, etc.) raise FileNotFoundError. Try to restore it via the
# ``git`` module first, then append /usr/bin as a last resort. We APPEND rather
# than prepend so the conda env python (aim3_rnn) still wins over any system
# python that might live under /usr/bin.
if ! command -v git >/dev/null 2>&1; then
  module load git 2>/dev/null || true
fi
if ! command -v git >/dev/null 2>&1; then
  export PATH="${PATH}:/usr/bin"
fi
command -v git >/dev/null 2>&1 \
  && echo "[runner] git=$(command -v git)" \
  || echo "[runner] WARNING: git still not on PATH; subprocess hooks may warn" >&2
echo "[runner] python=$(command -v python)"

echo "[runner] host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[runner] project_root=$PROJECT_ROOT"
echo "[runner] checkpoint=$CHECKPOINT"
echo "[runner] selectivity=$SELECTIVITY_NPZ"
echo "[runner] split_report=$SPLIT_REPORT_JSON"
echo "[runner] AIM3_NUM_WORKERS=$AIM3_NUM_WORKERS AIM3_PIN_MEMORY=$AIM3_PIN_MEMORY"

# --- preflight: fail fast if inputs are missing ------------------------------
missing=0
for path in "$CHECKPOINT" "$SELECTIVITY_NPZ" "$SPLIT_REPORT_JSON"; do
  if [[ ! -f "$path" ]]; then
    echo "[runner] MISSING: $path" >&2
    missing=1
  fi
done
if [[ ! -d "$STIMULI_ROOT" ]]; then
  echo "[runner] MISSING stimuli root: $STIMULI_ROOT" >&2
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "[runner] one or more prerequisite inputs are missing; aborting." >&2
  exit 2
fi

MPL_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/gawf-align-dist-matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "$MPL_CACHE_ROOT"
export MPLCONFIGDIR="$MPL_CACHE_ROOT"

# Anal + viz pairs share these common flags.
COMMON_ANAL_ARGS=(
  --ckpt "$CHECKPOINT"
  --data_dir "$STIMULI_ROOT"
  --data_suffix 40h-uint8
  --selectivity "$SELECTIVITY_NPZ"
  --split_report "$SPLIT_REPORT_JSON"
  --batch_size 16
  --num_workers "$AIM3_NUM_WORKERS"
  --chan_num 2
  --device cuda
)

# ---------------------------------------------------------------------------
# 1. Efferent 4-group (recurrent × sector/digit) analysis + viz
# ---------------------------------------------------------------------------
if [[ "${AIM3_PLOT_ONLY:-0}" != "1" ]]; then
  echo "[runner] --- efferent 4-group analysis ---"
  python utils_anal/gawf_recurrent_group_gate_distributions.py "${COMMON_ANAL_ARGS[@]}"
fi
echo "[runner] --- efferent 4-group viz ---"
python utils_viz/gawf_recurrent_group_gate_distributions.py

# ---------------------------------------------------------------------------
# 2. Afferent 2-group (input+recurrent × sector/digit) analysis + viz
# ---------------------------------------------------------------------------
if [[ "${AIM3_PLOT_ONLY:-0}" != "1" ]]; then
  echo "[runner] --- afferent 2-group analysis ---"
  python utils_anal/gawf_afferent_top10_gate_distributions.py "${COMMON_ANAL_ARGS[@]}"
fi
echo "[runner] --- afferent 2-group viz ---"
python utils_viz/gawf_afferent_top10_gate_distributions.py
echo "[runner] --- afferent aggregate Part-2 (cohen's d + alignment matrix) viz ---"
python utils_viz/gawf_afferent_relevance_alignment.py

# ---------------------------------------------------------------------------
# 3. Afferent 4-group (recurrent × sector/digit) analysis + viz
# ---------------------------------------------------------------------------
if [[ "${AIM3_PLOT_ONLY:-0}" != "1" ]]; then
  echo "[runner] --- afferent 4-group analysis ---"
  python utils_anal/gawf_recurrent_afferent_group_gate_distributions.py "${COMMON_ANAL_ARGS[@]}"
fi
echo "[runner] --- afferent 4-group viz ---"
python utils_viz/gawf_recurrent_afferent_group_gate_distributions.py

echo "[runner] finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
