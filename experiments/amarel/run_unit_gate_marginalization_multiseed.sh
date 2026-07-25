#!/usr/bin/env bash
#SBATCH --job-name=aim3-03-gate-multiseed
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --exclude=gpu018,gpu043
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/amarel/artifacts/unit_gate_marginalization_multiseed/%j.out
#SBATCH --error=experiments/amarel/artifacts/unit_gate_marginalization_multiseed/%j.err

# Compute the Figure-03 unit-level gate context-variance fractions across all best-model seeds
# (gawf/lstm/gru x 10), then redraw the 1x3 marginalization panel with cross-seed mean +/- sd.
# The driver is resumable: it saves the pooled JSON after every (model, seed) unit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

CONDA_INIT="${AIM3_CONDA_INIT:-/home/js3269/enter/etc/profile.d/conda.sh}"
source "$CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"

# Fail fast on a broken GPU node (e.g. gpu018, where torch cannot see CUDA) instead of running
# the whole analysis on CPU by accident.
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to torch; refusing to run on this node.")
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
PY

CHECKPOINT_ROOT="${AIM3_CAMPAIGN_ROOT:-/scratch/${USER}/results/train_data/clutter_best6_multiseed_40h_ep150}"
DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
DATA_SUFFIX="${AIM3_DATA_SUFFIX:-40h-uint8}"
SAVE_JSON="${AIM3_SAVE_JSON:-results/anal_data/D_variance_decomposition/rnn_unit_gate_context_specificity_multiseed/data/unit_gate_context_variance_multiseed.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python utils_anal/rnn_unit_gate_context_specificity_multiseed.py \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --models gawf lstm gru \
  --data_dir "$DATA_DIR" \
  --data_suffix "$DATA_SUFFIX" \
  --save_json "$SAVE_JSON" \
  --device cuda

python -m utils_viz.rnn_unit_gate_context_specificity \
  --report "$SAVE_JSON"

echo "Done. Cross-seed 03_unit_gate_marginalization_1x3 written under results/anal_figs/D_variance_decomposition/rnn_unit_gate_context_specificity/figs"
