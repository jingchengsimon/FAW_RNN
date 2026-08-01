#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-fs4s4-depth-match
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Build one depth-specific parameter-match table against the L1 LSTM(512) anchor.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/run_task.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
: "${MATCH_DIR:?MATCH_DIR is required}"
: "${NUM_LAYERS:?NUM_LAYERS is required}"
NUM_ACTIONS="${NUM_ACTIONS:-4}"
(( NUM_LAYERS >= 1 )) || { echo "NUM_LAYERS must be >= 1" >&2; exit 2; }

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

mkdir -p "$MATCH_DIR"
python -m experiments.rl.atari.atari_ssm_param_match \
  --models rnn gru lstm gawf \
  --num_actions "$NUM_ACTIONS" \
  --num_layers "$NUM_LAYERS" \
  --out_dir "$MATCH_DIR"

python - "$MATCH_DIR/atari_param_match.json" "$NUM_ACTIONS" "$NUM_LAYERS" <<'PY'
import json
import sys

path, num_actions, num_layers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
required = {"ann", "rnn", "gru", "lstm", "gawf"}
missing = sorted(required - set(data["matched"]))
if missing:
    raise RuntimeError(f"Parameter match table lacks {missing}")
if data.get("anchor") != "lstm" or data.get("anchor_num_layers") != 1:
    raise RuntimeError("Parameter match anchor is not the L1 LSTM")
if data.get("hidden_size") != 512:
    raise RuntimeError("Parameter match anchor hidden_size is not 512")
if data.get("num_actions") != num_actions:
    raise RuntimeError("Parameter match action count mismatch")
if data.get("candidate_num_layers") != num_layers:
    raise RuntimeError("Parameter match candidate depth mismatch")
if any(data["matched"][model].get("num_layers") != num_layers for model in required):
    raise RuntimeError("A matched model has an incorrect depth")
PY

echo "[$(date -Is)] L${NUM_LAYERS} match table -> $MATCH_DIR/atari_param_match.json"
