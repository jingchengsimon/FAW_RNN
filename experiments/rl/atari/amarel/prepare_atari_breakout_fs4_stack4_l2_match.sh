#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-fs4s4-l2-match
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Build the parameter-match table for strict 4-action, two-layer Breakout.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/run_task.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
MATCH_DIR="${MATCH_DIR:-$AIM3_RESULTS_PATH/data/rl/atari/parameter_match/breakout_fs4_stack4_l2}"
NUM_ACTIONS="${NUM_ACTIONS:-4}"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

mkdir -p "$MATCH_DIR"
python -m experiments.atari.atari_ssm_param_match \
  --models rnn gru lstm gawf \
  --num_actions "$NUM_ACTIONS" \
  --num_layers 2 \
  --out_dir "$MATCH_DIR"

python - "$MATCH_DIR/atari_param_match.json" "$NUM_ACTIONS" <<'PY'
import json
import sys

path, num_actions = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
required = {"ann", "rnn", "gru", "lstm", "gawf"}
missing = sorted(required - set(data["matched"]))
if missing:
    raise RuntimeError(f"Parameter match table lacks {missing}")
if data.get("num_actions") != num_actions:
    raise RuntimeError(
        f"Match table num_actions={data.get('num_actions')} != expected {num_actions}"
    )
if data.get("candidate_num_layers") != 2:
    raise RuntimeError("Parameter match table is not two-layer")
if any(data["matched"][model].get("num_layers") != 2 for model in required):
    raise RuntimeError("A matched model does not have num_layers=2")
PY

echo "[$(date -Is)] L2 match table -> $MATCH_DIR/atari_param_match.json"
