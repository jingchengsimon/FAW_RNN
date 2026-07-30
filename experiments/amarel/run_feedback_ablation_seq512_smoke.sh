#!/usr/bin/env bash
#SBATCH --job-name=aim3-fbabl-s512-smoke
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Compute-node smoke test for one 512-frame/16-sequence GaWF feedback-ablation batch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

if [[ -z "${AIM3_CONDA_INIT:-}" || ! -f "$AIM3_CONDA_INIT" ]]; then
  echo "AIM3_CONDA_INIT must identify the Amarel Conda initialization script." >&2
  exit 2
fi
source "$AIM3_CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"

DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
RESULTS_ROOT="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
CHECKPOINT_ROOT="${AIM3_CHECKPOINT_ROOT:-$RESULTS_ROOT/train_data/clutter_best6_multiseed_40h_ep150}"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/gawf-seed01"
SAVE_DIR="${AIM3_SMOKE_SAVE_DIR:-$RESULTS_ROOT/anal_data/G_behaviour/feedback_ablation_seq512_smoke/gawf-seed01}"

if [[ -e "$SAVE_DIR/ablation_metrics.json" ]]; then
  echo "Refusing to overwrite existing smoke result: $SAVE_DIR/ablation_metrics.json" >&2
  exit 3
fi
shopt -s nullglob
checkpoints=("$CHECKPOINT_DIR"/*_model.pth)
shopt -u nullglob
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one checkpoint in $CHECKPOINT_DIR; found ${#checkpoints[@]}." >&2
  exit 2
fi

mkdir -p "$SAVE_DIR"
python utils_anal/feedback_ablation.py \
  --ckpt "${checkpoints[0]}" \
  --shuffle \
  --save_dir "$SAVE_DIR" \
  --data_dir "$DATA_DIR" \
  --data_suffix 40h-uint8 \
  --device cuda \
  --batch_size 16 \
  --sequence_length 512 \
  --seed 42 \
  --use_mmap \
  --use_sector_mode \
  --max_batches 1

SAVE_DIR="$SAVE_DIR" python -c 'import json, os; from pathlib import Path; p = Path(os.environ["SAVE_DIR"]) / "ablation_metrics.json"; d = json.loads(p.read_text()); assert d["sequence_length"] == 512; assert d["batch_size"] == 16; assert d["conditions"]["baseline"]["n_frames"] == 8192; assert d["cuda_max_memory_allocated_bytes"] > 0; print(json.dumps({"status": "smoke_passed", "metrics": str(p), "peak_allocated_bytes": d["cuda_max_memory_allocated_bytes"], "peak_reserved_bytes": d["cuda_max_memory_reserved_bytes"]}, indent=2))'
