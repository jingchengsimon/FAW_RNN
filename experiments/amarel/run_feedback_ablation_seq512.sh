#!/usr/bin/env bash
#SBATCH --job-name=aim3-fbabl-s512
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Run one GaWF feedback-ablation seed with 512-frame recurrent/shuffle windows on a compute node.

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
if [[ ! "${AIM3_SEED:-}" =~ ^([1-9]|10)$ ]]; then
  echo "AIM3_SEED must be an integer from 1 through 10." >&2
  exit 2
fi
source "$AIM3_CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"

DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
RESULTS_ROOT="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
CHECKPOINT_ROOT="${AIM3_CHECKPOINT_ROOT:-$RESULTS_ROOT/train_data/clutter_best6_multiseed_40h_ep150}"
SEED_TAG="$(printf 'gawf-seed%02d' "$AIM3_SEED")"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$SEED_TAG"
OUTPUT_ROOT="${AIM3_ABLATION_OUTPUT_ROOT:-$RESULTS_ROOT/anal_data/G_behaviour/feedback_ablation_seq512_10seed}"
SAVE_DIR="$OUTPUT_ROOT/$SEED_TAG"
SEQUENCE_LENGTH="${AIM3_SEQUENCE_LENGTH:-512}"
BATCH_SIZE="${AIM3_BATCH_SIZE:-16}"

if [[ "$SEQUENCE_LENGTH" != "512" || "$BATCH_SIZE" != "16" ]]; then
  echo "This launcher is fixed to the registered protocol: sequence_length=512, batch_size=16." >&2
  exit 2
fi
if [[ ! -f "$DATA_DIR/stimulus_reg-test-40h-uint8.npy" ]]; then
  echo "Missing test stimuli: $DATA_DIR/stimulus_reg-test-40h-uint8.npy" >&2
  exit 2
fi
if [[ ! -f "$DATA_DIR/stimulus_reg-test-40h-uint8.tsv" ]]; then
  echo "Missing test labels: $DATA_DIR/stimulus_reg-test-40h-uint8.tsv" >&2
  exit 2
fi
if [[ -e "$SAVE_DIR/ablation_metrics.json" ]]; then
  echo "Refusing to overwrite existing analysis result: $SAVE_DIR/ablation_metrics.json" >&2
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
echo "seed=$AIM3_SEED checkpoint=${checkpoints[0]}"
echo "sequence_length=$SEQUENCE_LENGTH batch_size=$BATCH_SIZE"
echo "data_dir=$DATA_DIR results_root=$RESULTS_ROOT save_dir=$SAVE_DIR"

python utils_anal/feedback_ablation.py \
  --ckpt "${checkpoints[0]}" \
  --shuffle \
  --K 10 \
  --pre_K 5 \
  --save_dir "$SAVE_DIR" \
  --data_dir "$DATA_DIR" \
  --data_suffix 40h-uint8 \
  --device cuda \
  --batch_size "$BATCH_SIZE" \
  --sequence_length "$SEQUENCE_LENGTH" \
  --seed 42 \
  --use_mmap \
  --use_sector_mode

SAVE_DIR="$SAVE_DIR" python -c 'import json, os; from pathlib import Path; p = Path(os.environ["SAVE_DIR"]) / "ablation_metrics.json"; d = json.loads(p.read_text()); expected = {"sequence_length": 512, "batch_size": 16, "n_frames": 57344}; assert d["conditions"]["baseline"]["n_frames"] == expected["n_frames"]; assert all(d[k] == v for k, v in expected.items() if k != "n_frames"); assert d["cuda_max_memory_allocated_bytes"] > 0; print(json.dumps({"status": "complete", "metrics": str(p), "peak_allocated_bytes": d["cuda_max_memory_allocated_bytes"], "peak_reserved_bytes": d["cuda_max_memory_reserved_bytes"]}, indent=2))'
