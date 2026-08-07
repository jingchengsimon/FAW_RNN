#!/usr/bin/env bash
#SBATCH --job-name=aim3-gawf-fullscan-bench
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Run one deterministic L3 GaWF full-scan compile benchmark on a compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

: "${AIM3_SOURCE_SNAPSHOT:?AIM3_SOURCE_SNAPSHOT is required}"
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ -d "$AIM3_SOURCE_SNAPSHOT" ]] || { echo "Missing snapshot: $AIM3_SOURCE_SNAPSHOT" >&2; exit 2; }

case "$SLURM_ARRAY_TASK_ID" in
  0) TAG="b4t8"; BATCH_SIZE=4; SEQ_LEN=8; WARMUP=2; ITERATIONS=5 ;;
  1) TAG="b8t16"; BATCH_SIZE=8; SEQ_LEN=16; WARMUP=5; ITERATIONS=20 ;;
  *) echo "Unsupported benchmark task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

OUTPUT_DIR="$AIM3_RESULTS_PATH/data/rl/atari/5task_18action/gawf_fullscan_compile_benchmark"
OUTPUT_JSON="$OUTPUT_DIR/amarel_${TAG}.json"
[[ ! -e "$OUTPUT_JSON" ]] || { echo "Refusing to overwrite: $OUTPUT_JSON" >&2; exit 3; }
mkdir -p "$OUTPUT_DIR"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
cd "$AIM3_SOURCE_SNAPSHOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

timeout 25m python -m experiments.rl.atari.benchmark_gawf_fullscan_compile \
  --output "$OUTPUT_JSON" --batch-size "$BATCH_SIZE" --seq-len "$SEQ_LEN" \
  --warmup "$WARMUP" --iterations "$ITERATIONS" --amp-dtype bfloat16

[[ -s "$OUTPUT_JSON" ]] || { echo "Missing benchmark JSON: $OUTPUT_JSON" >&2; exit 4; }
