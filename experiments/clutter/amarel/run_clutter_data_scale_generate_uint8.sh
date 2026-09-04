#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-data
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00

# Generate one formal uint8 Clutter training scale on an Amarel compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
MNIST_ROOT="${AIM3_MNIST_ROOT:?AIM3_MNIST_ROOT is required}"
STATUS_DIR="${AIM3_STATUS_DIR:?AIM3_STATUS_DIR is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Slurm array task id is required}"
(( TASK_ID >= 0 && TASK_ID < 3 )) || { echo "Task id must be in [0, 2]" >&2; exit 2; }

HOURS=(4 10 20)
HOUR="${HOURS[TASK_ID]}"
SUFFIX="${HOUR}h-uint8"
EXPECTED_FRAMES=$(( HOUR * 3600 * 24 ))
TARGET_NPY="$DATA_DIR/stimulus_reg-train-$SUFFIX.npy"
TARGET_TSV="$DATA_DIR/stimulus_reg-train-$SUFFIX.tsv"
TARGET_MANIFEST="$DATA_DIR/generation-$SUFFIX.json"
STAGING_JOB_ID="${AIM3_STAGING_JOB_ID:-$SLURM_ARRAY_JOB_ID}"
STAGING="$DATA_DIR/.clutter_scale_generation/$STAGING_JOB_ID/task_$TASK_ID"
STAGED_NPY="$STAGING/stimulus_reg-train-$SUFFIX.npy"
STAGED_TSV="$STAGING/stimulus_reg-train-$SUFFIX.tsv"
for target in "$TARGET_NPY" "$TARGET_TSV" "$TARGET_MANIFEST"; do
  [[ ! -e "$target" ]] || { echo "Refusing to overwrite existing target: $target" >&2; exit 1; }
done
mkdir -p "$STATUS_DIR"

FAIL_FILE="$STATUS_DIR/task_$TASK_ID.$SLURM_ARRAY_JOB_ID.fail"
on_error() {
  status=$?
  trap - ERR
  printf 'status=failed task=%s scale=%s exit=%s staging=%s timestamp=%s\n' \
    "$TASK_ID" "$SUFFIX" "$status" "$STAGING" "$(date -Is)" > "$FAIL_FILE"
  exit "$status"
}
trap on_error ERR

cd "$ROOT"
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

if [[ -e "$STAGING" ]]; then
  [[ -s "$STAGED_NPY" && -s "$STAGED_TSV" ]] || {
    echo "Recovery staging leaf is incomplete: $STAGING" >&2
    exit 1
  }
  printf 'Reusing completed staging output: %s\n' "$STAGING"
else
  mkdir -p "$STAGING"
  python -B source/clutter/generate_movies.py \
    --hour "$HOUR" --storage-dtype uint8 --split train --output-mode simple \
    --switch-mode exclusive --seed 42 --output-dir "$STAGING" --mnist-root "$MNIST_ROOT"
fi

python -B -c '
import sys
import numpy as np
a = np.load(sys.argv[1], mmap_mode="r")
assert a.dtype == np.uint8
assert a.shape == (int(sys.argv[2]), 96, 96)
' "$STAGED_NPY" "$EXPECTED_FRAMES"
TSV_ROWS="$(wc -l < "$STAGED_TSV")"
(( TSV_ROWS == EXPECTED_FRAMES + 1 )) || {
  echo "Unexpected TSV row count: $TSV_ROWS (expected $(( EXPECTED_FRAMES + 1 )))" >&2
  exit 1
}

SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
NPY_BYTES="$(stat -c '%s' "$STAGED_NPY")"
TSV_BYTES="$(stat -c '%s' "$STAGED_TSV")"
{
  printf '{\n'
  printf '  "status": "complete",\n'
  printf '  "source_commit": "%s",\n' "$SOURCE_COMMIT"
  printf '  "generator": "source/clutter/generate_movies.py",\n'
  printf '  "scale": "%s",\n' "$SUFFIX"
  printf '  "seed": 42,\n'
  printf '  "dtype": "uint8",\n'
  printf '  "shape": [%s, 96, 96],\n' "$EXPECTED_FRAMES"
  printf '  "npy_bytes": %s,\n' "$NPY_BYTES"
  printf '  "tsv_bytes": %s\n' "$TSV_BYTES"
  printf '}\n'
} > "$STAGING/generation-$SUFFIX.json"

mv "$STAGED_NPY" "$TARGET_NPY"
mv "$STAGED_TSV" "$TARGET_TSV"
mv "$STAGING/generation-$SUFFIX.json" "$TARGET_MANIFEST"
rmdir "$STAGING"
printf 'status=done task=%s scale=%s frames=%s timestamp=%s\n' \
  "$TASK_ID" "$SUFFIX" "$EXPECTED_FRAMES" "$(date -Is)" \
  > "$STATUS_DIR/task_$TASK_ID.$SLURM_ARRAY_JOB_ID.done"
trap - ERR
