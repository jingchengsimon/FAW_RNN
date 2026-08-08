#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-scratch-usage
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00

# Read-only DSSK-side accounting and largest-directory report for one user.

set -euo pipefail

scratch_root="${SCRATCH_ROOT:-/scratch/js3269}"
results_root="$scratch_root/results"
train_root="$results_root/train_data"

printf '%s\n' '== DSSK quota =='
mmlsquota -u "${QUOTA_USER:-js3269}" scratch
printf '%s\n' '== scratch total =='
du -x -sk -- "$scratch_root"
printf '%s\n' '== largest scratch top-level entries (KiB) =='
shopt -s dotglob nullglob
top_entries=("$scratch_root"/*)
du -x -sk -- "${top_entries[@]}" | sort -rn | head -40
printf '%s\n' '== largest results/train_data entries (KiB) =='
train_entries=("$train_root"/*)
du -x -sk -- "${train_entries[@]}" | sort -rn | head -80
