#!/usr/bin/env bash
#SBATCH --job-name=aim3-dssk-scratch-attr
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00

# Read-only, race-tolerant attribution of one user's visible DSSK scratch usage.

set -uo pipefail

scratch_root="${SCRATCH_ROOT:-/scratch/js3269}"
results_root="$scratch_root/results"
train_root="$results_root/train_data"

report_tree() {
  local root="$1"
  local title="$2"
  local -a entries=()
  local entry
  printf '== %s ==\n' "$title"
  shopt -s dotglob nullglob
  entries=("$root"/*)
  for entry in "${entries[@]}"; do
    # Scratch cleanup may race this report.  Missing paths are recorded but do
    # not invalidate the remaining summary.
    if [[ ! -e "$entry" ]]; then
      printf 'MISSING\t%s\n' "$entry" >&2
      continue
    fi
    du -x -sk -- "$entry" 2>/dev/null || printf 'UNREADABLE\t%s\n' "$entry" >&2
  done | sort -rn | head -100
}

printf '%s\n' '== DSSK quota =='
mmlsquota -u "${QUOTA_USER:-js3269}" scratch
report_tree "$scratch_root" 'largest scratch top-level entries (KiB)'
report_tree "$results_root" 'largest results entries (KiB)'
report_tree "$train_root" 'largest results/train_data entries (KiB)'
printf '%s\n' '== deleted-but-open files owned by user on this node =='
if command -v lsof >/dev/null 2>&1; then
  lsof -nP -u "${QUOTA_USER:-js3269}" +L1 2>/dev/null | head -200 || true
else
  echo 'lsof unavailable'
fi
