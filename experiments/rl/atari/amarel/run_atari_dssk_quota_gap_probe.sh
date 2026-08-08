#!/usr/bin/env bash
#SBATCH --job-name=aim3-dssk-quota-gap
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00

# Compare DSSK user quota accounting with user-owned inodes visible below the user's scratch root.

set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1

scratch_root="${SCRATCH_ROOT:-/scratch/js3269}"
quota_user="${QUOTA_USER:-js3269}"
error_log="$(mktemp)"
trap 'rm -f -- "$error_log"' EXIT

printf 'hostname=%s started=%s\n' "$(hostname)" "$(date -Is)"
printf '%s\n' '== quota before scan =='
mmlsquota -u "$quota_user" scratch
printf '%s\n' '== unique user-owned inodes below scratch root =='
set +e
find "$scratch_root" -xdev -user "$quota_user" -printf '%D:%i\t%b\t%p\n' \
  2>"$error_log" |
  awk -F '\t' -v root="$scratch_root/" '
    {
      key = $1
      if (seen[key]++) {
        next
      }
      blocks = $2 + 0
      path = $3
      relative = path
      sub("^" root, "", relative)
      top = relative
      sub("/.*", "", top)
      if (top == path || top == "") {
        top = "."
      }
      total_inodes += 1
      total_blocks += blocks
      top_inodes[top] += 1
      top_blocks[top] += blocks
    }
    END {
      printf "TOTAL\tinodes=%d\tallocated_kib=%.0f\n", total_inodes, total_blocks / 2
      for (top in top_inodes) {
        printf "TOP\t%s\tinodes=%d\tallocated_kib=%.0f\n", \
          top, top_inodes[top], top_blocks[top] / 2
      }
    }
  '
find_status="${PIPESTATUS[0]}"
set -e
printf 'find_status=%s find_error_lines=%s\n' "$find_status" "$(wc -l < "$error_log")"
head -100 "$error_log"
printf '%s\n' '== quota after scan =='
mmlsquota -u "$quota_user" scratch
printf 'finished=%s\n' "$(date -Is)"
