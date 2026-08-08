#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-storage-probe
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

# Report GPU-node mounts and GPFS filesets without writing outside the Slurm log.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

printf 'hostname=%s\n' "$(hostname)"
printf '%s\n' '== user scratch quota =='
mmlsquota -u "${QUOTA_USER:-$USER}" scratch
printf '%s\n' '== GPFS filesystem mount table =='
mmlsfs all -T
printf '%s\n' '== mounted filesystems =='
df -PT
printf '%s\n' '== candidate path metadata =='
for path in /scratch /scache /tmp /local /work /projects; do
  if [[ -e "$path" ]]; then
    stat -c '%n|device=%d|mode=%A|owner=%U|group=%G' "$path"
  fi
done
