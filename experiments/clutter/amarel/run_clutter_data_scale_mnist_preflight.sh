#!/usr/bin/env bash
#SBATCH --job-name=aim3-mnist-preflight
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# Download or validate the shared MNIST cache on an Amarel compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
MNIST_ROOT="${AIM3_MNIST_ROOT:?AIM3_MNIST_ROOT is required}"
STATUS_DIR="${AIM3_STATUS_DIR:?AIM3_STATUS_DIR is required}"
mkdir -p "$STATUS_DIR" "$MNIST_ROOT"
cd "$ROOT"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

export AIM3_MNIST_ROOT="$MNIST_ROOT"
python -B -c '
import os
from torchvision.datasets import MNIST
MNIST(root=os.environ["AIM3_MNIST_ROOT"], train=True, download=True)
'
printf 'status=done timestamp=%s\n' "$(date -Is)" > "$STATUS_DIR/mnist.done"
