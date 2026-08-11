#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-5task-formal-smoke-cleanup
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

# Delete only accepted formal-10M smoke leaves after their gate has passed.

set -euo pipefail
: "${SMOKE_RESULT_DIR:?SMOKE_RESULT_DIR is required}"
: "${SMOKE_ARTIFACT_DIR:?SMOKE_ARTIFACT_DIR is required}"
: "${SMOKE_PASS_FILE:?SMOKE_PASS_FILE is required}"
: "${FORMAL_BASE:?FORMAL_BASE is required}"

[[ -f "$SMOKE_PASS_FILE" ]] || { echo "smoke pass record is missing" >&2; exit 2; }
EXPECTED_RESULT="$FORMAL_BASE/smoke/atari_dqn_5task_fs4_stack4_l3_buf1m_lrdecay1m_10m_gru_seed1_smoke"
[[ "$SMOKE_RESULT_DIR" == "$EXPECTED_RESULT" ]] || { echo "unexpected smoke result path" >&2; exit 2; }
[[ "$SMOKE_ARTIFACT_DIR" == */atari_5task_18action_formal_10m_2mpertask/smoke ]] || {
  echo "unexpected smoke artifact path" >&2
  exit 2
}
[[ -d "$SMOKE_RESULT_DIR" && -d "$SMOKE_ARTIFACT_DIR" ]] || {
  echo "accepted smoke leaf already absent" >&2
  exit 2
}
find "$SMOKE_RESULT_DIR" -mindepth 1 -maxdepth 1 -printf '%p\n'
find "$SMOKE_ARTIFACT_DIR" -mindepth 1 -maxdepth 1 -printf '%p\n'
rm -rf "$SMOKE_RESULT_DIR" "$SMOKE_ARTIFACT_DIR"
