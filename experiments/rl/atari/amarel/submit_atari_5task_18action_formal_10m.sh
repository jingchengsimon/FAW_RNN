#!/usr/bin/env bash
# Submit the five-task 10M formal GRU/LSTM array behind one recoverable smoke gate.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DRY_RUN=0
AFTER_SMOKE_ID=""
SMOKE_ATTEMPT=""
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --after-smoke)
      AFTER_SMOKE_ID="${2:-}"
      shift 2
      ;;
    --smoke-attempt)
      SMOKE_ATTEMPT="${2:-}"
      shift 2
      ;;
    *) echo "Usage: $0 [--dry-run] [--after-smoke JOB_ID] [--smoke-attempt TAG]" >&2; exit 2 ;;
  esac
done
[[ -z "$AFTER_SMOKE_ID" || "$AFTER_SMOKE_ID" =~ ^[0-9]+$ ]] || {
  echo "--after-smoke must be a Slurm job ID" >&2
  exit 2
}
[[ -z "$SMOKE_ATTEMPT" || "$SMOKE_ATTEMPT" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "--smoke-attempt must contain only letters, digits, underscores, or hyphens" >&2
  exit 2
}

BASE_REL="data/rl/atari/5task_18action/per_task_buf1m/formal_10m_2mpertask"
ARTIFACT_TAG="atari_5task_18action_formal_10m_2mpertask"
FORMAL_PREFIX="atari_dqn_5task_fs4_stack4_l3_buf1m_lrdecay1m_10m_"
if (( DRY_RUN )); then
  cat <<EOF
protocol: five-task full18; fs4/stack4; transition_balanced/task_balanced; per-task mmap=1M
formal: 10M global steps (=2M/task); GRU L3/h458 + LSTM L3/h373; seeds=1,2,3
acceleration: bfloat16 TF32 cudnn_benchmark fused_optimizer; torch.compile disabled
smoke: 500 steps, controlled SIGUSR1 checkpoint/resume, then exact smoke-leaf cleanup
formal dependency: afterok:<smoke_jobid>; array=0-5%1 (conservative: user quota unverified)
reuse: --after-smoke JOB_ID attaches cleanup/formal to an already-submitted smoke without a rerun
retry: --smoke-attempt TAG uses a distinct smoke leaf while preserving formal suffixes
result root: \$AIM3_RESULTS_PATH/$BASE_REL
formal suffix prefix: $FORMAL_PREFIX
EOF
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }
FORMAL_BASE="$AIM3_RESULTS_PATH/$BASE_REL"
ARTIFACT_ROOT="$ROOT/experiments/rl/atari/amarel/artifacts/$ARTIFACT_TAG"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_5task_18action_formal_10m_array.sh"
CLEANUP_RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_5task_18action_formal_10m_smoke_cleanup.sh"
SMOKE_SUFFIX="${FORMAL_PREFIX}gru_seed1_smoke"
[[ -z "$SMOKE_ATTEMPT" ]] || SMOKE_SUFFIX+="_$SMOKE_ATTEMPT"
SMOKE_RESULT_DIR="$FORMAL_BASE/smoke/$SMOKE_SUFFIX"
SMOKE_ARTIFACT_DIR="$ARTIFACT_ROOT/smoke"
SMOKE_PASS_FILE="$ARTIFACT_ROOT/status/${SMOKE_SUFFIX}.smoke_pass"

[[ -x "$RUNNER" ]] || { echo "runner is not executable: $RUNNER" >&2; exit 2; }
[[ -x "$CLEANUP_RUNNER" ]] || { echo "cleanup runner is not executable: $CLEANUP_RUNNER" >&2; exit 2; }
for model in gru lstm; do
  for seed in 1 2 3; do
    suffix="${FORMAL_PREFIX}${model}_seed${seed}"
    [[ ! -e "$FORMAL_BASE/$suffix" ]] || { echo "result suffix exists: $suffix" >&2; exit 3; }
  done
done
[[ ! -e "$SMOKE_RESULT_DIR" ]] || { echo "smoke result leaf exists: $SMOKE_RESULT_DIR" >&2; exit 3; }

while IFS= read -r job_id; do
  [[ "$job_id" == "$AFTER_SMOKE_ID" ]] && continue
  scontrol show job "$job_id" 2>/dev/null | grep -Fq "$FORMAL_PREFIX" && {
    echo "active Slurm writer references formal suffix prefix: $job_id" >&2
    exit 3
  }
done < <(squeue -h -u "${USER:-js3269}" -o '%i')
ps -fu "${USER:-js3269}" -o args= 2>/dev/null | grep -F "$FORMAL_PREFIX" | grep -v grep && {
  echo "active process references formal suffix prefix" >&2
  exit 3
} || true

mkdir -p "$ARTIFACT_ROOT/smoke" "$ARTIFACT_ROOT/formal" "$ARTIFACT_ROOT/cleanup" \
  "$ARTIFACT_ROOT/status"
COMMON="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,FORMAL_BASE=$FORMAL_BASE"
COMMON="$COMMON,ARTIFACT_ROOT=$ARTIFACT_ROOT,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
SMOKE_EXPORTS="$COMMON,RUN_PHASE=smoke"
if [[ -n "$AFTER_SMOKE_ID" ]]; then
  SMOKE_JOB_ID="$AFTER_SMOKE_ID"
else
  SMOKE_RAW="$(sbatch --parsable --job-name=aim3-atari-5task-formal-smoke --time=02:00:00 \
    --chdir="$ROOT" --output="$SMOKE_ARTIFACT_DIR/%j.out" --error="$SMOKE_ARTIFACT_DIR/%j.err" \
    --export="ALL,$SMOKE_EXPORTS" "$RUNNER")"
  SMOKE_JOB_ID="${SMOKE_RAW%%;*}"
fi

CLEANUP_EXPORTS="$COMMON,SMOKE_RESULT_DIR=$SMOKE_RESULT_DIR"
CLEANUP_EXPORTS="$CLEANUP_EXPORTS,SMOKE_ARTIFACT_DIR=$SMOKE_ARTIFACT_DIR"
CLEANUP_EXPORTS="$CLEANUP_EXPORTS,SMOKE_PASS_FILE=$SMOKE_PASS_FILE"
CLEANUP_RAW="$(sbatch --parsable --job-name=aim3-atari-5task-formal-smoke-cleanup \
  --chdir="$ROOT" --output="$ARTIFACT_ROOT/cleanup/%j.out" --error="$ARTIFACT_ROOT/cleanup/%j.err" \
  --dependency="afterok:$SMOKE_JOB_ID" --export="ALL,$CLEANUP_EXPORTS" "$CLEANUP_RUNNER")"

FORMAL_EXPORTS="$COMMON,RUN_PHASE=formal"
FORMAL_RAW="$(sbatch --parsable --job-name=aim3-atari-5task-formal-10m --array=0-5%1 \
  --chdir="$ROOT" --output="$ARTIFACT_ROOT/formal/%A_%a.out" \
  --error="$ARTIFACT_ROOT/formal/%A_%a.err" --dependency="afterok:$SMOKE_JOB_ID" \
  --export="ALL,$FORMAL_EXPORTS" "$RUNNER")"

echo "SMOKE_JOB_ID=$SMOKE_JOB_ID"
echo "SMOKE_CLEANUP_JOB_ID=${CLEANUP_RAW%%;*}"
echo "FORMAL_JOB_ID=${FORMAL_RAW%%;*}"
echo "ARRAY_MAPPING=0-2:gru_seed1-3,3-5:lstm_seed1-3"
echo "CONCURRENCY=1 (conservative; compute-node user quota was not parsable)"
echo "RESULT_ROOT=$FORMAL_BASE"
