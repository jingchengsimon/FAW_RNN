#!/usr/bin/env bash
# Schedule guarded SIGUSR1 pauses for the active 15-unit DSW Atari protocol.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config.sh"

usage() {
  cat <<'EOF'
Usage: ./experiments/remote/pause_dsw_atari_5task.sh --at-et 'YYYY-MM-DD HH:MM' \
  [--check|--status|--cancel] [--dry-run]

The local experiments/remote/config.sh must define DSW_SSH_TARGET, DSW_PROJECT_ROOT,
DSW_ATARI_RESULTS_ROOT, and DSW_PORTS.  Scheduling first discovers exactly one active
writer for every ann/rnn/gru/lstm/gawf × seed1/2/3 unit across those nodes.  At the
requested America/New_York time, each node timer re-discovers and signals only those
matching Python processes; other processes on the same GPU are untouched.

Options:
  --at-et DATETIME  Required America/New_York wall-clock time.
  --check           Verify the current 15-writer scope without scheduling a timer.
  --status          Show retained timer output for that time.
  --cancel          Remove only the not-yet-triggered timers for that time.
  --dry-run         Print resolved actions without connecting or changing state.
  -h, --help        Show this help.
EOF
}

AT_ET=""
MODE="schedule"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --at-et) AT_ET="${2:-}"; shift 2 ;;
    --check) MODE="check"; shift ;;
    --status) MODE="status"; shift ;;
    --cancel) MODE="cancel"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$AT_ET" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}\ [0-9]{2}:[0-9]{2}$ ]] || {
  echo "--at-et must be 'YYYY-MM-DD HH:MM'" >&2
  exit 2
}
[[ -f "$CONFIG_PATH" ]] || { echo "Missing local config: $CONFIG_PATH" >&2; exit 2; }
# shellcheck source=/dev/null
source "$CONFIG_PATH"
: "${DSW_SSH_TARGET:?Set DSW_SSH_TARGET in experiments/remote/config.sh}"
: "${DSW_PROJECT_ROOT:?Set DSW_PROJECT_ROOT in experiments/remote/config.sh}"
: "${DSW_ATARI_RESULTS_ROOT:?Set DSW_ATARI_RESULTS_ROOT in experiments/remote/config.sh}"
declare -p DSW_PORTS >/dev/null 2>&1 || {
  echo "Set DSW_PORTS=(...) in experiments/remote/config.sh" >&2
  exit 2
}
(( ${#DSW_PORTS[@]} > 0 )) || { echo "DSW_PORTS must not be empty" >&2; exit 2; }

stamp="${AT_ET//[-: ]/}"
session="atari_pause_${stamp}_et_managed"

run_ssh() {
  local port="$1"
  shift
  ssh -o BatchMode=yes -p "$port" "$DSW_SSH_TARGET" "$@"
}

list_node_writers() {
  local port="$1"
  run_ssh "$port" bash -s -- "$DSW_ATARI_RESULTS_ROOT" <<'REMOTE'
set -euo pipefail
results_root="$1"
{
  ps -eo pid=,args= | grep 'run_task.py atari-dqn' | grep -F "$results_root/" | grep -v grep | \
    grep -v 'tmux new-session' | while IFS= read -r line; do
      pid="$(sed -E 's/^ *([0-9]+).*/\1/' <<<"$line")"
      suffix="$(sed -nE 's/.*--result_suffix ([^ ]+).*/\1/p' <<<"$line")"
      if [[ "$suffix" =~ _formal_(ann|rnn|gru|lstm|gawf)_seed([123])$ ]]; then
        gpu="$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')"
        printf '%s\t%s_seed%s\t%s\n' "$pid" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "$gpu"
      fi
    done
} || true
REMOTE
}

if [[ "$DRY_RUN" == true ]]; then
  printf 'mode=%s at_et=%s session=%s\n' "$MODE" "$AT_ET" "$session"
  printf 'ssh_target=%s results_root=%s\n' "$DSW_SSH_TARGET" "$DSW_ATARI_RESULTS_ROOT"
  printf 'ports=%s\n' "${DSW_PORTS[*]}"
  exit 0
fi

if [[ "$MODE" == "status" || "$MODE" == "cancel" ]]; then
  for port in "${DSW_PORTS[@]}"; do
    if [[ "$MODE" == "cancel" ]]; then
      run_ssh "$port" "tmux kill-session -t $(printf '%q' "$session") 2>/dev/null || true"
      echo "node_port=$port session=$session cancelled"
    else
      echo "node_port=$port session=$session"
      session_q="$(printf '%q' "$session")"
      run_ssh "$port" "tmux has-session -t $session_q 2>/dev/null && "\
        "tmux capture-pane -pt $session_q -S -20 || echo MISSING"
    fi
  done
  exit 0
fi

seen_units=()
seen_locations=()
writer_count=0

unit_index() {
  local wanted="$1"
  local index
  for index in "${!seen_units[@]}"; do
    [[ "${seen_units[$index]}" == "$wanted" ]] && {
      printf '%s\n' "$index"
      return 0
    }
  done
  return 1
}

for port in "${DSW_PORTS[@]}"; do
  while IFS=$'\t' read -r pid unit gpu; do
    [[ -n "$pid" && -n "$unit" ]] || continue
    if existing_index="$(unit_index "$unit")"; then
      echo "Duplicate active writer for $unit: ${seen_locations[$existing_index]}" >&2
      echo "and $port:$pid" >&2
      exit 3
    fi
    seen_units+=("$unit")
    seen_locations+=("$port:$pid:gpu${gpu:-unset}")
    writer_count=$((writer_count + 1))
  done < <(list_node_writers "$port")
done

for model in ann rnn gru lstm gawf; do
  for seed in 1 2 3; do
    unit="${model}_seed${seed}"
    unit_index "$unit" >/dev/null || { echo "Missing active writer: $unit" >&2; exit 3; }
  done
done
[[ "$writer_count" -eq 15 ]] || {
  echo "Expected 15 active writers, found $writer_count" >&2
  exit 3
}
for index in "${!seen_units[@]}"; do
  echo "writer=${seen_units[$index]} location=${seen_locations[$index]}"
done | sort

[[ "$MODE" == "check" ]] && exit 0

for port in "${DSW_PORTS[@]}"; do
  session_q="$(printf '%q' "$session")"
  run_ssh "$port" "tmux has-session -t $session_q 2>/dev/null && exit 4 || exit 0" || {
    echo "Timer already exists on port $port: $session" >&2
    exit 4
  }
done

for port in "${DSW_PORTS[@]}"; do
  run_ssh "$port" bash -s -- "$DSW_ATARI_RESULTS_ROOT" "$AT_ET" "$session" <<'REMOTE'
set -euo pipefail
results_root="$1"
at_et="$2"
session="$3"
target_epoch="$(TZ=America/New_York date -d "${at_et}:00" +%s)"
delay="$((target_epoch - $(date +%s)))"
[[ "$delay" -gt 0 ]] || { echo "STOP target time has passed" >&2; exit 5; }
timer_body=$(cat <<'TIMER'
set -euo pipefail
target_rows() {
  ps -eo pid=,args= | grep 'run_task.py atari-dqn' | grep -F "$RESULTS_ROOT/" | grep -v grep | \
    grep -v 'tmux new-session' | while IFS= read -r line; do
      pid="$(sed -E 's/^ *([0-9]+).*/\1/' <<<"$line")"
      suffix="$(sed -nE 's/.*--result_suffix ([^ ]+).*/\1/p' <<<"$line")"
      if [[ "$suffix" =~ _formal_(ann|rnn|gru|lstm|gawf)_seed([123])$ ]]; then
        printf '%s\t%s_seed%s\t%s\n' "$pid" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "$suffix"
      fi
    done
}
echo "scheduled target_epoch=$TARGET_EPOCH delay_s=$DELAY"
sleep "$DELAY"
mapfile -t rows < <(target_rows || true)
if (( ${#rows[@]} > 15 )); then
  echo "STOP unexpected_writer_count=${#rows[@]}"
  exit 6
fi
pids=()
units=()
suffixes=()
for row in "${rows[@]}"; do
  IFS=$'\t' read -r pid unit suffix <<<"$row"
  pids+=("$pid")
  units+=("$unit")
  suffixes+=("$suffix")
done
echo "matched=${#pids[@]} units=${units[*]:-none}"
trigger_epoch="$(date +%s)"
if (( ${#pids[@]} > 0 )); then
  kill -USR1 "${pids[@]}"
  echo "SIGUSR1_sent=${pids[*]}"
else
  echo "no_active_matching_writers"
fi
sleep 75
mapfile -t remaining_rows < <(target_rows || true)
fresh=0
for suffix in "${suffixes[@]}"; do
  checkpoint="$RESULTS_ROOT/$suffix/checkpoint.pth"
  if [[ -f "$checkpoint" && "$(stat -c %Y "$checkpoint")" -ge "$trigger_epoch" ]]; then
    fresh=$((fresh + 1))
  fi
done
echo "remaining_after_pause=${#remaining_rows[@]} fresh_checkpoints=$fresh"
(( ${#remaining_rows[@]} == 0 )) || exit 7
(( fresh == ${#pids[@]} )) || exit 8
TIMER
)
printf -v timer_command '%q ' env "TARGET_EPOCH=$target_epoch" "DELAY=$delay" \
  "RESULTS_ROOT=$results_root" bash -lc "$timer_body"
tmux new-session -d -s "$session" -c "$(dirname "$results_root")" "$timer_command"
tmux set-window-option -t "$session" remain-on-exit on
echo "node_port=$port session=$session target_et=$at_et delay_s=$delay"
REMOTE
done
