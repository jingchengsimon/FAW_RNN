#!/usr/bin/env bash
#SBATCH --job-name=aim3-5task-videos
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/rl/atari/amarel/artifacts/atari_5task_18action_videos/%j.out
#SBATCH --error=experiments/rl/atari/amarel/artifacts/atari_5task_18action_videos/%j.err

# Evaluate all three seeds, choose the median seed per task, and render one episode per model/task.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/run_task.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must point to persistent Amarel storage}"
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

RUN_BASE="${AIM3_RESULTS_PATH}/data/rl/atari/5task_18action/per_task_buf500k/pilot"
VIDEO_BASE="${AIM3_RESULTS_PATH}/videos/5task_18action"
SELECTION_BASE="${VIDEO_BASE}/selection_eval${EVAL_SEED:-20260812}_n${EVAL_EPISODES:-30}"
EVAL_SEED="${EVAL_SEED:-20260812}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
VIDEO_DEVICE="${VIDEO_DEVICE:-cuda}"
VIDEO_AMP_DTYPE="${VIDEO_AMP_DTYPE:-bfloat16}"
mkdir -p "$VIDEO_BASE" "$SELECTION_BASE"

declare -a MODELS=(gawf lstm)
declare -a TASKS=(ALE/Pong-v5 ALE/Breakout-v5 ALE/Assault-v5 ALE/Seaquest-v5 ALE/Skiing-v5)

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    task_slug="${task//\//_}"
    for seed in 1 2 3; do
      run_dir="${RUN_BASE}/atari_dqn_5task_fs4_stack4_l3_buf0p5m_lrdecay1m_pilot_"
      run_dir+="${model}_seed${seed}"
      metrics_path="${run_dir}/metrics.json"
      checkpoint_path="$(find "$run_dir" -maxdepth 1 -type f -name 'dqn_*.pth' -print -quit)"
      metadata_path="${SELECTION_BASE}/${model}_${task_slug}_seed${seed}.json"
      [[ -f "$metrics_path" ]] || { echo "Missing metrics: $metrics_path" >&2; exit 2; }
      [[ -n "$checkpoint_path" && -f "$checkpoint_path" ]] || {
        echo "Missing final checkpoint: $run_dir" >&2
        exit 2
      }
      [[ ! -e "$metadata_path" ]] || {
        echo "Refusing existing selection: $metadata_path" >&2
        exit 3
      }
      python -m utils.analysis.rl.atari.evaluate_dqn_video \
        --metrics_path "$metrics_path" \
        --checkpoint "$checkpoint_path" \
        --task_env_id "$task" \
        --metadata_path "$metadata_path" \
        --num_episodes "$EVAL_EPISODES" \
        --eval_seed "$EVAL_SEED" \
        --selection_only \
        --device "$VIDEO_DEVICE" \
        --amp_dtype "$VIDEO_AMP_DTYPE"
    done
  done
done

python - "$SELECTION_BASE" "$VIDEO_BASE" "$EVAL_SEED" <<'PY'
import json
from pathlib import Path
import sys

selection_base = Path(sys.argv[1])
video_base = Path(sys.argv[2])
eval_seed = int(sys.argv[3])
models = ("gawf", "lstm")
tasks = ("ALE/Pong-v5", "ALE/Breakout-v5", "ALE/Assault-v5", "ALE/Seaquest-v5", "ALE/Skiing-v5")
selected = []
for model in models:
    for task in tasks:
        task_slug = task.replace("/", "_")
        rows = []
        for seed in (1, 2, 3):
            path = selection_base / f"{model}_{task_slug}_seed{seed}.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            returns = metadata["episode_returns"]
            rows.append((sum(returns) / len(returns), seed, str(path)))
        rows.sort(key=lambda row: (row[0], row[1]))
        mean_return, seed, path = rows[1]
        selected.append(
            {
                "model": model,
                "task_env_id": task,
                "training_seed": seed,
                "mean_return": mean_return,
                "selection_metadata": path,
                "selection_rule": "median training seed by mean return over fixed evaluation suite",
            }
        )
(video_base / "selected_seeds.json").write_text(
    json.dumps(
        {
            "eval_seed": eval_seed,
            "selection_rule": "median training seed by mean return over fixed evaluation suite",
            "selected": selected,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

while IFS=$'\t' read -r model task seed; do
  task_slug="${task//\//_}"
  run_dir="${RUN_BASE}/atari_dqn_5task_fs4_stack4_l3_buf0p5m_lrdecay1m_pilot_${model}_seed${seed}"
  checkpoint_path="$(find "$run_dir" -maxdepth 1 -type f -name 'dqn_*.pth' -print -quit)"
  output_dir="${VIDEO_BASE}/${model}/${task_slug}"
  output_video="${output_dir}/${model}_${task_slug}_seed${seed}_eval${EVAL_SEED}.mp4"
  output_metadata="${output_dir}/metadata.json"
  [[ ! -e "$output_video" && ! -e "$output_metadata" ]] || {
    echo "Refusing existing video artifact: $output_dir" >&2
    exit 4
  }
  mkdir -p "$output_dir"
  [[ -n "$checkpoint_path" && -f "$checkpoint_path" ]] || {
    echo "Missing final checkpoint: $run_dir" >&2
    exit 4
  }
  python -m utils.analysis.rl.atari.evaluate_dqn_video \
    --metrics_path "${run_dir}/metrics.json" \
    --checkpoint "$checkpoint_path" \
    --task_env_id "$task" \
    --output_path "$output_video" \
    --metadata_path "$output_metadata" \
    --num_episodes 1 \
    --eval_seed "$EVAL_SEED" \
    --episode_selection first \
    --fps "${VIDEO_FPS:-15}" \
    --video_title "5-task full18 | ${model} L3 | seed ${seed} | ${task_slug}" \
    --device "$VIDEO_DEVICE" \
    --amp_dtype "$VIDEO_AMP_DTYPE"
done < <(
  python - "$VIDEO_BASE/selected_seeds.json" <<'PY'
import json
from pathlib import Path
import sys
for row in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["selected"]:
    print(f"{row['model']}\t{row['task_env_id']}\t{row['training_seed']}")
PY
)

find "$VIDEO_BASE" -type f -name '*.mp4' ! -path '*/raw_episodes_*/*' -size +0c \
  | wc -l | grep -qx '10'
echo "videos=$VIDEO_BASE selected_seeds=$VIDEO_BASE/selected_seeds.json"
