"""Static coverage for the Amarel five-task formal 10M launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMITTER = ROOT / "experiments/rl/atari/amarel/submit_atari_5task_18action_formal_10m.sh"
RUNNER = ROOT / "experiments/rl/atari/amarel/run_atari_5task_18action_formal_10m_array.sh"


def test_formal_submitter_dry_run_locks_gate_mapping_and_conservative_throttle() -> None:
    output = subprocess.run(
        ["bash", str(SUBMITTER), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "10M global steps (=2M/task)" in output
    assert "GRU L3/h458 + LSTM L3/h373" in output
    assert "afterok:<smoke_jobid>" in output
    assert "array=0-5%1" in output
    assert "torch.compile disabled" in output
    assert "--after-smoke JOB_ID" in output
    assert "retry isolation: --run-tag TAG" in output


def test_formal_runner_locks_the_requested_replay_and_optimizer_protocol() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for required in (
        "MODELS=(gru lstm)",
        "HIDDEN_SIZES=(458 373)",
        "--total_timesteps \"$TOTAL_TIMESTEPS\"",
        "--replay_layout per_task",
        "--buffer_size 1000000",
        "--learning_rate_decay_step 0",
        "--learning_rate_decay_per_task_steps 1000000",
        "--learning_starts_per_task 20000",
        "--start_epsilon 1.0 --end_epsilon 0.01",
        "--exploration_steps 500000",
        "--amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer",
        "--required_gib 140",
        "kill -USR1 \"$TRAIN_PID\"",
        'os.path.join(task_dir, "meta.json")',
        "#SBATCH --constraint=adalovelace",
    ):
        assert required in text
    assert "replay_meta.json" not in text
    assert "compile_model" not in text


def test_formal_runner_validates_an_interrupted_smoke_before_resume() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    validation = 'controlled smoke interruption is missing its resumable checkpoint'
    assert validation in text
    assert text.index(validation) < text.index("TRAIN_ARGS=(")


def test_submitter_can_reuse_an_already_submitted_smoke() -> None:
    text = SUBMITTER.read_text(encoding="utf-8")
    assert "--after-smoke" in text
    assert 'SMOKE_JOB_ID="$AFTER_SMOKE_ID"' in text


def test_submitter_uses_distinct_retry_leaves() -> None:
    output = subprocess.run(
        ["bash", str(SUBMITTER), "--dry-run", "--run-tag", "r2"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "formal_10m_2mpertask_r2" in output
    assert "atari_5task_18action_formal_10m_2mpertask_r2" in output
    assert '[[ ! -e "$ARTIFACT_ROOT" ]]' in SUBMITTER.read_text(encoding="utf-8")
