"""Static and dry-run coverage for the sjc multi-task L3 GRU launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/remote/run_sjc_atari_multitask_l3_gru.sh"


def _dry_run(*args: str) -> str:
    result = subprocess.run(
        ["bash", str(RUNNER), *args, "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_two_task_dry_run_locks_requested_protocol() -> None:
    output = _dry_run("--mode", "two-task", "--cuda-device", "1")
    assert "ALE/Pong-v5" in output and "ALE/Breakout-v5" in output
    assert "--action_space_mode full18" in output
    assert "--num_layers 3" in output and "--hidden_size 458" in output
    assert "--replay_layout per_task" in output and "--buffer_size 1000000" in output
    assert "--task_schedule transition_balanced" in output
    assert "--replay_sampling task_balanced" in output
    assert "--learning_rate_decay_step 0" in output
    assert "--learning_rate_decay_per_task_steps 1000000" in output
    assert "--amp_dtype bfloat16" in output
    assert "--allow_tf32" in output and "--cudnn_benchmark" in output
    assert "--fused_optimizer" in output
    assert "--total_timesteps 4000000" in output
    assert "CUDA_VISIBLE_DEVICES=1" in output


def test_single_task_smoke_isolated_and_has_compatible_protocol() -> None:
    output = _dry_run("--mode", "breakout", "--cuda-device", "0", "--smoke")
    assert "breakout_only" in output and "_smoke" in output
    assert "--env_id ALE/Breakout-v5" in output
    assert "--total_timesteps 50000" in output
    assert "--replay_layout per_task" not in output
    assert "--task_schedule transition_balanced" not in output
    assert "--learning_rate_decay_step 1000000" in output
    assert "--learning_rate_decay_per_task_steps 0" in output
    assert "--learning_starts_per_task 0" in output


def test_history_without_checkpoint_is_refused(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    run_tag = "existing"
    result_dir = result_root / "data/rl/atari/multitask_18action" / run_tag
    result_dir.mkdir(parents=True)
    (result_dir / "metrics_history.jsonl").touch()
    result = subprocess.run(
        [
            "bash", str(RUNNER), "--mode", "pong", "--cuda-device", "0",
            "--results-root", str(result_root), "--run-tag", run_tag, "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "without resumable checkpoint" in result.stderr
