"""Unit tests for task-balanced Atari collection, replay, and learning warm-up."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from utils.training.atari.atari_envs import _EpisodeTaskScheduler
from utils.training.atari.atari_replay import AtariReplayBuffer, PerTaskAtariReplayBuffer
from utils.training.train_scripts.atari_dqn import _learning_ready


ROOT = Path(__file__).resolve().parents[2]


def _buffer(seed: int = 7) -> AtariReplayBuffer:
    buffer = AtariReplayBuffer(
        buffer_size=100,
        num_envs=1,
        obs_shape=(1, 2, 2),
        device="cpu",
        seed=seed,
        num_tasks=5,
        sampling_mode="task_balanced",
    )
    buffer._stored_task_counts[:] = 2
    return buffer


def test_replay_remainder_rotates_for_transition_and_sequence_batches() -> None:
    for batch_size in (32, 8):
        buffer = _buffer()
        counts = np.zeros(5, dtype=np.int64)
        for _ in range(5):
            counts += np.bincount(
                buffer._balanced_task_targets(batch_size),
                minlength=5,
            )
        assert counts.tolist() == [batch_size] * 5


def test_replay_remainder_cursor_resumes_exactly() -> None:
    uninterrupted = _buffer()
    uninterrupted._balanced_task_targets(32)
    state = uninterrupted.state_dict()

    resumed = _buffer(seed=999)
    resumed.load_state_dict(state)

    assert np.array_equal(
        uninterrupted._balanced_task_targets(8),
        resumed._balanced_task_targets(8),
    )
    assert uninterrupted.state_dict()["remainder_cursor"] == resumed.state_dict()[
        "remainder_cursor"
    ]


def test_old_replay_state_defaults_remainder_cursor_to_zero() -> None:
    source = _buffer()
    state = source.state_dict()
    state.pop("remainder_cursor")

    restored = _buffer(seed=999)
    restored.load_state_dict(state)

    assert restored.state_dict()["remainder_cursor"] == 0


def test_per_task_replay_keeps_independent_partitions() -> None:
    buffer = PerTaskAtariReplayBuffer(
        buffer_size_per_task=20,
        num_envs=1,
        obs_shape=(1, 2, 2),
        device="cpu",
        seed=7,
        num_tasks=5,
        sampling_mode="task_balanced",
    )
    for step in range(20):
        buffer.add(
            obs=np.full((1, 1, 2, 2), step, dtype=np.uint8),
            actions=np.zeros(1, dtype=np.int64),
            rewards=np.zeros(1, dtype=np.float32),
            dones=np.zeros(1, dtype=np.uint8),
            resets=np.zeros(1, dtype=np.uint8),
            task_ids=np.asarray([step % 5], dtype=np.int16),
        )

    state = buffer.state_dict()
    assert state["replay_layout"] == "per_task"
    assert [task["pos"] for task in state["task_states"]] == [4] * 5
    batch = buffer.sample_transitions(10)
    assert np.bincount(batch.task_ids.numpy(), minlength=5).tolist() == [2] * 5


def test_scheduler_state_round_trip_preserves_next_choice() -> None:
    uninterrupted = _EpisodeTaskScheduler(5, start_idx=2, mode="transition_balanced")
    for task_idx, count in enumerate((11, 3, 7, 3, 9)):
        for _ in range(count):
            uninterrupted.record_step(task_idx)
    uninterrupted.next_task()
    state = uninterrupted.state_dict()

    resumed = _EpisodeTaskScheduler(5, start_idx=0, mode="transition_balanced")
    resumed.load_state_dict(state)

    assert resumed.next_task() == uninterrupted.next_task()
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_learning_waits_for_every_task_threshold() -> None:
    args = Namespace(learning_starts=20_000, learning_starts_per_task=20_000)
    counts = {f"task_{index}": 20_000 for index in range(5)}

    assert not _learning_ready(args, 19_999, counts)
    counts["task_4"] = 19_999
    assert not _learning_ready(args, 100_000, counts)
    counts["task_4"] = 20_000
    assert _learning_ready(args, 100_000, counts)


def test_per_task_pilot_completion_validator_only_checks_required_artifacts() -> None:
    """Completion requires only final artifacts, requested steps, and finite loss."""

    runner = (
        ROOT / "experiments/rl/atari/amarel/run_atari_5task_18action_l3_array.sh"
    ).read_text(encoding="utf-8")
    validator = runner.split('python - "$RESULT_DIR" "$TOTAL_TIMESTEPS"', 1)[1]

    forbidden_metadata = (
        "env_ids",
        "multitask",
        "action_space_mode",
        "num_actions",
        "task_schedule",
        "replay_sampling",
        "model_type",
        "num_layers",
        "frame_skip",
        "frame_stack",
        "replay_layout",
        "buffer_size",
        "batch_size",
        "seq_len",
        "sequences_per_batch",
        "learning_starts_per_task",
        "learning_started_at_step",
        "task_scheduler_states",
        "replay_remainder_cursor",
        "learning_rate_decay_step",
        "learning_rate_decay_per_task_steps",
    )
    assert all(f'"{field}"' not in validator for field in forbidden_metadata)
    assert '"global_step"' in validator
    assert '"loss"' in validator
    assert '"metrics_history.jsonl"' in validator
    assert "Missing final or resumable checkpoint" in validator
    assert "!= 1" not in validator
