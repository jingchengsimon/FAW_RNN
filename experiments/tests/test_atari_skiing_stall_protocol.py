"""Regression tests for the versioned Skiing boundary and canonical action fix."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import utils.training.atari.atari_envs as atari_envs
from utils.analysis.rl.atari.evaluate_dqn_video import training_seed_from_metrics
from utils.training.atari.atari_envs import (
    SKIING_STALL_RETURN_FLOOR,
    SKIING_STALL_STEPS,
    _canonical_18_action_space,
    _skiing_stall_boundary,
)
from utils.training.train_scripts.atari_dqn import (
    RESUME_ARG_KEYS,
    _extract_episode_records,
    _load_initial_weights,
    _replay_boundary_flags,
    _resume_validation_keys,
    build_arg_parser,
)


class _FakeAle:
    def __init__(self) -> None:
        self.ram = np.zeros(128, dtype=np.uint8)

    def getRAM(self) -> np.ndarray:
        return self.ram


class _SkiingEnv(gym.Env):
    metadata: dict[str, object] = {}

    def __init__(self, *, progress: bool, terminate_at: int | None = None) -> None:
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(0, 255, (2,), dtype=np.uint8)
        self.ale = _FakeAle()
        self.progress = progress
        self.terminate_at = terminate_at
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.ale.ram.fill(0)
        return np.zeros(2, dtype=np.uint8), {}

    def step(self, action):
        self.steps += 1
        if self.progress:
            self.ale.ram[86] = self.steps % 251
        terminated = self.terminate_at == self.steps
        return np.zeros(2, dtype=np.uint8), -1.0, terminated, False, {}


class _ActionEnv(gym.Env):
    metadata: dict[str, object] = {}

    def __init__(self, action_values: list[int]) -> None:
        self._action_set = [SimpleNamespace(value=value) for value in action_values]
        self.action_space = gym.spaces.Discrete(len(action_values))
        self.observation_space = gym.spaces.Box(0, 1, (1,), dtype=np.uint8)
        self.last_action: int | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.uint8), {}

    def step(self, action):
        self.last_action = int(action)
        return np.zeros(1, dtype=np.uint8), 0.0, False, False, {}


def test_normal_downhill_progress_never_truncates() -> None:
    env = _skiing_stall_boundary(_SkiingEnv(progress=True), gym)
    env.reset()
    for _ in range(SKIING_STALL_STEPS + 20):
        _obs, _reward, terminated, truncated, info = env.step(0)
        assert not terminated and not truncated
        assert "end_reason" not in info


def test_stall_truncates_exactly_at_threshold_with_return_floor() -> None:
    env = _skiing_stall_boundary(_SkiingEnv(progress=False), gym)
    env.reset()
    total_return = 0.0
    for step in range(1, SKIING_STALL_STEPS + 1):
        _obs, reward, terminated, truncated, info = env.step(0)
        total_return += reward
        assert not terminated
        assert truncated is (step == SKIING_STALL_STEPS)
    assert total_return == SKIING_STALL_RETURN_FLOOR
    assert info["end_reason"] == "stalled"
    assert info["stall_steps"] == SKIING_STALL_STEPS
    assert info["course_progress_events"] == 0
    assert info["stall_reward_adjustment"] < 0


def test_natural_terminal_takes_precedence_over_stall_boundary() -> None:
    env = _skiing_stall_boundary(
        _SkiingEnv(progress=False, terminate_at=SKIING_STALL_STEPS), gym
    )
    env.reset()
    for _ in range(SKIING_STALL_STEPS):
        _obs, _reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert "end_reason" not in info


def test_skiing_maps_one_full18_head_to_nine_non_fire_actions() -> None:
    base = _ActionEnv([0, 2, 3, 4, 5, 6, 7, 8, 9])
    wrapped = _canonical_18_action_space(base, gym)
    expected = (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8)
    assert wrapped.action_space.n == 18
    assert wrapped.canonical_action_mapping == expected
    for canonical, legal_index in enumerate(expected):
        wrapped.step(canonical)
        assert base.last_action == legal_index


def test_full_ale_action_set_keeps_identity_mapping() -> None:
    wrapped = _canonical_18_action_space(_ActionEnv(list(range(18))), gym)
    assert wrapped.canonical_action_mapping == tuple(range(18))


def test_actionfix_multitask_skips_only_the_legacy_second_mapping(monkeypatch) -> None:
    extra_mapping_calls = 0

    def fake_factory(**_kwargs):
        return lambda: _ActionEnv(list(range(18)))

    def count_mapping(env, _gym):
        nonlocal extra_mapping_calls
        extra_mapping_calls += 1
        return env

    monkeypatch.setattr(atari_envs, "make_atari_env", fake_factory)
    monkeypatch.setattr(atari_envs, "_canonical_18_action_space", count_mapping)
    monkeypatch.setattr(atari_envs, "_register_ale_envs", lambda _gym: None)
    env_ids = tuple(f"task-{index}" for index in range(5))

    fixed = atari_envs.make_multitask_atari_env(
        env_ids, seed=1, idx=0, atari_env_protocol="skiing-stall-actionfix-v1"
    )()
    assert extra_mapping_calls == 0
    fixed.close()

    baseline = atari_envs.make_multitask_atari_env(env_ids, seed=1, idx=0)()
    assert extra_mapping_calls == 5
    baseline.close()


def test_stall_resets_episode_but_keeps_td_bootstrap() -> None:
    terminated = np.asarray([False, True, False])
    truncated = np.asarray([True, False, True])
    episode_ends, bootstrap_stops = _replay_boundary_flags(
        terminated, truncated, ["stalled", None, None]
    )
    assert episode_ends.tolist() == [1, 1, 1]
    assert bootstrap_stops.tolist() == [0, 1, 1]


def test_episode_without_end_reason_is_natural_not_string_none() -> None:
    infos = {
        "episode": {"r": np.asarray([-123.0]), "l": np.asarray([12])},
        "_episode": np.asarray([True]),
    }
    assert _extract_episode_records(infos, "ALE/Skiing-v5") == [
        ("ALE/Skiing-v5", -123.0, 12, None)
    ]


def test_completed_weights_initialize_without_training_state(tmp_path) -> None:
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "final.pth"
    torch.save(source.state_dict(), checkpoint)

    provenance = _load_initial_weights(target, str(checkpoint), torch.device("cpu"))

    assert all(
        torch.equal(source.state_dict()[key], target.state_dict()[key])
        for key in source.state_dict()
    )
    assert provenance["mode"] == "weights_only"
    assert len(provenance["source_checkpoint_sha256"]) == 64


def test_weights_only_rejects_resumable_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model": {}, "optimizer": {}, "replay": {}}, checkpoint)
    with pytest.raises(ValueError, match="completed final model state_dict"):
        _load_initial_weights(torch.nn.Linear(1, 1), str(checkpoint), torch.device("cpu"))


def test_baseline_cli_default_remains_unchanged() -> None:
    args = build_arg_parser().parse_args([])
    assert args.atari_env_protocol == "baseline"
    assert args.init_weights_from is None


def test_budget_extension_only_relaxes_an_increasing_total_timesteps() -> None:
    checkpoint = {"args": {"total_timesteps": 1_000_000}}
    args = SimpleNamespace(
        total_timesteps=2_000_000,
        allow_total_timesteps_extension=True,
    )
    keys, origin = _resume_validation_keys(checkpoint, args)
    assert "total_timesteps" not in keys
    assert origin == 1_000_000

    checkpoint["args"]["total_timesteps"] = 2_000_000
    checkpoint["extended_from_total_timesteps"] = origin
    keys, repeated_origin = _resume_validation_keys(checkpoint, args)
    assert "total_timesteps" not in keys
    assert repeated_origin == 1_000_000

    args.total_timesteps = 999_999
    with pytest.raises(ValueError, match="cannot reduce"):
        _resume_validation_keys(checkpoint, args)

    args.allow_total_timesteps_extension = False
    keys, repeated_origin = _resume_validation_keys(checkpoint, args)
    assert keys == RESUME_ARG_KEYS
    assert repeated_origin == 1_000_000


def test_video_seed_parser_accepts_retry_suffix_and_prefers_metrics(tmp_path) -> None:
    metrics_path = tmp_path / "run_seed1_r2" / "metrics.json"
    assert training_seed_from_metrics({}, metrics_path) == 1
    assert training_seed_from_metrics({"seed": 7}, metrics_path) == 7
