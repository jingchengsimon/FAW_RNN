"""Lazy Gymnasium/ALE environment helpers for Atari experiments.

This module intentionally imports Gymnasium only inside factory functions so model
tests can run on machines without Atari ROMs or Gymnasium extras installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ATARI_PILOT_ENVS = (
    "ALE/Pong-v5",
    "ALE/Breakout-v5",
    "ALE/Assault-v5",
    "ALE/Seaquest-v5",
    "ALE/Skiing-v5",
    "ALE/MsPacman-v5",
    "ALE/BeamRider-v5",
)

ATARI_TASK_SCHEDULES = ("transition_balanced", "round_robin")

# ALE action enum values use the canonical order 0..17. Games such as Skiing expose only
# non-fire actions even under ``full_action_space=True``. Map fire variants to the matching
# legal movement (and FIRE to NOOP) so every task still shares one task-blind 18-output head.
_CANONICAL_ACTION_FALLBACK = (0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9)


class _EpisodeTaskScheduler:
    """Choose tasks at episode boundaries while tracking collected transitions."""

    def __init__(self, num_tasks: int, start_idx: int, mode: str) -> None:
        if num_tasks < 2:
            raise ValueError("Episode task scheduling requires at least two tasks")
        if mode not in ATARI_TASK_SCHEDULES:
            raise ValueError(f"Unsupported Atari task schedule: {mode}")
        self.mode = mode
        self.task_steps = [0] * num_tasks
        self._cursor = start_idx % num_tasks

    def next_task(self) -> int:
        """Return the next task, breaking equal-step ties cyclically."""
        if self.mode == "round_robin":
            selected = self._cursor
        else:
            minimum = min(self.task_steps)
            selected = self._cursor
            for offset in range(len(self.task_steps)):
                candidate = (self._cursor + offset) % len(self.task_steps)
                if self.task_steps[candidate] == minimum:
                    selected = candidate
                    break
        self._cursor = (selected + 1) % len(self.task_steps)
        return selected

    def record_step(self, task_idx: int) -> None:
        """Record one valid transition for the active task."""
        if not 0 <= task_idx < len(self.task_steps):
            raise IndexError(f"task_idx out of range: {task_idx}")
        self.task_steps[task_idx] += 1

    def state_dict(self) -> dict[str, Any]:
        """Return the exact episode-boundary scheduling state."""
        return {
            "mode": self.mode,
            "task_steps": list(self.task_steps),
            "cursor": self._cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduling counts without restoring any ALE state."""
        if state.get("mode") != self.mode:
            raise ValueError(
                f"Scheduler mode mismatch: stored={state.get('mode')!r} expected={self.mode!r}"
            )
        task_steps = [int(value) for value in state.get("task_steps", [])]
        if len(task_steps) != len(self.task_steps) or any(value < 0 for value in task_steps):
            raise ValueError(
                "Scheduler task_steps must be non-negative and match the configured task count"
            )
        cursor = int(state.get("cursor", -1))
        if not 0 <= cursor < len(self.task_steps):
            raise ValueError(f"Scheduler cursor out of range: {cursor}")
        self.task_steps = task_steps
        self._cursor = cursor


def _register_ale_envs(gym) -> None:
    """Register ALE namespaces for Gymnasium versions that require explicit setup."""
    try:
        import ale_py
    except ImportError as exc:
        raise ImportError(
            "Atari experiments require ale-py, e.g. "
            "`pip install 'gymnasium[atari]' ale-py`."
        ) from exc
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)


def _frame_stack(env, gym, frame_stack: int):
    if frame_stack <= 1:
        return env
    if hasattr(gym.wrappers, "FrameStackObservation"):
        try:
            return gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
        except TypeError:
            return gym.wrappers.FrameStackObservation(env, frame_stack)
    if hasattr(gym.wrappers, "FrameStack"):
        return gym.wrappers.FrameStack(env, num_stack=frame_stack)
    raise RuntimeError("Gymnasium frame stack wrapper not found")


def _canonical_18_action_space(env, gym):
    """Expose Discrete(18), mapping unsupported canonical actions to legal equivalents."""
    action_set = getattr(env.unwrapped, "_action_set", None)
    if action_set is None:
        raise RuntimeError("ALE environment does not expose its legal action set")
    legal_to_index = {
        int(getattr(action, "value", action)): index
        for index, action in enumerate(action_set)
    }
    noop_index = legal_to_index.get(0)
    if noop_index is None:
        raise RuntimeError("ALE legal action set is missing NOOP")
    mapping = tuple(
        legal_to_index.get(action, legal_to_index.get(fallback, noop_index))
        for action, fallback in enumerate(_CANONICAL_ACTION_FALLBACK)
    )

    class _Canonical18Action(gym.ActionWrapper):
        def __init__(self, wrapped_env) -> None:
            super().__init__(wrapped_env)
            self.action_space = gym.spaces.Discrete(18)
            self.canonical_action_mapping = mapping

        def action(self, action: int) -> int:
            action_index = int(action)
            if not 0 <= action_index < 18:
                raise ValueError(f"Canonical Atari action out of range: {action_index}")
            return self.canonical_action_mapping[action_index]

    return _Canonical18Action(env)


def _flicker(env, gym, flicker_prob: float, seed: int):
    """Flickering-Atari partial observability (Hausknecht & Stone, 2015).

    At every timestep the entire screen is obscured (returned as an all-zero
    frame) with probability ``flicker_prob``, otherwise the true observation is
    passed through. This is applied on the preprocessed 84x84 frame and *before*
    frame stacking, so with ``frame_stack=1`` each single-frame observation is
    independently blanked, turning the MDP into a POMDP that requires temporal
    integration to recover the hidden game state.
    """
    if flicker_prob <= 0.0:
        return env

    import numpy as np  # local import: numpy is only needed inside the factory

    class _FlickerObservation(gym.ObservationWrapper):
        def __init__(self, env, prob: float, rng_seed: int) -> None:
            super().__init__(env)
            self.prob = float(prob)
            self._rng = np.random.default_rng(rng_seed)

        def observation(self, observation):
            if self._rng.random() < self.prob:
                return np.zeros_like(np.asarray(observation))
            return observation

    return _FlickerObservation(env, flicker_prob, seed)


def make_atari_env(
    env_id: str,
    seed: int,
    idx: int,
    frame_stack: int = 4,
    frame_skip: int = 4,
    flicker_prob: float = 0.0,
    capture_video: bool = False,
    video_dir: str | None = None,
    full_action_space: bool = False,
    render_mode: str | None = None,
) -> Callable[[], object]:
    """Return a thunk that creates one preprocessed Atari environment."""

    if frame_skip < 1:
        raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

    def thunk():
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "Atari experiments require gymnasium with Atari extras, e.g. "
                "`pip install 'gymnasium[atari,accept-rom-license]'`."
            ) from exc
        _register_ale_envs(gym)

        env = gym.make(
            env_id,
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=full_action_space,
            render_mode=render_mode or ("rgb_array" if capture_video else None),
        )
        if full_action_space:
            env = _canonical_18_action_space(env, gym)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            if video_dir is None:
                raise ValueError("video_dir must be set when capture_video=True")
            env = gym.wrappers.RecordVideo(env, video_dir)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=frame_skip,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            scale_obs=False,
        )
        env = _flicker(env, gym, flicker_prob, seed + idx)
        env = _frame_stack(env, gym, frame_stack)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def make_vector_atari_env(
    env_id: str,
    seed: int,
    num_envs: int,
    frame_stack: int = 4,
    frame_skip: int = 4,
    flicker_prob: float = 0.0,
    capture_video: bool = False,
    video_dir: str | None = None,
    full_action_space: bool = False,
    render_mode: str | None = None,
) -> Any:
    """Create a synchronous vector Atari environment."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "Atari experiments require gymnasium with Atari extras, e.g. "
            "`pip install 'gymnasium[atari,accept-rom-license]'`."
        ) from exc
    _register_ale_envs(gym)

    env_fns = [
        make_atari_env(
            env_id,
            seed,
            idx,
            frame_stack,
            frame_skip,
            flicker_prob,
            capture_video,
            video_dir,
            full_action_space,
            render_mode,
        )
        for idx in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)


def make_multitask_atari_env(
    env_ids: tuple[str, ...],
    seed: int,
    idx: int,
    frame_stack: int = 1,
    frame_skip: int = 1,
    flicker_prob: float = 0.0,
    task_schedule: str = "transition_balanced",
    scheduler_state: dict[str, Any] | None = None,
) -> Callable[[], object]:
    """Return one task-blind Atari env that switches games at episode resets.

    All component games expose ALE's canonical 18-action space. The active
    ``env_id`` and integer ``task_id`` are emitted in ``info`` for metrics only;
    neither is added to the observation consumed by the agent.
    """
    if len(env_ids) < 2:
        raise ValueError("Multi-task Atari requires at least two env_ids")
    if len(set(env_ids)) != len(env_ids):
        raise ValueError("Multi-task env_ids must be unique")
    if task_schedule not in ATARI_TASK_SCHEDULES:
        raise ValueError(f"Unsupported Phase0 task_schedule: {task_schedule}")

    def thunk():
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "Atari experiments require gymnasium with Atari extras, e.g. "
                "`pip install 'gymnasium[atari,accept-rom-license]'`."
            ) from exc
        _register_ale_envs(gym)

        component_envs = [
            _canonical_18_action_space(
                make_atari_env(
                    env_id=env_id,
                    seed=seed + task_idx * 10_000,
                    idx=idx,
                    frame_stack=frame_stack,
                    frame_skip=frame_skip,
                    flicker_prob=flicker_prob,
                    full_action_space=True,
                )(),
                gym,
            )
            for task_idx, env_id in enumerate(env_ids)
        ]

        class _EpisodeSwitchAtariEnv(gym.Env):
            metadata = component_envs[0].metadata

            def __init__(self) -> None:
                super().__init__()
                self._envs = component_envs
                self._env_ids = env_ids
                self._scheduler = _EpisodeTaskScheduler(
                    num_tasks=len(self._envs),
                    start_idx=idx,
                    mode=task_schedule,
                )
                if scheduler_state is not None:
                    self._scheduler.load_state_dict(scheduler_state)
                self._active_task_idx: int | None = None
                self._has_reset = [False] * len(self._envs)
                self.action_space = self._envs[0].action_space
                self.observation_space = self._envs[0].observation_space
                self.render_mode = getattr(self._envs[0], "render_mode", None)
                for env_id, env in zip(self._env_ids, self._envs):
                    if env.action_space != self.action_space:
                        raise RuntimeError(f"Action space mismatch for {env_id}")
                    if env.observation_space != self.observation_space:
                        raise RuntimeError(f"Observation space mismatch for {env_id}")
                if self.action_space.__class__.__name__ != "Discrete":
                    raise RuntimeError("Atari multi-task action space must be Discrete")
                if int(self.action_space.n) != 18:
                    raise RuntimeError(
                        "Atari multi-task Phase0 requires the canonical 18-action space"
                    )

            def _add_task_info(self, info: dict[str, Any]) -> dict[str, Any]:
                if self._active_task_idx is None:
                    raise RuntimeError("Multi-task environment has not been reset")
                enriched = dict(info)
                enriched["task_id"] = self._active_task_idx
                enriched["env_id"] = self._env_ids[self._active_task_idx]
                return enriched

            def reset(
                self,
                *,
                seed: int | None = None,
                options: dict[str, Any] | None = None,
            ):
                task_idx = self._scheduler.next_task()
                self._active_task_idx = task_idx
                reset_seed = seed
                if reset_seed is None and not self._has_reset[task_idx]:
                    reset_seed = seed_value + task_idx * 10_000 + idx
                obs, info = self._envs[task_idx].reset(seed=reset_seed, options=options)
                self._has_reset[task_idx] = True
                return obs, self._add_task_info(info)

            def step(self, action: int):
                if self._active_task_idx is None:
                    raise RuntimeError("Call reset() before step()")
                obs, reward, terminated, truncated, info = self._envs[
                    self._active_task_idx
                ].step(action)
                self._scheduler.record_step(self._active_task_idx)
                return (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    self._add_task_info(info),
                )

            def render(self):
                if self._active_task_idx is None:
                    return None
                return self._envs[self._active_task_idx].render()

            def task_scheduler_state(self) -> dict[str, Any]:
                """Return resumable scheduler metadata for this vector slot."""
                return self._scheduler.state_dict()

            def close(self) -> None:
                for env in self._envs:
                    env.close()

        seed_value = seed
        return _EpisodeSwitchAtariEnv()

    return thunk


def make_multitask_vector_atari_env(
    env_ids: tuple[str, ...],
    seed: int,
    num_envs: int,
    frame_stack: int = 1,
    frame_skip: int = 1,
    flicker_prob: float = 0.0,
    task_schedule: str = "transition_balanced",
    scheduler_states: tuple[dict[str, Any], ...] | None = None,
) -> Any:
    """Create task-blind Atari envs that select tasks only at episode boundaries."""
    if scheduler_states is not None and len(scheduler_states) != num_envs:
        raise ValueError(
            f"scheduler_states has {len(scheduler_states)} entries, expected num_envs={num_envs}"
        )
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "Atari experiments require gymnasium with Atari extras, e.g. "
            "`pip install 'gymnasium[atari,accept-rom-license]'`."
        ) from exc
    _register_ale_envs(gym)
    env_fns = [
        make_multitask_atari_env(
            env_ids=env_ids,
            seed=seed,
            idx=idx,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            flicker_prob=flicker_prob,
            task_schedule=task_schedule,
            scheduler_state=None if scheduler_states is None else scheduler_states[idx],
        )
        for idx in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)


def multitask_scheduler_states(vector_env: Any) -> tuple[dict[str, Any], ...]:
    """Return exact per-slot scheduler state from a synchronous multi-task vector env."""
    component_envs = getattr(vector_env, "envs", None)
    if component_envs is None:
        raise TypeError("Multi-task scheduler checkpointing requires SyncVectorEnv.envs")
    return tuple(env.task_scheduler_state() for env in component_envs)
