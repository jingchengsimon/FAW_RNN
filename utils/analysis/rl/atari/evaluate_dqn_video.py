"""Evaluate and render greedy Atari DQN episodes from completed checkpoints.

Inputs are a final training ``metrics.json`` and its checkpoint. The evaluator
reconstructs the exact saved network and strict Atari observation protocol,
records deterministic greedy-policy episodes, and optionally copies a selected
episode to the requested MP4 path. It supports the strict single-task protocol
and task-blind full-18 multi-task checkpoints. A companion JSON records the
source checkpoint, training/evaluation seeds, all episode returns, selected
episode, protocol fields, and output path.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, ContextManager

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import cv2

from utils.training.atari.atari_dqn_models import AtariQNetwork, normalize_atari_dqn_model_type
from utils.training.atari.atari_envs import make_atari_env
from utils.training.atari.atari_train_utils import (
    select_device,
    set_atari_seed,
    to_channel_first_obs,
)
from utils.analysis.anal_paths import output_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--eval_seed", type=int, default=20260718)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--video_title",
        default=None,
        help="Optional label rendered into every output-video frame.",
    )
    parser.add_argument(
        "--task_env_id",
        default=None,
        help="Task to evaluate for a full-18 multi-task checkpoint, e.g. ALE/Breakout-v5.",
    )
    parser.add_argument(
        "--episode_selection",
        choices=["first", "best_return"],
        default="best_return",
        help="Episode copied to MP4 when recording is enabled.",
    )
    parser.add_argument(
        "--selection_only",
        action="store_true",
        help="Evaluate and write metadata without encoding episode videos.",
    )
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--amp_dtype", choices=["none", "bfloat16", "float16"], default="bfloat16")
    return parser.parse_args()


def load_metrics(path: Path, task_env_id: str | None) -> tuple[dict[str, Any], str]:
    """Load metrics and resolve a valid single-task or full-18 evaluation task."""
    metrics = json.loads(path.read_text(encoding="utf-8"))
    is_multitask = bool(metrics.get("multitask"))
    expected = {"frame_skip": 4, "frame_stack": 4, "flicker_prob": 0.0}
    skiing_protocol = metrics.get("atari_env_protocol") == "skiing-stall-actionfix-v1"
    if is_multitask or skiing_protocol:
        expected.update({"action_space_mode": "full18", "num_actions": 18})
    else:
        expected.update({"action_space_mode": "minimal"})
    mismatches = {
        key: (metrics.get(key), value)
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported or mismatched Atari video protocol: {mismatches}")
    if is_multitask:
        env_ids = [str(env_id) for env_id in metrics.get("env_ids", [])]
        if task_env_id is None:
            raise ValueError("--task_env_id is required for a multi-task checkpoint")
        if task_env_id not in env_ids:
            raise ValueError(f"Task {task_env_id} is absent from checkpoint tasks: {env_ids}")
        selected_env_id = task_env_id
    else:
        expected_actions = {"ALE/Pong-v5": 6, "ALE/Breakout-v5": 4}
        selected_env_id = str(metrics.get("env_id"))
        if skiing_protocol:
            if selected_env_id != "ALE/Skiing-v5":
                raise ValueError(
                    "skiing-stall-actionfix-v1 is valid only for ALE/Skiing-v5"
                )
        elif selected_env_id not in expected_actions:
            raise ValueError(f"Unsupported Atari video environment: {selected_env_id}")
        if not skiing_protocol and int(metrics.get("num_actions", -1)) != expected_actions[
            selected_env_id
        ]:
            raise ValueError(
                f"Unexpected action count for {selected_env_id}: {metrics.get('num_actions')} "
                f"(expected {expected_actions[selected_env_id]})"
            )
        if task_env_id is not None and task_env_id != selected_env_id:
            raise ValueError(f"Single-task checkpoint is for {selected_env_id}, not {task_env_id}")
    if int(metrics.get("num_layers", 0)) < 1:
        raise ValueError(
            f"Expected at least one recurrent/readout layer, got {metrics.get('num_layers')}"
        )
    if int(metrics.get("global_step", 0)) < 1:
        raise ValueError("metrics.json does not describe a completed training run")
    return metrics, selected_env_id


def training_seed_from_metrics(metrics: dict[str, Any], metrics_path: Path) -> int:
    """Resolve the recorded seed, with a leaf-name fallback for historical metrics."""
    if metrics.get("seed") is not None:
        return int(metrics["seed"])
    match = re.search(r"(?:^|_)seed([0-9]+)(?:_|$)", metrics_path.parent.name)
    if match is None:
        raise ValueError(f"Cannot resolve training seed from {metrics_path}")
    return int(match.group(1))


def build_model(metrics: dict[str, Any], device: torch.device) -> AtariQNetwork:
    """Construct the Atari Q-network encoded by final metrics."""
    return AtariQNetwork(
        num_actions=int(metrics["num_actions"]),
        input_channels=int(metrics["frame_stack"]),
        model_type=normalize_atari_dqn_model_type(str(metrics["model_type"])),
        hidden_size=int(metrics["hidden_size"]),
        encoder_feature_dim=int(metrics.get("encoder_feature_dim", 512)),
        feedback_mode=str(metrics["feedback_mode"]),
        num_layers=int(metrics["num_layers"]),
    ).to(device)


def load_checkpoint(model: AtariQNetwork, path: Path, device: torch.device) -> None:
    """Load one final checkpoint while reporting compatibility evidence."""
    state_dict = torch.load(path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict mapping in {path}")
    state_dict = {
        key: value for key, value in state_dict.items() if not key.endswith("prev_feedback")
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    print("missing_keys:", incompatible.missing_keys)
    print("unexpected_keys:", incompatible.unexpected_keys)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint is incompatible with reconstructed model: {path}")


def autocast_context(device: torch.device, amp_dtype: str) -> ContextManager[Any]:
    """Return the requested CUDA autocast context or an eager no-op context."""
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def open_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    """Create a validated OpenCV MP4 writer matching one RGB render frame."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an RGB render frame, got shape {frame.shape}")
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"OpenCV could not open MP4 writer: {path}")
    return writer


def annotate_frame(frame: np.ndarray, title: str | None) -> np.ndarray:
    """Render an optional protocol-and-seed title onto a BGR video frame."""
    if not title:
        return frame
    annotated = frame.copy()
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (min(annotated.shape[1] - 8, 610), 42), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
    cv2.putText(
        annotated,
        title,
        (16, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Record greedy episodes, retain the highest-return MP4, and save metadata."""
    if args.num_episodes < 1:
        raise ValueError("--num_episodes must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    metrics_path = Path(args.metrics_path).resolve()
    metrics, task_env_id = load_metrics(metrics_path, args.task_env_id)
    checkpoint = Path(args.checkpoint or metrics["checkpoint"]).resolve()
    training_seed = training_seed_from_metrics(metrics, metrics_path)
    env_slug = "".join(
        character.lower() if character.isalnum() else "_" for character in task_env_id
    )
    env_slug = env_slug.strip("_")
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else output_dir("G_behaviour", "evaluate_atari_dqn_video", "figs")
        / f"{env_slug}_seed{training_seed}.mp4"
    )
    metadata_path = (
        Path(args.metadata_path).resolve()
        if args.metadata_path
        else output_dir("G_behaviour", "evaluate_atari_dqn_video", "data")
        / f"{env_slug}_seed{training_seed}.json"
    )
    if not args.selection_only and f"seed{training_seed}" not in output_path.name:
        raise ValueError("Output filename must include the selected training seed")
    if (not args.selection_only and output_path.exists()) or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing final Atari video artifact")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    raw_video_dir: Path | None = None
    if not args.selection_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_video_dir = output_path.parent / f"raw_episodes_eval{args.eval_seed}_{os.getpid()}"
        raw_video_dir.mkdir(parents=True, exist_ok=False)

    set_atari_seed(args.eval_seed)
    device = select_device(args.device)
    model = build_model(metrics, device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    env = make_atari_env(
        env_id=task_env_id,
        seed=args.eval_seed,
        idx=0,
        frame_stack=int(metrics["frame_stack"]),
        frame_skip=int(metrics["frame_skip"]),
        flicker_prob=float(metrics["flicker_prob"]),
        capture_video=False,
        full_action_space=str(metrics["action_space_mode"]) == "full18",
        render_mode="rgb_array",
        atari_env_protocol=str(metrics.get("atari_env_protocol", "baseline")),
    )()
    returns: list[float] = []
    episode_lengths: list[int] = []
    episode_terminated: list[bool] = []
    episode_truncated: list[bool] = []
    episode_end_reasons: list[str] = []
    videos: list[Path] = []
    try:
        if int(env.action_space.n) != int(metrics["num_actions"]):
            raise RuntimeError(
                f"Action count mismatch: env={env.action_space.n}, metrics={metrics['num_actions']}"
            )
        for episode in range(args.num_episodes):
            obs, _info = env.reset(seed=args.eval_seed + episode)
            episode_video = (
                raw_video_dir / f"{output_path.stem}-episode-{episode}.mp4"
                if raw_video_dir is not None
                else None
            )
            writer: cv2.VideoWriter | None = None
            state = None
            prev_done = torch.ones(1, device=device)
            episode_return = 0.0
            episode_length = 0
            terminated = truncated = False
            end_reason = "natural"
            try:
                while not (terminated or truncated):
                    obs_batch = to_channel_first_obs(np.expand_dims(np.asarray(obs), axis=0))
                    obs_tensor = torch.as_tensor(obs_batch, device=device)
                    with torch.no_grad(), autocast_context(device, args.amp_dtype):
                        q_values, state = model.step(obs_tensor, prev_done, state)
                    action = int(q_values.argmax(dim=-1).item())
                    obs, reward, terminated, truncated, _info = env.step(action)
                    end_reason = str(_info.get("end_reason", "natural"))
                    if episode_video is not None:
                        frame = np.asarray(env.render(), dtype=np.uint8)
                        if writer is None:
                            writer = open_video_writer(episode_video, frame, args.fps)
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(annotate_frame(bgr_frame, args.video_title))
                    prev_done = torch.zeros(1, device=device)
                    episode_return += float(reward)
                    episode_length += 1
            finally:
                if writer is not None:
                    writer.release()
            if episode_video is not None:
                if (
                    writer is None
                    or not episode_video.is_file()
                    or episode_video.stat().st_size <= 0
                ):
                    raise RuntimeError(f"Episode did not produce a valid MP4: {episode_video}")
                videos.append(episode_video)
            returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_terminated.append(bool(terminated))
            episode_truncated.append(bool(truncated))
            episode_end_reasons.append(end_reason)
            print(
                f"episode={episode} return={episode_return:.1f} "
                f"environment_steps={episode_length}"
            )
    finally:
        env.close()

    if not args.selection_only and len(videos) != args.num_episodes:
        raise RuntimeError(
            f"Expected {args.num_episodes} recorded videos, found {len(videos)} in {raw_video_dir}"
        )
    selected_episode = (
        0
        if args.episode_selection == "first"
        else int(np.argmax(np.asarray(returns, dtype=np.float64)))
    )
    source_video: Path | None = None
    if not args.selection_only:
        source_video = videos[selected_episode]
        shutil.copy2(source_video, output_path)
        if output_path.stat().st_size <= 0:
            raise RuntimeError(f"Generated empty video: {output_path}")

    metadata = {
        "env_id": task_env_id,
        "model_type": metrics["model_type"],
        "feedback_mode": metrics["feedback_mode"],
        "num_layers": int(metrics["num_layers"]),
        "frame_skip": int(metrics["frame_skip"]),
        "frame_stack": int(metrics["frame_stack"]),
        "action_space_mode": metrics["action_space_mode"],
        "num_actions": int(metrics["num_actions"]),
        "atari_env_protocol": metrics.get("atari_env_protocol", "baseline"),
        "training_seed": training_seed,
        "eval_seed": int(args.eval_seed),
        "num_episodes": int(args.num_episodes),
        "fps": float(args.fps),
        "video_title": args.video_title,
        "episode_returns": returns,
        "episode_lengths": episode_lengths,
        "episode_terminated": episode_terminated,
        "episode_truncated": episode_truncated,
        "episode_end_reasons": episode_end_reasons,
        "stall_rate": episode_end_reasons.count("stalled") / len(episode_end_reasons),
        "episode_selection": args.episode_selection,
        "selected_episode": selected_episode,
        "selected_return": returns[selected_episode],
        "source_metrics": str(metrics_path),
        "source_checkpoint": str(checkpoint),
        "source_video": str(source_video) if source_video is not None else None,
        "output_video": str(output_path) if not args.selection_only else None,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """Run the greedy-video evaluator."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    metadata = evaluate(parse_args())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
