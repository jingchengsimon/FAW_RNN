"""Evaluate DQN checkpoints greedily with fixed Atari seeds.

Inputs are one completed-run ``metrics.json`` and named model snapshots.  The
output is a compact JSON table of per-seed episode returns for every snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.analysis.rl.atari.evaluate_dqn_video import (
    autocast_context,
    build_model,
    load_checkpoint,
    load_metrics,
)
from utils.training.atari.atari_envs import make_atari_env
from utils.training.atari.atari_train_utils import (
    select_device,
    set_atari_seed,
    to_channel_first_obs,
)


def parse_args() -> argparse.Namespace:
    """Parse evaluator inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--eval_seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--amp_dtype", choices=["none", "bfloat16", "float16"], default="bfloat16")
    return parser.parse_args()


def evaluate_episode(model: torch.nn.Module, metrics: dict[str, object], device: torch.device,
                     seed: int, amp_dtype: str) -> tuple[float, int]:
    """Run one deterministic greedy episode and return its score and length."""
    env = make_atari_env(
        env_id=str(metrics["env_id"]), seed=seed, idx=0,
        frame_stack=int(metrics["frame_stack"]), frame_skip=int(metrics["frame_skip"]),
        flicker_prob=float(metrics["flicker_prob"]), capture_video=False,
        full_action_space=str(metrics["action_space_mode"]) == "full18", render_mode=None,
        atari_env_protocol=str(metrics.get("atari_env_protocol", "baseline")),
    )()
    try:
        observation, _ = env.reset(seed=seed)
        state = None
        previous_done = torch.ones(1, device=device)
        terminated = truncated = False
        episode_return = 0.0
        length = 0
        while not (terminated or truncated):
            batch = to_channel_first_obs(np.expand_dims(np.asarray(observation), axis=0))
            tensor = torch.as_tensor(batch, device=device)
            with torch.no_grad(), autocast_context(device, amp_dtype):
                q_values, state = model.step(tensor, previous_done, state)
            action = int(q_values.argmax(-1).item())
            observation, reward, terminated, truncated, _ = env.step(action)
            previous_done.zero_()
            episode_return += float(reward)
            length += 1
        return episode_return, length
    finally:
        env.close()


def main() -> None:
    """Evaluate all supplied immutable snapshots without overwriting outputs."""
    args = parse_args()
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    metrics, selected_env_id = load_metrics(Path(args.metrics_path).resolve(), None)
    metrics = dict(metrics)
    metrics["env_id"] = selected_env_id
    device = select_device(args.device)
    results: dict[str, object] = {"metrics_path": str(Path(args.metrics_path).resolve()),
                                  "eval_seeds": args.eval_seeds, "checkpoints": {}}
    for checkpoint_arg in args.checkpoints:
        checkpoint = Path(checkpoint_arg).resolve()
        model = build_model(metrics, device)
        load_checkpoint(model, checkpoint, device)
        model.eval()
        rows = []
        for seed in args.eval_seeds:
            set_atari_seed(seed)
            score, length = evaluate_episode(model, metrics, device, seed, args.amp_dtype)
            rows.append({"seed": seed, "return": score, "length": length})
        results["checkpoints"][checkpoint.name] = rows
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
