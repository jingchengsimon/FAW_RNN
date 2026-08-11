"""Render current L3 Pong/Breakout single-task and two-task learning curves.

Inputs are three Atari ``metrics_history.jsonl`` files.  Outputs are one two-panel PNG and a
JSON manifest; each panel overlays the raw single-task and two-task curves by task environment
steps, without averaging.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _curve(history_path: Path, environment: str) -> tuple[np.ndarray, np.ndarray]:
    """Return finite return-100 values against environment steps from one history."""

    xs: list[float] = []
    ys: list[float] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        values = record.get("per_env", {}).get(environment, {})
        try:
            environment_steps = float(values.get("environment_steps", 0))
            x = environment_steps
            if x <= 0:
                x = float(record["global_step"])
            y = float(values.get("episodic_return_100", float("nan")))
            if not np.isfinite(y) and environment_steps <= 0:
                y = float(record["episodic_return_100"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def main() -> None:
    """Parse histories and write the L3 two-panel current-progress figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-task", type=Path, required=True)
    parser.add_argument("--pong-only", type=Path, required=True)
    parser.add_argument("--breakout-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "Pong": (args.pong_only, "ALE/Pong-v5"),
        "Breakout": (args.breakout_only, "ALE/Breakout-v5"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(18, 7.2))
    for axis, (task, (single_history, environment)) in zip(axes, sources.items()):
        for history, label, color in ((single_history, f"{task}-only", "#1f77b4"),
                                      (args.two_task, "Pong + Breakout", "#ff7f0e")):
            x, y = _curve(history, environment)
            axis.plot(x / 1e6, y, lw=2.4, color=color, label=label)
        axis.set_title(task, fontsize=18)
        axis.set_xlabel("task environment steps (×10⁶)", fontsize=15)
        axis.grid(alpha=0.35)
        axis.legend(frameon=False, fontsize=14)
    axes[0].set_ylabel("episodic return (last 100)", fontsize=15)
    axes[1].set_ylabel("episodic return (last 100)", fontsize=15)
    figure.suptitle("GRU · full 18-action · skip 4, stack 4 · L3 h458 · seed 42", fontsize=18)
    figure.tight_layout()
    figure.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)
    args.output.with_suffix(".json").write_text(json.dumps({"sources": [str(args.two_task), str(args.pong_only), str(args.breakout_only)]}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
