"""Plot all six completed L3 GaWF seed-2 diagnostic training curves.

Inputs are six ``metrics_history.jsonl`` files saved by the diagnostic runs.
The output is one PNG overlaying their causally smoothed ``episodic_return_100``
series with a legend that identifies each intervention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


VARIANTS = {
    "baseline": "baseline (1M replay)",
    "double_dqn": "Double DQN",
    "lr_decay": "LR ×0.1 after 1M",
    "buffer_500k": "replay 0.5M",
    "buffer_2m": "replay 2M",
    "no_feedback": "no q-values feedback",
}


def parse_args() -> argparse.Namespace:
    """Parse diagnostic source and destination paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=10)
    return parser.parse_args()


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load finite step and online episodic-return records from one JSONL file."""
    steps: list[int] = []
    returns: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        value = record.get("episodic_return_100")
        step = record.get("global_step")
        if value is None or step is None or not np.isfinite(float(value)):
            continue
        steps.append(int(step))
        returns.append(float(value))
    if not steps:
        raise RuntimeError(f"No finite training returns in {path}")
    order = np.argsort(np.asarray(steps))
    return np.asarray(steps, dtype=np.int64)[order], np.asarray(returns, dtype=np.float64)[order]


def causal_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Return a causal moving average, without incorporating future updates."""
    if window < 1:
        raise ValueError("--smooth must be positive")
    if window == 1:
        return values
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    index = np.arange(values.size)
    starts = np.maximum(0, index - window + 1)
    return (cumulative[index + 1] - cumulative[starts]) / (index - starts + 1)


def main() -> None:
    """Render one legend-labelled overlay from all completed diagnostic histories."""
    args = parse_args()
    figure, axis = plt.subplots(figsize=(10.0, 5.7))
    for key, label in VARIANTS.items():
        history = args.data_root / f"atari_dqn_breakout_fs4_stack4_l3diag_{key}_gawf_seed2" / "metrics_history.jsonl"
        steps, returns = load_curve(history)
        axis.plot(steps / 1_000_000, causal_smooth(returns, args.smooth), linewidth=1.8, label=label)
    axis.set_title("Strict 4-action Breakout · fs4/stack4 · L3 GaWF · seed 2 diagnostics")
    axis.set_xlabel("environment steps (×10⁶)")
    axis.set_ylabel("episodic return (last 100 episodes)")
    axis.set_xlim(0.0, 3.0)
    axis.grid(True, alpha=0.3)
    axis.legend(title="run", fontsize=9)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


if __name__ == "__main__":
    main()
