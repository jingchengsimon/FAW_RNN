"""Aggregate all completed Breakout GaWF LR-decay/replay cells.

Inputs are ``metrics.json`` and ``metrics_history.jsonl`` for twelve completed L3/L4 cells.
The script validates the common 3M-step Atari protocol, aggregates each condition's rolling
histories on a shared step grid, and writes one PNG with a sample-standard-deviation band.
L3/1M combines sweep seeds 1 and 3 with the matching LR-decay diagnostic seed 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EXPECTED_STEPS = 3_000_000
CONDITIONS = (
    ("diagnostic_augmented", 3, "1m", "L3 · 1M replay + LR decay"),
    ("replicated", 3, "2m", "L3 · 2M replay + LR decay"),
    ("replicated", 4, "1m", "L4 · 1M replay + LR decay"),
    ("replicated", 4, "2m", "L4 · 2M replay + LR decay"),
)


def parse_args() -> argparse.Namespace:
    """Parse result roots and figure settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--diagnostic-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=190.0)
    parser.add_argument("--grid-points", type=int, default=301)
    return parser.parse_args()


def load_curve(run_path: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate one final run and load its finite episodic-return history."""
    metrics_path = run_path / "metrics.json"
    history_path = run_path / "metrics_history.jsonl"
    if not metrics_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Missing final metrics/history for {run_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {
        "global_step": EXPECTED_STEPS,
        "model_type": "gawf",
        "num_layers": layer,
        "frame_skip": 4,
        "frame_stack": 4,
        "flicker_prob": 0.0,
        "action_space_mode": "minimal",
        "num_actions": 4,
    }
    mismatches = {key: (metrics.get(key), value) for key, value in expected.items()
                  if metrics.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Unexpected final protocol in {metrics_path}: {mismatches}")

    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
               if line]
    pairs = [(int(record["global_step"]), float(record["episodic_return_100"]))
             for record in records
             if record.get("global_step") is not None
             and record.get("episodic_return_100") is not None
             and np.isfinite(float(record["episodic_return_100"]))]
    if not pairs:
        raise RuntimeError(f"No finite episodic_return_100 values in {history_path}")
    steps, values = zip(*sorted(pairs))
    return np.asarray(steps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    """Return a causal rolling mean without future-data leakage."""
    if window < 1:
        raise ValueError("--smooth must be >= 1")
    if window == 1 or values.size < 2:
        return values
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    indices = np.arange(values.size)
    starts = np.maximum(0, indices - window + 1)
    return (cumulative[indices + 1] - cumulative[starts]) / (indices - starts + 1)


def interpolate_smoothed_curve(run_path: Path, layer: int, smooth: int,
                               grid: np.ndarray) -> np.ndarray:
    """Load one validated curve, smooth it, and interpolate it onto ``grid``."""
    steps, values = load_curve(run_path, layer)
    return np.interp(grid, steps, smooth_curve(values, smooth))


def aggregate_condition(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return pointwise mean and sample SD for supplied aligned learning curves."""
    if len(curves) < 2:
        raise ValueError("At least two curves are required for a sample SD")
    stacked = np.stack(curves, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)


def replicated_condition(data_root: Path, layer: int, buffer_tag: str, smooth: int,
                         grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Aggregate all three seeds for one common LR-decay/buffer configuration."""
    curves: list[np.ndarray] = []
    for seed in (1, 2, 3):
        suffix = f"atari_dqn_breakout_fs4_stack4_l{layer}_lrdecay1m_buf{buffer_tag}_gawf_seed{seed}"
        curves.append(interpolate_smoothed_curve(data_root / suffix, layer, smooth, grid))
    mean, std = aggregate_condition(curves)
    return mean, std, len(curves)


def diagnostic_augmented_l3_1m_condition(
    data_root: Path,
    diagnostic_data_root: Path,
    smooth: int,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Aggregate L3/1M LR-decay sweep seeds 1/3 and diagnostic seed 2."""
    curves = []
    for seed in (1, 3):
        suffix = f"atari_dqn_breakout_fs4_stack4_l3_lrdecay1m_buf1m_gawf_seed{seed}"
        curves.append(interpolate_smoothed_curve(data_root / suffix, 3, smooth, grid))
    curves.append(
        interpolate_smoothed_curve(
            diagnostic_data_root / "atari_dqn_breakout_fs4_stack4_l3diag_lr_decay_gawf_seed2",
            3,
            smooth,
            grid,
        )
    )
    mean, std = aggregate_condition(curves)
    return mean, std, len(curves)


def main() -> None:
    """Render the complete-condition cross-seed aggregate figure."""
    args = parse_args()
    if args.y_min >= args.y_max:
        raise ValueError("--y-min must be below --y-max")
    if args.grid_points < 2:
        raise ValueError("--grid-points must be >= 2")

    grid = np.linspace(0, EXPECTED_STEPS, args.grid_points, dtype=np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    for index, (kind, layer, buffer_tag, label) in enumerate(CONDITIONS):
        if kind == "diagnostic_augmented":
            mean, std, count = diagnostic_augmented_l3_1m_condition(
                args.data_root, args.diagnostic_data_root, args.smooth, grid
            )
        else:
            if buffer_tag is None:
                raise ValueError("Replicated condition requires a replay-buffer tag")
            mean, std, count = replicated_condition(args.data_root, layer, buffer_tag, args.smooth, grid)
        color = f"C{index}"
        ax.plot(grid / 1_000_000.0, mean, color=color, linewidth=2.0, label=f"{label} (n={count})")
        ax.fill_between(grid / 1_000_000.0, mean - std, mean + std, color=color, alpha=0.20,
                        linewidth=0.0, label="_nolegend_")

    ax.set_title("Strict 4-action Breakout · fs4/stack4 · GaWF · LR-decay conditions")
    ax.set_xlabel("environment steps (×10⁶)")
    ax.set_ylabel("episodic return (last 100 episodes)")
    ax.set_xlim(0.0, EXPECTED_STEPS / 1_000_000.0)
    ax.set_ylim(args.y_min, args.y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(title="line: mean · shaded band: sample SD")
    ax.text(0.99, 0.02, "L3 · 1M uses sweep seeds 1/3 plus matching diagnostic seed 2;\n"
            "all displayed conditions have n=3. Fixed-LR baseline excluded.", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
