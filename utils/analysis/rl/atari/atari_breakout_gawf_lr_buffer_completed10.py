"""Plot eleven completed Breakout GaWF LR-decay/replay-sweep learning curves.

Inputs are final ``metrics.json`` and ``metrics_history.jsonl`` files for the four completed L3,
six completed L4, and one L3 seed-2 baseline diagnostic cells.  The script verifies the 3M-step
protocol and writes one PNG.  Colors identify the six ``(buffer, seed)`` tasks and line styles
identify L3 (solid) versus L4 (dashed).
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
CURVE_SPECS = (
    (3, "1m", 2, "diagnostic_baseline"),
    (3, "1m", 3, "sweep"),
    (3, "2m", 1, "sweep"),
    (3, "2m", 2, "sweep"),
    (3, "2m", 3, "sweep"),
    (4, "1m", 1, "sweep"),
    (4, "1m", 2, "sweep"),
    (4, "1m", 3, "sweep"),
    (4, "2m", 1, "sweep"),
    (4, "2m", 2, "sweep"),
    (4, "2m", 3, "sweep"),
)
TASK_COLORS = {
    ("1m", 1): "C0",
    ("1m", 2): "C1",
    ("1m", 3): "C2",
    ("2m", 1): "C3",
    ("2m", 2): "C4",
    ("2m", 3): "C5",
}
LAYER_STYLES = {3: "-", 4: "--"}


def parse_args() -> argparse.Namespace:
    """Parse data and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--diagnostic-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=190.0)
    return parser.parse_args()


def run_dir(
    data_root: Path,
    diagnostic_data_root: Path,
    layer: int,
    buffer_tag: str,
    seed: int,
    source: str,
) -> Path:
    """Return the exact result directory for one completed sweep or diagnostic cell."""
    if source == "diagnostic_baseline":
        if (layer, buffer_tag, seed) != (3, "1m", 2):
            raise ValueError(f"Unexpected diagnostic selector: {(layer, buffer_tag, seed)}")
        return diagnostic_data_root / "atari_dqn_breakout_fs4_stack4_l3diag_baseline_gawf_seed2"
    suffix = f"atari_dqn_breakout_fs4_stack4_l{layer}_lrdecay1m_buf{buffer_tag}_gawf_seed{seed}"
    return data_root / suffix


def load_curve(run_path: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate one final result and return finite rolling episodic-return history."""
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
    mismatches = {
        key: (metrics.get(key), value)
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Unexpected final protocol in {metrics_path}: {mismatches}")

    steps: list[int] = []
    values: list[float] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        value = record.get("episodic_return_100")
        step = record.get("global_step")
        if value is None or step is None or not np.isfinite(float(value)):
            continue
        steps.append(int(step))
        values.append(float(value))
    if not steps:
        raise RuntimeError(f"No finite episodic_return_100 values in {history_path}")
    order = np.argsort(np.asarray(steps, dtype=np.int64))
    return np.asarray(steps, dtype=np.int64)[order], np.asarray(values, dtype=np.float64)[order]


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


def main() -> None:
    """Render the combined ten-curve figure from validated result histories."""
    args = parse_args()
    if args.y_min >= args.y_max:
        raise ValueError("--y-min must be below --y-max")
    curves = []
    for layer, buffer_tag, seed, source in CURVE_SPECS:
        path = run_dir(args.data_root, args.diagnostic_data_root, layer, buffer_tag, seed, source)
        curves.append((layer, buffer_tag, seed, source, *load_curve(path, layer)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    for layer, buffer_tag, seed, source, steps, returns in curves:
        label = f"L{layer} · {buffer_tag} replay · seed {seed}"
        if source == "diagnostic_baseline":
            label += " (baseline diagnostic)"
        ax.plot(
            steps / 1_000_000.0,
            smooth_curve(returns, args.smooth),
            color=TASK_COLORS[(buffer_tag, seed)],
            linestyle=LAYER_STYLES[layer],
            linewidth=1.7,
            label=label,
        )
    ax.set_title("Strict 4-action Breakout · fs4/stack4 · GaWF · 11 completed runs")
    ax.set_xlabel("environment steps (×10⁶)")
    ax.set_ylabel("episodic return (last 100 episodes)")
    ax.set_xlim(0.0, EXPECTED_STEPS / 1_000_000.0)
    ax.set_ylim(args.y_min, args.y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(
        ncol=2,
        title="C0–C5: replay buffer + seed · solid: L3 · dashed: L4",
        fontsize=8.1,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
