"""Plot completed and explicitly labelled partial strict Breakout depth runs.

Inputs are ``metrics.json`` and ``metrics_history.jsonl`` files under Atari DQN result
directories. The script writes one PNG from the numeric histories, using mean +/- sample SD for
completed seeds of each model and a dashed line for any requested in-progress seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODELS = ("ann", "rnn", "gru", "lstm", "gawf")
MODEL_COLORS = {
    "ann": "#7f7f7f",
    "rnn": "#1f77b4",
    "gru": "#2ca02c",
    "lstm": "#9467bd",
    "gawf": "#d62728",
}
N_GRID = 300


def parse_args() -> argparse.Namespace:
    """Parse plotting options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="results/train_data")
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument(
        "--partial",
        action="append",
        default=[],
        metavar="MODEL:SEED",
        help="Explicit in-progress curve to show as a dashed line; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def run_dir(root: Path, layers: int, model: str, seed: int) -> Path:
    """Return the conventional plain Breakout result directory for one run."""
    return root / f"atari_dqn_breakout_fs4_stack4_l{layers}match_{model}_seed{seed}"


def load_curve(history_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load finite rolling returns from a metrics history."""
    steps: list[int] = []
    returns: list[float] = []
    with history_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get("episodic_return_100")
            step = record.get("global_step")
            if value is None or step is None or not np.isfinite(float(value)):
                continue
            steps.append(int(step))
            returns.append(float(value))
    if not steps:
        raise RuntimeError(f"No finite episodic_return_100 values in {history_path}")
    order = np.argsort(steps)
    return np.asarray(steps, dtype=np.int64)[order], np.asarray(returns, dtype=np.float64)[order]


def smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    """Return a causal moving average without using future values."""
    if window < 1:
        raise ValueError("--smooth must be >= 1")
    if window == 1 or values.size < 2:
        return values
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    indices = np.arange(values.size)
    starts = np.maximum(0, indices - window + 1)
    return (cumulative[indices + 1] - cumulative[starts]) / (indices - starts + 1)


def is_completed(path: Path, model: str, layers: int, expected_steps: int) -> bool:
    """Return whether a run has final metrics matching the requested protocol."""
    metrics_path = path / "metrics.json"
    history_path = path / "metrics_history.jsonl"
    if not metrics_path.is_file() or not history_path.is_file():
        return False
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {
        "global_step": expected_steps,
        "model_type": model,
        "num_layers": layers,
        "frame_skip": 4,
        "frame_stack": 4,
        "flicker_prob": 0.0,
        "action_space_mode": "minimal",
        "num_actions": 4,
    }
    return all(metrics.get(key) == value for key, value in expected.items())


def aggregate(curves: list[tuple[np.ndarray, np.ndarray]], smooth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate completed seed curves onto their shared domain."""
    lower = max(int(steps[0]) for steps, _ in curves)
    upper = min(int(steps[-1]) for steps, _ in curves)
    if upper <= lower:
        raise RuntimeError("Completed curves have no shared step range")
    grid = np.linspace(lower, upper, N_GRID)
    values = np.vstack(
        [np.interp(grid, steps, smooth_curve(returns, smooth)) for steps, returns in curves]
    )
    return grid, values.mean(axis=0), values.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(N_GRID)


def parse_partial(tokens: list[str]) -> list[tuple[str, int]]:
    """Parse ``MODEL:SEED`` partial-curve selectors."""
    parsed: list[tuple[str, int]] = []
    for token in tokens:
        model, separator, seed_text = token.partition(":")
        if separator != ":" or model not in MODELS or not seed_text.isdigit():
            raise ValueError(f"Invalid --partial value {token!r}; expected MODEL:SEED")
        parsed.append((model, int(seed_text)))
    return parsed


def completed_curves(
    root: Path, layers: int, model: str, seeds: list[int], expected_steps: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load every completed seed curve for one model."""
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in seeds:
        path = run_dir(root, layers, model, seed)
        if is_completed(path, model, layers, expected_steps):
            curves[seed] = load_curve(path / "metrics_history.jsonl")
    return curves


def style_axis(ax: plt.Axes, title: str) -> None:
    """Apply the shared strict-Breakout curve style."""
    ax.set_title(title)
    ax.set_xlabel("environment steps (×10⁶)")
    ax.set_ylabel("episodic return (last 100 episodes)")
    ax.grid(True, alpha=0.3)


def save_seed_figure(
    output_path: Path,
    *,
    root: Path,
    args: argparse.Namespace,
    completed: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    partials: set[tuple[str, int]],
    seed: int,
) -> None:
    """Plot all available models for one seed, including explicitly requested partial curves."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    plotted = 0
    for model in MODELS:
        curve = completed[model].get(seed)
        partial = (model, seed) in partials and curve is None
        if curve is None and partial:
            history_path = run_dir(root, args.num_layers, model, seed) / "metrics_history.jsonl"
            if not history_path.is_file():
                raise FileNotFoundError(f"Missing requested partial history: {history_path}")
            curve = load_curve(history_path)
        if curve is None:
            continue
        steps, returns = curve
        values = smooth_curve(returns, args.smooth)
        label = model if not partial else f"{model} in progress ({steps[-1] / 1_000_000:.3f}M)"
        ax.plot(
            steps / 1_000_000.0,
            values,
            color=MODEL_COLORS[model],
            linestyle="--" if partial else "-",
            linewidth=2.0,
            label=label,
        )
        if partial:
            ax.scatter(steps[-1] / 1_000_000.0, values[-1], color=MODEL_COLORS[model], s=28)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    style_axis(
        ax,
        f"Strict 4-action Breakout · fs4/stack4 · L{args.num_layers} · seed {seed}",
    )
    ax.legend(title="model", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def save_mean_std_figure(
    output_path: Path,
    *,
    args: argparse.Namespace,
    completed: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> int:
    """Plot only models with every declared seed complete, using mean +/- sample SD."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    groups = 0
    for model in MODELS:
        curves_by_seed = completed[model]
        if set(curves_by_seed) != set(args.seeds):
            continue
        steps, mean, std = aggregate(list(curves_by_seed.values()), args.smooth)
        x_axis = steps / 1_000_000.0
        color = MODEL_COLORS[model]
        ax.plot(x_axis, mean, color=color, linewidth=2.0, label=model)
        ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
        groups += 1
    if groups == 0:
        plt.close(fig)
        return 0
    style_axis(
        ax,
        f"Strict 4-action Breakout · fs4/stack4 · L{args.num_layers} · "
        f"{len(args.seeds)}-seed mean ± SD",
    )
    ax.legend(title="model", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return groups


def main() -> None:
    """Write per-seed figures plus one all-complete-seed mean/std figure for one depth."""
    args = parse_args()
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partials = set(parse_partial(args.partial))
    completed = {
        model: completed_curves(root, args.num_layers, model, args.seeds, args.expected_steps)
        for model in MODELS
    }
    for seed in args.seeds:
        save_seed_figure(
            output_dir / f"seed{seed}.png",
            root=root,
            args=args,
            completed=completed,
            partials=partials,
            seed=seed,
        )
    mean_groups = save_mean_std_figure(output_dir / "mean_std.png", args=args, completed=completed)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "completed": {model: sorted(curves) for model, curves in completed.items()},
                "mean_std_groups": mean_groups,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
