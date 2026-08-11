"""Render a provisional lower-projected GaWF Breakout progress snapshot.

Inputs are locally synchronized ``metrics.json`` and ``metrics_history.jsonl`` files for the
fixed Breakout minimal-4-action L3 lower-projected feedback run.  The script writes per-seed PNG
curves, a completed-seed mean-plus-sample-SD PNG, and a JSON manifest.  In-progress curves are
shown only in per-seed figures and never enter the aggregate statistic.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DZ_VALUES = (32, 64)
DEFAULT_SEEDS = (1, 2, 3)
COLORS = {32: "#1f77b4", 64: "#ff7f0e"}
N_GRID = 300


def parse_args() -> argparse.Namespace:
    """Parse plotting locations and smoothing controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "results/data/rl/atari/breakout_4action/"
            "fs4_stack4_l3_lower_projected_1m_progress"
        ),
        help="Local snapshot root containing result-suffix directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/figs/rl/atari/breakout_4action/"
            "fs4_stack4_l3_lower_projected_1m_progress"
        ),
        help="Destination directory for provisional progress figures.",
    )
    parser.add_argument("--smooth", type=int, default=10, help="Causal smoothing window.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def suffix(dz: int, seed: int) -> str:
    """Return the fixed result suffix for one projected-feedback unit."""
    return (
        "atari_dqn_breakout_fs4_stack4_l3_lrdecay1m_buf1m_lowerproj_"
        f"dz{dz}_gawf_seed{seed}"
    )


def load_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load finite ``episodic_return_100`` values from one JSONL history."""
    steps: list[int] = []
    returns: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = record.get("global_step")
        value = record.get("episodic_return_100")
        if step is None or value is None or not np.isfinite(float(value)):
            continue
        steps.append(int(step))
        returns.append(float(value))
    if not steps:
        raise RuntimeError(f"No finite episodic_return_100 values in {path}")
    order = np.argsort(np.asarray(steps, dtype=np.int64))
    return np.asarray(steps, dtype=np.int64)[order], np.asarray(returns, dtype=np.float64)[order]


def is_complete(run_dir: Path, dz: int) -> bool:
    """Return whether a run has final metrics for the fixed 1M-step protocol."""
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "metrics_history.jsonl"
    if not metrics_path.is_file() or not history_path.is_file():
        return False
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {
        "global_step": 1_000_000,
        "model_type": "gawf",
        "num_layers": 3,
        "feedback_mode": "qvalues",
        "feedback_dim": dz,
        "lower_feedback_projected": True,
        "frame_skip": 4,
        "frame_stack": 4,
        "action_space_mode": "minimal",
        "num_actions": 4,
        "replay_backing": "mmap",
        "learning_rate_decay_step": 1_000_000,
        "learning_rate_decay_scale": 0.1,
    }
    actual = {key: metrics.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"Protocol mismatch in {metrics_path}: {actual}")
    return True


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Return a causal rolling mean with a shorter leading window."""
    if window < 1:
        raise ValueError("--smooth must be at least one")
    if window == 1 or values.size < 2:
        return values
    totals = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    indices = np.arange(values.size)
    starts = np.maximum(0, indices - window + 1)
    return (totals[indices + 1] - totals[starts]) / (indices - starts + 1)


def style_axis(axis: plt.Axes, title: str) -> None:
    """Apply the shared progress-curve axis style."""
    axis.set_title(title)
    axis.set_xlabel("environment steps (×10⁶)")
    axis.set_ylabel("episodic return (last 100 episodes)")
    axis.grid(axis="y", alpha=0.28)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_seed_figure(
    output_path: Path,
    seed: int,
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    completed: dict[int, bool],
    window: int,
) -> None:
    """Write one seed-specific figure for the available dz trajectories."""
    fig, axis = plt.subplots(figsize=(9.4, 5.5))
    for dz, (steps, returns) in curves.items():
        values = smooth(returns, window)
        final = completed[dz]
        label = f"dz {dz} complete" if final else f"dz {dz} in progress ({steps[-1] / 1e6:.3f}M)"
        axis.plot(
            steps / 1_000_000.0,
            values,
            color=COLORS[dz],
            linestyle="-" if final else "--",
            linewidth=2.0,
            label=label,
        )
        if not final:
            axis.scatter(steps[-1] / 1_000_000.0, values[-1], color=COLORS[dz], s=28, zorder=3)
    style_axis(axis, f"Breakout minimal 4-action · GaWF L3 lower-projected · seed {seed}")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def aggregate(curves: list[tuple[np.ndarray, np.ndarray]], window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate complete curves onto their shared interval and return mean/sample SD."""
    lower = max(int(steps[0]) for steps, _ in curves)
    upper = min(int(steps[-1]) for steps, _ in curves)
    if upper <= lower:
        raise RuntimeError("Completed curves have no shared step interval")
    grid = np.linspace(lower, upper, N_GRID)
    aligned = np.vstack(
        [np.interp(grid, steps, smooth(returns, window)) for steps, returns in curves]
    )
    deviation = np.zeros_like(grid) if len(curves) == 1 else aligned.std(axis=0, ddof=1)
    return grid, aligned.mean(axis=0), deviation


def save_mean_std_figure(
    output_path: Path,
    complete: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]],
    window: int,
) -> dict[int, int]:
    """Write provisional completed-seed means and sample-SD bands for each dz."""
    fig, axis = plt.subplots(figsize=(9.4, 5.5))
    counts: dict[int, int] = {}
    for dz in DZ_VALUES:
        seed_curves = complete[dz]
        counts[dz] = len(seed_curves)
        if not seed_curves:
            continue
        steps, mean, deviation = aggregate(list(seed_curves.values()), window)
        x_axis = steps / 1_000_000.0
        label = f"dz {dz} completed seeds (n={len(seed_curves)})"
        axis.plot(x_axis, mean, color=COLORS[dz], linewidth=2.0, label=label)
        axis.fill_between(x_axis, mean - deviation, mean + deviation, color=COLORS[dz], alpha=0.17)
    style_axis(axis, "Breakout minimal 4-action · GaWF L3 lower-projected · provisional mean ± SD")
    axis.text(
        0.01,
        0.02,
        "Only completed seeds enter mean ± sample SD; in-progress curves are seed-only.",
        transform=axis.transAxes,
        fontsize=8.5,
        va="bottom",
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return counts


def main() -> None:
    """Render all available seed figures and the completed-seed provisional aggregate."""
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    available: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {seed: {} for seed in args.seeds}
    complete: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {dz: {} for dz in DZ_VALUES}
    completion: dict[int, dict[int, bool]] = {seed: {} for seed in args.seeds}
    for dz in DZ_VALUES:
        for seed in args.seeds:
            run_dir = data_root / suffix(dz, seed)
            history_path = run_dir / "metrics_history.jsonl"
            if not history_path.is_file():
                continue
            curve = load_history(history_path)
            final = is_complete(run_dir, dz)
            available[seed][dz] = curve
            completion[seed][dz] = final
            if final:
                complete[dz][seed] = curve
    written: list[str] = []
    for seed, curves in available.items():
        if not curves:
            continue
        output_path = output_dir / f"seed{seed}.png"
        save_seed_figure(output_path, seed, curves, completion[seed], args.smooth)
        written.append(output_path.name)
    if not any(complete.values()):
        raise RuntimeError("No completed seed is available for the provisional mean/std figure")
    mean_path = output_dir / "mean_std.png"
    counts = save_mean_std_figure(mean_path, complete, args.smooth)
    written.append(mean_path.name)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "smooth": args.smooth,
        "available_seeds": {str(seed): sorted(curves) for seed, curves in available.items()},
        "completed_seed_counts": {str(dz): count for dz, count in counts.items()},
        "files_written": written,
        "aggregation": "completed seeds only; sample SD is zero for n=1",
    }
    (data_root / "progress_snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
