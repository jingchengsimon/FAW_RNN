"""Render descriptive figures for the fixed five-task full-18 Atari pilot.

Inputs are the saved ``metrics.json`` and ``metrics_history.jsonl`` files from the
five-task pilot.  The script does not load models or environments.  It produces
per-task learning curves, task/episode balance diagnostics, relative final-score
and learning-AUC heatmaps, and a data-coverage figure.  A partially completed
pilot is valid input: every plotted legend reports its available seed count.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TASKS = ("Pong", "Breakout", "Assault", "Seaquest", "Skiing")
MODELS = ("ann", "rnn", "gru", "lstm", "gawf")
MODEL_LABELS = {"ann": "ANN", "rnn": "RNN", "gru": "GRU", "lstm": "LSTM", "gawf": "GaWF"}
MODEL_COLORS = {"ann": "#7f7f7f", "rnn": "#1f77b4", "gru": "#2ca02c", "lstm": "#9467bd", "gawf": "#d62728"}
N_GRID = 250


@dataclass(frozen=True)
class Run:
    """One seed's structured pilot result."""

    model: str
    seed: int
    metrics: dict
    history: list[dict]


def parse_args() -> argparse.Namespace:
    """Parse input and output roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Pilot result directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG outputs.")
    return parser.parse_args()


def _task_name(env_id: str) -> str:
    """Map an ALE id to its compact task name."""
    return env_id.removeprefix("ALE/").removesuffix("-v5")


def _load_runs(data_root: Path) -> list[Run]:
    """Read only complete pilot result bundles present under ``data_root``."""
    runs: list[Run] = []
    for model in MODELS:
        for seed in range(1, 4):
            directory = data_root / (
                f"atari_dqn_5task_fs4_stack4_l3_buf1m_lrdecay1m_pilot_{model}_seed{seed}"
            )
            metrics_path, history_path = directory / "metrics.json", directory / "metrics_history.jsonl"
            if not metrics_path.is_file() or not history_path.is_file():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
            except (OSError, json.JSONDecodeError):
                continue
            if metrics.get("global_step", 0) < 5_000_000 or not history:
                continue
            runs.append(Run(model=model, seed=seed, metrics=metrics, history=history))
    return runs


def _curve(run: Run, task: str, x_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return finite per-task return curves against global or task-local steps."""
    xs, ys = [], []
    for record in run.history:
        per_env = record.get("per_env", {})
        value = per_env.get(f"ALE/{task}-v5", {}).get("episodic_return_100")
        if value is None or not np.isfinite(value):
            continue
        x = record.get("global_step") if x_key == "global_step" else per_env[f"ALE/{task}-v5"].get("environment_steps")
        if x is not None:
            xs.append(float(x))
            ys.append(float(value))
    if len(xs) < 2:
        return None
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def _aggregate(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate available seeds over their common range and compute mean/SEM."""
    if not curves:
        return None
    lo, hi = max(curve[0][0] for curve in curves), min(curve[0][-1] for curve in curves)
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, N_GRID)
    values = np.vstack([np.interp(grid, curve[0], curve[1]) for curve in curves])
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros(N_GRID)
    return grid, values.mean(axis=0), sem


def _style_axis(axis: plt.Axes) -> None:
    """Apply the shared compact plotting style."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)


def _learning_curves(runs: list[Run], output_dir: Path, x_key: str) -> None:
    """Plot five raw-score learning panels against a chosen step axis."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(18, 3.8))
    for axis, task in zip(axes, TASKS):
        for model in MODELS:
            curves = [curve for run in runs if run.model == model if (curve := _curve(run, task, x_key))]
            aggregate = _aggregate(curves)
            if aggregate is None:
                continue
            x, mean, sem = aggregate
            axis.plot(x / 1e6, mean, color=MODEL_COLORS[model], lw=1.6, label=f"{MODEL_LABELS[model]} (n={len(curves)})")
            if len(curves) > 1:
                axis.fill_between(x / 1e6, mean - sem, mean + sem, color=MODEL_COLORS[model], alpha=0.17, lw=0)
        axis.set_title(task)
        axis.set_xlabel("global steps (M)" if x_key == "global_step" else "task environment steps (M)")
        _style_axis(axis)
    axes[0].set_ylabel("episodic return (last 100 episodes)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.11))
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.19, top=0.79, wspace=0.27)
    fig.savefig(output_dir / f"01_learning_curves_{x_key}.png", dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _balance(runs: list[Run], output_dir: Path) -> None:
    """Visualize final task-step equality and intentionally unequal episode counts."""
    step_values = np.full((len(runs), len(TASKS)), np.nan)
    episode_values = np.full_like(step_values, np.nan)
    for row, run in enumerate(runs):
        per_env = run.metrics.get("per_env", {})
        for col, task in enumerate(TASKS):
            data = per_env.get(f"ALE/{task}-v5", {})
            step_values[row, col] = data.get("environment_steps", np.nan)
            episode_values[row, col] = data.get("episodes", np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    for row in range(len(runs)):
        axes[0].plot(TASKS, step_values[row] / 1e6, color="#808080", alpha=0.22, lw=0.75)
        axes[1].plot(TASKS, episode_values[row], color="#808080", alpha=0.22, lw=0.75)
    axes[0].plot(TASKS, np.nanmean(step_values, axis=0) / 1e6, color="#111111", marker="o", lw=2, label="mean across completed runs")
    axes[0].axhline(1.0, color="#d62728", ls="--", lw=1.0, label="target: 1.0M/task")
    axes[1].plot(TASKS, np.nanmean(episode_values, axis=0), color="#111111", marker="o", lw=2)
    axes[0].set_ylabel("final task environment steps (M)")
    axes[1].set_ylabel("final episodes")
    for axis, title in zip(axes, ("Frame-level task balance", "Episode counts (not balanced)")):
        axis.set_title(title)
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "02_task_balance_and_episode_counts.png", dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _relative_heatmap(values: np.ndarray, title: str, output_path: Path) -> None:
    """Plot per-task model-relative values, preserving missing cells."""
    normalized = np.full_like(values, np.nan, dtype=float)
    for column in range(values.shape[1]):
        valid = values[:, column][np.isfinite(values[:, column])]
        if valid.size == 1:
            normalized[np.isfinite(values[:, column]), column] = 1.0
        elif valid.size > 1 and valid.max() > valid.min():
            normalized[:, column] = (values[:, column] - valid.min()) / (valid.max() - valid.min())
    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    image = axis.imshow(np.ma.masked_invalid(normalized), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(TASKS)), TASKS)
    axis.set_yticks(range(len(MODELS)), [MODEL_LABELS[model] for model in MODELS])
    axis.set_title(title)
    for row in range(len(MODELS)):
        for col in range(len(TASKS)):
            if np.isfinite(normalized[row, col]):
                axis.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center", color="white" if normalized[row, col] < 0.55 else "black", fontsize=9)
            else:
                axis.text(col, row, "—", ha="center", va="center", color="#555555", fontsize=11)
    fig.colorbar(image, ax=axis, label="within-task relative score (0 = lowest, 1 = highest)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _summary_heatmaps(runs: list[Run], output_dir: Path) -> None:
    """Plot seed-mean final return and trajectory AUC relative to each task."""
    final_values = np.full((len(MODELS), len(TASKS)), np.nan)
    auc_values = np.full_like(final_values, np.nan)
    for model_index, model in enumerate(MODELS):
        model_runs = [run for run in runs if run.model == model]
        for task_index, task in enumerate(TASKS):
            finals, aucs = [], []
            for run in model_runs:
                curve = _curve(run, task, "environment_steps")
                if curve is None:
                    continue
                x, y = curve
                finals.append(y[-1])
                area = np.sum((y[1:] + y[:-1]) * np.diff(x) * 0.5)
                aucs.append(area / (x[-1] - x[0]))
            if finals:
                final_values[model_index, task_index] = np.mean(finals)
                auc_values[model_index, task_index] = np.mean(aucs)
    _relative_heatmap(final_values, "Final performance (relative within each task)", output_dir / "03_final_return_relative_heatmap.png")
    _relative_heatmap(auc_values, "Learning AUC (relative within each task)", output_dir / "04_learning_auc_relative_heatmap.png")


def _coverage(runs: list[Run], output_dir: Path) -> None:
    """Show which model/seed cells have a complete 5M-step result."""
    matrix = np.full((len(MODELS), 3), np.nan)
    for run in runs:
        matrix[MODELS.index(run.model), run.seed - 1] = run.metrics.get("global_step", np.nan) / 1e6
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap="Blues", vmin=0, vmax=5, aspect="auto")
    axis.set_xticks(range(3), ["seed 1", "seed 2", "seed 3"])
    axis.set_yticks(range(len(MODELS)), [MODEL_LABELS[model] for model in MODELS])
    axis.set_title("Completed pilot data coverage")
    for row in range(len(MODELS)):
        for col in range(3):
            label = f"{matrix[row, col]:.1f}M" if np.isfinite(matrix[row, col]) else "missing"
            axis.text(col, row, label, ha="center", va="center", color="white" if np.isfinite(matrix[row, col]) else "#555555")
    fig.colorbar(image, ax=axis, label="final global steps (M)")
    fig.tight_layout()
    fig.savefig(output_dir / "05_data_coverage.png", dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _write_summary(runs: list[Run], output_dir: Path) -> None:
    """Persist a small auditable table listing the plotted complete result bundles."""
    with (output_dir / "figure_inputs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "seed", "global_step", "history_records"))
        for run in runs:
            writer.writerow((run.model, run.seed, run.metrics.get("global_step"), len(run.history)))


def main() -> None:
    """Render all descriptive pilot figures."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(args.data_root)
    if not runs:
        raise SystemExit(f"No complete 5M pilot result bundles under {args.data_root}")
    _learning_curves(runs, args.output_dir, "environment_steps")
    _learning_curves(runs, args.output_dir, "global_step")
    _balance(runs, args.output_dir)
    _summary_heatmaps(runs, args.output_dir)
    _coverage(runs, args.output_dir)
    _write_summary(runs, args.output_dir)
    print(f"Rendered {len(runs)} complete pilot bundles to {args.output_dir}")


if __name__ == "__main__":
    main()
