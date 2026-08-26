"""Render raw per-run five-task Atari learning curves from JSONL histories.

Inputs are one or more explicitly supplied five-task result directories.  Each directory must
contain ``metrics_history.jsonl``; ``metrics.json`` is optional so running experiments can be
rendered.  The module writes a shared-loss panel plus five return panels for the requested
x-axes, defaulting to both ``environment_steps`` and ``global_step``.  It can overlay raw runs
or aggregate each model's available or completed seed curves as mean plus sample SD while retaining
running seeds as raw curves.  The script also writes
auditable ``figure_inputs`` CSV/JSON files and a manifest in the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TASKS = ("Pong", "Breakout", "Assault", "Seaquest", "Skiing")
MODELS = ("ann", "rnn", "gru", "lstm", "gawf")
FORMAL_COMPLETED_STEPS = 20_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_LABELS = {"ann": "ANN", "rnn": "RNN", "gru": "GRU", "lstm": "LSTM", "gawf": "GaWF"}
MODEL_COLORS = {
    "ann": "#7f7f7f",
    "rnn": "#1f77b4",
    "gru": "#2ca02c",
    "lstm": "#9467bd",
    "gawf": "#d62728",
}
MODEL_SEED_PATTERN = re.compile(r"(?:^|_)(ann|rnn|gru|lstm|gawf)_seed(\d+)(?:_|$)")


@dataclass(frozen=True)
class RunHistory:
    """One explicit result directory and its parsed training-history records."""

    directory: Path
    label: str
    display_label: str
    history: list[dict[str, Any]]
    has_metrics: bool
    is_completed: bool


def parse_args() -> argparse.Namespace:
    """Parse explicit run directories and the output directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help="Five-task result directory; repeat once for each run to plot.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Optional display label; repeat once per --run-dir in the same order.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Destination for PNGs and inputs."
    )
    parser.add_argument(
        "--x-axis",
        action="append",
        choices=("environment_steps", "global_step"),
        dest="x_axes",
        help="Axis to render; repeat for both. Defaults to both axes.",
    )
    parser.add_argument(
        "--model-mean-std",
        action="store_true",
        help="Aggregate each model's available seed curves as mean plus sample-SD.",
    )
    parser.add_argument(
        "--completed-mean-std",
        action="store_true",
        help="Aggregate completed formal seeds per model; retain running seeds as raw curves.",
    )
    args = parser.parse_args()
    if args.label is not None and len(args.label) != len(args.run_dir):
        parser.error("--label must be supplied once for every --run-dir.")
    if args.model_mean_std and args.completed_mean_std:
        parser.error("--model-mean-std and --completed-mean-std are mutually exclusive.")
    return args


def _safe_label(directory: Path, index: int) -> str:
    """Return a stable filename label that cannot collide for ordinary result directories."""

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", directory.name).strip("_.-")
    return f"run{index:02d}_{stem or 'result'}"


def _display_label(directory: Path, index: int) -> str:
    """Infer a compact model/seed legend label when its result-directory name provides one."""

    match = MODEL_SEED_PATTERN.search(directory.name)
    if match is None:
        return f"run {index}: {directory.name}"
    return f"{MODEL_LABELS[match.group(1)]} seed {match.group(2)}"


def _is_completed_run(directory: Path) -> bool:
    """Return whether a formal run has final metrics at the 20M-step target."""

    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return int(metrics.get("global_step", -1)) >= FORMAL_COMPLETED_STEPS
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def load_run_histories(
    run_dirs: list[Path], display_labels: list[str] | None = None
) -> list[RunHistory]:
    """Read histories from the user-selected directories without requiring final metrics."""

    runs: list[RunHistory] = []
    for index, directory in enumerate(run_dirs, start=1):
        resolved = directory.expanduser().resolve()
        history_path = resolved / "metrics_history.jsonl"
        if not history_path.is_file():
            raise FileNotFoundError(f"Missing metrics_history.jsonl under {resolved}")
        history: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            history_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {history_path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {history_path}:{line_number}")
            history.append(record)
        if not history:
            raise RuntimeError(f"No history records in {history_path}")
        runs.append(
            RunHistory(
                directory=resolved,
                label=_safe_label(resolved, index),
                display_label=(
                    display_labels[index - 1]
                    if display_labels is not None
                    else _display_label(resolved, index)
                ),
                history=history,
                has_metrics=(resolved / "metrics.json").is_file(),
                is_completed=_is_completed_run(resolved),
            )
        )
    return runs


def curve_for_task(
    run: RunHistory,
    task: str,
    x_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract finite ``episodic_return_100`` values for one environment and x-axis."""

    xs: list[float] = []
    ys: list[float] = []
    environment_key = f"ALE/{task}-v5"
    for record in run.history:
        per_env = record.get("per_env")
        if not isinstance(per_env, dict):
            continue
        values = per_env.get(environment_key)
        if not isinstance(values, dict):
            continue
        x_value = record.get("global_step") if x_key == "global_step" else values.get(x_key)
        y_value = values.get("episodic_return_100")
        try:
            x, y = float(x_value), float(y_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    return np.asarray(xs, dtype=np.float64)[order], np.asarray(ys, dtype=np.float64)[order]


def curve_for_shared_loss(run: RunHistory, x_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract the shared TD loss against global or mean task environment steps."""

    xs: list[float] = []
    ys: list[float] = []
    for record in run.history:
        per_env = record.get("per_env")
        if not isinstance(per_env, dict):
            continue
        if x_key == "global_step":
            x_value = record.get("global_step")
        else:
            task_steps = [
                values.get("environment_steps")
                for values in per_env.values()
                if isinstance(values, dict)
            ]
            if len(task_steps) != len(TASKS):
                continue
            x_value = np.mean(task_steps)
        try:
            x, y = float(x_value), float(record.get("loss"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    return np.asarray(xs, dtype=np.float64)[order], np.asarray(ys, dtype=np.float64)[order]


def _style_axis(axis: plt.Axes) -> None:
    """Apply the compact learning-curve style used by the pilot figures."""

    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)


def _model_seed(directory: Path) -> tuple[str | None, int]:
    """Extract the model and seed encoded by a run-directory name."""

    match = MODEL_SEED_PATTERN.search(directory.name)
    return (match.group(1), int(match.group(2))) if match else (None, 1)


def _mean_std(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate seed curves over their shared interval and return mean/sample SD."""

    if not curves:
        return None
    lower = max(curve[0][0] for curve in curves)
    upper = min(curve[0][-1] for curve in curves)
    if upper <= lower:
        return None
    grid = np.linspace(lower, upper, 250)
    values = np.vstack([np.interp(grid, x_values, y_values) for x_values, y_values in curves])
    deviation = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(grid)
    return grid, values.mean(axis=0), deviation


def _plot_curves(
    axis: plt.Axes,
    runs: list[RunHistory],
    curve: Callable[[RunHistory], tuple[np.ndarray, np.ndarray]],
    *,
    model_mean_std: bool,
    completed_mean_std: bool,
) -> bool:
    """Plot raw runs, all-seed aggregates, or completed-seed aggregates."""

    has_data = False
    if model_mean_std or completed_mean_std:
        for model in MODELS:
            selected_runs = [
                run
                for run in runs
                if _model_seed(run.directory)[0] == model
                and (model_mean_std or run.is_completed)
            ]
            curves = [curve(run) for run in selected_runs]
            aggregate = _mean_std([item for item in curves if item[0].size])
            if aggregate is None:
                continue
            x_values, mean, deviation = aggregate
            count = sum(item[0].size > 0 for item in curves)
            label = f"{MODEL_LABELS[model]} (n={count})"
            if completed_mean_std:
                label = f"{MODEL_LABELS[model]} completed (n={count})"
            axis.plot(x_values / 1_000_000.0, mean, color=MODEL_COLORS[model], linewidth=1.5,
                      label=label)
            if count > 1:
                axis.fill_between(x_values / 1_000_000.0, mean - deviation, mean + deviation,
                                  color=MODEL_COLORS[model], alpha=0.17, linewidth=0)
            has_data = True
    if not model_mean_std:
        duplicate_indices: defaultdict[tuple[str | None, int], int] = defaultdict(int)
        line_styles = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)))
        for run in runs:
            if completed_mean_std and run.is_completed:
                continue
            x_values, y_values = curve(run)
            if not x_values.size:
                continue
            model, seed = _model_seed(run.directory)
            key = (model, seed)
            duplicate_index = duplicate_indices[key]
            duplicate_indices[key] += 1
            axis.plot(
                x_values / 1_000_000.0,
                y_values,
                color=MODEL_COLORS.get(model),
                linestyle=line_styles[(seed - 1 + duplicate_index) % len(line_styles)],
                linewidth=1.35,
                label=run.display_label,
            )
            has_data = True
    return has_data


def render_learning_curves(
    runs: list[RunHistory],
    output_dir: Path,
    x_key: str,
    model_mean_std: bool = False,
    completed_mean_std: bool = False,
) -> Path:
    """Write shared-loss plus five-return panels and return the PNG path."""

    if x_key not in {"environment_steps", "global_step"}:
        raise ValueError(f"Unsupported x-axis key: {x_key}")
    figure, axes = plt.subplots(1, len(TASKS) + 1, figsize=(21, 3.8))
    x_label = "global steps (M)" if x_key == "global_step" else "task environment steps (M)"
    loss_axis, task_axes = axes[0], axes[1:]
    _plot_curves(
        loss_axis,
        runs,
        lambda run: curve_for_shared_loss(run, x_key),
        model_mean_std=model_mean_std,
        completed_mean_std=completed_mean_std,
    )
    loss_axis.set_title("Shared TD loss")
    loss_axis.set_xlabel(
        "global steps (M)" if x_key == "global_step" else "mean task environment steps (M)"
    )
    loss_axis.set_ylabel("last TD loss")
    _style_axis(loss_axis)

    for axis, task in zip(task_axes, TASKS):
        has_data = _plot_curves(
            axis,
            runs,
            lambda run: curve_for_task(run, task, x_key),
            model_mean_std=model_mean_std,
            completed_mean_std=completed_mean_std,
        )
        if not has_data:
            axis.text(
                0.5, 0.5, "no finite data", ha="center", va="center", transform=axis.transAxes
            )
        axis.set_title(task)
        axis.set_xlabel(x_label)
        _style_axis(axis)
    task_axes[0].set_ylabel("episodic return (last 100 episodes)")
    handles, labels = loss_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)), frameon=False,
                 bbox_to_anchor=(0.5, 1.11))
    figure.subplots_adjust(left=0.05, right=0.995, bottom=0.19, top=0.79, wspace=0.28)
    output_path = output_dir / f"01_learning_curves_{x_key}.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)
    return output_path


def _git_commit() -> str:
    """Return the current commit when available, without making it a rendering dependency."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def write_inputs_and_manifest(
    runs: list[RunHistory],
    output_dir: Path,
    figures: list[Path],
    model_mean_std: bool = False,
    completed_mean_std: bool = False,
) -> None:
    """Write auditable selected-run records and a manifest for the rendered PNGs."""

    input_rows = [
        {
            "label": run.label,
            "display_label": run.display_label,
            "model": _model_seed(run.directory)[0],
            "seed": _model_seed(run.directory)[1],
            "run_directory": str(run.directory),
            "metrics_history": str(run.directory / "metrics_history.jsonl"),
            "metrics_present": run.has_metrics,
            "completed": run.is_completed,
            "history_records": len(run.history),
        }
        for run in runs
    ]
    csv_path = output_dir / "figure_inputs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(input_rows[0]))
        writer.writeheader()
        writer.writerows(input_rows)
    json_path = output_dir / "figure_inputs.json"
    json_path.write_text(json.dumps(input_rows, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "script_path": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "git_commit": _git_commit(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_directories": [str(run.directory) for run in runs],
        "files_written": [path.name for path in figures] + [csv_path.name, json_path.name],
        "key_numerical_results": {
            "run_count": len(runs),
            "figure_count": len(figures),
            "history_record_count": sum(len(run.history) for run in runs),
            "aggregation": (
                "mean_sample_std"
                if model_mean_std
                else "completed_mean_std_with_partial_raw"
                if completed_mean_std
                else "raw_runs"
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Render raw figures for every explicitly supplied five-task result directory."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_run_histories(args.run_dir, args.label)
    x_axes = args.x_axes or ["environment_steps", "global_step"]
    figures = [
        render_learning_curves(
            runs,
            args.output_dir,
            x_key,
            args.model_mean_std,
            args.completed_mean_std,
        )
        for x_key in x_axes
    ]
    write_inputs_and_manifest(
        runs,
        args.output_dir,
        figures,
        args.model_mean_std,
        args.completed_mean_std,
    )
    print(f"Rendered {len(figures)} raw learning-curve figures to {args.output_dir}")


if __name__ == "__main__":
    main()
