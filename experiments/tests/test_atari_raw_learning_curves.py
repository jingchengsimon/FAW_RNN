"""Regression coverage for the raw five-task Atari learning-curve CLI."""

from __future__ import annotations

import json
from pathlib import Path

from utils.analysis.rl.atari.atari_5task_raw_learning_curves import (
    TASKS,
    load_run_histories,
    render_learning_curves,
    write_inputs_and_manifest,
)


def test_raw_curve_renderer_accepts_running_runs_without_metrics(tmp_path: Path) -> None:
    """Four JSONL-only running results write two shared raw-curve figures and inputs."""

    records = []
    for global_step in (1_000, 2_000):
        records.append(
            {
                "global_step": global_step,
                "per_env": {
                    f"ALE/{task}-v5": {
                        "environment_steps": global_step // len(TASKS),
                        "episodic_return_100": float(global_step + index),
                    }
                    for index, task in enumerate(TASKS)
                },
            }
        )
    run_dirs = []
    for model, seed in (("gawf", 1), ("gawf", 2), ("ann", 1), ("ann", 2)):
        run_dir = tmp_path / f"atari_5task_{model}_seed{seed}"
        run_dir.mkdir()
        (run_dir / "metrics_history.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        run_dirs.append(run_dir)

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    runs = load_run_histories(run_dirs)
    figures = [
        render_learning_curves(runs, output_dir, "environment_steps"),
        render_learning_curves(runs, output_dir, "global_step"),
    ]
    aggregate_figure = render_learning_curves(
        runs, output_dir, "environment_steps", model_mean_std=True
    )
    write_inputs_and_manifest(runs, output_dir, figures)

    assert all(not run.has_metrics for run in runs)
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    assert aggregate_figure.is_file() and aggregate_figure.stat().st_size > 0
    assert (output_dir / "figure_inputs.csv").is_file()
    inputs = json.loads((output_dir / "figure_inputs.json").read_text(encoding="utf-8"))
    assert [row["display_label"] for row in inputs] == [
        "GaWF seed 1",
        "GaWF seed 2",
        "ANN seed 1",
        "ANN seed 2",
    ]
    assert all(row["metrics_present"] is False for row in inputs)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["key_numerical_results"]["run_count"] == 4
    assert manifest["key_numerical_results"]["figure_count"] == 2
