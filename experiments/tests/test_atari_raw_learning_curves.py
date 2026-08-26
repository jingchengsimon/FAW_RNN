"""Regression coverage for the raw five-task Atari learning-curve CLI."""

from __future__ import annotations

import json
from pathlib import Path

from utils.analysis.rl.atari.atari_5task_raw_learning_curves import (
    TASKS,
    curve_for_shared_loss,
    load_run_histories,
    render_learning_curves,
    write_inputs_and_manifest,
)


def test_raw_curve_renderer_accepts_running_runs_without_metrics(tmp_path: Path) -> None:
    """Completed seeds aggregate while running seeds remain raw in the mixed figure."""

    records = []
    for global_step in (1_000, 2_000):
        records.append(
            {
                "global_step": global_step,
                "loss": global_step / 1_000.0,
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
        if model == "gawf":
            (run_dir / "metrics.json").write_text(
                json.dumps({"global_step": 20_000_000}) + "\n", encoding="utf-8"
            )
        run_dirs.append(run_dir)

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    runs = load_run_histories(run_dirs)
    loss_steps, losses = curve_for_shared_loss(runs[0], "environment_steps")
    figures = [
        render_learning_curves(runs, output_dir, "environment_steps"),
        render_learning_curves(runs, output_dir, "global_step"),
    ]
    aggregate_figure = render_learning_curves(
        runs, output_dir, "environment_steps", model_mean_std=True
    )
    mixed_figure = render_learning_curves(
        runs, output_dir, "environment_steps", completed_mean_std=True
    )
    write_inputs_and_manifest(runs, output_dir, figures)

    assert [run.is_completed for run in runs] == [True, True, False, False]
    assert loss_steps.tolist() == [200.0, 400.0]
    assert losses.tolist() == [1.0, 2.0]
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    assert aggregate_figure.is_file() and aggregate_figure.stat().st_size > 0
    assert mixed_figure.is_file() and mixed_figure.stat().st_size > 0
    assert (output_dir / "figure_inputs.csv").is_file()
    inputs = json.loads((output_dir / "figure_inputs.json").read_text(encoding="utf-8"))
    assert [row["display_label"] for row in inputs] == [
        "GaWF seed 1",
        "GaWF seed 2",
        "ANN seed 1",
        "ANN seed 2",
    ]
    assert [row["completed"] for row in inputs] == [True, True, False, False]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["key_numerical_results"]["run_count"] == 4
    assert manifest["key_numerical_results"]["figure_count"] == 2
