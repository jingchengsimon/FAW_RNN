"""Regression coverage for the raw five-task Atari learning-curve CLI."""

from __future__ import annotations

import json
from pathlib import Path

from utils.analysis.rl.atari.atari_5task_raw_learning_curves import (
    TASKS,
    curve_for_task,
    curve_for_shared_loss,
    load_run_histories,
    render_learning_curves,
    render_task_learning_curves,
    write_inputs_and_manifest,
)


def test_step_offset_stitches_fresh_extension_phase(tmp_path: Path) -> None:
    """A fresh phase can continue the displayed x-axis without changing its raw history."""

    run_dirs = []
    for phase in (1, 2):
        run_dir = tmp_path / f"atari_5task_gru_seed1_phase{phase}"
        run_dir.mkdir()
        records = [
            {
                "global_step": step,
                "per_env": {
                    "ALE/Skiing-v5": {
                        "environment_steps": step,
                        "episodic_return_100": float(-step),
                    }
                },
            }
            for step in (1_000, 2_000)
        ]
        (run_dir / "metrics_history.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        run_dirs.append(run_dir)

    runs = load_run_histories(
        run_dirs,
        display_labels=["GRU seed 1", "GRU seed 1"],
        step_offsets=[0, 2_000],
    )
    phase1_x, _ = curve_for_task(runs[0], "Skiing", "environment_steps")
    phase2_x, _ = curve_for_task(runs[1], "Skiing", "environment_steps")

    assert phase1_x.tolist() == [1_000.0, 2_000.0]
    assert phase2_x.tolist() == [3_000.0, 4_000.0]


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
    skiing_figure = render_task_learning_curves(
        runs, output_dir, "Skiing", "environment_steps"
    )
    write_inputs_and_manifest(runs, output_dir, figures)

    assert [run.is_completed for run in runs] == [True, True, False, False]
    assert loss_steps.tolist() == [200.0, 400.0]
    assert losses.tolist() == [1.0, 2.0]
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    assert aggregate_figure.is_file() and aggregate_figure.stat().st_size > 0
    assert mixed_figure.is_file() and mixed_figure.stat().st_size > 0
    assert skiing_figure.name == "01_skiing_learning_curves_environment_steps.png"
    assert skiing_figure.stat().st_size > 0
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
