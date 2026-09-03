"""Regression checks for the restored Clutter data-scale comparison."""

from __future__ import annotations

import argparse
import csv
import json

from utils.analysis.clutter.data_scale_comparison import MODELS, SCALES, run


def test_workflow_selects_best_runs_and_separates_data_from_figures(tmp_path) -> None:
    """The restored workflow writes numeric summaries and figures to their own roots."""

    input_root = tmp_path / "input"
    task_id = 0
    for scale in SCALES:
        for model in MODELS:
            task_dir = input_root / f"task_{task_id:04d}"
            task_dir.mkdir(parents=True)
            metrics = {
                "dataset_suffix": f"{scale}-float32",
                "eval_dataset_suffix": "40h-float32",
                "model_type": model,
                "hidden_size": 64,
                "lr": 0.0001,
                "weight_decay": 0.0,
                "actual_epochs": 100,
                "train_acc_at_best_val": 85.0,
                "val_acc_at_best": 75.0,
                "train_acc_sector_at_best_val_sector": 92.0,
                "val_acc_sector_at_best": 88.0,
                "best_epoch_val_acc_1based": 90,
                "best_epoch_val_acc_sector_1based": 80,
            }
            (task_dir / f"{model}_metrics.json").write_text(json.dumps(metrics))
            task_id += 1
    better = input_root / f"task_{task_id:04d}"
    better.mkdir()
    metrics.update(
        {
            "dataset_suffix": "4h-float32",
            "model_type": "rnn",
            "hidden_size": 128,
            "val_acc_at_best": 80.0,
        }
    )
    (better / "rnn_metrics.json").write_text(json.dumps(metrics))
    data_dir = tmp_path / "data"
    figure_dir = tmp_path / "figures"

    data_files, figure_files = run(
        argparse.Namespace(
            input_root=input_root,
            data_dir=data_dir,
            figure_dir=figure_dir,
            save_pdf=False,
        )
    )

    assert len(data_files) == 3
    assert all(path.parent == data_dir for path in data_files)
    assert len(figure_files) == 3
    assert all(path.parent == figure_dir and path.suffix == ".png" for path in figure_files)
    with (data_dir / "data_scale_best_runs.csv").open(newline="") as handle:
        best = list(csv.DictReader(handle))
    selected = next(row for row in best if row["scale"] == "4h" and row["model"] == "rnn")
    assert float(selected["val_acc_char"]) == 80.0
    assert int(selected["hidden_size"]) == 128
