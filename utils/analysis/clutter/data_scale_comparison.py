"""Summarize and plot the Clutter 4h/10h/20h/40h full-grid comparison.

Input is the historical ``gen_hparam_full_grid`` directory containing one metrics JSON per
task. The script writes compact CSV/JSON summaries below ``results/data/analysis`` and renders
the corresponding accuracy and overfit-gap figures below ``results/figs``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.fg_switch_offset_acc import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
)


CATEGORY = "G_behaviour"
SCRIPT_NAME = "data_scale_comparison"
SCALES = ("4h", "10h", "20h", "40h")
MODELS = ("rnn", "lstm", "gru", "gawf")
HIDDEN_SIZES = (64, 128, 256, 512)
LEARNING_RATES = (1e-4, 5e-4, 1e-3, 5e-3)
WEIGHT_DECAYS = (0.0, 1e-5, 1e-4, 1e-3)
EVAL_DATASET_SUFFIX = "40h-float32"
CSV_FIELDS = (
    "task_id",
    "scale",
    "model",
    "hidden_size",
    "lr",
    "weight_decay",
    "actual_epochs",
    "train_acc_char",
    "val_acc_char",
    "overfit_gap_char",
    "train_acc_sector",
    "val_acc_sector",
    "overfit_gap_sector",
    "best_epoch_char",
    "best_epoch_sector",
    "source_metrics",
)
Row = dict[str, str | int | float]


def parse_args() -> argparse.Namespace:
    """Parse the historical result root and optional canonical output overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Historical gen_hparam_full_grid directory containing task_* leaves.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--save-pdf", action="store_true")
    return parser.parse_args()


def _number(payload: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"Metrics are missing all required fields: {keys}")


def _task_id(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("task_"))
    except ValueError as exc:
        raise ValueError(f"Metrics parent must be task_<integer>: {path}") from exc


def load_runs(input_root: Path) -> list[Row]:
    """Load and validate the historical four-scale, four-model grid metrics."""

    paths = sorted(input_root.glob("task_*/*_metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No task metrics found under {input_root}")
    scale_by_suffix = {f"{scale}-float32": scale for scale in SCALES}
    rows: list[Row] = []
    seen: set[tuple[str, str, int, float, float]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        suffix = str(payload.get("dataset_suffix"))
        model = str(payload.get("model_type"))
        if suffix not in scale_by_suffix or model not in MODELS:
            raise ValueError(f"Unexpected dataset/model in {path}: {suffix}, {model}")
        if payload.get("eval_dataset_suffix") != EVAL_DATASET_SUFFIX:
            raise ValueError(
                f"Expected eval_dataset_suffix={EVAL_DATASET_SUFFIX!r} in {path}, "
                f"found {payload.get('eval_dataset_suffix')!r}"
            )
        hidden_size = int(payload["hidden_size"])
        lr = float(payload["lr"])
        weight_decay = float(payload["weight_decay"])
        if (
            hidden_size not in HIDDEN_SIZES
            or lr not in LEARNING_RATES
            or weight_decay not in WEIGHT_DECAYS
        ):
            raise ValueError(f"Unexpected grid configuration in {path}")
        scale = scale_by_suffix[suffix]
        identity = (scale, model, hidden_size, lr, weight_decay)
        if identity in seen:
            raise ValueError(f"Duplicate grid configuration {identity} in {path}")
        seen.add(identity)
        train_char = _number(payload, "train_acc_at_best_val", "best_train_acc_char")
        val_char = _number(payload, "val_acc_at_best", "best_val_acc_char")
        train_sector = _number(
            payload,
            "train_acc_sector_at_best_val_sector",
            "best_train_acc_pos",
        )
        val_sector = _number(payload, "val_acc_sector_at_best", "best_val_acc_pos")
        rows.append(
            {
                "task_id": _task_id(path),
                "scale": scale,
                "model": model,
                "hidden_size": hidden_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "actual_epochs": int(_number(payload, "actual_epochs", "num_epochs")),
                "train_acc_char": train_char,
                "val_acc_char": val_char,
                "overfit_gap_char": float(payload.get("overfit_gap", train_char - val_char)),
                "train_acc_sector": train_sector,
                "val_acc_sector": val_sector,
                "overfit_gap_sector": float(
                    payload.get("overfit_gap_sector", train_sector - val_sector)
                ),
                "best_epoch_char": int(
                    _number(payload, "best_epoch_val_acc_1based", "best_epoch_char")
                ),
                "best_epoch_sector": int(
                    _number(payload, "best_epoch_val_acc_sector_1based", "best_epoch_pos")
                ),
                "source_metrics": str(path.resolve()),
            }
        )
    return rows


def select_best_runs(rows: list[Row]) -> list[Row]:
    """Select the highest validation-character-accuracy run per scale and model."""

    best: list[Row] = []
    for scale, model in product(SCALES, MODELS):
        candidates = [row for row in rows if row["scale"] == scale and row["model"] == model]
        if not candidates:
            raise RuntimeError(f"No valid runs for scale={scale}, model={model}")
        best.append(
            max(
                candidates,
                key=lambda row: (float(row["val_acc_char"]), -int(row["task_id"])),
            )
        )
    return best


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _missing_configurations(rows: list[Row]) -> list[dict[str, str | int | float]]:
    observed = {
        (
            str(row["scale"]),
            str(row["model"]),
            int(row["hidden_size"]),
            float(row["lr"]),
            float(row["weight_decay"]),
        )
        for row in rows
    }
    expected = product(SCALES, MODELS, HIDDEN_SIZES, LEARNING_RATES, WEIGHT_DECAYS)
    return [
        {
            "scale": scale,
            "model": model,
            "hidden_size": hidden_size,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        for scale, model, hidden_size, lr, weight_decay in expected
        if (scale, model, hidden_size, lr, weight_decay) not in observed
    ]


def _write_summary(path: Path, input_root: Path, rows: list[Row], best: list[Row]) -> None:
    counts = Counter((str(row["scale"]), str(row["model"])) for row in rows)
    payload = {
        "input_root": str(input_root.resolve()),
        "scales": list(SCALES),
        "models": list(MODELS),
        "eval_dataset_suffix": EVAL_DATASET_SUFFIX,
        "expected_runs": len(SCALES)
        * len(MODELS)
        * len(HIDDEN_SIZES)
        * len(LEARNING_RATES)
        * len(WEIGHT_DECAYS),
        "observed_runs": len(rows),
        "counts_by_scale_model": {
            scale: {model: counts[(scale, model)] for model in MODELS} for scale in SCALES
        },
        "missing_configurations": _missing_configurations(rows),
        "best_runs": best,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_best_csv(path: Path) -> list[Row]:
    numeric = set(CSV_FIELDS) - {"scale", "model", "source_metrics"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[Row] = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    key: (float(value) if key in numeric else value)
                    for key, value in raw.items()
                }
            )
    return rows


def _plot_metric_pair(
    rows: list[Row],
    figure_dir: Path,
    stem: str,
    char_field: str,
    sector_field: str,
    ylabel: str,
    *,
    accuracy: bool,
    save_pdf: bool,
) -> list[Path]:
    by_identity = {(str(row["scale"]), str(row["model"])): row for row in rows}
    x = list(range(len(SCALES)))
    with plt.rc_context(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=accuracy)
        for axis, field, title in zip(axes, (char_field, sector_field), ("Char", "Sector")):
            for model in MODELS:
                values = [float(by_identity[(scale, model)][field]) for scale in SCALES]
                axis.plot(
                    x,
                    values,
                    color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model],
                    linewidth=2.2,
                    markersize=6.5,
                    label=MODEL_LABELS[model],
                )
            axis.set_title(title)
            axis.set_xticks(x, SCALES)
            axis.set_xlabel("Training data (hours)")
            axis.grid(axis="y", linewidth=0.7, alpha=0.25)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if accuracy:
                axis.set_ylim(0.0, 100.0)
            else:
                axis.axhline(0.0, color="#777777", linewidth=0.8, zorder=0)
        fig.supylabel(ylabel, x=0.02)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(MODELS), frameon=False)
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.82, wspace=0.18)
        figure_dir.mkdir(parents=True, exist_ok=True)
        outputs = [figure_dir / f"{stem}.png"]
        fig.savefig(outputs[0], dpi=150, bbox_inches="tight", pad_inches=0.06)
        if save_pdf:
            outputs.append(figure_dir / f"{stem}.pdf")
            fig.savefig(outputs[-1], bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    return outputs


def run(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """Write structured summaries, then render all figures from the saved best-run CSV."""

    data_dir = args.data_dir or output_dir(CATEGORY, SCRIPT_NAME, "data")
    figure_dir = args.figure_dir or output_dir(CATEGORY, SCRIPT_NAME, "figs")
    rows = load_runs(args.input_root)
    best = select_best_runs(rows)
    all_runs_csv = data_dir / "data_scale_all_runs.csv"
    best_runs_csv = data_dir / "data_scale_best_runs.csv"
    summary_json = data_dir / "data_scale_summary.json"
    _write_csv(all_runs_csv, rows)
    _write_csv(best_runs_csv, best)
    _write_summary(summary_json, args.input_root, rows, best)
    saved_best = _read_best_csv(best_runs_csv)
    figures: list[Path] = []
    figures.extend(
        _plot_metric_pair(
            saved_best,
            figure_dir,
            "data_scale_validation_accuracy",
            "val_acc_char",
            "val_acc_sector",
            "Validation accuracy (%)",
            accuracy=True,
            save_pdf=args.save_pdf,
        )
    )
    figures.extend(
        _plot_metric_pair(
            saved_best,
            figure_dir,
            "data_scale_training_accuracy",
            "train_acc_char",
            "train_acc_sector",
            "Training accuracy (%)",
            accuracy=True,
            save_pdf=args.save_pdf,
        )
    )
    figures.extend(
        _plot_metric_pair(
            saved_best,
            figure_dir,
            "data_scale_overfit_gap",
            "overfit_gap_char",
            "overfit_gap_sector",
            "Overfit gap (percentage points)",
            accuracy=False,
            save_pdf=args.save_pdf,
        )
    )
    return [all_runs_csv, best_runs_csv, summary_json], figures


def main() -> None:
    """Run the restored data-scale summary and visualization workflow."""

    data_files, figure_files = run(parse_args())
    print("Data:")
    for path in data_files:
        print(path)
    print("Figures:")
    for path in figure_files:
        print(path)


if __name__ == "__main__":
    main()
