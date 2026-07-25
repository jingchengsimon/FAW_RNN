"""Merged 2x4 summary: the best6 multiseed 2x3 panels plus a GaWF shuffle-ablation column.

Columns 0-2 reuse the exact panels of ``clutter_multiseed_summary`` (test accuracy, validation
loss, target-switch recovery) for the Digit (row 0) and Sector (row 1) readouts. Column 3 adds
the GaWF feedback shuffle ablation: three bars (Baseline, Shuffle digit, Shuffle sector) per
readout, with the baseline taken from the same multiseed test CSV as column 0. The two shuffle
panels share one y-axis. All four columns are equal width and 30% narrower than the 2x3 columns.

The original 2x3 summary and the standalone shuffle figure are left untouched; this writes a new
``best6_multiseed_shuffle_2x4.png`` alongside them. PNG only for now.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_viz.clutter_multiseed_summary import (  # noqa: E402
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    _mean_sd,
    _plot_loss_axis,
    _plot_recovery_axis,
    _plot_test_axis,
    _style_axis,
    load_recovery_curves,
    load_test_metrics,
)

# Digit/sector bar colours align with core_objects_aggregate_2x2 (digit #E76F51, sector #264653).
DIGIT_COLOR = "#E76F51"
SECTOR_COLOR = "#264653"
DATA_ROOT = PROJECT_ROOT / "results" / "data" / "anal_data" / "G_behaviour"
FIG_DIR = PROJECT_ROOT / "results" / "train_figs" / "clutter" / "clutter_best6_multiseed_40h_ep150"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=DATA_ROOT
        / "clutter_multiseed_best_acc_bars"
        / "clutter_best6_multiseed_40h_ep150"
        / "best_acc_test_mean_std.csv",
    )
    parser.add_argument("--loss_png", type=Path, default=FIG_DIR / "loss_mean_std.png")
    parser.add_argument(
        "--recovery_dir",
        type=Path,
        default=DATA_ROOT
        / "export_fg_switch_offset_acc"
        / "fg_switch_offset_acc_clutter_best_jointswitch_balanced_10digit_unique_sector_covered"
        / "fg10",
    )
    parser.add_argument(
        "--ablation_dir", type=Path, default=DATA_ROOT / "feedback_ablation_multiseed"
    )
    parser.add_argument(
        "--output_png", type=Path, default=FIG_DIR / "best6_multiseed_shuffle_2x4.png"
    )
    parser.add_argument("--output_pdf", type=Path, default=None)
    return parser.parse_args()


def _shuffle_from_ablation(ablation_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return per-seed shuffle_digit/shuffle_sector arrays for both readouts."""

    files = sorted(glob.glob(str(ablation_dir / "gawf-seed*" / "ablation_metrics.json")))
    if not files:
        raise FileNotFoundError(f"No gawf-seed*/ablation_metrics.json under {ablation_dir}")
    collected: dict[str, dict[str, list[float]]] = {}
    for path in files:
        conditions = json.load(open(path))["conditions"]
        for cond in ("shuffle_digit", "shuffle_sector"):
            collected.setdefault(cond, {"char_acc": [], "sector_acc": []})
            for key in ("char_acc", "sector_acc"):
                collected[cond][key].append(float(conditions[cond][key]))
    return {c: {k: np.asarray(v) for k, v in d.items()} for c, d in collected.items()}


def _plot_shuffle_axis(
    axis: plt.Axes,
    baseline: np.ndarray,
    shuffle_digit: np.ndarray,
    shuffle_sector: np.ndarray,
    color: str,
    show_xticks: bool,
    y_min: float,
    y_max: float,
    y_step: float,
) -> None:
    """Plot one readout's GaWF shuffle-ablation bars (Baseline, Shuffle digit, Shuffle sector)."""

    conds = (
        ("Baseline", baseline),
        ("Shuffle\ndigit", shuffle_digit),
        ("Shuffle\nsector", shuffle_sector),
    )
    positions = np.arange(len(conds), dtype=np.float64)
    rng = np.random.default_rng(0)
    for index, (_label, values) in enumerate(conds):
        mean, sd = _mean_sd(values)
        axis.bar(
            positions[index],
            mean,
            width=0.72,
            yerr=sd,
            color=color,
            edgecolor="none",
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"},
        )
        jitter = rng.uniform(-0.16, 0.16, size=values.size)
        axis.scatter(
            np.full(values.size, positions[index]) + jitter,
            values,
            s=10,
            color="#333333",
            alpha=0.52,
            linewidths=0,
            zorder=3,
        )
    axis.set_xticks(positions, [label for label, _ in conds])
    if not show_xticks:
        axis.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    axis.set_ylim(y_min, y_max)
    axis.set_yticks(np.arange(y_min, y_max + 0.001, y_step))
    _style_axis(axis)


def main() -> None:
    args = parse_args()
    test_metrics = load_test_metrics(args.test_csv)
    recovery_offsets, recovery_curves = load_recovery_curves(args.recovery_dir)
    shuffle = _shuffle_from_ablation(args.ablation_dir)
    if "gawf" not in test_metrics:
        raise RuntimeError("Multiseed test metrics must include gawf for the shuffle baseline.")

    with plt.rc_context(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    ):
        # 3-column 2x3 was figsize (13.2, 6.9); four columns each 30% narrower keeps the same
        # per-column ratio: width = 13.2 * (4 * 0.7) / 3 = 12.32.
        fig, axes = plt.subplots(2, 4, figsize=(12.32, 6.9))
        _plot_test_axis(axes[0, 0], test_metrics, "char", show_xticks=False)
        _plot_test_axis(axes[1, 0], test_metrics, "sector", show_xticks=True)
        # Override the shared helper's default ylim/ticks for this merged figure only. Bounds
        # track the tick range with a 1-point margin where real seeds sit just outside it (digit
        # char_acc spans 73.2-86.6 across models; sector spans 87.9-93.2).
        axes[0, 0].set_ylim(73.0, 87.0)
        axes[0, 0].set_yticks((74.0, 78.0, 82.0, 86.0))
        axes[1, 0].set_ylim(87.0, 94.0)
        axes[1, 0].set_yticks((88.0, 90.0, 92.0, 94.0))
        _plot_loss_axis(axes[0, 1], args.loss_png, "char", show_xlabel=False, show_xticks=False)
        _plot_loss_axis(axes[1, 1], args.loss_png, "sector", show_xlabel=True, show_xticks=True)
        _plot_recovery_axis(
            axes[0, 2], recovery_offsets, recovery_curves, "char", show_xlabel=False,
            show_xticks=False,
        )
        _plot_recovery_axis(
            axes[1, 2], recovery_offsets, recovery_curves, "sector", show_xlabel=True,
            show_xticks=True,
        )
        axes[0, 2].set_yticks((0, 20, 40, 60, 80, 100))
        axes[1, 2].set_yticks((0, 20, 40, 60, 80, 100))
        # y_min/y_max here just seed the shared helper's internal set_ylim call; both axes get
        # their final ylim/yticks overridden below to match the tick-range-plus-margin rule.
        _plot_shuffle_axis(
            axes[0, 3],
            test_metrics["gawf"]["char"],
            shuffle["shuffle_digit"]["char_acc"],
            shuffle["shuffle_sector"]["char_acc"],
            DIGIT_COLOR,
            show_xticks=False,
            y_min=70.0,
            y_max=90.0,
            y_step=4.0,
        )
        _plot_shuffle_axis(
            axes[1, 3],
            test_metrics["gawf"]["sector"],
            shuffle["shuffle_digit"]["sector_acc"],
            shuffle["shuffle_sector"]["sector_acc"],
            SECTOR_COLOR,
            show_xticks=True,
            y_min=75.0,
            y_max=95.0,
            y_step=4.0,
        )
        # Match the digit ablation's ticks to the digit test-accuracy panel, and the sector
        # ablation's ticks to a comparable 4-point span. Bounds get a 1-2 point margin beyond the
        # tick range where a real condition sits outside it (digit char_acc across
        # baseline/shuffle_digit/shuffle_sector spans 72.4-86.8; sector sector_acc spans
        # 79.0-93.2).
        axes[0, 3].set_ylim(72.0, 87.0)
        axes[0, 3].set_yticks((74.0, 78.0, 82.0, 86.0))
        axes[1, 3].set_ylim(78.0, 94.0)
        axes[1, 3].set_yticks((80.0, 84.0, 88.0, 92.0))

        # The 30% narrower columns crowd the recovery tick labels; rotate that column's bottom
        # x labels so pre10/switch/post4/post10 no longer overlap. Only this merged figure is
        # affected -- the shared helper and the original 2x3 keep their horizontal labels.
        for tick_label in axes[1, 2].get_xticklabels():
            tick_label.set_rotation(30)
            tick_label.set_ha("right")
            tick_label.set_rotation_mode("anchor")

        # Validation-loss column: the digit loss keeps its 0.3-1.2 range; the sector loss zooms
        # to 0.15-0.45 (its raster spans 0.15-0.5, so the tighter ylim simply clips the 0.45-0.5
        # top).
        axes[0, 1].set_yticks((0.3, 0.6, 0.9, 1.2))
        axes[1, 1].set_ylim(0.15, 0.45)
        axes[1, 1].set_yticks((0.15, 0.25, 0.35, 0.45))

        fig.subplots_adjust(
            left=0.075, right=0.995, bottom=0.12, top=0.81, hspace=0.40, wspace=0.24
        )
        column_centers = [
            np.mean(
                [
                    axes[row, column].get_position().x0
                    + axes[row, column].get_position().width / 2
                    for row in range(2)
                ]
            )
            for column in range(4)
        ]
        title_y = max(axes[0, column].get_position().y1 for column in range(4)) + 0.05
        for x, title in zip(
            column_centers,
            ("Test accuracy", "Validation loss", "Target switch recovery", "GaWF shuffle ablation"),
        ):
            fig.text(x, title_y, title, ha="center", va="bottom", fontsize=15)
        row_centers = [
            np.mean(
                [
                    axes[row, column].get_position().y0
                    + axes[row, column].get_position().height / 2
                    for column in range(4)
                ]
            )
            for row in range(2)
        ]
        for y, label in zip(row_centers, ("Digit", "Sector")):
            fig.text(0.038, y, label, rotation=90, ha="center", va="center", fontsize=15)

        models = [model for model in MODEL_ORDER if model in test_metrics]
        handles = [Line2D([0], [0], color=MODEL_COLORS[model], linewidth=2.2) for model in models]
        fig.legend(
            handles,
            [MODEL_LABELS[model] for model in models],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.992),
            ncol=len(models),
            fontsize=13,
        )
        output_pdf = (
            args.output_pdf if args.output_pdf is not None else args.output_png.with_suffix(".pdf")
        )
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_png, dpi=180, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    print(f"Saved {args.output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()
