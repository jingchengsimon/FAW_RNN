"""Render the combined ICLR recurrent-gate and gate-dependent-current Figure 7.

The top row contains the retained ten-seed Digit and Sector delta-g bar panels plus the
stacked Digit/Sector T-to-T sign-magnitude curves. The bottom row contains the connection-
normalized Digit and Sector gate-dependent-current bars. All panels are regenerated from
the compact Figure 7 arrays and Figure 8 long tables; outputs are one development PNG and
one 5.5-inch-wide manuscript PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from utils.analysis.anal_paths import output_dir  # noqa: E402
from utils.analysis.clutter.fig6_net_recurrent_current import (  # noqa: E402
    CURRENT_COLORS,
    _load_fig8_long,
    _plot_fig8_bars,
)
from utils.analysis.clutter.fig7_overall_recurrent_gate_disinhibition import (  # noqa: E402
    CURVE_GROUP,
    DATA_ROOT,
    POSTER_RC,
    VARIABLES,
    _draw_poster_panel,
    _draw_supple3_panel,
    _load_inputs,
    _poster_limits,
)
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (  # noqa: E402
    NEG_COLOR,
    POS_COLOR,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_NAME = Path(__file__).stem
FIG8_ROOT = PROJECT_ROOT / "results/save_data/fig8/recurrent_current/connection"
OUTPUT_STEM = "Fig7_recurrent_gate_disinhibition_and_current_2x3_10seed"


def parse_args() -> argparse.Namespace:
    """Parse the retained Figure 7/Figure 8 inputs and combined-figure destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig7_data_root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--digit_long",
        type=Path,
        default=FIG8_ROOT / "digit/net_recurrent_current_connection_10seed_long.csv",
    )
    parser.add_argument(
        "--sector_long",
        type=Path,
        default=FIG8_ROOT / "sector/net_recurrent_current_connection_10seed_long.csv",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=(
            PROJECT_ROOT / "results/figs/E_relevance_alignment" / f"{OUTPUT_STEM}.png"
        ),
    )
    parser.add_argument(
        "--output_pdf",
        type=Path,
        default=PROJECT_ROOT / "results/save/iclr_figs" / f"{OUTPUT_STEM}.pdf",
    )
    return parser.parse_args()


def _compact_axis(axis: plt.Axes, *, hide_y_tick_labels: bool = False) -> None:
    """Restyle an imported standalone panel for the 5.5-inch two-row layout."""

    axis.title.set_fontsize(8.2)
    axis.xaxis.label.set_fontsize(7.5)
    axis.yaxis.label.set_fontsize(7.5)
    axis.tick_params(labelsize=6.7, length=2.5, width=0.7)
    if hide_y_tick_labels:
        axis.tick_params(axis="y", labelleft=False)
    for label in axis.get_xticklabels():
        label.set_rotation(0)
    for text in axis.texts:
        if text.get_text() == "*":
            text.set_fontsize(8.0)


def render(
    fig7_stats: dict,
    fig7_tests: dict,
    fig7_pooled: dict,
    fig8_reports: dict,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Render three centered top blocks and two centered wider bottom panels."""

    style = dict(POSTER_RC)
    style.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    with plt.rc_context(style):
        figure = plt.figure(figsize=(5.5, 4.35))
        grid = figure.add_gridspec(
            2,
            6,
            left=0.105,
            right=0.99,
            bottom=0.085,
            top=0.91,
            height_ratios=(1.0, 1.03),
            hspace=0.58,
            wspace=0.95,
        )
        top_axes = (
            figure.add_subplot(grid[0, :2]),
            figure.add_subplot(grid[0, 2:4]),
        )
        curve_grid = grid[0, 4:].subgridspec(2, 1, hspace=0.62)
        curve_axes = (
            figure.add_subplot(curve_grid[0, 0]),
            figure.add_subplot(curve_grid[1, 0]),
        )
        bottom_axes = (
            figure.add_subplot(grid[1, :3]),
            figure.add_subplot(grid[1, 3:]),
        )

        limits = _poster_limits(fig7_stats)
        for index, (axis, family) in enumerate(zip(top_axes, VARIABLES)):
            _draw_poster_panel(
                axis,
                fig7_stats[family],
                fig7_tests[family],
                family.capitalize(),
                limits,
                show_ylabel=index == 0,
            )
            axis.set_xticks(axis.get_xticks(), ("T→T", "T→R", "R→T", "R→R"))
            _compact_axis(axis, hide_y_tick_labels=index > 0)

        for axis, family in zip(curve_axes, VARIABLES):
            _draw_supple3_panel(
                axis,
                fig7_pooled[family][CURVE_GROUP],
                f"{family.capitalize()} T→T",
                show_ylabel=True,
            )
            for line in axis.lines:
                if line.get_marker() == "o":
                    line.set_markersize(2.8)
                if line.get_linestyle() not in ("", "None", "none"):
                    line.set_linewidth(1.2)
            _compact_axis(axis)
        curve_axes[0].set_xlabel("")

        bar_limits = (-0.02, 0.08)
        bar_ticks = (-0.02, 0.0, 0.04, 0.08)
        for index, (axis, family) in enumerate(zip(bottom_axes, VARIABLES)):
            _plot_fig8_bars(
                axis,
                fig8_reports[family],
                family,
                "connection",
                bar_limits,
                bar_ticks,
                "%.2f",
                show_legend=False,
            )
            axis.set_title(family.capitalize())
            axis.set_xlabel("Group")
            if index > 0:
                axis.set_ylabel("")
            _compact_axis(axis, hide_y_tick_labels=index > 0)

        figure.legend(
            handles=(
                Patch(facecolor=POS_COLOR, edgecolor="none", label="W > 0 (+)"),
                Patch(facecolor=NEG_COLOR, edgecolor="none", label="W < 0 (-)"),
                Patch(
                    facecolor=CURRENT_COLORS["total"],
                    edgecolor="none",
                    label="Balanced",
                ),
            ),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            handlelength=1.0,
            columnspacing=1.2,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_png, dpi=300)
        figure.savefig(output_pdf)
        plt.close(figure)


def main() -> None:
    """Load retained structured results and render the combined ICLR figure."""

    args = parse_args()
    stats, _gaps, tests, pooled = _load_inputs(args.fig7_data_root)
    reports = {
        "digit": _load_fig8_long(args.digit_long, "digit"),
        "sector": _load_fig8_long(args.sector_long, "sector"),
    }
    data_dir = output_dir("E_relevance_alignment", SCRIPT_NAME, "data")
    output_dir("E_relevance_alignment", SCRIPT_NAME, "figs")
    (data_dir / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(
            {
                "training_seeds": 10,
                "layout": {
                    "top": ["Digit delta-g", "Sector delta-g", "Digit/Sector TT curves"],
                    "bottom": ["Digit gate-dependent current", "Sector gate-dependent current"],
                },
                "current_normalization": "per nonzero recurrent connection",
                "fig7_data_root": str(args.fig7_data_root),
                "digit_current_long": str(args.digit_long),
                "sector_current_long": str(args.sector_long),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    render(stats, tests, pooled, reports, args.output_png, args.output_pdf)
    print(f"Saved {args.output_png}")
    print(f"Saved {args.output_pdf}")


if __name__ == "__main__":
    main()
