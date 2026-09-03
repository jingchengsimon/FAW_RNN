"""Render an overall Figure 7 from retained ten-seed delta-g summaries.

The PDF places the Figure 7 Digit and Sector sign-split bar panels in the first two columns.
Its third column contains the Supplementary 3 Digit and Sector T-to-T panels.
All panels are regenerated from compact per-seed numerical arrays, not assembled from rasters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.fig7_recurrent_gate_disinhibition import (
    POSTER_RC,
    _cross_seed_stats,
    exact_sign_flip_p,
    overlap_band_sign_stats,
)
from utils.analysis.clutter.fig7_recurrent_gate_multiseed import (
    _compact_paths,
    _overlap_stats,
    _pooled_records,
)
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (
    GROUP_NAMES,
    NEG_COLOR,
    POS_COLOR,
    binned_mean_curve,
    quantile_bin_edges,
)
from utils.analysis.clutter.multiseed_plotting import add_seed_points


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_NAME = Path(__file__).stem
DATA_ROOT = PROJECT_ROOT / "results/data/analysis/fig7_recurrent_gate_10seed"
SAVE_FIGURE = PROJECT_ROOT / "results/save/Fig7_overall_recurrent_gate_disinhibition_1x4_10seed.pdf"
VARIABLES = ("digit", "sector")
CURVE_GROUP = "TT"


def parse_args() -> argparse.Namespace:
    """Parse the compact Figure 7 data root and overall PDF destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--figure", type=Path, default=SAVE_FIGURE)
    return parser.parse_args()


def _load_inputs(
    data_root: Path,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, float]], dict[str, dict[str, dict]],
           dict[str, dict[str, pd.DataFrame]]]:
    """Load exactly ten compact files into poster statistics and pooled Supple3 records."""

    paths = _compact_paths(data_root)
    if len(paths) != 10:
        raise RuntimeError(
            f"Expected ten compact Figure 7 files in {data_root}, found {len(paths)}."
        )
    per_seed: dict[str, list[dict[str, dict]]] = {kind: [] for kind in VARIABLES}
    pooled: dict[str, dict[str, list[pd.DataFrame]]] = {
        kind: {group: [] for group in GROUP_NAMES} for kind in VARIABLES
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for kind in VARIABLES:
                seed_records = _pooled_records(arrays, kind)
                seed_overlap = _overlap_stats(seed_records)
                per_seed[kind].append(
                    overlap_band_sign_stats(seed_records, seed_overlap, y_col="delta_of")
                )
                for group in GROUP_NAMES:
                    pooled[kind][group].append(seed_records[group])
    stats = {kind: _cross_seed_stats(per_seed[kind]) for kind in VARIABLES}
    gaps = {
        kind: {
            group: stats[kind][group]["+"]["mean"] - stats[kind][group]["-"]["mean"]
            for group in GROUP_NAMES
        }
        for kind in VARIABLES
    }
    tests: dict[str, dict[str, dict]] = {}
    for kind in VARIABLES:
        tests[kind] = {}
        for group in GROUP_NAMES:
            positive = np.asarray([row[group]["+"]["mean"] for row in per_seed[kind]])
            negative = np.asarray([row[group]["-"]["mean"] for row in per_seed[kind]])
            tests[kind][group] = {
                "+": exact_sign_flip_p(positive),
                "-": exact_sign_flip_p(negative),
                "gap": exact_sign_flip_p(positive - negative),
            }
    merged = {
        kind: {group: pd.concat(frames, ignore_index=True) for group, frames in groups.items()}
        for kind, groups in pooled.items()
    }
    return stats, gaps, tests, merged


def _star(p_value: float) -> str:
    """Return one uncorrected significance marker for the established poster convention."""

    return "*" if np.isfinite(p_value) and p_value < 0.05 else ""


def _poster_limits(stats: dict[str, dict[str, dict]]) -> tuple[float, float]:
    """Return the existing poster's shared delta-g y limits."""

    edges = [
        stats[kind][group][sign]["mean"] + direction * stats[kind][group][sign]["sem"]
        for kind in VARIABLES
        for group in GROUP_NAMES
        for sign in ("+", "-")
        for direction in (-1.0, 1.0)
    ]
    minimum, maximum = min(edges), max(edges)
    return minimum - 0.28 * (maximum - minimum), 0.1


def _draw_poster_panel(
    axis: plt.Axes,
    stats: dict[str, dict],
    tests: dict[str, dict],
    title: str,
    limits: tuple[float, float],
    *,
    show_ylabel: bool,
) -> None:
    """Draw one poster-style delta-g bar panel from training-seed statistics."""

    positions = np.arange(len(GROUP_NAMES))
    width = 0.32
    span = limits[1] - limits[0]
    for index, group in enumerate(GROUP_NAMES):
        positive, negative = stats[group]["+"], stats[group]["-"]
        for x_pos, sign, record, color in (
            (positions[index] - width / 2, "+", positive, POS_COLOR),
            (positions[index] + width / 2, "-", negative, NEG_COLOR),
        ):
            axis.bar(x_pos, record["mean"], width, yerr=record["sem"], capsize=3,
                     color=color, edgecolor="none", zorder=3)
            add_seed_points(
                axis,
                np.asarray([x_pos]),
                np.asarray(record["values"], dtype=np.float64)[:, None],
                bar_width=width,
                rng=np.random.default_rng(index),
            )
            edge = record["mean"] + (record["sem"] if record["mean"] >= 0.0 else -record["sem"])
            marker = _star(tests[group][sign])
            if marker:
                axis.text(
                    x_pos,
                    edge + (0.04 * span if edge >= 0.0 else -0.04 * span),
                    marker,
                    ha="center",
                    va="bottom" if edge >= 0.0 else "top",
                    fontsize=10,
                    fontweight="bold",
                )
    axis.axhline(0.0, color="black", linewidth=1.0, zorder=2)
    axis.set_xticks(positions, [f"{group[0]}->{group[1]}" for group in GROUP_NAMES])
    axis.set_ylim(*limits)
    axis.set_yticks((0.1, 0.0, -0.2, -0.4))
    axis.set_title(title)
    axis.set_xlabel("Group")
    if show_ylabel:
        axis.set_ylabel("Delta gate open fraction")
    axis.spines[["top", "right"]].set_visible(False)


def _weight_zero_baseline(
    curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> float:
    """Estimate the raw delta-g value at zero weight from both sign curves."""

    intercepts: list[float] = []
    for mean, _sem, center in curves:
        if mean.size < 2:
            raise RuntimeError("Each sign curve needs at least two nonempty |W| bins.")
        slope = (mean[1] - mean[0]) / (center[1] - center[0])
        intercepts.append(float(mean[0] - slope * center[0]))
    return float(np.mean(intercepts))


def _draw_supple3_panel(
    axis: plt.Axes,
    values: pd.DataFrame,
    title: str,
    *,
    show_ylabel: bool,
) -> None:
    """Render one compact Supplementary 3 binned delta-g panel without raw connections."""

    positive = values.loc[values["signpos"] == 1]
    negative = values.loc[values["signpos"] == 0]
    edges = quantile_bin_edges(values["absW"].to_numpy(dtype=np.float64))
    plotted: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for frame, color in ((positive, POS_COLOR), (negative, NEG_COLOR)):
        center, mean, sem, count = binned_mean_curve(
            frame["absW"].to_numpy(dtype=np.float64),
            frame["delta_of"].to_numpy(dtype=np.float64),
            edges,
        )
        valid = count > 0
        axis.errorbar(
            center[valid], mean[valid], yerr=sem[valid], color=color, linewidth=2.2,
            marker="o", markersize=5, capsize=2.5, zorder=3,
        )
        plotted.append((mean[valid], sem[valid], center[valid]))
    baseline = _weight_zero_baseline(plotted)
    y_low, y_high = baseline - 0.30, baseline + 0.30
    tick_start = np.ceil(y_low * 10.0 - 1e-12) / 10.0
    axis.set_xlim(0.0, 1.08 * max(float(center.max()) for _mean, _sem, center in plotted))
    axis.set_ylim(y_low, y_high)
    axis.set_yticks(np.arange(tick_start, y_high + 1e-12, 0.1))
    axis.set_xlabel("|W|", fontsize=13)
    axis.set_ylabel("Δg" if show_ylabel else "", fontsize=13)
    axis.set_title(title, fontsize=13)
    axis.tick_params(labelsize=10)
    axis.spines[["top", "right"]].set_visible(False)


def render(
    stats: dict[str, dict[str, dict]],
    tests: dict[str, dict[str, dict]],
    pooled: dict[str, dict[str, pd.DataFrame]],
    destinations: tuple[Path, ...],
) -> None:
    """Render the two poster panels plus the right-side Digit/Sector T-to-T column."""

    with plt.rc_context(POSTER_RC):
        figure = plt.figure(figsize=(15.4, 5.8))
        poster_grid = figure.add_gridspec(
            1, 2, left=0.06, right=0.62, bottom=0.15, top=0.83, wspace=0.30
        )
        curve_grid = figure.add_gridspec(
            2, 1, left=0.70, right=0.99, bottom=0.15, top=0.84, hspace=0.62
        )
        limits = _poster_limits(stats)
        _draw_poster_panel(
            figure.add_subplot(poster_grid[0, 0]),
            stats["digit"],
            tests["digit"],
            "Digit",
            limits,
            show_ylabel=True,
        )
        _draw_poster_panel(
            figure.add_subplot(poster_grid[0, 1]),
            stats["sector"],
            tests["sector"],
            "Sector",
            limits,
            show_ylabel=False,
        )
        for row, kind in enumerate(VARIABLES):
            _draw_supple3_panel(
                figure.add_subplot(curve_grid[row, 0]),
                pooled[kind][CURVE_GROUP],
                f"{kind.capitalize()} T→T",
                show_ylabel=True,
            )
        handles = [
            Line2D([], [], marker="o", linestyle="-", color=POS_COLOR, linewidth=2.2,
                   markersize=6, label="W > 0 (+)"),
            Line2D([], [], marker="o", linestyle="-", color=NEG_COLOR, linewidth=2.2,
                   markersize=6, label="W < 0 (-)"),
        ]
        figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                      bbox_to_anchor=(0.5, 1.01))
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, bbox_inches="tight", pad_inches=0.06)
        plt.close(figure)


def main() -> None:
    """Load compact ten-seed inputs, write layout metadata, and render the Figure 7 PDF."""

    args = parse_args()
    stats, _gaps, tests, pooled = _load_inputs(args.data_root)
    data_dir = output_dir("E_relevance_alignment", SCRIPT_NAME, "data")
    figure_dir = output_dir("E_relevance_alignment", SCRIPT_NAME, "figs")
    (data_dir / "fig7_overall_recurrent_gate_disinhibition_1x4_10seed.json").write_text(
        json.dumps(
            {
                "training_seeds": 10,
                "layout": ["Digit poster", "Sector poster", "Digit TT / Sector TT"],
                "right_grid": {"rows": ["Digit TT", "Sector TT"], "columns": ["T-to-T"]},
                "source_data_root": str(args.data_root),
                "right_grid_style": (
                    "Figure-6-style binned mean +/- SEM; raw delta-g, with no scatter, "
                    "overlap shading, or gap annotation"
                ),
                "right_grid_y_axis": {
                    "definition": "raw delta-g",
                    "limits": "estimated |W|=0 baseline +/- 0.30 per panel",
                    "tick_step": 0.1,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    canonical = figure_dir / args.figure.name
    render(stats, tests, pooled, (canonical, args.figure))
    print(f"Saved {args.figure}")


if __name__ == "__main__":
    main()
