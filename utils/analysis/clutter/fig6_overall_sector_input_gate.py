"""Render the three-part Sector input-gate Figure 6 from retained ten-seed summaries.

The PDF contains the encoder Sector spatial maps, reset-excluded sequential input-gate delta
maps, plus matching- and other-source sign-versus-|W| binned curves. It reads compact saved
arrays and does not reuse raster figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (
    NEG_COLOR,
    POS_COLOR,
    binned_mean_curve,
    quantile_bin_edges,
)
from utils.analysis.clutter.supple2_input_gate_sign_magnitude_sector import (
    _load_all_sector_by_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_NAME = Path(__file__).stem
ENCODER_SUMMARY = (
    PROJECT_ROOT
    / "results/data/analysis/fig6_encoder_sector_patterns_gawf_10seed/final"
    / "encoder_sector_patterns_10seed_summary.npz"
)
GATE_ROOT = PROJECT_ROOT / "results/save_data/fig6"
SUPPLE2_ROOT = (
    PROJECT_ROOT / "results/data/analysis"
    / "supple2_input_gate_sign_magnitude_9sector_reset_excluded_10seed"
)
SAVE_FIGURE = PROJECT_ROOT / "results/save/Fig6_overall_sector_input_gate_1x3_10seed.pdf"


def parse_args() -> argparse.Namespace:
    """Parse retained-summary paths and the one publication PDF destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder_summary", type=Path, default=ENCODER_SUMMARY)
    parser.add_argument("--gate_root", type=Path, default=GATE_ROOT)
    parser.add_argument("--supple2_root", type=Path, default=SUPPLE2_ROOT)
    parser.add_argument("--figure", type=Path, default=SAVE_FIGURE)
    return parser.parse_args()


def _load_encoder_maps(path: Path) -> np.ndarray:
    """Return the retained ten-seed mean encoder Sector maps."""

    with np.load(path, allow_pickle=False) as arrays:
        maps = np.asarray(arrays["spatial_maps"], dtype=np.float64)
    if maps.shape != (9, 6, 6) or not np.isfinite(maps).all():
        raise RuntimeError(f"Expected finite encoder maps (9, 6, 6) in {path}, got {maps.shape}.")
    return maps


def _load_gate_delta_maps(root: Path) -> np.ndarray:
    """Average the reset-excluded point maps across exactly ten training seeds."""

    paths = sorted(root.glob("seed*/sector_gate_mean_sequential_equal_n.npz"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten Fig6 gate files in {root}, found {len(paths)}.")
    maps = []
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            value = np.asarray(arrays["point_excluded"], dtype=np.float64)
        if value.shape != (9, 6, 6) or not np.isfinite(value).all():
            raise RuntimeError(f"Expected finite point-excluded maps (9, 6, 6) in {path}.")
        maps.append(value)
    mean_gate = np.mean(np.stack(maps), axis=0)
    return mean_gate - mean_gate.mean(axis=0, keepdims=True)


def _source_curves(root: Path, group: str) -> tuple[np.ndarray, np.ndarray]:
    """Return positive- and negative-weight nine-bin curves for one Supple2 source group."""

    by_seed = _load_all_sector_by_seed(root)
    frames = by_seed[group]
    if len(frames) != 10:
        raise RuntimeError(f"Expected ten Supple2 seed tables for {group}.")
    abs_weight = np.concatenate([frame["absW"].to_numpy(dtype=np.float64) for frame in frames])
    delta_gate = np.concatenate(
        [frame["delta_gate"].to_numpy(dtype=np.float64) for frame in frames]
    )
    sign_positive = np.concatenate(
        [frame["signpos"].to_numpy(dtype=bool) for frame in frames]
    )
    edges = quantile_bin_edges(abs_weight)
    curves = []
    for select in (sign_positive, ~sign_positive):
        center, mean, sem, count = binned_mean_curve(abs_weight[select], delta_gate[select], edges)
        curves.append(np.stack((center, mean, sem, count.astype(np.float64))))
    return curves[0], curves[1]


def _zero_weight_baseline(positive: np.ndarray, negative: np.ndarray) -> float:
    """Estimate the shared ``|W|=0`` level by linear extrapolation from each curve's first bins."""

    intercepts = []
    for curve in (positive, negative):
        center, mean, _sem, count = curve
        valid = np.flatnonzero(count > 0)
        if valid.size < 2:
            raise RuntimeError("Need two non-empty bins to estimate the |W|=0 baseline.")
        first, second = valid[:2]
        slope = (mean[second] - mean[first]) / (center[second] - center[first])
        intercepts.append(float(mean[first] - slope * center[first]))
    return float(np.mean(intercepts))


def _draw_map_grid(
    figure: plt.Figure,
    grid: matplotlib.gridspec.SubplotSpec,
    maps: np.ndarray,
    *,
    cmap: str,
    norm: matplotlib.colors.Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
) -> None:
    """Draw one Fig6-style 3-by-3 spatial map block with its own colorbar."""

    inner = grid.subgridspec(3, 4, width_ratios=(1, 1, 1, 0.10), wspace=0.08, hspace=0.18)
    image = None
    for sector in range(9):
        axis = figure.add_subplot(inner[sector // 3, sector % 3])
        image = axis.pcolormesh(
            maps[sector],
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
            edgecolors="face",
            linewidth=0.01,
            antialiased=False,
            rasterized=False,
            snap=True,
        )
        axis.set_xlim(0, 6)
        axis.set_ylim(6, 0)
        axis.set_aspect("equal")
        axis.set_title(f"Sector {sector}", fontsize=16)
        axis.set_xticks([])
        axis.set_yticks([])
    assert image is not None
    colorbar = figure.colorbar(image, cax=figure.add_subplot(inner[:, 3]))
    if colorbar_label is not None:
        colorbar.set_label(colorbar_label, fontsize=16)
    colorbar.ax.tick_params(labelsize=14)


def _draw_curves(
    axis: plt.Axes,
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    baseline: float,
    y_label: str,
    show_legend: bool,
) -> None:
    """Draw only the retained Supple2 binned mean plus SEM curves, without point clouds."""

    curves = ((positive, POS_COLOR, "W > 0 (+)"), (negative, NEG_COLOR, "W < 0 (-)"))
    for curve, color, label in curves:
        center, mean, sem, count = curve
        valid = count > 0
        axis.errorbar(
            center[valid],
            mean[valid],
            yerr=sem[valid],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5,
            capsize=2.5,
            label=label,
        )
    axis.set_xlim(0.0, 1.08 * max(np.nanmax(positive[0]), np.nanmax(negative[0])))
    y_low, y_high = baseline - 0.15, baseline + 0.15
    axis.set_ylim(y_low, y_high)
    tick_start = np.ceil(y_low * 10.0 - 1e-12) / 10.0
    axis.set_yticks(np.arange(tick_start, y_high + 1e-12, 0.1))
    axis.set_xlabel("|W|", fontsize=16)
    axis.set_ylabel(y_label, fontsize=16)
    axis.tick_params(labelsize=14)
    axis.spines[["top", "right"]].set_visible(False)
    if show_legend:
        axis.legend(frameon=False, fontsize=14, loc="best")


def render(
    encoder_maps: np.ndarray,
    gate_delta_maps: np.ndarray,
    matching_positive: np.ndarray,
    matching_negative: np.ndarray,
    other_positive: np.ndarray,
    other_negative: np.ndarray,
    matching_baseline: float,
    other_baseline: float,
    destinations: tuple[Path, ...],
) -> None:
    """Render the title-free Figure 6 three-panel layout at every requested PDF path."""

    with plt.rc_context({"font.size": 13, "axes.titlesize": 16}):
        figure = plt.figure(figsize=(20.5, 6.8), layout="constrained")
        outer = figure.add_gridspec(1, 3, width_ratios=(1, 1, 0.94), wspace=0.20)
        _draw_map_grid(
            figure,
            outer[0, 0],
            encoder_maps,
            cmap="Reds",
            vmin=float(encoder_maps.min()),
            vmax=float(encoder_maps.max()),
            colorbar_label="Mean encoder activation",
        )
        limit = float(np.abs(gate_delta_maps).max())
        _draw_map_grid(
            figure,
            outer[0, 1],
            gate_delta_maps,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        curve_grid = outer[0, 2].subgridspec(2, 1, hspace=0.34)
        _draw_curves(
            figure.add_subplot(curve_grid[0, 0]),
            matching_positive,
            matching_negative,
            baseline=matching_baseline,
            y_label="Matching Δg",
            show_legend=True,
        )
        _draw_curves(
            figure.add_subplot(curve_grid[1, 0]),
            other_positive,
            other_negative,
            baseline=other_baseline,
            y_label="Other Δg",
            show_legend=False,
        )
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, bbox_inches="tight", pad_inches=0.06)
        plt.close(figure)


def main() -> None:
    """Load retained summaries, save a compact numeric record, and render the final PDF."""

    args = parse_args()
    encoder_maps = _load_encoder_maps(args.encoder_summary)
    gate_delta_maps = _load_gate_delta_maps(args.gate_root)
    matching_positive, matching_negative = _source_curves(args.supple2_root, "sector0_sources")
    other_positive, other_negative = _source_curves(args.supple2_root, "other_sources")
    matching_baseline = _zero_weight_baseline(matching_positive, matching_negative)
    other_baseline = _zero_weight_baseline(other_positive, other_negative)
    data_dir = output_dir("B_gate_by_context", SCRIPT_NAME, "data")
    figure_dir = output_dir("B_gate_by_context", SCRIPT_NAME, "figs")
    np.savez_compressed(
        data_dir / "fig6_overall_sector_input_gate_1x3_10seed.npz",
        encoder_spatial_maps=encoder_maps.astype(np.float32),
        gate_delta_maps=gate_delta_maps.astype(np.float32),
        matching_positive_curve=matching_positive.astype(np.float32),
        matching_negative_curve=matching_negative.astype(np.float32),
        other_positive_curve=other_positive.astype(np.float32),
        other_negative_curve=other_negative.astype(np.float32),
        matching_weight_zero_baseline=np.float32(matching_baseline),
        other_weight_zero_baseline=np.float32(other_baseline),
    )
    (data_dir / "fig6_overall_sector_input_gate_1x3_10seed.json").write_text(
        json.dumps(
            {
                "training_seeds": 10,
                "panels": [
                    "mean encoder activation by Sector",
                    "sequential reset-excluded delta input-gate mean by Sector",
                    "matching- and other-source delta input-gate versus |W|, binned mean plus SEM",
                ],
                "supple2_removed": ["scatter", "shared-|W| shading", "n/gap annotation", "title"],
                "curve_y_axis": {
                    "definition": "raw delta gate; limits centered on |W|=0 baseline",
                    "half_range": 0.15,
                    "tick_step": 0.1,
                    "matching_weight_zero_baseline": matching_baseline,
                    "matching_limits": [matching_baseline - 0.15, matching_baseline + 0.15],
                    "other_weight_zero_baseline": other_baseline,
                    "other_limits": [other_baseline - 0.15, other_baseline + 0.15],
                },
                "inputs": {
                    "encoder_summary": str(args.encoder_summary),
                    "gate_root": str(args.gate_root),
                    "supple2_root": str(args.supple2_root),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    canonical = figure_dir / args.figure.name
    render(
        encoder_maps,
        gate_delta_maps,
        matching_positive,
        matching_negative,
        other_positive,
        other_negative,
        matching_baseline,
        other_baseline,
        (canonical, args.figure),
    )
    print(f"Saved {args.figure}")


if __name__ == "__main__":
    main()
