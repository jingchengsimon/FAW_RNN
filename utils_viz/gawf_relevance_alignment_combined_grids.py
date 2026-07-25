"""Merge the per-context top-10% vs remaining SOURCE-gate distribution figures into grids.

Reads the same NPZ histograms as ``gawf_recurrent_sector_relevance_distributions.py`` and
``gawf_remaining_relevance_distributions.py`` and renders one grid figure per (gate, factor)
combination — recurrent/sector (3x3), input/sector (3x3), input/digit (2x5), and
recurrent/digit (2x5) — instead of one PNG per context. Each panel keeps the same stairs
density, mean lines, title, and legend as the original standalone figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir

COLORS = ("#3b82f6", "#f97316")
GROUP_LABELS = ("Top 10% units", "Remaining 90% units")

GRID_SPECS = {
    "recurrent_sector": (3, 3),
    "input_sector": (3, 3),
    "input_digit": (2, 5),
    "recurrent_digit": (2, 5),
}


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sector_data",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_recurrent_sector_relevance_distributions",
                "data",
            )
            / "recurrent_sector_top10_gate_distributions.npz"
        ),
    )
    parser.add_argument(
        "--remaining_data",
        default=str(
            output_dir("E_relevance_alignment", "gawf_remaining_relevance_distributions", "data")
            / "remaining_top10_gate_distributions.npz"
        ),
    )
    parser.add_argument(
        "--fig_dir",
        default=str(output_dir("E_relevance_alignment", "gawf_relevance_alignment_combined_grids", "figs")),
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _plot_panel(
    axis: plt.Axes,
    gate_name: str,
    factor: str,
    context: int,
    bin_edges: np.ndarray,
    hist_counts: np.ndarray,
    group_mean: np.ndarray,
    group_count: np.ndarray,
    context_d: float,
    relevant_units: int,
    remaining_units: int,
    *,
    density_limit: float,
) -> None:
    """Draw one context's normalized SOURCE-gate density comparison onto ``axis``."""

    widths = np.diff(bin_edges)
    density = hist_counts / (group_count[:, None] * widths[None, :])
    for group_index, (label, color, unit_count) in enumerate(
        zip(GROUP_LABELS, COLORS, (relevant_units, remaining_units))
    ):
        axis.stairs(
            density[group_index],
            bin_edges,
            color=color,
            linewidth=1.4,
            fill=True,
            alpha=0.22,
            label=f"{label} ({unit_count} units; mean={group_mean[group_index]:.3f})",
        )
        axis.axvline(
            group_mean[group_index],
            color=color,
            linewidth=1.0,
            linestyle="--",
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, density_limit)
    axis.set_xlabel(f"Raw {gate_name} gate (mean over 256 destinations)", fontsize=8)
    axis.set_ylabel("Density", fontsize=8)
    context_label = factor.capitalize()
    axis.set_title(
        f"{context_label} {context} — {gate_name} SOURCE-gate distribution\n"
        f"Top 10% versus remaining 90% eligible units; Cohen's d = {context_d:.3f}",
        fontsize=8.5,
    )
    axis.tick_params(labelsize=7)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=6.5, loc="upper right")


def _save_grid(
    cell: str,
    gate_name: str,
    factor: str,
    levels: int,
    bin_edges: np.ndarray,
    hist_counts: np.ndarray,
    group_mean: np.ndarray,
    group_count: np.ndarray,
    context_d: np.ndarray,
    relevant_mask: np.ndarray,
    eligible: np.ndarray,
    fig_dir: Path,
    dpi: int,
) -> Path:
    """Render every context of one (gate, factor) family into a single grid figure."""

    nrows, ncols = GRID_SPECS[cell]
    widths = np.diff(bin_edges)
    all_density = hist_counts / (group_count[..., None] * widths[None, None, :])
    density_limit = 1.08 * float(all_density.max())
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.6 * nrows))
    axes_flat = np.asarray(axes).reshape(-1)
    for context in range(levels):
        relevant_units = int(relevant_mask[context].sum())
        _plot_panel(
            axes_flat[context],
            gate_name,
            factor,
            context,
            bin_edges,
            hist_counts[context],
            group_mean[context],
            group_count[context],
            float(context_d[context]),
            relevant_units,
            int(eligible.sum() - relevant_units),
            density_limit=density_limit,
        )
    for extra_axis in axes_flat[levels:]:
        extra_axis.set_visible(False)
    fig.tight_layout()
    destination = fig_dir / f"{cell}_top10_vs_remaining_distribution_grid.png"
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {destination}")
    return destination


def main() -> None:
    """Load saved histograms and render the four combined grid figures."""

    args = parse_args()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    with np.load(Path(args.sector_data).expanduser().resolve(), allow_pickle=False) as data:
        bin_edges = np.asarray(data["bin_edges"], dtype=np.float64)
        _save_grid(
            "recurrent_sector",
            "recurrent",
            "sector",
            9,
            bin_edges,
            np.asarray(data["hist_counts"], dtype=np.int64),
            np.asarray(data["group_mean"], dtype=np.float64),
            np.asarray(data["group_count"], dtype=np.int64),
            np.asarray(data["sector_cohens_d"], dtype=np.float64),
            np.asarray(data["relevant_mask"], dtype=bool),
            np.asarray(data["eligible_mask"], dtype=bool),
            fig_dir,
            args.dpi,
        )

    with np.load(Path(args.remaining_data).expanduser().resolve(), allow_pickle=False) as data:
        bin_edges = np.asarray(data["bin_edges"], dtype=np.float64)
        for cell, (gate_name, factor, levels) in (
            ("input_sector", ("input", "sector", 9)),
            ("input_digit", ("input", "digit", 10)),
            ("recurrent_digit", ("recurrent", "digit", 10)),
        ):
            _save_grid(
                cell,
                gate_name,
                factor,
                levels,
                bin_edges,
                np.asarray(data[f"{cell}_hist_counts"], dtype=np.int64),
                np.asarray(data[f"{cell}_group_mean"], dtype=np.float64),
                np.asarray(data[f"{cell}_group_count"], dtype=np.int64),
                np.asarray(data[f"{cell}_context_cohens_d"], dtype=np.float64),
                np.asarray(data[f"{cell}_relevant_mask"], dtype=bool),
                np.asarray(data[f"{cell}_eligible_mask"], dtype=bool),
                fig_dir,
                args.dpi,
            )


if __name__ == "__main__":
    main()
