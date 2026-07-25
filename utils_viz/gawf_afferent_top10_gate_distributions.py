"""Plot afferent top-10% versus remaining DESTINATION-gate distribution grids.

Reads ``afferent_top10_gate_distributions.npz`` produced by
``utils_anal.gawf_afferent_top10_gate_distributions`` and renders one grid figure per cell:
``recurrent_sector`` (3x3), ``input_sector`` (3x3), ``input_digit`` (2x5), and
``recurrent_digit`` (2x5). Each panel shows the two per-destination distributions (top-10%
vs remaining eligible hidden units) with mean lines and the per-context Cohen's d annotation.
Figures are saved directly under ``results/anal_figs/E_relevance_alignment/afferent/``.
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
GROUP_LABELS = ("Top 10% hidden destinations", "Remaining 90% hidden destinations")

# (nrows, ncols, gate_name, factor, levels, factor_label, source_dim_label)
GRID_SPECS = {
    "recurrent_sector": (3, 3, "recurrent", "sector", 9, "Sector", "256 hidden units"),
    "input_sector":     (3, 3, "input",     "sector", 9, "Sector", "1152 encoder features"),
    "input_digit":      (2, 5, "input",     "digit", 10, "Digit",  "1152 encoder features"),
    "recurrent_digit":  (2, 5, "recurrent", "digit", 10, "Digit",  "256 hidden units"),
}


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_afferent_top10_gate_distributions",
                "data",
            )
            / "afferent_top10_gate_distributions.npz"
        ),
    )
    parser.add_argument(
        "--fig_dir",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_afferent_top10_gate_distributions",
                "figs",
            )
            / "afferent"
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def plot_afferent_top10_panel(
    axis: plt.Axes,
    factor_label: str,
    context: int,
    gate_name: str,
    source_dim_label: str,
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
    """Draw one context's top-10% vs remaining afferent gate density onto ``axis``."""

    if hist_counts.shape[0] != 2:
        raise ValueError("hist_counts must have exactly two rows (top10, remaining)")
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
    axis.set_xlabel(
        f"Raw {gate_name} gate (afferent: mean over {source_dim_label})",
        fontsize=8,
    )
    axis.set_ylabel("Density", fontsize=8)
    axis.set_title(
        f"{factor_label} {context} — {gate_name} AFFERENT gate distribution\n"
        f"Top 10% versus remaining 90% eligible hidden destinations; "
        f"Cohen's d = {context_d:.3f}",
        fontsize=8.5,
    )
    axis.tick_params(labelsize=7)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=6.5, loc="upper right")


def render_grid(
    cell: str,
    bin_edges: np.ndarray,
    hist_counts: np.ndarray,
    group_mean: np.ndarray,
    group_count: np.ndarray,
    context_d: np.ndarray,
    relevant_mask: np.ndarray,
    eligible_mask: np.ndarray,
    fig_dir: Path,
    dpi: int,
) -> Path:
    """Render every context of one cell into a single afferent grid figure."""

    if cell not in GRID_SPECS:
        raise KeyError(f"Unknown cell {cell!r}")
    nrows, ncols, gate_name, _factor, levels, factor_label, source_dim_label = GRID_SPECS[cell]
    widths = np.diff(bin_edges)
    all_density = hist_counts / (group_count[..., None] * widths[None, None, :])
    density_limit = 1.08 * float(all_density.max())
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.6 * nrows))
    axes_flat = np.asarray(axes).reshape(-1)
    eligible_units = int(eligible_mask.sum())
    for context in range(levels):
        relevant_units = int(relevant_mask[context].sum())
        plot_afferent_top10_panel(
            axes_flat[context],
            factor_label,
            context,
            gate_name,
            source_dim_label,
            bin_edges,
            hist_counts[context],
            group_mean[context],
            group_count[context],
            float(context_d[context]),
            relevant_units,
            eligible_units - relevant_units,
            density_limit=density_limit,
        )
    for extra_axis in axes_flat[levels:]:
        extra_axis.set_visible(False)
    fig.tight_layout()
    destination = fig_dir / f"{cell}_top10_vs_remaining_afferent_distribution_grid.png"
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {destination}")
    return destination


def main() -> None:
    """Load saved histograms and render the four afferent grid figures."""

    args = parse_args()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(args.data).expanduser().resolve(), allow_pickle=False) as data:
        bin_edges = np.asarray(data["bin_edges"], dtype=np.float64)
        for cell in GRID_SPECS:
            render_grid(
                cell,
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
