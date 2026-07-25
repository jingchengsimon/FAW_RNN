"""Plot four-group recurrent AFFERENT gate distributions per context on a single grid.

Reads ``recurrent_afferent_group_gate_distributions.npz`` produced by
``utils_anal.gawf_recurrent_afferent_group_gate_distributions`` and renders one grid figure
per factor: ``recurrent_sector`` (3x3) and ``recurrent_digit`` (2x5). Each panel shows the
four per-context afferent distributions with mean lines and Cohen's d annotations comparing
each within-group distribution to its paired between-group distribution that shares the same
destination group.
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


GROUP_NAMES = ("within_top", "top_to_rem", "rem_to_top", "within_rem")
GROUP_LABELS = {
    "within_top": "Top10 → Top10 (within-top afferent)",
    "top_to_rem": "Top10 → Rem (top src into rem dest)",
    "rem_to_top": "Rem → Top10 (rem src into top dest)",
    "within_rem": "Rem → Rem (within-remaining)",
}
GROUP_COLORS = {
    "within_top": "#1d4ed8",
    "top_to_rem": "#0ea5e9",
    "rem_to_top": "#f97316",
    "within_rem": "#dc2626",
}
GROUP_STYLES = {
    "within_top": "-",
    "top_to_rem": "--",
    "rem_to_top": "--",
    "within_rem": "-",
}

GRID_SPECS = {
    "recurrent_sector": (3, 3, 9, "sector", "Sector"),
    "recurrent_digit": (2, 5, 10, "digit", "Digit"),
}


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_recurrent_afferent_group_gate_distributions",
                "data",
            )
            / "recurrent_afferent_group_gate_distributions.npz"
        ),
    )
    parser.add_argument(
        "--fig_dir",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_recurrent_afferent_group_gate_distributions",
                "figs",
            )
            / "afferent"
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def plot_afferent_group_panel(
    axis: plt.Axes,
    factor_label: str,
    context: int,
    bin_edges: np.ndarray,
    hist_counts: np.ndarray,
    group_mean: np.ndarray,
    group_count: np.ndarray,
    within_between_d: np.ndarray,
    *,
    density_limit: float,
    small: bool = True,
) -> None:
    """Draw one context's four-group afferent gate density comparison onto ``axis``.

    ``hist_counts`` has shape ``(4, bins)``; ``group_mean`` and ``group_count`` share shape
    ``(4,)``; ``within_between_d`` has shape ``(2,)`` and holds Cohen's d for
    (within_top vs rem_to_top, within_rem vs top_to_rem). Both pairs are grouped by
    destination unit, so the comparison isolates the effect of the source group on afferent
    gate strength.
    """

    if hist_counts.shape[0] != len(GROUP_NAMES):
        raise ValueError(f"hist_counts must have {len(GROUP_NAMES)} rows")
    widths = np.diff(bin_edges)
    density = hist_counts / (group_count[:, None] * widths[None, :])
    label_font = 8 if small else 10
    tick_font = 7 if small else 9
    title_font = 8.5 if small else 11
    legend_font = 6.3 if small else 8
    for group_index, group_name in enumerate(GROUP_NAMES):
        axis.stairs(
            density[group_index],
            bin_edges,
            color=GROUP_COLORS[group_name],
            linewidth=1.4,
            linestyle=GROUP_STYLES[group_name],
            fill=False,
            alpha=0.9,
            label=(
                f"{GROUP_LABELS[group_name]} "
                f"(N={int(group_count[group_index])}; mean={group_mean[group_index]:.3f})"
            ),
        )
        axis.axvline(
            group_mean[group_index],
            color=GROUP_COLORS[group_name],
            linewidth=0.9,
            linestyle=":",
            alpha=0.8,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, density_limit)
    axis.set_xlabel(
        "Recurrent gate averaged over source subset (afferent)",
        fontsize=label_font,
    )
    axis.set_ylabel("Density", fontsize=label_font)
    d_top = float(within_between_d[0])
    d_rem = float(within_between_d[1])
    axis.set_title(
        f"{factor_label} {context} — four-group AFFERENT gate distribution\n"
        f"d(within_top vs rem→top)={d_top:.3f}; d(within_rem vs top→rem)={d_rem:.3f}",
        fontsize=title_font,
    )
    axis.tick_params(labelsize=tick_font)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=legend_font, loc="upper right")


def render_grid(
    cell: str,
    bin_edges: np.ndarray,
    hist_counts: np.ndarray,
    group_mean: np.ndarray,
    group_count: np.ndarray,
    within_between_d: np.ndarray,
    fig_dir: Path,
    dpi: int,
) -> Path:
    """Render every context of one factor's afferent view into a single grid figure."""

    if cell not in GRID_SPECS:
        raise KeyError(f"Unknown cell {cell!r}")
    nrows, ncols, levels, _factor, factor_label = GRID_SPECS[cell]
    widths = np.diff(bin_edges)
    all_density = hist_counts / (group_count[..., None] * widths[None, None, :])
    density_limit = 1.08 * float(all_density.max())
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.6 * nrows))
    axes_flat = np.asarray(axes).reshape(-1)
    for context in range(levels):
        plot_afferent_group_panel(
            axes_flat[context],
            factor_label,
            context,
            bin_edges,
            hist_counts[context],
            group_mean[context],
            group_count[context],
            within_between_d[context],
            density_limit=density_limit,
        )
    for extra_axis in axes_flat[levels:]:
        extra_axis.set_visible(False)
    fig.tight_layout()
    destination = fig_dir / f"{cell}_afferent_group_split_distribution_grid.png"
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {destination}")
    return destination


def main() -> None:
    """Load saved histograms and render the two afferent grid figures."""

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
                np.asarray(data[f"{cell}_within_between_cohens_d"], dtype=np.float64),
                fig_dir,
                args.dpi,
            )


if __name__ == "__main__":
    main()
