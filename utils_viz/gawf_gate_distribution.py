"""Plot the seven requested GaWF gate-distribution figures from saved analysis arrays."""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments.

    Directory defaults are left ``None`` here and resolved lazily in ``main()`` so that
    ``output_dir(...)`` (which creates its directory as a side effect) only runs for a category
    actually needed by this invocation. Building the parser must never, by itself, recreate
    figure directories for categories a ``--save_dir`` override makes irrelevant.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--digit_data_dir", default=None)
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--context_dir", default=None)
    parser.add_argument("--delta_dir", default=None)
    parser.add_argument("--relevance_dir", default=None)
    parser.add_argument(
        "--save_dir",
        default="",
        help="Deprecated compatibility override; sends every figure to one directory.",
    )
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    return parser.parse_args()


def _resolve_output_dirs(args: argparse.Namespace) -> None:
    """Fill in any directory left unset by the CLI, calling ``output_dir`` only when needed."""

    if args.save_dir:
        args.raw_dir = args.context_dir = args.delta_dir = args.relevance_dir = args.save_dir
    if args.data_dir is None:
        args.data_dir = str(output_dir("A_raw_gate", "gawf_gate_distribution", "data"))
    if args.digit_data_dir is None:
        args.digit_data_dir = str(
            output_dir("B_gate_by_context", "gawf_gate_digit_distribution", "data")
        )
    if args.raw_dir is None:
        args.raw_dir = str(output_dir("A_raw_gate", "gawf_gate_distribution", "figs"))
    if args.context_dir is None:
        args.context_dir = str(output_dir("B_gate_by_context", "gawf_gate_distribution", "figs"))
    if args.delta_dir is None:
        args.delta_dir = str(output_dir("C_delta_gate", "gawf_gate_distribution", "figs"))
    if args.relevance_dir is None:
        args.relevance_dir = str(
            output_dir("E_relevance_alignment", "gawf_gate_distribution", "figs")
        )


def _density(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    widths = np.diff(edges)
    total = float(np.asarray(counts).sum())
    return np.asarray(counts, dtype=np.float64) / (total * widths)


def _probability_percent(counts: np.ndarray) -> np.ndarray:
    """Return each bin's share of the total count, in percent, matching 01's pooled histogram."""

    counts_float = np.asarray(counts, dtype=np.float64)
    return 100.0 * counts_float / counts_float.sum()


def _finish(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {path}")


def _gate_axes(
    axes: np.ndarray,
    edges: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    centers = (edges[:-1] + edges[1:]) / 2.0
    for axis, kind, title in zip(axes, ("input", "recurrent"), ("Input gate", "Recurrent gate")):
        counts = arrays[f"hist_{kind}_all"]
        stats = metadata["distribution"][kind]
        axis.plot(centers, _density(counts, edges), color="#2b6cb0", linewidth=1.5)
        axis.axvline(0.5, color="black", linestyle="--", label="0.5")
        axis.axvline(stats["mean"], color="#d53f8c", label=f"mean={stats['mean']:.4f}")
        axis.axvline(
            stats["median"], color="#38a169", linestyle=":", label=f"median={stats['median']:.4f}"
        )
        axis.set(title=title, xlabel="Gate value", ylabel="Density", xlim=(0.0, 1.0))
        axis.legend(fontsize=8)


def _style_probability_axis(axis: plt.Axes) -> None:
    """Apply the shared grid/spine style used by the pooled gate and weight panels."""

    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


GATE_YTICKS = {"input": (0, 10, 20, 30, 40), "recurrent": (0, 5, 10, 15)}
WEIGHT_YTICKS = {"input": (0, 12, 24, 36), "recurrent": (0, 7, 14, 21)}
GATE_XTICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
WEIGHT_XTICKS = (-2, -1, 0, 1, 2)
GATE_REBIN_FACTOR = 4  # 400 saved bins at width 0.0025 -> 100 bins at width 0.01.


def _rebin_counts(
    counts: np.ndarray, edges: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sum consecutive equal-width bins by ``factor``, widening the bin width accordingly."""

    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape[-1] % factor != 0:
        raise ValueError(f"bin count {counts.shape[-1]} not divisible by rebin factor {factor}")
    rebinned_counts = counts.reshape(-1, factor).sum(axis=-1)
    rebinned_edges = edges[::factor]
    return rebinned_counts, rebinned_edges


def _pooled_gate_probability_axes(
    axes: np.ndarray,
    edges: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    """Draw the input/recurrent pooled-gate probability panels (row 1 of the 2-by-2 summary).

    The saved histogram uses 0.0025-wide bins; those are re-binned here to 0.01-wide bins
    (``GATE_REBIN_FACTOR``) so each bar carries enough mass to read clearly. The gate curve is
    labeled ``G`` and the reference line is labeled ``$\tilde{G}$`` (median) so both can join
    the single legend shared with row 2 (G, median, W, G⊙W). Median is kept over mean because
    the gate distribution is heavily skewed (most mass sits very close to 0 or 1), which drags
    the mean away from where samples actually concentrate; the median stays representative of
    that bulk.
    """

    for axis, kind, title in zip(axes, ("input", "recurrent"), ("Input gate", "Recurrent gate")):
        counts, coarse_edges = _rebin_counts(arrays[f"hist_{kind}_all"], edges, GATE_REBIN_FACTOR)
        centers = (coarse_edges[:-1] + coarse_edges[1:]) / 2.0
        stats = metadata["distribution"][kind]
        axis.plot(centers, _probability_percent(counts), color="#2b6cb0", linewidth=1.5, label="G")
        axis.axvline(stats["median"], color="#38a169", linestyle=":", label=r"$\tilde{G}$")
        axis.set(title=title, xlabel="Gate value", ylabel="Probability (%)", xlim=(-0.01, 1.01))
        axis.set_xticks(GATE_XTICKS)
        axis.set_yticks(GATE_YTICKS[kind])
        _style_probability_axis(axis)


def _weight_probability_axes(axes: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    """Draw the base/effective input & recurrent weight probability panels (row 2)."""

    titles = ("Input weights", "Recurrent weights")
    for axis, kind, title in zip(axes, ("input", "recurrent"), titles):
        effective_edges = arrays[f"effective_edges_{kind}"]
        effective_centers = (effective_edges[:-1] + effective_edges[1:]) / 2.0
        axis.plot(
            effective_centers,
            _probability_percent(arrays[f"hist_weight_{kind}"]),
            label="W",
            color="black",
        )
        axis.plot(
            effective_centers,
            _probability_percent(arrays[f"hist_effective_{kind}"]),
            label=r"$G\odot W$",
            color="#dd6b20",
        )
        axis.set(title=title, xlabel="Weight", ylabel="Probability (%)", xlim=(-2.0, 2.0))
        axis.set_xticks(WEIGHT_XTICKS)
        axis.set_yticks(WEIGHT_YTICKS[kind])
        _style_probability_axis(axis)


def main() -> None:
    """Read analysis outputs and save seven independent figures."""

    args = parse_args()
    _resolve_output_dirs(args)
    for directory in (args.raw_dir, args.context_dir, args.delta_dir, args.relevance_dir):
        os.makedirs(directory, exist_ok=True)
    stats_path = os.path.join(args.data_dir, "gawf_gate_distribution_stats.npz")
    metadata_path = os.path.join(args.data_dir, "gawf_gate_distribution_meta.json")
    with np.load(stats_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    with open(metadata_path, encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    suffix = args.format

    edges = arrays["gate_edges"]
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    _gate_axes(axes, edges, arrays, metadata)
    fig.suptitle("GaWF gate values pooled across all test frames")
    _finish(fig, os.path.join(args.raw_dir, f"01_pooled_histogram.{suffix}"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, kind, title in zip(axes, ("input", "recurrent"), ("Input gate", "Recurrent gate")):
        counts = arrays[f"hist_{kind}_sign"]
        axis.plot(centers, _density(counts[0], edges), label="W > 0", color="#c53030")
        axis.plot(centers, _density(counts[1], edges), label="W < 0", color="#2b6cb0")
        axis.axvline(0.5, color="black", linestyle="--", linewidth=0.9)
        axis.set(title=title, xlabel="Gate value", ylabel="Density", xlim=(0.0, 1.0))
        axis.legend()
    fig.suptitle("Gate distributions split by corresponding weight sign")
    _finish(fig, os.path.join(args.raw_dir, f"02_weight_sign_histogram.{suffix}"))

    delta_edges = arrays["delta_edges"]
    delta_centers = (delta_edges[:-1] + delta_edges[1:]) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, kind, title in zip(axes, ("input", "recurrent"), ("Input gate", "Recurrent gate")):
        counts = arrays[f"hist_{kind}_delta"]
        axis.plot(delta_centers, _density(counts, delta_edges), color="#805ad5")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        axis.set(
            title=title,
            xlabel=r"Group-mean gate $\Delta g$",
            ylabel="Density",
            xlim=(-0.75, 0.75),
        )
    fig.suptitle("Sector-centered gate distributions")
    sector_centered_name = f"03_sector_centered_gate_histogram.{suffix}"
    _finish(fig, os.path.join(args.delta_dir, sector_centered_name))

    digit_stats_path = os.path.join(args.digit_data_dir, "gawf_gate_digit_stats.npz")
    digit_meta_path = os.path.join(args.digit_data_dir, "gawf_gate_digit_meta.json")
    digit_arrays: dict[str, np.ndarray] | None = None
    digit_metadata: dict[str, object] | None = None
    if os.path.isfile(digit_stats_path):
        with np.load(digit_stats_path) as loaded:
            digit_arrays = {key: loaded[key] for key in loaded.files}
        digit_delta_edges = digit_arrays["delta_edges"]
        if not np.array_equal(delta_edges, digit_delta_edges):
            raise RuntimeError("Sector and digit delta histograms must use identical bin edges")
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 7.2),
            sharex=True,
            sharey="row",
        )
        for row, kind in enumerate(("input", "recurrent")):
            for col, (conditioning, source) in enumerate(
                (("Sector", arrays), ("Digit", digit_arrays))
            ):
                axis = axes[row, col]
                counts = source[f"hist_{kind}_delta"]
                axis.plot(delta_centers, _density(counts, delta_edges), color="#805ad5")
                axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
                axis.set_xlim(-0.75, 0.75)
                axis.set_title(f"{kind.title()} gate — {conditioning}")
                axis.set_xlabel(r"Group-mean gate $\Delta g$")
                axis.set_ylabel("Density")
        fig.suptitle("Corrected group-mean gate deviations on shared axes")
        fig.tight_layout()
        combined_name = f"03_sector_digit_group_mean_delta_histogram.{suffix}"
        _finish(fig, os.path.join(args.delta_dir, combined_name))
        if os.path.isfile(digit_meta_path):
            with open(digit_meta_path, encoding="utf-8") as file_obj:
                digit_metadata = json.load(file_obj)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, 9))
    for axis, kind, title in zip(axes, ("input", "recurrent"), ("Input gate", "Recurrent gate")):
        context_counts = arrays[f"hist_{kind}_context"]
        for sector in range(9):
            axis.plot(
                centers,
                _density(context_counts[sector], edges),
                color=colors[sector],
                linewidth=1.0,
                label=str(sector),
            )
        axis.set(title=title, xlabel="Gate value", ylabel="Density", xlim=(0.0, 1.0))
    axes[1].legend(title="Sector", ncol=3, fontsize=8)
    fig.suptitle("Gate distributions by foreground sector")
    _finish(fig, os.path.join(args.context_dir, f"04_per_context_histogram.{suffix}"))

    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    relevance = arrays["hist_input_relevance"]
    axis.plot(centers, _density(relevance[0], edges), label="Relevant spatial sector")
    axis.plot(centers, _density(relevance[1], edges), label="Other spatial sectors")
    effect = metadata["task_relevance"]["cohens_d_relevant_minus_irrelevant"]
    axis.set(
        xlabel="Input-gate value",
        ylabel="Density",
        title=f"Task-relevance proxy (Cohen's d={effect:.4f})",
        xlim=(0.0, 1.0),
    )
    axis.legend()
    _finish(fig, os.path.join(args.relevance_dir, f"05_task_relevance_histogram.{suffix}"))

    sectors = np.arange(9)
    digits = np.arange(10)
    factor_columns = [("Sector", sectors, "Sector", metadata["sparsity"])]
    if digit_metadata is not None:
        factor_columns.append(("Digit", digits, "Foreground digit", digit_metadata["sparsity"]))
    # Row 1 (mass fraction) and row 2 (index) use distinct color pairs so the two series pairs
    # stay visually separable once their legends are merged into one shared legend.
    mass_colors = ("#2b6cb0", "#dd6b20")
    index_colors = ("#38a169", "#805ad5")
    fig, axes = plt.subplots(
        2, 2 * len(factor_columns), figsize=(5 * len(factor_columns), 7.4), sharex="col"
    )
    axes = np.atleast_2d(axes)
    for factor_index, (factor_label, x_values, x_label, sparsity_by_kind) in enumerate(
        factor_columns
    ):
        for kind_index, kind in enumerate(("input", "recurrent")):
            column = 2 * factor_index + kind_index
            records = sparsity_by_kind[kind]
            axes[0, column].plot(
                x_values,
                [row["top_5pct_mass_fraction"] for row in records],
                marker="o",
                color=mass_colors[0],
                label="top 5%",
            )
            axes[0, column].plot(
                x_values,
                [row["top_10pct_mass_fraction"] for row in records],
                marker="s",
                color=mass_colors[1],
                label="top 10%",
            )
            axes[0, column].set(
                title=f"{kind.capitalize()} gate mass ({factor_label})", ylabel="Mass fraction"
            )
            axes[1, column].plot(
                x_values,
                [row["gini"] for row in records],
                marker="o",
                color=index_colors[0],
                label="Gini",
            )
            axes[1, column].plot(
                x_values,
                [row["normalized_participation_ratio"] for row in records],
                marker="s",
                color=index_colors[1],
                label="Normalized PR",
            )
            axes[1, column].set(xlabel=x_label, ylabel="Index", xticks=x_values)
    # One text-line height (10pt, the default legend/tick font) expressed as a fraction of this
    # figure's fixed 7.4-inch height. Both the title and the shared legend drop by two of these
    # units; the title drops by one extra unit so the title-to-legend gap tightens as well.
    char_height_frac = (10.0 / 72.0) / 7.4
    title_y = 0.985 - 3 * char_height_frac
    legend_y = 0.93 - 2 * char_height_frac
    fig.suptitle("Gate sparsity and concentration by sector and digit", y=title_y)
    handles = (
        axes[0, 0].get_legend_handles_labels()[0] + axes[1, 0].get_legend_handles_labels()[0]
    )
    labels = (
        axes[0, 0].get_legend_handles_labels()[1] + axes[1, 0].get_legend_handles_labels()[1]
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=4,
        frameon=False,
    )
    _finish(fig, os.path.join(args.context_dir, f"06_sparsity_by_sector_digit.{suffix}"))

    # Per-panel width:height ratio matches core_objects_aggregate_2x2.png (5.05in x 4.1in per
    # panel), just laid out as two columns instead of one, then narrowed by another 30% in
    # width only. No composite title: the four panel titles plus the shared
    # G/median/W/G-odot-W legend already carry everything it would add.
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(0.7 * 2 * 5.05, 2 * 4.1))
        _pooled_gate_probability_axes(axes[0], edges, arrays, metadata)
        _weight_probability_axes(axes[1], arrays)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        handles = (
            axes[0, 0].get_legend_handles_labels()[0] + axes[1, 0].get_legend_handles_labels()[0]
        )
        labels = (
            axes[0, 0].get_legend_handles_labels()[1] + axes[1, 0].get_legend_handles_labels()[1]
        )
        # One text-line height (16pt, matching the rc override above) as a fraction of this
        # figure's fixed 8.2-inch height; the legend drops by 0.4 of that unit below its prior
        # position.
        char_height_frac = (16.0 / 72.0) / (2 * 4.1)
        legend_y = 0.965 - 0.4 * char_height_frac
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=4,
            frameon=False,
        )
        gate_weight_png = os.path.join(args.raw_dir, "07_gate_and_weight_distributions_2x2.png")
        gate_weight_pdf = os.path.join(args.raw_dir, "07_gate_and_weight_distributions_2x2.pdf")
        fig.savefig(gate_weight_png, dpi=150, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(gate_weight_pdf, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    print(f"Saved figure: {gate_weight_png}")
    print(f"Saved figure: {gate_weight_pdf}")


if __name__ == "__main__":
    main()
