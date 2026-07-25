"""Plot afferent Part-2 style aggregate Cohen's d and continuous alignment matrices.

Reads ``afferent_top10_gate_distributions.npz`` and its metadata JSON produced by
``utils_anal.gawf_afferent_top10_gate_distributions`` and renders two aggregate figures that
mirror the symmetric relevance-timing Part-2 outputs on the destination-view (afferent) side:

* ``part2_afferent_continuous_alignment.{png,pdf}`` — 2x2 grid of cosine alignment heatmaps
  between hidden activation tuning and per-context afferent gate tuning, with the diagonal
  minus off-diagonal contrast annotated per cell.
* ``part2_afferent_relevance_effects_top10.png`` — grouped bar chart of the top-10% bootstrap
  Cohen's d for input vs recurrent afferent gates, split by sector and digit factors, with
  95% bootstrap CIs.

Both figures are saved under ``results/anal_figs/E_relevance_alignment/afferent/`` alongside the
per-context grid distributions produced by the sibling viz module.
"""

from __future__ import annotations

import argparse
import json
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


COLORS = {"sector": "#3b82f6", "digit": "#f97316"}
ALIGNMENT_COLOR_LIMIT = 0.6

# Ordered so heatmaps and bar plots keep the same cell order used by the efferent viz module.
CELLS: tuple[tuple[str, str], ...] = (
    ("input", "sector"),
    ("input", "digit"),
    ("recurrent", "sector"),
    ("recurrent", "digit"),
)


def parse_args() -> argparse.Namespace:
    """Parse visualisation arguments."""

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
        "--report",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_afferent_top10_gate_distributions",
                "data",
            )
            / "afferent_top10_gate_distributions_meta.json"
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
    parser.add_argument(
        "--top_percent",
        type=int,
        default=10,
        help="Which top-k%% bootstrap Cohen's d to render on the bar chart.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def plot_alignment_grid(
    data: np.lib.npyio.NpzFile,
    metadata: dict[str, object],
    fig_dir: Path,
    dpi: int,
) -> Path:
    """Render the 2x2 heatmap of activation vs afferent gate cosine alignment."""

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.7))
    for axis, (gate, factor) in zip(axes.flat, CELLS):
        cell = f"{gate}_{factor}"
        matrix = np.asarray(data[f"{cell}_alignment_matrix"], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Alignment matrix for {cell} must be square, got {matrix.shape}")
        cell_report = metadata["cells"][cell]["continuous_alignment"]
        diag_off = float(cell_report["diagonal_minus_off_diagonal"])
        p_value = float(cell_report["permutation_p_value"])
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-ALIGNMENT_COLOR_LIMIT,
            vmax=ALIGNMENT_COLOR_LIMIT,
        )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(
            f"{gate} afferent / {factor}\n"
            f"diag-offdiag = {diag_off:.3f} (p = {p_value:.3f})"
        )
        axis.set_xlabel("gate context (afferent)")
        axis.set_ylabel("activation context (hidden)")
    fig.suptitle("Part 2 afferent — hidden activation vs destination-view gate cosine alignment")
    fig.tight_layout()
    png_path = fig_dir / "part2_afferent_continuous_alignment.png"
    pdf_path = fig_dir / "part2_afferent_continuous_alignment.pdf"
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return png_path


def plot_top_percent_bars(
    metadata: dict[str, object],
    fig_dir: Path,
    top_percent: int,
    dpi: int,
) -> Path:
    """Render the aggregate top-``k%`` bootstrap Cohen's d bar chart with 95% CI."""

    positions = np.arange(2)
    width = 0.34
    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    percent_key = str(top_percent)
    for factor_index, factor in enumerate(("sector", "digit")):
        points, lower, upper = [], [], []
        for gate in ("input", "recurrent"):
            cell = metadata["cells"][f"{gate}_{factor}"]["top_percent"].get(percent_key)
            if cell is None:
                raise KeyError(
                    f"metadata for {gate}_{factor} does not contain top_percent[{percent_key!r}]"
                )
            point = float(cell["cohens_d"])
            ci_lo, ci_hi = (float(x) for x in cell["bootstrap_ci95"])
            points.append(point)
            lower.append(point - ci_lo)
            upper.append(ci_hi - point)
        offset = (factor_index - 0.5) * width
        bars = axis.bar(
            positions + offset,
            points,
            width,
            color=COLORS[factor],
            alpha=0.82,
            label=factor.capitalize(),
        )
        axis.errorbar(
            positions + offset,
            points,
            yerr=np.asarray([lower, upper]),
            fmt="none",
            ecolor="black",
            capsize=2,
            linewidth=0.8,
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, ["Input afferent", "Recurrent afferent"])
    axis.set_ylabel("Cohen's d: relevant minus other eligible destinations")
    axis.set_title(f"Top {top_percent}% afferent relevance effects (interaction excluded)")
    axis.grid(axis="y", alpha=0.2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    axis.margins(y=0.16)
    fig.tight_layout()
    destination = fig_dir / f"part2_afferent_relevance_effects_top{top_percent}.png"
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {destination}")
    return destination


def main() -> None:
    """Load the afferent aggregate NPZ and metadata, render both aggregate figures."""

    args = parse_args()
    if not (0 < args.top_percent < 100):
        raise ValueError("--top_percent must lie in the open interval (0, 100)")
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    with report_path.open(encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    with np.load(data_path, allow_pickle=False) as data:
        plot_alignment_grid(data, metadata, fig_dir, args.dpi)
    plot_top_percent_bars(metadata, fig_dir, args.top_percent, args.dpi)


if __name__ == "__main__":
    main()
