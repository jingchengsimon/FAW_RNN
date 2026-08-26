"""Delta-g version of gawf_recurrent_gate_disinhibition_poster.py.

Same grouped-bar distillation (TT/TR/RT/RR x sign(W), |W|-matched overlap band, digit vs
sector side by side, TT highlighted), but the bar heights are ``delta_of`` instead of raw
``of``: each connection's open fraction is first recentered against its own cross-context
(cross-digit or cross-sector) grand mean (see collect_pooled_records's ``delta_of`` column in
the digit disinhibition module) before being averaged into a group/sign bar. This makes the
bars comparable against a well-defined 0 -- "no deviation from this connection's own average
behavior" -- instead of against an unstated criterion on the raw [0, 1] open-fraction scale.

Reuses the raw poster module's overlap-band aggregation and exact paired seed-level sign-flip
test; only the figure renderer and delta-g wiring are delta-specific.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.multiseed_plotting import add_seed_points
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (
    GROUP_NAMES,
    NEG_COLOR,
    POS_COLOR,
    analyze,
)
from utils.analysis.clutter.fig7_recurrent_gate_disinhibition import (
    CONTEXTS_BY_VARIABLE,
    POSTER_RC,
    POSTER_RC_VERTICAL,
    VARIABLES,
    _cross_seed_stats,
    overlap_band_sign_stats,
    print_overlap_bands,
    seed_level_gap_p,
)

CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem
DPI = 150
Y_COL = "delta_of"


# --------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------
def _render_poster_delta(
    stats_by_variable: dict[str, dict[str, dict]],
    gap_by_variable: dict[str, dict[str, float]],
    p_by_variable: dict[str, dict[str, float]],
    fig_path: Path, dpi: int = DPI,
    *,
    vertical: bool = False,
    significance_tests: dict[str, dict[str, dict[str, float]]] | None = None,
    show_seed_points: bool = True,
) -> None:
    rc = POSTER_RC_VERTICAL if vertical else POSTER_RC
    with plt.rc_context(rc):
        if vertical:
            fig, axes = plt.subplots(2, 1, figsize=(6.0, 9.6), sharex=True)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
        _draw_delta_panels(fig, np.asarray(axes).reshape(-1), stats_by_variable,
                           gap_by_variable, p_by_variable, fig_path, dpi, vertical,
                           significance_tests, show_seed_points)


def _draw_delta_panels(
    fig: plt.Figure,
    axes: np.ndarray,
    stats_by_variable: dict[str, dict[str, dict]],
    gap_by_variable: dict[str, dict[str, float]],
    p_by_variable: dict[str, dict[str, float]],
    fig_path: Path,
    dpi: int,
    vertical: bool,
    significance_tests: dict[str, dict[str, dict[str, float]]] | None,
    show_seed_points: bool,
) -> None:
    x = np.arange(len(GROUP_NAMES))
    width = 0.32

    # Shared symmetric-ish y-range derived from the actual bar +/- SEM extents across both
    # panels (delta values have no fixed [0, 1] scale the way raw open fraction does).
    all_edges: list[float] = []
    for variable in VARIABLES:
        stats = stats_by_variable[variable]
        for g in GROUP_NAMES:
            for sign in ("+", "-"):
                s = stats[g][sign]
                if s["n"] > 0:
                    all_edges.append(s["mean"] + s["sem"])
                    all_edges.append(s["mean"] - s["sem"])
    y_min, y_max = min(all_edges), max(all_edges)
    span = y_max - y_min
    y_lo, y_hi = y_min - 0.28 * span, 0.1

    for row, (ax, variable) in enumerate(zip(axes, VARIABLES)):
        stats = stats_by_variable[variable]
        for gi, g in enumerate(GROUP_NAMES):
            pos, neg = stats[g]["+"], stats[g]["-"]
            ax.bar(x[gi] - width / 2, pos["mean"], width, yerr=pos["sem"], capsize=3,
                   color=POS_COLOR, edgecolor="none", zorder=3)
            ax.bar(x[gi] + width / 2, neg["mean"], width, yerr=neg["sem"], capsize=3,
                   color=NEG_COLOR, edgecolor="none", zorder=3)
            for x_pos, values in (
                (x[gi] - width / 2, pos.get("values")),
                (x[gi] + width / 2, neg.get("values")),
            ):
                if values is not None:
                    add_seed_points(
                        ax,
                        np.asarray([x_pos]),
                        np.asarray(values, dtype=np.float64)[:, None],
                        bar_width=width,
                        show=show_seed_points,
                        rng=np.random.default_rng(gi),
                    )

            if significance_tests is not None:
                for x_pos, sign, bar in (
                    (x[gi] - width / 2, "+", pos),
                    (x[gi] + width / 2, "-", neg),
                ):
                    stars = _p_stars(significance_tests[variable][g][sign])
                    if stars:
                        edge = (
                            bar["mean"] + bar["sem"]
                            if bar["mean"] >= 0
                            else bar["mean"] - bar["sem"]
                        )
                        direction = 1.0 if edge >= 0 else -1.0
                        ax.text(x_pos, edge + direction * 0.04 * span, stars,
                                ha="center", va="bottom" if direction > 0 else "top",
                                fontsize=10, fontweight="bold")

            gap = gap_by_variable[variable][g]
            p = p_by_variable[variable][g]

            # Place the label just beyond whichever bar tip sits farthest from 0, on that
            # same side -- unlike the raw-of poster, delta bars routinely extend *below* 0,
            # so "above the bar" is wrong whenever the farthest tip is negative.
            pos_edge = pos["mean"] + pos["sem"] if pos["mean"] >= 0 else pos["mean"] - pos["sem"]
            neg_edge = neg["mean"] + neg["sem"] if neg["mean"] >= 0 else neg["mean"] - neg["sem"]
            farthest = pos_edge if abs(pos_edge) >= abs(neg_edge) else neg_edge
            offset = (0.13 if significance_tests is not None else 0.08) * span
            if farthest >= 0:
                y_text = min(farthest + offset, y_hi - 0.03 * span)
                va = "bottom"
            else:
                y_text = max(farthest - offset, y_lo + 0.03 * span)
                va = "top"
            gap_star = _p_stars(p)
            label = f"gap={gap:.3f}" if not gap_star else f"gap={gap:.3f}\n{gap_star}"
            ax.text(x[gi], y_text, label,
                   ha="center", va=va, fontsize=8.5, alpha=0.85)

        ax.axhline(0.0, color="black", linewidth=1.0, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g[0]}->{g[1]}" for g in GROUP_NAMES])
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks((0.1, 0.0, -0.2, -0.4))
        ax.set_title(variable.capitalize())
        ax.set_xlabel("Group")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if vertical and row == 0:
            ax.set_xlabel("")
            plt.setp(ax.get_xticklabels(), visible=False)

    axes[0].set_ylabel("Delta gate open fraction")
    if vertical:
        axes[1].set_ylabel("Delta gate open fraction")

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=POS_COLOR,
               markeredgecolor="none", markersize=10, label="W > 0 (+)"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=NEG_COLOR,
               markeredgecolor="none", markersize=10, label="W < 0 (-)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.98))
    if vertical:
        fig.subplots_adjust(top=0.85, bottom=0.07, hspace=0.22)
    else:
        fig.subplots_adjust(top=0.87, bottom=0.11, wspace=0.08)
    fig.savefig(fig_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path.with_suffix('.pdf')}")


def _p_stars(p_value: float) -> str:
    """Return one star for an uncorrected significant test."""

    if not np.isfinite(p_value) or p_value >= 0.05:
        return ""
    return "*"


def _connection_group_stats(
    df, overlap_low: float, overlap_high: float,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return unique-connection bars and three uncorrected connection-level p-values."""

    from scipy.stats import ttest_1samp, ttest_ind

    in_band = df[(df["absW"] >= overlap_low) & (df["absW"] <= overlap_high)]
    per_connection = in_band.groupby(["conn", "signpos"], as_index=False)[Y_COL].mean()
    values = {
        "+": per_connection.loc[per_connection["signpos"] == 1, Y_COL].to_numpy(),
        "-": per_connection.loc[per_connection["signpos"] == 0, Y_COL].to_numpy(),
    }
    if any(group.size < 2 for group in values.values()):
        raise RuntimeError("Each sign requires at least two unique connections.")
    stats = {
        sign: {
            "mean": float(group.mean()),
            "sem": float(group.std(ddof=1) / np.sqrt(group.size)),
            "n": int(group.size),
        }
        for sign, group in values.items()
    }
    raw_p = {
        "+": float(ttest_1samp(values["+"], 0.0).pvalue),
        "-": float(ttest_1samp(values["-"], 0.0).pvalue),
        "gap": float(ttest_ind(values["+"], values["-"], equal_var=False).pvalue),
    }
    return stats, raw_p


def _holm_adjust(p_values: dict[tuple[str, str, str], float]) -> dict[tuple[str, str, str], float]:
    """Apply Holm correction jointly to all finite pre-planned p-values."""

    finite = sorted((p, key) for key, p in p_values.items() if np.isfinite(p))
    adjusted: dict[tuple[str, str, str], float] = {
        key: float("nan") for key in p_values
    }
    running = 0.0
    total = len(finite)
    for rank, (p_value, key) in enumerate(finite):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[key] = running
    return adjusted


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse optional ten-seed cache locations and a PDF-only destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dirs", nargs="+", type=Path, default=None)
    parser.add_argument("--fig_dir", type=Path, default=None)
    parser.add_argument("--output_stem", default="recurrent_gate_disinhibition_poster_delta")
    parser.add_argument("--single_seed_connection_tests", action="store_true")
    parser.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay one neutral-gray point per training seed on each 10-seed bar.",
    )
    return parser.parse_args()


def main() -> None:
    """Render single-seed or ten-seed delta-g poster PDFs from structured caches."""

    args = parse_args()
    if args.cache_dirs is not None and len(args.cache_dirs) < 2:
        raise ValueError("--cache_dirs requires at least two per-seed cache directories")
    if args.single_seed_connection_tests and args.cache_dirs is not None:
        raise ValueError("--single_seed_connection_tests cannot be combined with --cache_dirs")
    stats_by_variable: dict[str, dict[str, dict]] = {}
    gap_by_variable: dict[str, dict[str, float]] = {}
    p_by_variable: dict[str, dict[str, float]] = {}
    raw_connection_p: dict[tuple[str, str, str], float] = {}

    for kind in VARIABLES:
        if args.cache_dirs is None:
            pooled, _meta, overlap_stats, _regs_full_of, _regs_clean_of = analyze(
                CONTEXTS_BY_VARIABLE[kind], kind, run_regressions=False
            )
            print_overlap_bands(kind, overlap_stats)
            if args.single_seed_connection_tests:
                stats = {}
                for g in GROUP_NAMES:
                    low = overlap_stats[g]["overlap_low"]
                    high = overlap_stats[g]["overlap_high"]
                    stats[g], raw_p = _connection_group_stats(pooled[g], low, high)
                    raw_connection_p.update(
                        {(kind, g, test): p_value for test, p_value in raw_p.items()}
                    )
            else:
                stats = overlap_band_sign_stats(pooled, overlap_stats, y_col=Y_COL)
            p_by_variable[kind] = {g: float("nan") for g in GROUP_NAMES}
        else:
            per_seed_stats = []
            for cache_dir in args.cache_dirs:
                pooled, _meta, overlap_stats, _regs_full_of, _regs_clean_of = analyze(
                    CONTEXTS_BY_VARIABLE[kind], kind, cache_dir=cache_dir,
                    run_regressions=False,
                )
                per_seed_stats.append(overlap_band_sign_stats(pooled, overlap_stats, y_col=Y_COL))
            stats = _cross_seed_stats(per_seed_stats)
            p_by_variable[kind] = {
                g: seed_level_gap_p(per_seed_stats, g) for g in GROUP_NAMES
            }
        stats_by_variable[kind] = stats
        gap_by_variable[kind] = {
            g: stats[g]["+"]["mean"] - stats[g]["-"]["mean"] for g in GROUP_NAMES
        }

    connection_tests = None
    if args.single_seed_connection_tests:
        connection_tests = {
            kind: {
                g: {test: raw_connection_p[(kind, g, test)] for test in ("+", "-", "gap")}
                for g in GROUP_NAMES
            }
            for kind in VARIABLES
        }
        p_by_variable = {
            kind: {g: raw_connection_p[(kind, g, "gap")] for g in GROUP_NAMES}
            for kind in VARIABLES
        }
        print("\n=== single-seed connection-level raw gap p-values (no correction) ===")
        for kind in VARIABLES:
            for g in GROUP_NAMES:
                print(f"{kind:>7} {g:>2}  p_gap={p_by_variable[kind][g]:.6g}")

    if args.single_seed_connection_tests:
        summary_scope = "unique-connection means within the |W| overlap band"
        p_heading = "p (raw gap)"
    else:
        summary_scope = "|W|-matched overlap band"
        p_heading = "p (paired seeds)"
    print(f"\n=== summary ({summary_scope}, {Y_COL}) ===")
    header = (
        f"{'kind':>7} {'group':>6} {'n_+':>8} {'n_-':>8} "
        f"{Y_COL + '+':>10} {Y_COL + '-':>10} {'gap':>8} {p_heading:>22}"
    )
    print(header)
    for kind in VARIABLES:
        stats = stats_by_variable[kind]
        for g in GROUP_NAMES:
            pos, neg = stats[g]["+"], stats[g]["-"]
            print(f"{kind:>7} {g:>6} {pos['n']:>8} {neg['n']:>8} {pos['mean']:>10.4f} "
                  f"{neg['mean']:>10.4f} {gap_by_variable[kind][g]:>8.4f} "
                  f"{p_by_variable[kind][g]:>22.3e}")

    fig_dir = (
        args.fig_dir
        if args.fig_dir is not None
        else output_dir(CATEGORY, SCRIPT_NAME, "figs")
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / args.output_stem
    _render_poster_delta(
        stats_by_variable, gap_by_variable, p_by_variable, output_path,
        significance_tests=connection_tests,
        show_seed_points=args.show_seed_points,
    )
    _render_poster_delta(
        stats_by_variable, gap_by_variable, p_by_variable,
        fig_dir / f"{args.output_stem}_vertical", vertical=True,
        significance_tests=connection_tests,
        show_seed_points=args.show_seed_points,
    )


if __name__ == "__main__":
    main()
