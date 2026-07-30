"""Poster-style distillation of the recurrent-gate sign-vs-magnitude disinhibition result.

Collapses the connection-level TT/TR/RT/RR x sign(W) analysis (see
gawf_recurrent_gate_sign_vs_magnitude_disinhibition.py / ..._sector.py in utils_anal) into one
grouped-bar figure: for each group, restricted to that group's own |W| overlap band (so + and -
are |W|-matched), mean open fraction +/- SEM for W>0 (blue) vs W<0 (red), digit and sector shown
as two side-by-side subplots. TT is highlighted (bold border + shaded background, other groups
dimmed) because it is the group where the sign gap is largest and most robust to the |W| +
context control; putting digit and sector side by side is meant to make "digit's TT gap is
large, sector's is much smaller" legible at a glance.

Re-derives the pooled data and clean (overlap-band-only) regression in-memory by calling each
module's analyze(contexts, kind) -- this only replays cheap pandas/statsmodels work against the
already-collected npz caches, not the expensive torch forward pass, so it is fast enough to not
warrant a separate persisted-data intermediate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir
from utils_anal.clutter.fig7_recurrent_gate_sign_magnitude import (
    DIGITS,
    GROUP_NAMES,
    NEG_COLOR,
    POS_COLOR,
    analyze,
)
from utils_anal.clutter.fig7_recurrent_gate_sector_sign_magnitude import SECTORS

CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem
DPI = 150
VARIABLES = ("digit", "sector")
CONTEXTS_BY_VARIABLE = {"digit": DIGITS, "sector": SECTORS}
HIGHLIGHT_GROUP = "TT"
DIM_ALPHA = 0.45  # non-highlighted groups' bar alpha
HIGHLIGHT_BG = "#fef9c3"


# --------------------------------------------------------------------------------------------
# |W|-matched (overlap-band-only) sign stats
# --------------------------------------------------------------------------------------------
def print_overlap_bands(kind: str, overlap_stats: dict[str, dict]) -> None:
    print(f"\n=== {kind}: |W| overlap band per group ===")
    for g in GROUP_NAMES:
        low, high = overlap_stats[g]["overlap_low"], overlap_stats[g]["overlap_high"]
        if np.isfinite(low) and np.isfinite(high) and high > low:
            print(f"  {g}: [{low:.4f}, {high:.4f}]")
        else:
            print(f"  {g}: empty overlap band")


def overlap_band_sign_stats(
    pooled: dict[str, pd.DataFrame], overlap_stats: dict[str, dict], y_col: str = "of"
) -> dict[str, dict]:
    """{group -> {"+"/"-" -> {"mean", "sem", "n"}}}, restricted to that group's overlap band.

    ``y_col`` is "of" (default, raw open fraction) or "delta_of" (per-connection, cross-context
    demeaned; see collect_pooled_records in the digit disinhibition module).
    """

    result: dict[str, dict] = {}
    for g in GROUP_NAMES:
        df = pooled[g]
        low, high = overlap_stats[g]["overlap_low"], overlap_stats[g]["overlap_high"]
        if not (np.isfinite(low) and np.isfinite(high) and high > low):
            result[g] = {s: {"mean": float("nan"), "sem": float("nan"), "n": 0} for s in ("+", "-")}
            continue
        in_band = df[(df["absW"] >= low) & (df["absW"] <= high)]
        stats: dict = {}
        for sign_label, signpos in (("+", 1), ("-", 0)):
            sub = in_band.loc[in_band["signpos"] == signpos, y_col]
            n = int(sub.size)
            mean = float(sub.mean()) if n else float("nan")
            sem = float(sub.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            stats[sign_label] = {"mean": mean, "sem": sem, "n": n}
        result[g] = stats
    return result


def sign_p_value(reg: dict | None) -> float:
    """Prefer the conn-clustered robust p-value; fall back to the plain OLS p."""

    if reg is None:
        return float("nan")
    return reg.get("p_cluster", reg.get("p", float("nan")))


# --------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------
# Tick labels spell out src->dst directionality directly (T->T, T->R, R->T, R->R) instead of
# the terse TT/TR/RT/RR group codes, so the axis no longer needs a "(src -> dst)" gloss.
GROUP_TICK_LABELS = {g: f"{g[0]}->{g[1]}" for g in GROUP_NAMES}


# Enlarged-font rc overrides, matching rnn_unit_gate_context_specificity.py's
# plot_marginalization_summary (produces 03_unit_gate_marginalization_1x3).
POSTER_RC = {
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
}

# Vertical layout only: every font bumped to 16 (separate from POSTER_RC so the horizontal
# layout's sizing is untouched).
POSTER_RC_VERTICAL = {
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}


def _draw_poster_panel(
    ax: plt.Axes, stats: dict[str, dict], title: str, *,
    show_xlabel: bool = True, show_ylabel: bool = True,
) -> None:
    """Draw one Digit/Sector bar panel (shared by the horizontal and vertical layouts)."""

    x = np.arange(len(GROUP_NAMES))
    width = 0.32
    for gi, g in enumerate(GROUP_NAMES):
        is_highlight = g == HIGHLIGHT_GROUP
        alpha = 1.0 if is_highlight else DIM_ALPHA
        edge_color = "black" if is_highlight else "0.5"
        edge_lw = 2.2 if is_highlight else 0.6
        if is_highlight:
            ax.axvspan(x[gi] - 0.5, x[gi] + 0.5, color=HIGHLIGHT_BG, zorder=0)

        pos, neg = stats[g]["+"], stats[g]["-"]
        ax.bar(x[gi] - width / 2, pos["mean"], width, yerr=pos["sem"], capsize=3,
               color=POS_COLOR, alpha=alpha, edgecolor=edge_color, linewidth=edge_lw, zorder=3)
        ax.bar(x[gi] + width / 2, neg["mean"], width, yerr=neg["sem"], capsize=3,
               color=NEG_COLOR, alpha=alpha, edgecolor=edge_color, linewidth=edge_lw, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_TICK_LABELS[g] for g in GROUP_NAMES])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("Group")
    if show_ylabel:
        ax.set_ylabel("Gate open fraction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=POS_COLOR,
               markeredgecolor="none", markersize=10, label="W > 0 (+)"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=NEG_COLOR,
               markeredgecolor="none", markersize=10, label="W < 0 (-)"),
    ]


def _save_png_and_pdf(fig: plt.Figure, fig_path: Path, dpi: int) -> None:
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {fig_path}")
    pdf_path = fig_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    plt.close(fig)


def render_poster(
    stats_by_variable: dict[str, dict[str, dict]],
    fig_path: Path, dpi: int = DPI,
) -> None:
    """Horizontal layout: Digit and Sector side by side, sharing the y-axis."""

    with plt.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
        for ax, variable in zip(axes, VARIABLES):
            _draw_poster_panel(
                ax, stats_by_variable[variable], variable.capitalize(),
                show_ylabel=(ax is axes[0]),
            )
        fig.legend(handles=_legend_handles(), loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.98))
        fig.subplots_adjust(top=0.87, bottom=0.11, wspace=0.08)
        _save_png_and_pdf(fig, fig_path, dpi)


def render_poster_vertical(
    stats_by_variable: dict[str, dict[str, dict]],
    fig_path: Path, dpi: int = DPI,
) -> None:
    """Vertical layout: Digit on top, Sector below, sharing the x-axis (each keeps its own
    y-axis -- ticks/gridlines/label are not shared, per the horizontal version's request).
    Uses POSTER_RC_VERTICAL (all fonts at 16) instead of the horizontal layout's POSTER_RC."""

    with plt.rc_context(POSTER_RC_VERTICAL):
        fig, axes = plt.subplots(2, 1, figsize=(6.0, 9.6), sharex=True)
        for row, (ax, variable) in enumerate(zip(axes, VARIABLES)):
            _draw_poster_panel(ax, stats_by_variable[variable], variable.capitalize())
            if row < len(axes) - 1:
                ax.set_xlabel("")
                plt.setp(ax.get_xticklabels(), visible=False)

        # Legend row nudged down by 0.5 "character height" (taken as half the legend font size
        # in points, converted to a figure-fraction offset via this figure's actual height).
        fig_height_in = fig.get_size_inches()[1]
        char_height_in = POSTER_RC_VERTICAL["legend.fontsize"] / 72.0
        legend_y = 0.98 - 0.5 * char_height_in / fig_height_in
        fig.legend(handles=_legend_handles(), loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, legend_y))
        # All text at 16pt needs more headroom than POSTER_RC's mixed 13/15pt did, or the
        # legend (now taller) collides with the top ("Digit") panel's title.
        fig.subplots_adjust(top=0.85, bottom=0.07, hspace=0.22)
        _save_png_and_pdf(fig, fig_path, dpi)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main() -> None:
    stats_by_variable: dict[str, dict[str, dict]] = {}
    gap_by_variable: dict[str, dict[str, float]] = {}
    p_by_variable: dict[str, dict[str, float]] = {}

    for kind in VARIABLES:
        pooled, _meta, overlap_stats, _regs_full, regs_clean = analyze(CONTEXTS_BY_VARIABLE[kind], kind)
        print_overlap_bands(kind, overlap_stats)

        stats = overlap_band_sign_stats(pooled, overlap_stats)
        stats_by_variable[kind] = stats
        gap_by_variable[kind] = {
            g: stats[g]["+"]["mean"] - stats[g]["-"]["mean"] for g in GROUP_NAMES
        }
        p_by_variable[kind] = {g: sign_p_value(regs_clean.get(g)) for g in GROUP_NAMES}

    print("\n=== summary (|W|-matched overlap band, controlling for |W| + context) ===")
    header = f"{'kind':>7} {'group':>6} {'n_+':>8} {'n_-':>8} {'of+':>8} {'of-':>8} {'gap':>8} {'p (clean, sign coef)':>22}"
    print(header)
    for kind in VARIABLES:
        stats = stats_by_variable[kind]
        for g in GROUP_NAMES:
            pos, neg = stats[g]["+"], stats[g]["-"]
            print(f"{kind:>7} {g:>6} {pos['n']:>8} {neg['n']:>8} {pos['mean']:>8.4f} "
                  f"{neg['mean']:>8.4f} {gap_by_variable[kind][g]:>8.4f} "
                  f"{p_by_variable[kind][g]:>22.3e}")

    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    render_poster(stats_by_variable, fig_dir / "recurrent_gate_disinhibition_poster.png")
    render_poster_vertical(
        stats_by_variable, fig_dir / "recurrent_gate_disinhibition_poster_vertical.png"
    )


if __name__ == "__main__":
    main()
