"""Delta-g version of gawf_recurrent_gate_disinhibition_poster.py.

Same grouped-bar distillation (TT/TR/RT/RR x sign(W), |W|-matched overlap band, digit vs
sector side by side, TT highlighted), but the bar heights are ``delta_of`` instead of raw
``of``: each connection's open fraction is first recentered against its own cross-context
(cross-digit or cross-sector) grand mean (see collect_pooled_records's ``delta_of`` column in
the digit disinhibition module) before being averaged into a group/sign bar. This makes the
bars comparable against a well-defined 0 -- "no deviation from this connection's own average
behavior" -- instead of against an unstated criterion on the raw [0, 1] open-fraction scale.

Reuses print_overlap_bands / overlap_band_sign_stats / sign_p_value from the raw poster module
(both already accept a y_col argument) and run_ols from the digit disinhibition module; only
the figure renderer and the gap/regression wiring are delta-specific.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir
from utils_anal.gawf_recurrent_gate_sign_vs_magnitude_disinhibition import (
    GROUP_NAMES,
    NEG_COLOR,
    POS_COLOR,
    analyze,
    run_ols,
)
from utils_viz.gawf_recurrent_gate_disinhibition_poster import (
    CONTEXTS_BY_VARIABLE,
    DIM_ALPHA,
    HIGHLIGHT_BG,
    HIGHLIGHT_GROUP,
    VARIABLES,
    overlap_band_sign_stats,
    print_overlap_bands,
    sign_p_value,
)

CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem
DPI = 150
Y_COL = "delta_of"


# --------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------
def render_poster_delta(
    stats_by_variable: dict[str, dict[str, dict]],
    gap_by_variable: dict[str, dict[str, float]],
    p_by_variable: dict[str, dict[str, float]],
    fig_path: Path, dpi: int = DPI,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
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
    y_lo, y_hi = y_min - 0.22 * span, y_max + 0.22 * span

    for ax, variable in zip(axes, VARIABLES):
        stats = stats_by_variable[variable]
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

            gap = gap_by_variable[variable][g]
            p = p_by_variable[variable][g]
            p_text = "p<0.001" if np.isfinite(p) and p < 1e-3 else f"p={p:.3f}"

            # Place the label just beyond whichever bar tip sits farthest from 0, on that
            # same side -- unlike the raw-of poster, delta bars routinely extend *below* 0,
            # so "above the bar" is wrong whenever the farthest tip is negative.
            pos_edge = pos["mean"] + pos["sem"] if pos["mean"] >= 0 else pos["mean"] - pos["sem"]
            neg_edge = neg["mean"] + neg["sem"] if neg["mean"] >= 0 else neg["mean"] - neg["sem"]
            farthest = pos_edge if abs(pos_edge) >= abs(neg_edge) else neg_edge
            offset = 0.08 * span
            if farthest >= 0:
                y_text = min(farthest + offset, y_hi - 0.03 * span)
                va = "bottom"
            else:
                y_text = max(farthest - offset, y_lo + 0.03 * span)
                va = "top"
            ax.text(x[gi], y_text, f"gap={gap:.3f}\n{p_text}",
                   ha="center", va=va, fontsize=9.5 if is_highlight else 8.0,
                   fontweight="bold" if is_highlight else "normal",
                   alpha=1.0 if is_highlight else 0.85)

        ax.axhline(0.0, color="black", linewidth=1.0, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_NAMES)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(variable, fontsize=12)
        ax.set_xlabel("group (src -> dst)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "delta gate open fraction\n(mean +/- SEM vs. 0, |W|-matched overlap band)"
    )

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=POS_COLOR,
               markeredgecolor="none", markersize=10, label="W > 0 (+)"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=NEG_COLOR,
               markeredgecolor="none", markersize=10, label="W < 0 (-)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(
        "Recurrent gate disinhibition (delta-g): deviation from each connection's own\n"
        "cross-context grand mean, |W|-matched, by group and sign -- compared against 0",
        fontsize=11.5, y=0.99,
    )
    fig.subplots_adjust(top=0.80, bottom=0.11, wspace=0.08)
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main() -> None:
    stats_by_variable: dict[str, dict[str, dict]] = {}
    gap_by_variable: dict[str, dict[str, float]] = {}
    p_by_variable: dict[str, dict[str, float]] = {}

    for kind in VARIABLES:
        pooled, _meta, overlap_stats, _regs_full_of, _regs_clean_of = analyze(
            CONTEXTS_BY_VARIABLE[kind], kind
        )
        print_overlap_bands(kind, overlap_stats)

        stats = overlap_band_sign_stats(pooled, overlap_stats, y_col=Y_COL)
        stats_by_variable[kind] = stats
        gap_by_variable[kind] = {
            g: stats[g]["+"]["mean"] - stats[g]["-"]["mean"] for g in GROUP_NAMES
        }

        p_by_variable[kind] = {}
        for g in GROUP_NAMES:
            df = pooled[g]
            low, high = overlap_stats[g]["overlap_low"], overlap_stats[g]["overlap_high"]
            if np.isfinite(low) and np.isfinite(high) and high > low:
                clean_df = df[(df["absW"] >= low) & (df["absW"] <= high)]
                print(f"--- regression [{kind} {g}] overlap-only (clean, {Y_COL}) ---")
                reg = run_ols(clean_df, f"{kind} {g} clean ({Y_COL})", y_col=Y_COL)
            else:
                print(f"--- regression [{kind} {g}] overlap-only skipped: no overlap band ---")
                reg = None
            p_by_variable[kind][g] = sign_p_value(reg)

    print(f"\n=== summary (|W|-matched overlap band, controlling for |W| + context, {Y_COL}) ===")
    header = (
        f"{'kind':>7} {'group':>6} {'n_+':>8} {'n_-':>8} "
        f"{Y_COL + '+':>10} {Y_COL + '-':>10} {'gap':>8} {'p (clean, sign coef)':>22}"
    )
    print(header)
    for kind in VARIABLES:
        stats = stats_by_variable[kind]
        for g in GROUP_NAMES:
            pos, neg = stats[g]["+"], stats[g]["-"]
            print(f"{kind:>7} {g:>6} {pos['n']:>8} {neg['n']:>8} {pos['mean']:>10.4f} "
                  f"{neg['mean']:>10.4f} {gap_by_variable[kind][g]:>8.4f} "
                  f"{p_by_variable[kind][g]:>22.3e}")

    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    render_poster_delta(
        stats_by_variable, gap_by_variable, p_by_variable,
        fig_dir / "recurrent_gate_disinhibition_poster_delta.png",
    )


if __name__ == "__main__":
    main()
