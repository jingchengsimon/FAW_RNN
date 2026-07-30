"""Recurrent gate sign-vs-magnitude ("disinhibition") test, pooled across digits.

Tests whether the frozen recurrent gate opens according to sign(W) rather than |W|. For each
of the four source/destination tuned-vs-remaining groups (TT/TR/RT/RR, naming: src->dst), the
per-connection open fraction is regressed on signpos = 1{W>0}, |W|, and a digit fixed effect.
A positive, robust signpos coefficient after controlling for |W| and digit is evidence for
disinhibition (sign-gated); a coefficient that vanishes once |W| is controlled for instead
points to a magnitude effect.

Convention: gate[..., i, j] = src j -> dst i (row=dst, col=src), same for W[i, j]. Groups are
named src->dst: TT=(src in T, dst in T), TR=(src in T, dst in R), RT=(src in R, dst in T),
RR=(src in R, dst in R), with T = the per-digit top-10% tuned set and R its complement.

Reuses the real per-digit gate/act/W caches (digit{d}_gate_act_cache.npz, d=0..9) written by
gawf_recurrent_gate_single_digit_diagnostic_collect.py and gawf_recurrent_gate_multi_digit_collect.py.
T is read from the cache's T_old (FDR-selective top-10% among eligible units) rather than
reconstructed naively from mean activation -- gawf_recurrent_gate_raw_group_sign_grid.py found
that the naive reconstruction does not reproduce this project's own reference TT statistic.
Falls back to synthetic multi-digit data end-to-end when no real cache is found on disk, so the
script always runs; the synthetic gate is built with both a sign effect and a magnitude effect
baked in, so every code path (including the regression) is exercised meaningfully.

The load/pool/regression/plot pipeline (``load_all_contexts`` .. ``analyze``) is written to be
context-kind-agnostic (digit or sector; anything with a ``context{i}_gate_act_cache.npz``-style
cache via ``cache_path_for_context(context, kind=...)``), so this module also backs the sector
companion script gawf_recurrent_gate_sign_vs_magnitude_disinhibition_sector.py, which calls
``analyze(SECTORS, "sector")`` and only supplies its own CATEGORY/SCRIPT_NAME/output filenames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir
from utils_anal.clutter.fig7_recurrent_gate_cache import cache_path_for_context, group_masks

CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem
DIGITS = tuple(range(10))
TOP_FRACTION = 0.10
EXCLUDE_DIAGONAL = False
STRICT_FRACTION = False  # of = mean(gate); True -> mean(gate > 0.5)
N_BINS = 9
MIN_GROUP_SIGN_N = 30
OVERLAP_WARN_FRAC = 0.10  # overlap width < 10% of the combined |W| range -> WARN
MAX_SCATTER_PER_SIGN = 8000  # figure only; regressions always use the full pooled data
DPI = 150
POS_COLOR = "#1d4ed8"
NEG_COLOR = "#dc2626"
GROUP_NAMES = ("TT", "TR", "RT", "RR")
RNG_SEED = 0


# --------------------------------------------------------------------------------------------
# DATA LOADING -- real per-context (digit or sector) caches, synthetic fallback if none found.
# --------------------------------------------------------------------------------------------
def load_context_data(context: int, kind: str = "digit") -> dict | None:
    """Load one digit/sector's (W, gate_raw, act, T) from the real cache; None if absent."""

    cache_path = cache_path_for_context(context, kind=kind)
    if not cache_path.is_file():
        return None
    with np.load(cache_path, allow_pickle=False) as cached:
        W = np.asarray(cached["W"], dtype=np.float64)
        gate_raw = np.asarray(cached["gate"], dtype=np.float64)
        act = np.asarray(cached["act"], dtype=np.float64)
        T = np.asarray(cached["T_old"], dtype=bool)
    return {"W": W, "gate_raw": gate_raw, "act": act, "T": T, "context": context}


def build_synthetic_dataset(contexts: tuple[int, ...], hidden_size: int = 64,
                             n_samples: int = 400) -> dict[int, dict]:
    """Deterministic synthetic (W, gate_raw, act, T) per context; W is shared across contexts.

    The gate logit is built from BOTH sign(W) (disinhibition, coefficient ~1.5) and |W|
    (magnitude leak, coefficient ~2.0) plus sample noise, so a real mixed effect is present
    for the regression below to (correctly) recover. Context-agnostic: works as a digit or a
    sector placeholder, since it never reads anything context-semantic, only the context id.
    """

    rng = np.random.default_rng(RNG_SEED)
    W = rng.normal(0.0, 0.08, size=(hidden_size, hidden_size))
    data: dict[int, dict] = {}
    for ctx in contexts:
        crng = np.random.default_rng(RNG_SEED + 1 + ctx)
        act_mean = crng.gamma(2.0, 1.0, size=hidden_size)
        top_k = max(1, int(round(TOP_FRACTION * hidden_size)))
        T = np.zeros(hidden_size, dtype=bool)
        T[np.argsort(-act_mean)[:top_k]] = True

        base_logit = 1.5 * np.sign(W) + 2.0 * np.abs(W) - 0.6
        sample_noise = crng.normal(0.0, 1.0, size=(n_samples, hidden_size, hidden_size))
        gate = 1.0 / (1.0 + np.exp(-(base_logit[None, :, :] + sample_noise)))
        gate_raw = gate.reshape(1, n_samples, 1, hidden_size, hidden_size)

        act = np.broadcast_to(act_mean, (n_samples, hidden_size)).copy()
        act += crng.normal(0.0, 0.1, size=act.shape)
        act = act.reshape(1, n_samples, 1, hidden_size)

        data[ctx] = {"W": W, "gate_raw": gate_raw, "act": act, "T": T, "context": ctx}
    return data


def load_all_contexts(contexts: tuple[int, ...], kind: str = "digit") -> dict[int, dict]:
    found: dict[int, dict] = {}
    for ctx in contexts:
        loaded = load_context_data(ctx, kind=kind)
        if loaded is not None:
            found[ctx] = loaded
    if found:
        cache_dir = cache_path_for_context(contexts[0], kind=kind).parent
        print(f"Loaded real gate/act caches for {kind}s {sorted(found)} from {cache_dir}")
        return found
    print(f"No real {kind} gate/act caches found on disk; using synthetic multi-{kind} "
          "placeholder data so the script runs end-to-end.")
    return build_synthetic_dataset(contexts)


# --------------------------------------------------------------------------------------------
# Per-context preprocessing
# --------------------------------------------------------------------------------------------
def per_connection_open_fraction(gate_raw: np.ndarray, strict: bool = STRICT_FRACTION) -> np.ndarray:
    """of[i, j] = fraction/mean-open of gate_raw[0, :, 0, i, j] over samples."""

    if gate_raw.ndim != 5 or gate_raw.shape[0] != 1 or gate_raw.shape[2] != 1:
        raise AssertionError(f"gate_raw must be (1, n_samples, 1, H, H); got {gate_raw.shape}")
    g = gate_raw[0, :, 0]  # (n_samples, H, H)
    if strict:
        return (g > 0.5).mean(axis=0)
    return g.mean(axis=0)


# --------------------------------------------------------------------------------------------
# Pool (i, j, context) connections into the 4 groups
# --------------------------------------------------------------------------------------------
def collect_pooled_records(
    all_contexts: dict[int, dict], exclude_diagonal: bool = EXCLUDE_DIAGONAL
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return ({group -> DataFrame[absW, of, delta_of, signpos, sign, context, conn]}, meta).

    ``delta_of[ctx, i, j] = of[ctx, i, j] - grand[i, j]``, where ``grand[i, j]`` is the plain
    (unweighted) mean of ``of[ctx, i, j]`` over every loaded context -- each context counts as
    one vote regardless of its own raw sample count (so digit 0's larger cache does not get
    extra weight), mirroring the "equal_n" delta convention already used elsewhere in this repo
    (gawf_gate_context_parts123.py's ``equal_delta = equal_mean - equal_mean.mean(axis=0)``).
    ``delta_of`` is still connection-level (per (i, j, context) triple) at this point; it is
    only a per-connection recentering, not an aggregate. This lets a later mean over many
    connections compare against a well-defined 0 (this connection's own cross-context average)
    instead of against ``of``'s un-anchored [0, 1] scale.

    ``meta`` carries per-context n_samples/n_top and per-group diagonal connection counts
    (with and without exclusion) for the printed summary. ``all_contexts`` keys are digit
    ids or sector ids depending on the caller; this function itself is context-agnostic.
    """

    of_by_context: dict[int, np.ndarray] = {
        ctx: per_connection_open_fraction(data["gate_raw"]) for ctx, data in all_contexts.items()
    }
    grand = np.mean(np.stack(list(of_by_context.values()), axis=0), axis=0)  # (H, H)

    per_group_rows: dict[str, list[pd.DataFrame]] = {g: [] for g in GROUP_NAMES}
    n_samples_by_context: dict[int, int] = {}
    n_top_by_context: dict[int, int] = {}
    diag_counts = {g: {"with_diagonal": 0, "without_diagonal": 0} for g in GROUP_NAMES}

    for ctx, data in sorted(all_contexts.items()):
        W = data["W"]
        hidden_size = W.shape[0]
        gate_raw = data["gate_raw"]
        T = data["T"]
        R = ~T
        n_samples_by_context[ctx] = int(gate_raw.shape[1])
        n_top_by_context[ctx] = int(T.sum())

        of = of_by_context[ctx]
        groups = group_masks(T, R)

        for group_name, (src_mask, dst_mask) in groups.items():
            full_mask = dst_mask[:, None] & src_mask[None, :] & (W != 0)
            not_diag = ~np.eye(hidden_size, dtype=bool)
            # Always tally both counts for transparency, independent of which one is
            # actually used below (that choice is governed by exclude_diagonal).
            diag_counts[group_name]["with_diagonal"] += int(full_mask.sum())
            diag_counts[group_name]["without_diagonal"] += int((full_mask & not_diag).sum())
            if exclude_diagonal:
                full_mask = full_mask & not_diag

            i_idx, j_idx = np.where(full_mask)
            if i_idx.size == 0:
                continue
            w_vals = W[i_idx, j_idx]
            of_vals = of[i_idx, j_idx]
            delta_vals = of_vals - grand[i_idx, j_idx]
            conn_id = i_idx.astype(np.int64) * hidden_size + j_idx.astype(np.int64)
            per_group_rows[group_name].append(pd.DataFrame({
                "absW": np.abs(w_vals),
                "of": of_vals,
                "delta_of": delta_vals,
                "signpos": (w_vals > 0).astype(np.int64),
                "sign": np.where(w_vals > 0, "+", "-"),
                "context": ctx,
                "conn": conn_id,
            }))

    pooled = {
        g: (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["absW", "of", "delta_of", "signpos", "sign", "context", "conn"]))
        for g, rows in per_group_rows.items()
    }
    meta = {
        "n_samples_by_context": n_samples_by_context,
        "n_top_by_context": n_top_by_context,
        "diag_counts": diag_counts,
        "exclude_diagonal": exclude_diagonal,
    }
    return pooled, meta


# --------------------------------------------------------------------------------------------
# Overlap check
# --------------------------------------------------------------------------------------------
def compute_overlap_band(x_pos: np.ndarray, x_neg: np.ndarray) -> dict:
    stats: dict = {}
    for name, arr in (("pos", x_pos), ("neg", x_neg)):
        if arr.size == 0:
            stats[name] = {"min": float("nan"), "max": float("nan"),
                           "p5": float("nan"), "p95": float("nan")}
        else:
            stats[name] = {
                "min": float(arr.min()), "max": float(arr.max()),
                "p5": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95)),
            }
    if x_pos.size == 0 or x_neg.size == 0:
        low, high = float("nan"), float("nan")
    else:
        low = max(stats["pos"]["min"], stats["neg"]["min"])
        high = min(stats["pos"]["max"], stats["neg"]["max"])
    stats["overlap_low"], stats["overlap_high"] = low, high
    return stats


def check_overlap(group_name: str, stats: dict, x_pos: np.ndarray, x_neg: np.ndarray) -> tuple[float, float]:
    low, high = stats["overlap_low"], stats["overlap_high"]
    p = stats["pos"]
    n = stats["neg"]
    print(f"[{group_name}] |W| ranges: "
          f"+ min={p['min']:.4f} max={p['max']:.4f} p5={p['p5']:.4f} p95={p['p95']:.4f}  |  "
          f"- min={n['min']:.4f} max={n['max']:.4f} p5={n['p5']:.4f} p95={n['p95']:.4f}")

    if x_pos.size == 0 or x_neg.size == 0:
        print(f"  WARN [{group_name}]: one sign has zero connections -- cannot compare + vs -.")
        return low, high

    width = high - low
    combined = np.concatenate([x_pos, x_neg])
    total_range = float(combined.max() - combined.min())
    if width <= 0:
        print(f"  WARN [{group_name}]: empty |W| overlap band [{low:.4f}, {high:.4f}] -- "
              "sign 与 |W| 共线, 此组无法干净区分符号与大小, 结论只在重叠带内有效(此组无重叠带).")
    elif total_range > 0 and width / total_range < OVERLAP_WARN_FRAC:
        print(f"  WARN [{group_name}]: narrow |W| overlap band [{low:.4f}, {high:.4f}] "
              f"({width / total_range:.1%} of combined range) -- "
              "sign 与 |W| 共线, 此组无法干净区分符号与大小, 结论只在重叠带内有效.")
    else:
        frac = width / total_range if total_range > 0 else float("nan")
        print(f"  overlap band [{low:.4f}, {high:.4f}] ({frac:.1%} of combined range)")
    return low, high


# --------------------------------------------------------------------------------------------
# Binned mean curves
# --------------------------------------------------------------------------------------------
def quantile_bin_edges(x: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([float(x.min()), float(x.max()) + 1e-12])
    return edges


def binned_mean_curve(
    x: np.ndarray, y: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (x-mean center, y-mean, y-SEM, n), using x's own in-bin mean as the plotted center."""

    n_bins = bin_edges.size - 1
    idx = np.clip(np.digitize(x, bin_edges[1:-1], right=False), 0, n_bins - 1)
    centers = np.full(n_bins, np.nan)
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    ns = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = idx == b
        ns[b] = int(sel.sum())
        if ns[b] > 0:
            centers[b] = float(x[sel].mean())
            means[b] = float(y[sel].mean())
            sems[b] = float(y[sel].std(ddof=1) / np.sqrt(ns[b])) if ns[b] > 1 else 0.0
    return centers, means, sems, ns


# --------------------------------------------------------------------------------------------
# Regression: y_col ~ signpos + absW + C(context)  (y_col is "of" by default, or "delta_of")
# --------------------------------------------------------------------------------------------
def run_ols(df: pd.DataFrame, label: str, y_col: str = "of") -> dict | None:
    if df["signpos"].nunique() < 2:
        print(f"  [{label}] skipped: only one sign present, signpos coefficient is undefined")
        return None
    formula = f"{y_col} ~ signpos + absW" + (" + C(context)" if df["context"].nunique() > 1 else "")
    model = smf.ols(formula, data=df).fit()
    ci = model.conf_int().loc["signpos"]
    result = {
        "n": int(len(df)), "coef": float(model.params["signpos"]),
        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
        "p": float(model.pvalues["signpos"]), "r2": float(model.rsquared),
    }
    print(f"  [{label}] n={result['n']} sign coef={result['coef']:.4f} "
          f"95%CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}] "
          f"p={result['p']:.3e} R^2={result['r2']:.3f}")
    try:
        clustered = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["conn"]}
        )
        cci = clustered.conf_int().loc["signpos"]
        print(f"    conn-clustered robust SE: coef={clustered.params['signpos']:.4f} "
              f"95%CI=[{cci[0]:.4f}, {cci[1]:.4f}] p={clustered.pvalues['signpos']:.3e}")
        result.update({
            "coef_cluster": float(clustered.params["signpos"]),
            "ci_low_cluster": float(cci[0]), "ci_high_cluster": float(cci[1]),
            "p_cluster": float(clustered.pvalues["signpos"]),
        })
    except Exception as exc:  # degenerate clustering (e.g. too few distinct connections)
        print(f"    conn-clustered robust SE failed: {exc}")
    return result


# --------------------------------------------------------------------------------------------
# Figure: 2x2, one panel per group
# --------------------------------------------------------------------------------------------
def _draw_panel(
    ax: plt.Axes, df: pd.DataFrame, overlap_low: float, overlap_high: float,
    rng: np.random.Generator, y_col: str = "of", y_label: str = "gate open fraction",
) -> tuple[float, float]:
    """Draw one group panel (scatter + binned curves + overlap shading + n/gap annotation).

    ``y_col`` selects which pooled column to plot on the y-axis -- "of" (default) or
    "delta_of" (per-connection, cross-context demeaned; see collect_pooled_records).

    Returns (pos_center_max, neg_center_max): the largest plotted bin-center |W| for each
    sign's binned curve (nan if that sign has no valid bin), so callers can zoom the x-axis
    to just the curves' own range instead of the full scatter/overlap-band extent.
    """

    pos = df[df["signpos"] == 1]
    neg = df[df["signpos"] == 0]

    for sub, color in ((pos, POS_COLOR), (neg, NEG_COLOR)):
        if len(sub) == 0:
            continue
        plotted = sub if len(sub) <= MAX_SCATTER_PER_SIGN else sub.sample(
            n=MAX_SCATTER_PER_SIGN, random_state=rng.integers(0, 2**31 - 1)
        )
        ax.scatter(plotted["absW"], plotted[y_col], s=6, color=color, alpha=0.12, linewidths=0)

    pos_center_max = float("nan")
    neg_center_max = float("nan")
    combined_x = df["absW"].to_numpy()
    if combined_x.size >= 2:
        edges = quantile_bin_edges(combined_x)
        for sub, color, is_pos in ((pos, POS_COLOR, True), (neg, NEG_COLOR, False)):
            if len(sub) == 0:
                continue
            centers, means, sems, ns = binned_mean_curve(
                sub["absW"].to_numpy(), sub[y_col].to_numpy(), edges
            )
            valid = ns > 0
            ax.errorbar(centers[valid], means[valid], yerr=sems[valid], color=color,
                       linewidth=1.8, marker="o", markersize=3.5, capsize=2, zorder=3)
            if valid.any():
                center_max = float(np.nanmax(centers[valid]))
                if is_pos:
                    pos_center_max = center_max
                else:
                    neg_center_max = center_max

    if np.isfinite(overlap_low) and np.isfinite(overlap_high) and overlap_high > overlap_low:
        ax.axvspan(overlap_low, overlap_high, color="0.85", zorder=0)
        in_band = df[(df["absW"] >= overlap_low) & (df["absW"] <= overlap_high)]
        mean_pos = in_band.loc[in_band["signpos"] == 1, y_col].mean()
        mean_neg = in_band.loc[in_band["signpos"] == 0, y_col].mean()
        gap_text = f"overlap gap ({y_col}+ - {y_col}-) = {mean_pos - mean_neg:.3f}"
    else:
        gap_text = "no |W| overlap band"

    if y_col == "delta_of":
        ax.axhline(0.0, color="0.3", linewidth=0.9, linestyle="--", zorder=1)
    ax.text(0.02, 0.98, f"n+={len(pos)}  n-={len(neg)}\n{gap_text}",
           transform=ax.transAxes, ha="left", va="top", fontsize=8,
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"))
    ax.set_xlabel("|W|")
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return pos_center_max, neg_center_max


def _finalize_figure(
    fig: plt.Figure, subtitle_extra: str = "",
    title: str = "Recurrent gate open fraction vs. |W|, split by sign(W) -- disinhibition check",
) -> None:
    """Shared color legend (between suptitle and grid) + suptitle + spacing, for both figures."""

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=POS_COLOR,
               markeredgecolor="none", markersize=8, label="W > 0 (+)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=NEG_COLOR,
               markeredgecolor="none", markersize=8, label="W < 0 (-)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.885))
    fig.suptitle(
        f"{title}\n"
        f"(binned mean +/- SEM curves; shaded = |W| range where both signs have data){subtitle_extra}",
        fontsize=11, y=0.99,
    )
    fig.subplots_adjust(top=0.82, bottom=0.07, hspace=0.32, wspace=0.25)


def render_figure(
    pooled: dict[str, pd.DataFrame], overlap_stats: dict[str, dict], fig_path: Path,
    dpi: int = DPI, rng_seed: int = RNG_SEED, y_col: str = "of",
    y_label: str = "gate open fraction",
    title: str = "Recurrent gate open fraction vs. |W|, split by sign(W) -- disinhibition check",
) -> None:
    """Full-range 2x2: x-axis auto-scales to the scatter (i.e. the raw |W| extent)."""

    rng = np.random.default_rng(rng_seed)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.2))
    layout = {"TT": (0, 0), "TR": (0, 1), "RT": (1, 0), "RR": (1, 1)}

    for group_name, (row, col) in layout.items():
        ax = axes[row, col]
        df = pooled[group_name]
        low, high = overlap_stats[group_name]["overlap_low"], overlap_stats[group_name]["overlap_high"]
        _draw_panel(ax, df, low, high, rng, y_col=y_col, y_label=y_label)
        ax.set_title(group_name, fontsize=11)

    _finalize_figure(fig, title=title)
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


def render_figure_zoomed(
    pooled: dict[str, pd.DataFrame], overlap_stats: dict[str, dict], fig_path: Path,
    dpi: int = DPI, rng_seed: int = RNG_SEED, margin: float = 1.08, y_col: str = "of",
    y_label: str = "gate open fraction",
    title: str = "Recurrent gate open fraction vs. |W|, split by sign(W) -- disinhibition check",
) -> None:
    """Same panels, but each x-axis is cropped to just past the two binned curves' own
    range (i.e. where the N_BINS quantile-bin markers actually sit), not the full
    scatter/overlap-band extent. The scatter cloud and gray overlap band are still drawn
    at full width and simply get clipped by the tighter xlim -- this is a viewing crop
    only, no data is dropped or recomputed."""

    rng = np.random.default_rng(rng_seed)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.2))
    layout = {"TT": (0, 0), "TR": (0, 1), "RT": (1, 0), "RR": (1, 1)}

    for group_name, (row, col) in layout.items():
        ax = axes[row, col]
        df = pooled[group_name]
        low, high = overlap_stats[group_name]["overlap_low"], overlap_stats[group_name]["overlap_high"]
        pos_center_max, neg_center_max = _draw_panel(ax, df, low, high, rng, y_col=y_col, y_label=y_label)

        candidates = [v for v in (pos_center_max, neg_center_max) if np.isfinite(v)]
        curve_max = max(candidates) if candidates else (
            float(df["absW"].max()) if len(df) else 1.0
        )
        ax.set_xlim(0.0, max(curve_max * margin, 1e-6))
        ax.set_title(f"{group_name} (zoomed to the {N_BINS} bins)", fontsize=11)

    _finalize_figure(
        fig, subtitle_extra=f"\nzoomed to each panel's own {N_BINS}-bin curve range", title=title,
    )
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


# --------------------------------------------------------------------------------------------
# Printed summary
# --------------------------------------------------------------------------------------------
def print_summary_table(
    pooled: dict[str, pd.DataFrame], overlap_stats: dict[str, dict],
    regressions_full: dict[str, dict | None], regressions_clean: dict[str, dict | None],
    meta: dict, kind: str = "digit",
) -> None:
    print(f"\n=== n_samples / n_top per {kind} ===")
    for ctx in sorted(meta["n_top_by_context"]):
        print(f"  {kind} {ctx}: n_samples={meta['n_samples_by_context'][ctx]} "
              f"n_top={meta['n_top_by_context'][ctx]}")

    print(f"\n=== diagonal (i==j) connection counts per group (exclude_diagonal={meta['exclude_diagonal']}) ===")
    for g, counts in meta["diag_counts"].items():
        print(f"  {g}: with_diagonal={counts['with_diagonal']} without_diagonal={counts['without_diagonal']}")

    header = f"{'group':>6} {'n_+':>8} {'n_-':>8} {'overlap band':>22} {'full sign coef':>16} {'clean sign coef':>16}"
    print(f"\n{header}")
    for g in GROUP_NAMES:
        df = pooled[g]
        n_pos = int((df["signpos"] == 1).sum())
        n_neg = int((df["signpos"] == 0).sum())
        if n_pos < MIN_GROUP_SIGN_N or n_neg < MIN_GROUP_SIGN_N:
            print(f"  WARN [{g}]: n_+={n_pos} n_-={n_neg} -- fewer than {MIN_GROUP_SIGN_N} points "
                  "in one sign, estimates may be noisy")
        low, high = overlap_stats[g]["overlap_low"], overlap_stats[g]["overlap_high"]
        overlap_str = (f"[{low:.3f},{high:.3f}]"
                        if np.isfinite(low) and np.isfinite(high) and high > low else "empty")
        full_coef = regressions_full[g]["coef"] if regressions_full[g] else float("nan")
        clean_coef = regressions_clean[g]["coef"] if regressions_clean.get(g) else float("nan")
        print(f"{g:>6} {n_pos:>8} {n_neg:>8} {overlap_str:>22} {full_coef:>16.4f} {clean_coef:>16.4f}")


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def analyze(
    contexts: tuple[int, ...], kind: str,
) -> tuple[dict[str, pd.DataFrame], dict, dict[str, dict], dict[str, dict | None], dict[str, dict | None]]:
    """Shared pipeline (load -> pool -> overlap check -> regressions) for digit or sector.

    Callers (this module's own main() and the sector companion script) each save their own
    figures afterwards via their own output_dir(...) call, since anal_paths.output_dir keys
    its run manifest off the immediate caller's filename.
    """

    print("Reminder: gate[..., i, j] / W[i, j] convention = src j -> dst i (row=dst, col=src).")
    print(f"kind={kind}  exclude_diagonal={EXCLUDE_DIAGONAL}  strict_fraction={STRICT_FRACTION}\n")

    all_contexts = load_all_contexts(contexts, kind=kind)
    pooled, meta = collect_pooled_records(all_contexts, exclude_diagonal=EXCLUDE_DIAGONAL)

    overlap_stats: dict[str, dict] = {}
    regressions_full: dict[str, dict | None] = {}
    regressions_clean: dict[str, dict | None] = {}
    for g in GROUP_NAMES:
        df = pooled[g]
        assert len(df) > 0, f"group {g} has zero pooled (i, j, {kind}) connections -- check T/R masks"

        x_pos = df.loc[df["signpos"] == 1, "absW"].to_numpy()
        x_neg = df.loc[df["signpos"] == 0, "absW"].to_numpy()
        stats = compute_overlap_band(x_pos, x_neg)
        low, high = check_overlap(g, stats, x_pos, x_neg)
        overlap_stats[g] = stats

        print(f"--- regression [{g}] full range ---")
        regressions_full[g] = run_ols(df, f"{g} full")

        if np.isfinite(low) and np.isfinite(high) and high > low:
            clean_df = df[(df["absW"] >= low) & (df["absW"] <= high)]
            print(f"--- regression [{g}] overlap-only (clean) ---")
            regressions_clean[g] = run_ols(clean_df, f"{g} clean")
        else:
            print(f"--- regression [{g}] overlap-only skipped: no overlap band ---")
            regressions_clean[g] = None

    print_summary_table(pooled, overlap_stats, regressions_full, regressions_clean, meta, kind=kind)
    return pooled, meta, overlap_stats, regressions_full, regressions_clean


DELTA_Y_LABEL = "delta gate open fraction\n(vs. this connection's own cross-{kind} grand mean)"
DELTA_TITLE = (
    "Recurrent gate delta-g vs. |W|, split by sign(W) -- disinhibition check "
    "(per-connection, cross-{kind} demeaned)"
)


def main() -> None:
    pooled, _meta, overlap_stats, _regressions_full, _regressions_clean = analyze(DIGITS, "digit")

    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    render_figure(pooled, overlap_stats, fig_dir / "recurrent_gate_sign_vs_magnitude_disinhibition.png")
    render_figure_zoomed(
        pooled, overlap_stats,
        fig_dir / "recurrent_gate_sign_vs_magnitude_disinhibition_zoom.png",
    )
    render_figure_zoomed(
        pooled, overlap_stats,
        fig_dir / "recurrent_gate_sign_vs_magnitude_disinhibition_delta_zoom.png",
        y_col="delta_of", y_label=DELTA_Y_LABEL.format(kind="digit"), title=DELTA_TITLE.format(kind="digit"),
    )


if __name__ == "__main__":
    main()
