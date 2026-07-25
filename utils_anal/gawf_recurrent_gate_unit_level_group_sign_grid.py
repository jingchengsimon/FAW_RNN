"""Unit-level (source-averaged) recurrent gate / W / g*W distributions, small-multiples grid.

Companion to gawf_recurrent_gate_raw_group_sign_grid.py (the connection-level 3x4 grid, left
untouched by this module -- both scripts/figures coexist). Same digit, same T/R groups, same
3x4 layout (rows={g,W,g*W}, cols={TT,TR,RT,RR}), but every source unit's outgoing connections
within a group are first averaged over the destination axis ("what gate does this unit emit,
on average, to this destination group") before histogramming. So this grid's population is
(sample, source-unit) pairs, not (sample, i, j) connections -- no time-averaging is introduced,
only the destination axis is collapsed per the source-oriented convention already used
elsewhere in this repo (e.g. gawf_recurrent_group_gate_distributions.py's mean_h G[h, i]).

Panel n/mean (title) use the unsigned reduction -- average over ALL destinations in the
group's destination set, regardless of W's sign. The +/- overlay curves then further split by
sign: for source unit i, the "+" value averages only over destinations j with W[i,j]>0 within
the group, and "-" only over W[i,j]<0. A unit with no connection of a given sign in that group
simply doesn't contribute a point to that sign's distribution (not an error).
"""

from __future__ import annotations

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
from utils_anal.gawf_recurrent_gate_raw_group_sign_grid import (
    DIGIT,
    DPI,
    GROUP_NAMES,
    ROW_NAMES,
    RNG_SEED,
    group_masks,
    load_data,
    panel_hist,
    prepare,
    print_connection_table,
    subsample_sample_indices,
)

CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem


# --------------------------------------------------------------------------------------------
# Unit-level (source-averaged) reduction
# --------------------------------------------------------------------------------------------
def unit_level_group_values(g: np.ndarray, W: np.ndarray, sample_idx: np.ndarray,
                             dst_idx: np.ndarray, src_idx: np.ndarray) -> dict:
    """Source-averaged g / W / g*W for one group: each source unit emits one averaged value
    per sample (or, for W, one static value), reduced over the group's destination axis."""

    g_block = g[np.ix_(sample_idx, dst_idx, src_idx)]  # (n_used, n_dst, n_src)
    W_block = W[np.ix_(dst_idx, src_idx)]  # (n_dst, n_src)
    gW_block = g_block * W_block[None, :, :]

    # Unsigned: average over ALL destinations in the group (panel n/mean statistic).
    all_g = g_block.mean(axis=1).reshape(-1)
    all_W = W_block.mean(axis=0).reshape(-1)
    all_gW = gW_block.mean(axis=1).reshape(-1)

    pos_conn, neg_conn = W_block > 0, W_block < 0
    count_pos = pos_conn.sum(axis=0).astype(np.float64)
    count_neg = neg_conn.sum(axis=0).astype(np.float64)
    has_pos, has_neg = count_pos > 0, count_neg > 0

    def reduce_sign(block: np.ndarray, conn_mask: np.ndarray, count: np.ndarray, has: np.ndarray) -> np.ndarray:
        if block.ndim == 3:
            summed = np.einsum("sdc,dc->sc", block, conn_mask)
            return (summed[:, has] / count[None, has]).reshape(-1)
        summed = np.einsum("dc,dc->c", block, conn_mask)
        return (summed[has] / count[has]).reshape(-1)

    return {
        "all_g": all_g, "all_W": all_W, "all_gW": all_gW,
        "g_pos": reduce_sign(g_block, pos_conn, count_pos, has_pos),
        "g_neg": reduce_sign(g_block, neg_conn, count_neg, has_neg),
        "W_pos": reduce_sign(W_block, pos_conn, count_pos, has_pos),
        "W_neg": reduce_sign(W_block, neg_conn, count_neg, has_neg),
        "gW_pos": reduce_sign(gW_block, pos_conn, count_pos, has_pos),
        "gW_neg": reduce_sign(gW_block, neg_conn, count_neg, has_neg),
        "n_src": int(len(src_idx)),
        "n_src_with_pos": int(has_pos.sum()),
        "n_src_with_neg": int(has_neg.sum()),
    }


# --------------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------------
def render_unit_level_grid(prepared: dict, fig_path: Path, dpi: int = DPI) -> None:
    H, g, W = prepared["H"], prepared["g"], prepared["W"]
    T, R, n_samples = prepared["T"], prepared["R"], prepared["n_samples"]
    global_mean_gate = prepared["global_mean_gate"]
    digit = prepared.get("digit", DIGIT)
    context_label = prepared.get("context_label", f"digit {digit}")

    groups = group_masks(T, R)
    print_connection_table(groups, H)
    rng = np.random.default_rng(RNG_SEED)

    g_bins = np.linspace(0.0, 1.0, 61)
    per_group: dict[str, dict] = {}
    wgw_abs_samples: list[np.ndarray] = []
    for name, (src_mask, dst_mask) in groups.items():
        dst_idx, src_idx = np.where(dst_mask)[0], np.where(src_mask)[0]
        n_dst, n_src = dst_idx.size, src_idx.size
        sample_idx = subsample_sample_indices(n_samples, n_dst, n_src, rng)
        print(f"  [{name}] source-averaging over {n_dst} destinations, using "
              f"{sample_idx.size}/{n_samples} samples -> {sample_idx.size * n_src} "
              f"(sample, source-unit) points per sign-unrestricted row")

        values = unit_level_group_values(g, W, sample_idx, dst_idx, src_idx)
        print(f"    n_src={values['n_src']}  n_src_with_pos_conn={values['n_src_with_pos']}  "
              f"n_src_with_neg_conn={values['n_src_with_neg']}")
        per_group[name] = values
        wgw_abs_samples.append(np.abs(values["all_W"]))
        wgw_abs_samples.append(np.abs(np.concatenate([values["W_pos"], values["W_neg"]])))

    wgw_all_abs = np.concatenate(wgw_abs_samples)
    wgw_max = float(wgw_all_abs.max())
    wgw_lim = 1.05 * wgw_max
    wgw_bins = np.linspace(-wgw_lim, wgw_lim, 61)
    wgw_view_lim = min(wgw_lim, 1.2 * float(np.percentile(wgw_all_abs, 99)))
    print(f"  unit-level W/g*W bins span +/-{wgw_lim:.3f} (true max); view cropped to "
          f"+/-{wgw_view_lim:.3f} (99th percentile)")

    fig, axes = plt.subplots(3, 4, figsize=(4.3 * 4, 3.5 * 3))
    for col, name in enumerate(GROUP_NAMES):
        d = per_group[name]
        panel_hist(axes[0, col], d["g_pos"], d["g_neg"], g_bins, global_mean_gate,
                   d["all_g"].size, float(d["all_g"].mean()), f"{name} — g (unit-level)",
                   "source-averaged recurrent gate")
        axes[0, col].set_xlim(0.0, 1.0)
        panel_hist(axes[1, col], d["W_pos"], d["W_neg"], wgw_bins, 0.0,
                   d["all_W"].size, float(d["all_W"].mean()), f"{name} — W (unit-level)",
                   "source-averaged frozen weight")
        axes[1, col].set_xlim(-wgw_view_lim, wgw_view_lim)
        panel_hist(axes[2, col], d["gW_pos"], d["gW_neg"], wgw_bins, 0.0,
                   d["all_gW"].size, float(d["all_gW"].mean()), f"{name} — g*W (unit-level)",
                   "source-averaged effective weight")
        axes[2, col].set_xlim(-wgw_view_lim, wgw_view_lim)
    for row in range(3):
        axes[row, 0].set_ylabel(f"density ({ROW_NAMES[row]}, unit-level)", fontsize=9)

    fig.suptitle(
        f"{context_label.capitalize()} unit-level (source-averaged) distributions, 3x4 grid — "
        f"n_top={int(T.sum())}, n_rem={int(R.sum())}, n_samples={n_samples}\n"
        f"each point = one source unit's mean gate/weight emitted to the group's destination set",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main() -> None:
    data = load_data()
    prepared = prepare(data["W"], data["gate_raw"], data["act"], data["T"])
    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    render_unit_level_grid(prepared, fig_dir / f"digit{DIGIT}_unit_level_gate_W_gW_group_sign_grid.png")


if __name__ == "__main__":
    main()
