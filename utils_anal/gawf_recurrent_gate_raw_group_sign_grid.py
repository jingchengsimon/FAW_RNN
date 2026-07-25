"""Raw (no time-averaging) recurrent gate / W / g*W distributions, small-multiples grid.

Single digit (default d=0), 3x4 grid: rows = {g, W, g*W}, columns = {TT, TR, RT, RR}. Every
panel is a raw-value histogram over (sample, i, j) triples -- no averaging over samples/time,
no mean-bar-chart summary. Uses the real, checkpoint-verified digit-0 cache produced by
gawf_recurrent_gate_single_digit_diagnostic_collect.py.

Convention: gate_raw[..., i, j] = src j -> dst i (row=dst, col=src), same for W[i, j].
Groups are named src->dst: TT=(src in T, dst in T), TR=(src in T, dst in R),
RT=(src in R, dst in T), RR=(src in R, dst in R), with T = top-10% tuned, R = the rest.

Note on the tuned set: the task description's "T = top-10% by mean-over-samples activation"
is the naive reconstruction, which the earlier single-digit diagnostic (Check3) showed gives
TT (mean) = 0.292, not the ~0.238 target. The set that reproduces 0.238 exactly is T_old from
the cache (FDR-selective top-10% among eligible units, from part1_selectivity.npz). Since the
task's own intent is explicitly "that reproduces 0.238", T_old is used here by default; the
naive activation-based reconstruction is used only as a fallback when T_old is unavailable.
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

CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem
DIGIT = 0
TOP_FRACTION = 0.10
DPI = 150
POS_COLOR = "#1d4ed8"
NEG_COLOR = "#dc2626"
GROUP_NAMES = ("TT", "TR", "RT", "RR")  # naming: src->dst
ROW_NAMES = ("g", "W", "gW")
MAX_VALUES = int(2e7)
TARGET_VALUES = int(5e6)
RNG_SEED = 0

def cache_path_for_context(index: int, kind: str = "digit") -> Path:
    """Location of the real gate/act cache for one digit or sector context. All collection
    scripts (single-digit, multi-digit, sector) write the same {kind}{index}_gate_act_cache.npz
    schema into this shared directory."""

    return (
        PROJECT_ROOT / "results" / "anal_data" / "E_relevance_alignment"
        / "gawf_recurrent_gate_single_digit_diagnostic_collect" / f"{kind}{index}_gate_act_cache.npz"
    )


def cache_path_for_digit(digit: int) -> Path:
    """Backward-compatible alias for cache_path_for_context(digit, kind='digit')."""

    return cache_path_for_context(digit, kind="digit")


REAL_DATA_CACHE = cache_path_for_digit(DIGIT)  # kept for backward compat; default digit=0


# --------------------------------------------------------------------------------------------
# DATA LOADING -- replace this section with your actual W / gate_raw / act / T.
# --------------------------------------------------------------------------------------------
def load_data(digit: int = DIGIT, *, kind: str = "digit") -> dict:
    """Return dict with W (H,H), gate_raw (1,n_samples,1,H,H), act (1,n_samples,1,H), T (H,).

    ``digit`` is the context index (a digit 0-9 or, with kind="sector", a sector 0-8);
    ``kind`` selects which cache family to read.
    """

    cache_path = cache_path_for_context(digit, kind=kind)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            W = np.asarray(cached["W"], dtype=np.float64)
            gate_raw = np.asarray(cached["gate"], dtype=np.float64)
            act = np.asarray(cached["act"], dtype=np.float64)
            T = np.asarray(cached["T_old"], dtype=bool)
        note = (
            " for digit 0 this exactly reproduces TT≈0.238, naive top-10%-by-mean-activation "
            "would give TT≈0.292 instead."
            if kind == "digit" and digit == 0 else ""
        )
        print(f"{kind} {digit}: T = T_old（FDR-selective top-10% among eligible units.{note})")
        return {"W": W, "gate_raw": gate_raw, "act": act, "T": T, "digit": digit, "kind": kind}

    # --- small synthetic placeholder so the script runs end-to-end without the real cache ---
    rng = np.random.default_rng(0)
    hidden_size, n_samples = 64, 500
    W = rng.normal(0.0, 0.06, size=(hidden_size, hidden_size)).astype(np.float64)
    logits = rng.normal(0.0, 1.5, size=(1, n_samples, 1, hidden_size, hidden_size))
    gate_raw = 1.0 / (1.0 + np.exp(-logits))
    act = rng.gamma(2.0, 1.0, size=(1, n_samples, 1, hidden_size))
    return {"W": W, "gate_raw": gate_raw, "act": act, "T": None, "digit": digit, "kind": kind}


# --------------------------------------------------------------------------------------------
# Prep
# --------------------------------------------------------------------------------------------
def prepare(W: np.ndarray, gate_raw: np.ndarray, act: np.ndarray, T: np.ndarray | None,
            digit: int = DIGIT, *, kind: str = "digit") -> dict:
    """Validate shapes, drop the pseudo time axis, and resolve the tuned mask T/R.

    ``kind`` is purely cosmetic (used to label prints/titles as "digit N" vs "sector N").
    """

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise AssertionError(f"W must be square (H, H); got {W.shape}")
    hidden_size = W.shape[0]
    if gate_raw.ndim != 5 or gate_raw.shape[0] != 1 or gate_raw.shape[2] != 1 or \
            gate_raw.shape[3:] != (hidden_size, hidden_size):
        raise AssertionError(
            f"gate_raw must be (1, n_samples, 1, {hidden_size}, {hidden_size}); got {gate_raw.shape}"
        )
    if act.ndim != 4 or act.shape[0] != 1 or act.shape[2] != 1 or act.shape[3] != hidden_size:
        raise AssertionError(f"act must be (1, n_samples, 1, {hidden_size}); got {act.shape}")

    g = gate_raw[0, :, 0]  # (n_samples, H, H); no time/sample averaging
    act_flat = act[0, :, 0]  # (n_samples, H)
    n_samples = g.shape[0]

    if T is None:
        n_top = int(round(TOP_FRACTION * hidden_size))
        mean_act = act_flat.mean(axis=0)
        T = np.zeros(hidden_size, dtype=bool)
        T[np.argsort(-mean_act)[:n_top]] = True
        print(f"T reconstructed as naive top-{TOP_FRACTION:.0%} by mean-over-samples activation "
              f"(no T_old available).")
    T = np.asarray(T, dtype=bool)
    if T.shape != (hidden_size,):
        raise AssertionError(f"T must be ({hidden_size},); got {T.shape}")
    R = ~T

    global_mean_gate = float(g.mean())
    context_label = f"{kind} {digit}"
    print(f"H={hidden_size} n_samples={n_samples} n_top={int(T.sum())} n_rem={int(R.sum())}")
    print(f"global mean gate ({context_label}, raw, no time-averaging) = {global_mean_gate:.4f}")

    return {
        "H": hidden_size, "g": g, "W": W, "act_flat": act_flat, "T": T, "R": R,
        "n_samples": n_samples, "global_mean_gate": global_mean_gate, "digit": digit,
        "kind": kind, "context_label": context_label,
    }


def group_masks(T: np.ndarray, R: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {group_name: (src_mask, dst_mask)} for TT/TR/RT/RR (naming: src->dst)."""

    return {"TT": (T, T), "TR": (T, R), "RT": (R, T), "RR": (R, R)}


def print_connection_table(groups: dict[str, tuple[np.ndarray, np.ndarray]], hidden_size: int) -> dict[str, int]:
    """Print each group's connection count and assert the split isn't degenerate."""

    conn_counts: dict[str, int] = {}
    print("\ngroup connection counts (n_dst * n_src):")
    for name, (src_mask, dst_mask) in groups.items():
        n_dst, n_src = int(dst_mask.sum()), int(src_mask.sum())
        if (n_dst, n_src) == (hidden_size, hidden_size):
            raise AssertionError(f"{name}: block shape degenerated to full (H, H) -- mask not applied")
        conn_counts[name] = n_dst * n_src
        print(f"  {name}: n_dst={n_dst} n_src={n_src} n_connections={conn_counts[name]}")
    if conn_counts["TT"] == conn_counts["RR"]:
        raise AssertionError("TT and RR connection counts are equal -- tuned/remaining split looks degenerate")
    if conn_counts["TR"] != conn_counts["RT"]:
        raise AssertionError(
            "TR and RT connection counts differ (should always match: n_top*n_rem either way) -- mask bug"
        )
    print(f"  check: TT smallest={conn_counts['TT'] == min(conn_counts.values())}, "
          f"RR largest={conn_counts['RR'] == max(conn_counts.values())}")
    return conn_counts


# --------------------------------------------------------------------------------------------
# Sub-sampling for large blocks
# --------------------------------------------------------------------------------------------
def subsample_sample_indices(n_samples: int, n_dst: int, n_src: int, rng: np.random.Generator) -> np.ndarray:
    """Return sample indices to use so n_used * n_dst * n_src stays near TARGET_VALUES."""

    total = n_samples * n_dst * n_src
    if total <= MAX_VALUES:
        return np.arange(n_samples)
    n_sub = max(1, min(n_samples, TARGET_VALUES // max(1, n_dst * n_src)))
    idx = rng.choice(n_samples, size=n_sub, replace=False)
    idx.sort()
    return idx


# --------------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------------
def panel_hist(axis: plt.Axes, pos_vals: np.ndarray, neg_vals: np.ndarray, bins: np.ndarray,
               vline: float, n_total: int, mean_total: float, title: str, xlabel: str) -> None:
    axis.hist(pos_vals, bins=bins, density=True, histtype="step", color=POS_COLOR, linewidth=1.4,
              label=f"+ (n={pos_vals.size}, mean={pos_vals.mean() if pos_vals.size else float('nan'):.3f})")
    axis.hist(neg_vals, bins=bins, density=True, histtype="step", color=NEG_COLOR, linewidth=1.4,
              label=f"- (n={neg_vals.size}, mean={neg_vals.mean() if neg_vals.size else float('nan'):.3f})")
    axis.axvline(vline, color="gray", linestyle="--", linewidth=1.0)
    axis.set_title(f"{title}\nn={n_total}, mean={mean_total:.4f}", fontsize=9.5)
    axis.set_xlabel(xlabel, fontsize=8.5)
    axis.tick_params(labelsize=7.5)
    axis.legend(frameon=False, fontsize=6.8, loc="upper right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def render_grid(prepared: dict, fig_path: Path, dpi: int = DPI) -> None:
    H, g, W = prepared["H"], prepared["g"], prepared["W"]
    T, R, n_samples = prepared["T"], prepared["R"], prepared["n_samples"]
    global_mean_gate = prepared["global_mean_gate"]
    digit = prepared.get("digit", DIGIT)
    context_label = prepared.get("context_label", f"digit {digit}")

    groups = group_masks(T, R)
    conn_counts = print_connection_table(groups, H)
    rng = np.random.default_rng(RNG_SEED)

    g_bins = np.linspace(0.0, 1.0, 61)
    per_group: dict[str, dict] = {}
    wgw_extent = 0.0
    wgw_abs_samples: list[np.ndarray] = []
    for name, (src_mask, dst_mask) in groups.items():
        dst_idx, src_idx = np.where(dst_mask)[0], np.where(src_mask)[0]
        n_dst, n_src = dst_idx.size, src_idx.size
        sample_idx = subsample_sample_indices(n_samples, n_dst, n_src, rng)
        n_used = sample_idx.size * n_dst * n_src
        print(f"  [{name}] using {sample_idx.size}/{n_samples} samples -> {n_used} values "
              f"(g and g*W rows)")

        g_block = g[np.ix_(sample_idx, dst_idx, src_idx)]
        W_block = W[np.ix_(dst_idx, src_idx)]
        gW_block = g_block * W_block[None, :, :]

        pos_conn, neg_conn = W_block > 0, W_block < 0
        per_group[name] = {
            "g_pos": g_block[:, pos_conn].reshape(-1), "g_neg": g_block[:, neg_conn].reshape(-1),
            "g_n": g_block.size, "g_mean": float(g_block.mean()),
            "W_pos": W_block[pos_conn], "W_neg": W_block[neg_conn],
            "W_n": W_block.size, "W_mean": float(W_block.mean()),
            "gW_pos": gW_block[:, pos_conn].reshape(-1), "gW_neg": gW_block[:, neg_conn].reshape(-1),
            "gW_n": gW_block.size, "gW_mean": float(gW_block.mean()),
        }
        wgw_extent = max(wgw_extent, float(np.abs(W_block).max()), float(np.abs(gW_block).max()))
        wgw_abs_samples.append(np.abs(W_block).reshape(-1))

    # Bins span the true full extent (density=True needs this for a correct normalization),
    # but the displayed view is cropped to a robust percentile so the bulk of the mass is
    # readable instead of being squashed by a handful of outlier |W| connections.
    wgw_lim = 1.05 * wgw_extent
    wgw_bins = np.linspace(-wgw_lim, wgw_lim, 61)
    wgw_view_lim = 1.2 * float(np.percentile(np.concatenate(wgw_abs_samples), 99))
    wgw_view_lim = min(wgw_view_lim, wgw_lim)
    print(f"  W/g*W bins span +/-{wgw_lim:.2f} (true max); view cropped to +/-{wgw_view_lim:.2f} "
          f"(99th percentile of |W|) for readability")

    fig, axes = plt.subplots(3, 4, figsize=(4.3 * 4, 3.5 * 3))
    for col, name in enumerate(GROUP_NAMES):
        d = per_group[name]
        panel_hist(axes[0, col], d["g_pos"], d["g_neg"], g_bins, global_mean_gate,
                   d["g_n"], d["g_mean"], f"{name} — g (raw)", "recurrent gate g")
        axes[0, col].set_xlim(0.0, 1.0)
        panel_hist(axes[1, col], d["W_pos"], d["W_neg"], wgw_bins, 0.0,
                   d["W_n"], d["W_mean"], f"{name} — W", "frozen weight W")
        axes[1, col].set_xlim(-wgw_view_lim, wgw_view_lim)
        panel_hist(axes[2, col], d["gW_pos"], d["gW_neg"], wgw_bins, 0.0,
                   d["gW_n"], d["gW_mean"], f"{name} — g*W", "effective weight g*W")
        axes[2, col].set_xlim(-wgw_view_lim, wgw_view_lim)
    for row in range(3):
        axes[row, 0].set_ylabel(f"density ({ROW_NAMES[row]})", fontsize=9)

    fig.suptitle(
        f"{context_label.capitalize()} raw distributions (no time-averaging), 3x4 grid — "
        f"n_top={int(T.sum())}, n_rem={int(R.sum())}, n_samples={n_samples}\n"
        f"connection counts: TT={conn_counts['TT']} TR={conn_counts['TR']} "
        f"RT={conn_counts['RT']} RR={conn_counts['RR']}",
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
    data = load_data(DIGIT)
    prepared = prepare(data["W"], data["gate_raw"], data["act"], data["T"], digit=DIGIT)
    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    render_grid(prepared, fig_dir / f"digit{DIGIT}_raw_gate_W_gW_group_sign_grid.png")


if __name__ == "__main__":
    main()
