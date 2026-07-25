"""Four-group x sign diagnostics for the frozen recurrent gate, plus a 2x2 effective-weight
spectral summary.

Convention (edit the DATA LOADING section below to match your variables):
- W[i, j]      : frozen recurrent weight, source unit j -> target unit i (h_i = sum_j W[i,j] h_j).
- G[d, i, j]    : recurrent gate in (0, 1), digit-conditioned mean over trials/timesteps,
                  aligned to W's [i, j] convention.
- tuning[d, u]  : per-digit tuning score for unit u (e.g. mean activation), used to pick the
                  top-10% "tuned" set T_d per digit; the rest is R_d = complement of T_d.

If W[i, j] actually means i -> j (target-first is source in your code), transpose W (and G)
before calling anything below -- everything here assumes row=target(dst), col=source(src).

Eight cells = (src group, dst group) in {T, R}^2, further split by sign(W[i, j]):
    TT = src in T, dst in T   |   TR = src in T, dst in R
    RT = src in R, dst in T   |   RR = src in R, dst in R
Because T_d / R_d are per-digit, the same (i, j) connection can land in different cells for
different digits -- all pooled statistics below pool over (i, j, d) triples, not just (i, j).
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


GROUP_NAMES = ("TT", "TR", "RT", "RR")
SIGN_NAMES = ("+", "-")
CELL_NAMES = tuple(f"{group}{sign}" for group in GROUP_NAMES for sign in SIGN_NAMES)
POS_COLOR = "#1d4ed8"
NEG_COLOR = "#dc2626"
CELL_COLOR = {cell: (POS_COLOR if cell.endswith("+") else NEG_COLOR) for cell in CELL_NAMES}


# --------------------------------------------------------------------------------------------
# DATA LOADING -- replace this section with your actual W / G / tuning arrays.
# --------------------------------------------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (W, G, tuning). Replace the body with your real arrays.

    W       : (H, H) float, frozen recurrent weight, W[i, j] = source j -> target i.
    G       : (D, H, H) float in (0, 1), digit-conditioned mean recurrent gate.
    tuning  : (D, H) float, per-digit tuning score used to define the top-10% set.
    """

    # --- demo data so the script runs end-to-end; delete once real arrays are wired in ---
    rng = np.random.default_rng(0)
    hidden_size, num_digits = 256, 10
    W = rng.normal(0.0, 0.06, size=(hidden_size, hidden_size)).astype(np.float64)
    tuning = rng.gamma(2.0, 1.0, size=(num_digits, hidden_size)).astype(np.float64)
    logits = rng.normal(0.0, 1.2, size=(num_digits, hidden_size, hidden_size))
    G = 1.0 / (1.0 + np.exp(-logits))
    return W, G.astype(np.float64), tuning


def reduce_raw_gate_to_conditional_mean(raw_gate: np.ndarray) -> np.ndarray:
    """Collapse raw (D, n_trials, T, H, H) gates to the digit-conditioned mean G (D, H, H)."""

    raw_gate = np.asarray(raw_gate, dtype=np.float64)
    if raw_gate.ndim != 5:
        raise ValueError(f"raw_gate must be 5-D (D, n_trials, T, H, H); got shape {raw_gate.shape}")
    return raw_gate.mean(axis=(1, 2))


# Figures land under results/anal_figs/E_relevance_alignment/ (the project convention keeps
# figs flat per category and data under results/anal_data/<category>/<script_name>/); this
# script produces no data artifacts, so only the figs root is requested.
CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem
DPI = 150
TOP_FRACTION = 0.10


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------
def validate_inputs(W: np.ndarray, G: np.ndarray, tuning: np.ndarray) -> None:
    """Shape/range asserts for W, G, tuning."""

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise AssertionError(f"W must be square (H, H); got {W.shape}")
    hidden_size = W.shape[0]
    if G.ndim != 3 or G.shape[1:] != (hidden_size, hidden_size):
        raise AssertionError(f"G must be (D, {hidden_size}, {hidden_size}); got {G.shape}")
    num_digits = G.shape[0]
    if tuning.shape != (num_digits, hidden_size):
        raise AssertionError(f"tuning must be ({num_digits}, {hidden_size}); got {tuning.shape}")
    if not np.isfinite(W).all():
        raise AssertionError("W contains non-finite values")
    if not np.isfinite(G).all():
        raise AssertionError("G contains non-finite values")
    if G.min() < -1e-6 or G.max() > 1.0 + 1e-6:
        print(
            f"  [warn] G is expected in (0, 1) but observed range "
            f"[{G.min():.4f}, {G.max():.4f}]"
        )


# --------------------------------------------------------------------------------------------
# Precompute: effective weight, top/remaining masks, sign masks
# --------------------------------------------------------------------------------------------
def compute_effective_weights(W: np.ndarray, G: np.ndarray) -> np.ndarray:
    """E[d] = G[d] * W elementwise; since g > 0, sign(E[d]) == sign(W) wherever W != 0."""

    E = G * W[None, :, :]
    nonzero = W != 0
    if not np.array_equal(np.sign(E)[:, nonzero], np.broadcast_to(np.sign(W)[nonzero], E[:, nonzero].shape)):
        raise AssertionError("sign(E) != sign(W) somewhere G <= 0 was found; check G's range")
    return E


def compute_tuned_masks(tuning: np.ndarray, frac: float = TOP_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Per-digit top-`frac` tuned mask T (D, H) and its complement R (D, H)."""

    num_digits, hidden_size = tuning.shape
    top_k = max(1, int(round(frac * hidden_size)))
    order = np.argsort(-tuning, axis=1)
    T = np.zeros((num_digits, hidden_size), dtype=bool)
    rows = np.repeat(np.arange(num_digits), top_k)
    cols = order[:, :top_k].reshape(-1)
    T[rows, cols] = True
    R = ~T
    return T, R


def sign_masks(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (H, H) masks for W > 0 and W < 0 (zeros excluded from both)."""

    return W > 0, W < 0


def group_mask(src_mask: np.ndarray, dst_mask: np.ndarray) -> np.ndarray:
    """(H, H) mask[i, j] = dst_mask[i] & src_mask[j] (row = target/dst, col = source/src)."""

    return dst_mask[:, None] & src_mask[None, :]


# --------------------------------------------------------------------------------------------
# Pooling across digits
# --------------------------------------------------------------------------------------------
def accumulate_cell_stats(
    W: np.ndarray, G: np.ndarray, E: np.ndarray, T: np.ndarray, R: np.ndarray
) -> dict:
    """Pool (i, j, d) connections into the 4 groups and 8 sign-split cells.

    Returns a dict with:
      group_pooled_g   : {group -> 1-D array} pooled gate values, sign-agnostic (for TT vs RR).
      cell_pooled_g/w/gw : {cell -> 1-D array} pooled gate / weight / effective-weight values.
      per_digit_cell_mean_g/w/gw : (D, 8) per-digit-then-cell mean (nan if a cell is empty).
      per_digit_cell_count : (D, 8) int connection counts.
    """

    num_digits = G.shape[0]
    pos_mask, neg_mask = sign_masks(W)
    sign_lookup = {"+": pos_mask, "-": neg_mask}

    group_pooled_g = {g: [] for g in GROUP_NAMES}
    cell_pooled_g = {c: [] for c in CELL_NAMES}
    cell_pooled_w = {c: [] for c in CELL_NAMES}
    cell_pooled_gw = {c: [] for c in CELL_NAMES}
    per_digit_cell_mean_g = np.full((num_digits, len(CELL_NAMES)), np.nan)
    per_digit_cell_mean_w = np.full((num_digits, len(CELL_NAMES)), np.nan)
    per_digit_cell_mean_gw = np.full((num_digits, len(CELL_NAMES)), np.nan)
    per_digit_cell_count = np.zeros((num_digits, len(CELL_NAMES)), dtype=np.int64)

    for d in range(num_digits):
        Td, Rd = T[d], R[d]
        groups = {"TT": group_mask(Td, Td), "TR": group_mask(Td, Rd),
                  "RT": group_mask(Rd, Td), "RR": group_mask(Rd, Rd)}
        for group_name, gmask in groups.items():
            group_pooled_g[group_name].append(G[d][gmask])
            for sign_name, smask in sign_lookup.items():
                cell = f"{group_name}{sign_name}"
                ci = CELL_NAMES.index(cell)
                cmask = gmask & smask
                vg, vw, vgw = G[d][cmask], W[cmask], E[d][cmask]
                cell_pooled_g[cell].append(vg)
                cell_pooled_w[cell].append(vw)
                cell_pooled_gw[cell].append(vgw)
                per_digit_cell_count[d, ci] = vg.size
                if vg.size:
                    per_digit_cell_mean_g[d, ci] = vg.mean()
                    per_digit_cell_mean_w[d, ci] = vw.mean()
                    per_digit_cell_mean_gw[d, ci] = vgw.mean()

    group_pooled_g = {g: np.concatenate(v) for g, v in group_pooled_g.items()}
    cell_pooled_g = {c: np.concatenate(v) for c, v in cell_pooled_g.items()}
    cell_pooled_w = {c: np.concatenate(v) for c, v in cell_pooled_w.items()}
    cell_pooled_gw = {c: np.concatenate(v) for c, v in cell_pooled_gw.items()}

    return {
        "group_pooled_g": group_pooled_g,
        "cell_pooled_g": cell_pooled_g,
        "cell_pooled_w": cell_pooled_w,
        "cell_pooled_gw": cell_pooled_gw,
        "per_digit_cell_mean_g": per_digit_cell_mean_g,
        "per_digit_cell_mean_w": per_digit_cell_mean_w,
        "per_digit_cell_mean_gw": per_digit_cell_mean_gw,
        "per_digit_cell_count": per_digit_cell_count,
    }


def mean_sem_across_digits(per_digit_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise mean and SEM (ddof=1) across the digit axis (axis 0)."""

    num_digits = per_digit_matrix.shape[0]
    mean = np.nanmean(per_digit_matrix, axis=0)
    std = np.nanstd(per_digit_matrix, axis=0, ddof=1)
    sem = std / np.sqrt(num_digits)
    return mean, sem


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-variance Cohen's d, x minus y."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled_std)


# --------------------------------------------------------------------------------------------
# Printed summary
# --------------------------------------------------------------------------------------------
def print_summary(W: np.ndarray, G: np.ndarray, tuning: np.ndarray, stats: dict) -> None:
    hidden_size = W.shape[0]
    num_digits = G.shape[0]
    print(f"W: {W.shape}, G: {G.shape}, tuning: {tuning.shape} (H={hidden_size}, D={num_digits})")
    print(f"global mean gate = {G.mean():.4f}")

    mean_g, _ = mean_sem_across_digits(stats["per_digit_cell_mean_g"])
    mean_w, _ = mean_sem_across_digits(stats["per_digit_cell_mean_w"])
    mean_gw, _ = mean_sem_across_digits(stats["per_digit_cell_mean_gw"])
    print(f"\n{'cell':>5} {'n':>10} {'mean_g':>10} {'mean_W':>10} {'mean_gW':>10}")
    for ci, cell in enumerate(CELL_NAMES):
        n = stats["cell_pooled_g"][cell].size
        print(f"{cell:>5} {n:>10d} {mean_g[ci]:>10.4f} {mean_w[ci]:>10.5f} {mean_gw[ci]:>10.5f}")

    d_tt_rr = cohens_d(stats["group_pooled_g"]["TT"], stats["group_pooled_g"]["RR"])
    d_ttpm = cohens_d(stats["cell_pooled_g"]["TT+"], stats["cell_pooled_g"]["TT-"])
    print(f"\nCohen's d (gate, pooled over connections):")
    print(f"  TT vs RR   = {d_tt_rr:.3f}")
    print(f"  TT+ vs TT- = {d_ttpm:.3f}")


# --------------------------------------------------------------------------------------------
# Figure 1: gate mechanism, 4 groups x sign
# --------------------------------------------------------------------------------------------
def plot_gate_mechanism(stats: dict, global_mean_gate: float, out_path: Path, dpi: int = DPI) -> None:
    mean_g, sem_g = mean_sem_across_digits(stats["per_digit_cell_mean_g"])
    x = np.arange(len(GROUP_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for sign_idx, sign in enumerate(SIGN_NAMES):
        offsets = x + (width / 2 if sign_idx == 0 else -width / 2)
        color = POS_COLOR if sign == "+" else NEG_COLOR
        heights, errs, ns = [], [], []
        for group in GROUP_NAMES:
            ci = CELL_NAMES.index(f"{group}{sign}")
            heights.append(mean_g[ci])
            errs.append(sem_g[ci])
            ns.append(stats["cell_pooled_g"][f"{group}{sign}"].size)
        bars = ax.bar(offsets, heights, width, yerr=errs, capsize=3, color=color, alpha=0.9,
                       label=f"sign {sign}", edgecolor="black", linewidth=0.5)
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={n}",
                    ha="center", va="bottom", fontsize=7)
    ax.axhline(global_mean_gate, color="gray", linestyle="--", linewidth=1.0,
               label=f"global mean gate = {global_mean_gate:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_NAMES)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean recurrent gate g[i, j]")
    ax.set_xlabel("group (src -> dst)")
    ax.set_title("Recurrent gate by source/destination group and weight sign\n"
                  "(mean +/- SEM across digits; TT+ vs TT- = self-excite gating)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------------------------
# Figure 2: weight vs effective weight, 8 cells
# --------------------------------------------------------------------------------------------
def plot_weight_vs_effective(stats: dict, out_path: Path, dpi: int = DPI) -> None:
    mean_w, sem_w = mean_sem_across_digits(stats["per_digit_cell_mean_w"])
    mean_gw, sem_gw = mean_sem_across_digits(stats["per_digit_cell_mean_gw"])
    x = np.arange(len(CELL_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.bar(x - width / 2, mean_w, width, yerr=sem_w, capsize=3,
           color=[CELL_COLOR[c] for c in CELL_NAMES], alpha=0.4, hatch="//",
           edgecolor="black", linewidth=0.5, label="W")
    ax.bar(x + width / 2, mean_gw, width, yerr=sem_gw, capsize=3,
           color=[CELL_COLOR[c] for c in CELL_NAMES], alpha=0.95,
           edgecolor="black", linewidth=0.5, label="g*W")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_NAMES)
    ax.set_ylabel("mean weight (same units as W)")
    ax.set_xlabel("cell (group x sign)")
    ax.set_title("Frozen weight W vs. gated effective weight g*W, per cell\n"
                  "(mean +/- SEM across digits; |g*W| < |W| = suppression)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------------------------
# Figure 3 (optional): decay ratio
# --------------------------------------------------------------------------------------------
def plot_decay_ratio(stats: dict, out_path: Path, dpi: int = DPI, eps: float = 1e-8) -> None:
    mean_w, _ = mean_sem_across_digits(stats["per_digit_cell_mean_w"])
    mean_gw, _ = mean_sem_across_digits(stats["per_digit_cell_mean_gw"])
    ratio = mean_gw / np.where(np.abs(mean_w) < eps, np.nan, mean_w)
    x = np.arange(len(CELL_NAMES))
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.bar(x, ratio, color=[CELL_COLOR[c] for c in CELL_NAMES], alpha=0.9,
           edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="no decay (ratio=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_NAMES)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mean(g*W) / mean(W)  (~|W|-weighted mean gate)")
    ax.set_xlabel("cell (group x sign)")
    ax.set_title("Per-cell decay ratio of the effective weight relative to W")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------------------------
# Dynamics: 2x2 effective-matrix spectrum (net signed, no sign split)
# --------------------------------------------------------------------------------------------
def block_mean_matrix(tensor: np.ndarray, T: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2x2 block-mean matrix in {T, R} order, averaged over digits.

    ``tensor`` is either (D, H, H) (e.g. E, digit-varying) or (H, H) (e.g. W, broadcast over
    digits since only the T/R masks vary by digit). First-order coarse-graining: block-internal
    homogeneity is assumed, i.e. every unit within a block is treated as interchangeable.
    Returns (M, M_per_digit) where M = mean_d M_d and M_per_digit has shape (D, 2, 2).
    """

    num_digits = T.shape[0]
    if tensor.ndim == 2:
        tensor = np.broadcast_to(tensor, (num_digits,) + tensor.shape)
    if tensor.shape[0] != num_digits:
        raise AssertionError("tensor's digit axis must match T/R's digit axis")
    order = ("T", "R")
    M_per_digit = np.zeros((num_digits, 2, 2), dtype=np.float64)
    for d in range(num_digits):
        idx = {"T": T[d], "R": R[d]}
        for ai, a in enumerate(order):
            block_rows = tensor[d][idx[a]]  # (|A|, H): fan-in rows for units in A
            for bi, b in enumerate(order):
                block = block_rows[:, idx[b]]  # (|A|, |B|)
                row_fanin_sums = block.sum(axis=1)  # sum over j in B (fan-in from B)
                M_per_digit[d, ai, bi] = row_fanin_sums.mean()  # mean over i in A
    M = M_per_digit.mean(axis=0)
    return M, M_per_digit


def spectral_summary(M: np.ndarray, label: str) -> None:
    if M.shape != (2, 2):
        raise AssertionError(f"M must be 2x2; got {M.shape}")
    eigvals, eigvecs = np.linalg.eig(M)
    order = np.argsort(-np.abs(eigvals))
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    print(f"\n{label} 2x2 effective matrix (rows/cols = [T, R]):")
    print(f"  {M}")
    print(f"  eigenvalues: {eigvals}")
    print(f"  |lambda_max| = {abs(eigvals[0]):.4f}  "
          f"(>=1 without input -> non-decaying/growing memory; <1 -> leak/decay)")
    print(f"  eigenvector for lambda_max (T, R basis): {np.real_if_close(eigvecs[:, 0])}")


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main() -> None:
    print("Reminder: this script assumes W[i, j] = source j -> target i (row=dst, col=src).")
    print("If your convention is reversed, transpose W and G before calling anything here.\n")

    W, G, tuning = load_data()
    validate_inputs(W, G, tuning)

    E = compute_effective_weights(W, G)
    T, R = compute_tuned_masks(tuning, TOP_FRACTION)
    stats = accumulate_cell_stats(W, G, E, T, R)
    print_summary(W, G, tuning, stats)

    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    plot_gate_mechanism(stats, float(G.mean()), fig_dir / "fig1_gate_group_sign.png")
    plot_weight_vs_effective(stats, fig_dir / "fig2_weight_vs_effective.png")
    plot_decay_ratio(stats, fig_dir / "fig3_decay_ratio.png")

    M_E, _ = block_mean_matrix(E, T, R)
    M_W, _ = block_mean_matrix(W, T, R)
    spectral_summary(M_E, "Effective weight (g*W)")
    spectral_summary(M_W, "Static weight (W)")


if __name__ == "__main__":
    main()
