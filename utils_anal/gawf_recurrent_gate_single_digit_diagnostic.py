"""Single-digit diagnostic: why did the four-group TT gate mean move from the old
analysis's ~0.238 to the new analysis's ~0.5?

Only digit d=0 is processed (kept fast on purpose). Runs Check0-4 from the diagnosis plan and
prints the matched decision branch(es) at the end. numpy + matplotlib only, functionalized,
shape-asserted.

Convention (edit the DATA LOADING section below to match your variables):
- W[i, j]  : frozen recurrent weight, source unit j -> target unit i (row=dst, col=src).
             If your convention is reversed, transpose W (and any raw gate array) first.
- gate     : one of three shapes -- print its shape first and branch:
             * G_avg (D, H, H)                         : trial/timestep-averaged static gate.
             * G_raw (D, n_trials, T_steps, H, H)       : raw per-sample gate.
             * (D, H)                                   : per-unit vector gate (NOT per-connection
               -- the connection-level four-group analysis does not apply to this case).
- act      : unit activation. Ideally per-sample (D, n_trials, T_steps, H); a static (D, H) is
             accepted too (only usable to reconstruct T_old, not for Check4).
- T_new    : new-pipeline top-10% tuned indices/mask for digit d (index array or (H,) bool mask).
- T_old    : old-analysis tuned indices/mask that produced ~0.238, if you still have them. If
             unavailable, this script reconstructs it as "top-10% by mean activation for digit d".
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
TARGET_OLD = 0.238
CLOSE_TOL = 0.03
PEAK_FRAC_THRESHOLD = 0.4  # frac(0.4<g<0.6) above this counts as "spiked at 0.5"
DPI = 150

BRANCH_DESC = {
    "A": "门被时间平均掉 + 用静态集 -> 需要改成逐样本瞬时算法 (Check1 spiked@0.5 & Check4 instant≈0.238)",
    "B": "新 tuning 指标选错了集合 -> 检查 top-10% 排序/激活口径 (Check3: T_old≈0.238, T_new≈0.5)",
    "C": "静态化问题：集合和门都要瞬时 (Check3: T_old也≈0.5, 但 Check4 instant≈0.238)",
    "D": "旧的 0.238 可能才是 artifact，需要回查旧分析 (Check3: T_old也≈0.5, Check4 instant也≈0.5)",
}


# --------------------------------------------------------------------------------------------
# DATA LOADING -- replace this section with your actual W / gate / act / T_new / T_old.
# --------------------------------------------------------------------------------------------
REAL_DATA_CACHE = (
    PROJECT_ROOT / "results" / "anal_data" / "E_relevance_alignment"
    / "gawf_recurrent_gate_single_digit_diagnostic_collect" / f"digit{DIGIT}_gate_act_cache.npz"
)


def load_data() -> dict:
    """Return a dict with keys W, gate, act, T_new, T_old (act/T_old may be None).

    Loads the real digit-0 cache produced by
    ``gawf_recurrent_gate_single_digit_diagnostic_collect.py`` (a separate, torch-dependent
    script kept out of this numpy-only module). Falls back to a small synthetic placeholder if
    the cache hasn't been generated yet, so this script still runs end-to-end on its own.
    """

    if REAL_DATA_CACHE.is_file():
        with np.load(REAL_DATA_CACHE, allow_pickle=False) as cached:
            return {
                "W": np.asarray(cached["W"], dtype=np.float64),
                "gate": np.asarray(cached["gate"], dtype=np.float64),
                "act": np.asarray(cached["act"], dtype=np.float64),
                "T_new": np.asarray(cached["T_new"], dtype=bool),
                "T_old": np.asarray(cached["T_old"], dtype=bool),
            }

    # --- small synthetic placeholder so the script runs end-to-end without the real cache ---
    rng = np.random.default_rng(0)
    hidden_size, num_digits, n_trials, t_steps = 64, 3, 10, 8
    W = rng.normal(0.0, 0.06, size=(hidden_size, hidden_size)).astype(np.float64)
    logits = rng.normal(0.0, 1.5, size=(num_digits, n_trials, t_steps, hidden_size, hidden_size))
    gate = 1.0 / (1.0 + np.exp(-logits))  # G_raw (D, n_trials, T_steps, H, H)
    act = rng.gamma(2.0, 1.0, size=(num_digits, n_trials, t_steps, hidden_size))
    top_k = max(1, int(round(TOP_FRACTION * hidden_size)))
    T_new = np.zeros(hidden_size, dtype=bool)
    T_new[np.argsort(-act[DIGIT].mean(axis=(0, 1)))[:top_k]] = True
    T_old = None  # unavailable -> reconstructed from act inside main()
    return {"W": W, "gate": gate, "act": act, "T_new": T_new, "T_old": T_old}


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
def to_bool_mask(indices_or_mask: np.ndarray, hidden_size: int, *, name: str = "mask") -> np.ndarray:
    """Accept either a (H,) bool mask or an integer index array; return a (H,) bool mask.

    Hardened against the bug class that produces a TT mean near the global mean gate (~0.5):
    a 0/1 mask stored as float/int silently falls through to the "index array" branch and
    degenerates into picking units 0 and 1 only, or -- depending on caller code upstream --
    a mask that never actually restricts anything. Steps 1-3 debugging checks (dtype, resolved
    size, and later the indexed submatrix shape) are printed here and in check3 so a mask that
    silently degraded to "everything" is caught immediately instead of quietly producing ~0.5.
    """

    arr = np.asarray(indices_or_mask)
    looks_like_zero_one_mask = (
        arr.dtype != bool
        and arr.shape == (hidden_size,)
        and arr.size > 0
        and np.array_equal(np.unique(arr), np.array([0, 1]))
    )
    if looks_like_zero_one_mask:
        raise AssertionError(
            f"{name} has dtype {arr.dtype} but looks like a 0/1 boolean mask (shape "
            f"({hidden_size},), values only in {{0, 1}}). Refusing to guess whether this is a "
            f"mask or an index array -- cast it to bool explicitly before calling. Treating it "
            f"as indices here would silently select only unit 0 and unit 1."
        )
    if arr.dtype == bool:
        if arr.shape != (hidden_size,):
            raise AssertionError(f"{name} bool mask must be ({hidden_size},); got {arr.shape}")
        mask = arr
    else:
        mask = np.zeros(hidden_size, dtype=bool)
        mask[arr.astype(np.int64)] = True

    n_true = int(mask.sum())
    print(f"  [mask check] {name}: dtype={arr.dtype}, raw_shape={arr.shape}, "
          f"resolved n_true={n_true}/{hidden_size} ({100 * n_true / hidden_size:.1f}%)")
    if n_true > hidden_size // 2:
        print(
            f"  [mask check][WARN] {name} selects more than half of all units -- this is the "
            f"exact signature of a mask that silently degraded to 'everything', which pulls "
            f"any block mean toward the global mean gate (~0.5)."
        )
    return mask


def reconstruct_top_by_activation(act: np.ndarray, d: int, frac: float, hidden_size: int) -> np.ndarray:
    """Fallback T_old: top-`frac` units by mean activation for digit d."""

    if act is None:
        raise ValueError("T_old is missing and no activation data is available to reconstruct it")
    act = np.asarray(act)
    if act.ndim == 4:  # (D, n_trials, T_steps, H)
        scores = act[d].mean(axis=(0, 1))
    elif act.ndim == 2:  # (D, H), static only
        scores = act[d]
    else:
        raise AssertionError(f"act must be (D, n_trials, T_steps, H) or (D, H); got {act.shape}")
    if scores.shape != (hidden_size,):
        raise AssertionError(f"activation scores must be ({hidden_size},); got {scores.shape}")
    top_k = max(1, int(round(frac * hidden_size)))
    mask = np.zeros(hidden_size, dtype=bool)
    mask[np.argsort(-scores)[:top_k]] = True
    return mask


def near(value: float | None, target: float, tol: float = CLOSE_TOL) -> bool:
    return value is not None and abs(value - target) <= tol


# --------------------------------------------------------------------------------------------
# Check 0 -- shapes and gate-kind branch
# --------------------------------------------------------------------------------------------
def check0_shapes(W: np.ndarray, gate: np.ndarray, act: np.ndarray | None,
                   T_new: np.ndarray, T_old: np.ndarray | None, d: int) -> str:
    """Print all shapes/convention and return gate kind in {'avg', 'raw', 'per_unit'}."""

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise AssertionError(f"W must be square (H, H); got {W.shape}")
    hidden_size = W.shape[0]
    print("=== Check 0: shapes & convention ===")
    print(f"W: {W.shape}  (assumed W[i, j] = src j -> dst i, row=dst/col=src; "
          f"transpose W & gate if your convention is reversed)")
    print(f"gate: {gate.shape}")
    print(f"act: {None if act is None else act.shape}")
    print(f"T_new: {np.asarray(T_new).shape}")
    print(f"T_old: {None if T_old is None else np.asarray(T_old).shape}")
    print(f"H={hidden_size}, digit d={d}")

    if gate.ndim == 3 and gate.shape[1:] == (hidden_size, hidden_size):
        kind = "avg"
    elif gate.ndim == 5 and gate.shape[3:] == (hidden_size, hidden_size):
        kind = "raw"
    elif gate.ndim == 2 and gate.shape[1] == hidden_size:
        kind = "per_unit"
    else:
        raise AssertionError(f"gate shape {gate.shape} does not match any recognized convention")
    print(f"gate kind detected: {kind}")
    if kind == "per_unit":
        print(
            "\n门是 per-unit (D, H)，不是 per-connection (D, H, H)。"
            "之前基于 W[i,j]/G[i,j] 的 connection 级 four-group（TT/TR/RT/RR）分析在这个对象上"
            "不适用 -- 停下等你确认门对象是否传错了。"
        )
    return kind


# --------------------------------------------------------------------------------------------
# Check 1 -- is the gate time-averaged toward ~0.5?
# --------------------------------------------------------------------------------------------
def derive_static_gate(gate: np.ndarray, kind: str, d: int, hidden_size: int) -> np.ndarray:
    """Return the (H, H) static gate for digit d actually used by the new pipeline."""

    if kind == "avg":
        G_static_d = gate[d]
    elif kind == "raw":
        G_static_d = gate[d].mean(axis=(0, 1))
    else:
        raise ValueError(f"derive_static_gate does not support kind={kind!r}")
    if G_static_d.shape != (hidden_size, hidden_size):
        raise AssertionError(f"static gate must be ({hidden_size}, {hidden_size}); got {G_static_d.shape}")
    return G_static_d


def check1_gate_histogram(G_static_d: np.ndarray, fig_path: Path, dpi: int = DPI) -> dict:
    """Histogram + summary stats of the flattened static gate; save PNG, print, return stats."""

    flat = G_static_d.reshape(-1)
    mean, std = float(flat.mean()), float(flat.std())
    frac_mid = float(np.mean((flat > 0.4) & (flat < 0.6)))
    frac_low = float(np.mean(flat < 0.1))
    frac_high = float(np.mean(flat > 0.9))
    peaked_at_half = frac_mid > PEAK_FRAC_THRESHOLD
    print("\n=== Check 1: static gate histogram ===")
    print(f"mean={mean:.4f} std={std:.4f} frac(0.4<g<0.6)={frac_mid:.3f} "
          f"frac(g<0.1)={frac_low:.3f} frac(g>0.9)={frac_high:.3f}")
    print(f"peaked_at_half={peaked_at_half} (threshold frac_mid>{PEAK_FRAC_THRESHOLD})")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(flat, bins=60, range=(0.0, 1.0), color="#1d4ed8", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("static recurrent gate value")
    ax.set_ylabel("count")
    ax.set_title(f"Check1: digit {DIGIT} static gate distribution\nmean={mean:.3f} std={std:.3f}")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    return {"mean": mean, "std": std, "frac_mid": frac_mid, "frac_low": frac_low,
            "frac_high": frac_high, "peaked_at_half": peaked_at_half}


# --------------------------------------------------------------------------------------------
# Check 2 -- tuned-set agreement
# --------------------------------------------------------------------------------------------
def check2_tuned_overlap(T_new_mask: np.ndarray, T_old_mask: np.ndarray) -> float:
    n_new, n_old = int(T_new_mask.sum()), int(T_old_mask.sum())
    inter = int((T_new_mask & T_old_mask).sum())
    union = int((T_new_mask | T_old_mask).sum())
    jaccard = inter / union if union > 0 else float("nan")
    print("\n=== Check 2: tuned-set agreement ===")
    print(f"|T_new|={n_new} |T_old|={n_old} intersection={inter} union={union} Jaccard={jaccard:.3f}")
    return jaccard


# --------------------------------------------------------------------------------------------
# Check 3 -- reproduce TT mean from the current static gate
# --------------------------------------------------------------------------------------------
def check3_tt_block_mean(G_static_d: np.ndarray, mask: np.ndarray, label: str) -> float:
    if mask.sum() == 0:
        raise AssertionError(f"{label}: mask is empty, cannot compute TT block mean")
    idx = np.where(mask)[0]
    block = G_static_d[np.ix_(idx, idx)]
    expected_shape = (idx.size, idx.size)
    print(f"  [submatrix check] {label}: indexed block shape={block.shape} "
          f"(expected {expected_shape}; full matrix would be {G_static_d.shape})")
    if block.shape != expected_shape:
        raise AssertionError(f"{label}: indexed block shape {block.shape} != {expected_shape}")
    mean_val = float(block.mean())
    print(f"Check3 [{label}]: TT block mean = {mean_val:.4f}  (n_units={idx.size}, n_conn={block.size})")
    return mean_val


# --------------------------------------------------------------------------------------------
# Check 4 -- instantaneous reproduction (requires per-sample gate + activation)
# --------------------------------------------------------------------------------------------
def check4_instantaneous(G_raw_d: np.ndarray, act_d: np.ndarray, T_new_mask: np.ndarray,
                          top_fraction: float) -> tuple[float, float]:
    """Per-sample active-set TT gate mean vs. static-set-but-instantaneous-gate control.

    ``G_raw_d`` is (n_trials, T_steps, H, H), ``act_d`` is (n_trials, T_steps, H).
    Returns (mean_t(gTT_instant_set_instant_gate), mean_t(gTT_static_set_instant_gate)).
    """

    if G_raw_d.ndim != 4 or act_d.ndim != 3:
        raise AssertionError("G_raw_d must be (n_trials, T, H, H) and act_d must be (n_trials, T, H)")
    n_trials, t_steps, hidden_size, _ = G_raw_d.shape
    if G_raw_d.shape[:2] != act_d.shape[:2] or act_d.shape[2] != hidden_size:
        raise AssertionError("G_raw_d and act_d must share (n_trials, T_steps, H)")
    top_k = max(1, int(round(top_fraction * hidden_size)))
    new_idx = np.where(T_new_mask)[0]

    instant_vals = np.empty(n_trials * t_steps, dtype=np.float64)
    control_vals = np.empty(n_trials * t_steps, dtype=np.float64)
    flat_act = act_d.reshape(n_trials * t_steps, hidden_size)
    flat_gate = G_raw_d.reshape(n_trials * t_steps, hidden_size, hidden_size)
    for sample in range(flat_act.shape[0]):
        active_idx = np.argpartition(-flat_act[sample], top_k - 1)[:top_k]
        g = flat_gate[sample]
        instant_vals[sample] = g[np.ix_(active_idx, active_idx)].mean()
        control_vals[sample] = g[np.ix_(new_idx, new_idx)].mean()

    mean_instant = float(instant_vals.mean())
    mean_control = float(control_vals.mean())
    print("\n=== Check 4: instantaneous reproduction ===")
    print(f"instantaneous active-set + instantaneous gate: mean_t(gTT) = {mean_instant:.4f}  "
          f"(n_samples={instant_vals.size})")
    print(f"control -- static T_new set + instantaneous gate: mean_t(gTT) = {mean_control:.4f}")
    return mean_instant, mean_control


# --------------------------------------------------------------------------------------------
# Final decision table
# --------------------------------------------------------------------------------------------
def print_decision(peaked_at_half: bool | None, tt_new: float | None, tt_old: float | None,
                    instant_mean: float | None) -> None:
    print("\n=== 判定 ===")
    print(f"peaked_at_half={peaked_at_half}  tt_new={tt_new}  tt_old={tt_old}  instant_mean={instant_mean}")

    matched = []
    if peaked_at_half and near(instant_mean, TARGET_OLD):
        matched.append("A")
    if near(tt_old, TARGET_OLD) and near(tt_new, 0.5):
        matched.append("B")
    if near(tt_old, 0.5) and near(instant_mean, TARGET_OLD):
        matched.append("C")
    if near(tt_old, 0.5) and near(instant_mean, 0.5):
        matched.append("D")

    if matched:
        for branch in matched:
            print(f"命中分支 {branch}: {BRANCH_DESC[branch]}")
    else:
        print("未清晰命中任何分支 A-D，请人工核对上面打印的四个数值，并检查 tol/threshold 是否需要放宽。")


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main() -> None:
    print("提醒：约定 W[i, j] = src j -> dst i；若相反请转置 W 和 gate 再跑。\n")

    data = load_data()
    W, gate = data["W"], data["gate"]
    act, T_new, T_old = data.get("act"), data["T_new"], data.get("T_old")
    hidden_size = W.shape[0]

    kind = check0_shapes(W, gate, act, T_new, T_old, DIGIT)
    if kind == "per_unit":
        return

    print("\n=== Check 1-3 prep: mask integrity (steps 1-3) ===")
    T_new_mask = to_bool_mask(T_new, hidden_size, name="T_new")
    t_old_available = T_old is not None
    if t_old_available:
        T_old_mask = to_bool_mask(T_old, hidden_size, name="T_old")
        old_label = "T_old"
    else:
        T_old_mask = reconstruct_top_by_activation(act, DIGIT, TOP_FRACTION, hidden_size)
        old_label = "T_old(重建: digit d mean-activation top-10%)"
        print(f"\nT_old 未提供，已用 {old_label} 代替。")

    G_static_d = derive_static_gate(gate, kind, DIGIT, hidden_size)

    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    check1 = check1_gate_histogram(G_static_d, fig_dir / f"check1_static_gate_hist_digit{DIGIT}.png")

    print("\n=== Check 2: tuned-set agreement ===")
    if t_old_available:
        check2_tuned_overlap(T_new_mask, T_old_mask)
    else:
        print("跳过：T_old 不是独立给出的（已用重建集合代替），Jaccard 对比没有意义。")

    print("\n=== Check 3: TT block mean under the current static gate ===")
    tt_new = check3_tt_block_mean(G_static_d, T_new_mask, "T_new")
    tt_old = check3_tt_block_mean(G_static_d, T_old_mask, old_label)

    instant_mean = None
    if kind == "raw" and act is not None and np.asarray(act).ndim == 4:
        G_raw_d = gate[DIGIT]
        act_d = act[DIGIT]
        instant_mean, _control_mean = check4_instantaneous(G_raw_d, act_d, T_new_mask, TOP_FRACTION)
    else:
        print("\n=== Check 4: instantaneous reproduction ===")
        print("跳过：需要逐样本门 G_raw (D, n_trials, T_steps, H, H) 和逐样本 act "
              "(D, n_trials, T_steps, H)，当前数据不满足。")

    print_decision(check1["peaked_at_half"], tt_new, tt_old, instant_mean)


if __name__ == "__main__":
    main()
