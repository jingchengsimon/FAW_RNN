"""Aggregate GaWF feedback-ablation test accuracy across seeds into a two-row bar figure.

Reads one ``ablation_metrics.json`` (from ``utils_anal/feedback_ablation.py``) per seed and
draws a single PNG with two stacked panels that share the same digit/sector readout bars:

- top panel: the zero (clear) ablations    -- baseline, clear_digit, clear_sector, clear_all
- bottom panel: the shuffle ablations       -- baseline, shuffle_digit, shuffle_sector

Each condition shows the digit-readout and sector-readout test accuracy as grouped bars with
the across-seed mean height, a sample-SD error bar, and one dot per seed. PNG only (no PDF).
"""
from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Panel layout: which conditions belong to the zero row and the shuffle row. ``baseline`` is
# shared by both rows as the unablated reference.
ZERO_CONDITIONS = ("baseline", "clear_digit", "clear_sector", "clear_all")
SHUFFLE_CONDITIONS = ("baseline", "shuffle_digit", "shuffle_sector", "shuffle_all")
# Digit/sector colours aligned with core_objects_aggregate_2x2 (digit #E76F51, sector #264653).
READOUTS = (("char_acc", "Digit readout", "#E76F51"), ("sector_acc", "Sector readout", "#264653"))
PRETTY = {
    "baseline": "baseline",
    "clear_digit": "clear\ndigit",
    "clear_sector": "clear\nsector",
    "clear_all": "clear\nall",
    "shuffle_digit": "shuffle\ndigit",
    "shuffle_sector": "shuffle\nsector",
    "shuffle_all": "shuffle\nall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed_dirs",
        nargs="+",
        default=None,
        help="Per-seed directories each holding an ablation_metrics.json.",
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        default=None,
        help="Parent directory searched (one level deep) for */ablation_metrics.json.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(output_dir("G_behaviour", "viz_feedback_ablation", "figs")),
        help="Directory for the PNG output.",
    )
    parser.add_argument("--out_name", type=str, default="fig_ablation_multiseed_2row.png")
    return parser.parse_args()


def _resolve_metric_files(args: argparse.Namespace) -> List[str]:
    """Return the ablation_metrics.json paths from either explicit dirs or a parent glob."""

    files: List[str] = []
    if args.seed_dirs:
        for directory in args.seed_dirs:
            candidate = os.path.join(directory, "ablation_metrics.json")
            if not os.path.isfile(candidate):
                raise FileNotFoundError(f"No ablation_metrics.json in {directory}")
            files.append(candidate)
    elif args.parent_dir:
        files = sorted(glob.glob(os.path.join(args.parent_dir, "*", "ablation_metrics.json")))
        if not files:
            raise FileNotFoundError(
                f"No */ablation_metrics.json under {args.parent_dir}"
            )
    else:
        raise ValueError("Provide either --seed_dirs or --parent_dir")
    return files


def _collect(files: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Return per-condition, per-readout arrays of one accuracy value per seed."""

    per_seed: List[Dict[str, Any]] = []
    for path in files:
        with open(path, "r") as handle:
            per_seed.append(json.load(handle)["conditions"])

    collected: Dict[str, Dict[str, np.ndarray]] = {}
    all_conditions = sorted({c for seed in per_seed for c in seed})
    for condition in all_conditions:
        collected[condition] = {}
        for key, _label, _color in READOUTS:
            values = [
                float(seed[condition][key])
                for seed in per_seed
                if condition in seed and key in seed[condition]
            ]
            collected[condition][key] = np.asarray(values, dtype=np.float64)
    return collected


def _plot_panel(
    axis: plt.Axes,
    collected: Dict[str, Dict[str, np.ndarray]],
    conditions: Sequence[str],
    title: str,
    y_min: float = 60.0,
) -> None:
    """Draw one row grouped by readout: the N condition bars for the digit readout sit flush
    together, then a gap, then the N condition bars for the sector readout. This makes the
    across-condition trend within each readout directly comparable.
    """

    present = [c for c in conditions if c in collected and collected[c]["char_acc"].size]
    n = len(present)
    width = 1.0            # bars within a readout group touch (no gap)
    group_gap = 1.4        # blank space between the digit and sector groups, in bar widths
    rng = np.random.default_rng(0)
    group_base = {"char_acc": 0.0, "sector_acc": n * width + group_gap}
    xticks: List[float] = []
    xticklabels: List[str] = []
    for key, label, color in READOUTS:
        base = group_base[key]
        for j, c in enumerate(present):
            xpos = base + j * width
            values = collected[c][key]
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
            axis.bar(
                xpos,
                mean,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=label if j == 0 else None,
                yerr=sd,
                capsize=3,
                error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
            )
            jitter = (rng.random(values.size) - 0.5) * width * 0.4
            axis.scatter(
                xpos + jitter, values, s=14, color="#222222", alpha=0.7, zorder=3, linewidths=0
            )
            xticks.append(xpos)
            xticklabels.append(PRETTY.get(c, c))
    axis.set_ylabel("Test accuracy (%)")
    axis.set_xticks(xticks)
    axis.set_xticklabels(xticklabels, fontsize=8)
    axis.set_xlim(-0.7, group_base["sector_acc"] + (n - 1) * width + 0.7)
    axis.set_ylim(y_min, 95.0)
    axis.set_yticks(np.arange(y_min, 95.1, 5.0))
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()
    files = _resolve_metric_files(args)
    collected = _collect(files)
    n_seeds = len(files)

    # Independent y-axes: the zero panel must reach clear_all (~65% digit) so it starts at 60,
    # while the shuffle panel's lowest bar is ~75% and reads better starting at 70.
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.4), sharey=False)
    _plot_panel(
        axes[0], collected, ZERO_CONDITIONS, f"Zero ablation (n={n_seeds} seeds)", y_min=60.0
    )
    _plot_panel(
        axes[1], collected, SHUFFLE_CONDITIONS, f"Shuffle ablation (n={n_seeds} seeds)", y_min=70.0
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {out_path}  (from {n_seeds} seed metrics)")


if __name__ == "__main__":
    main()
