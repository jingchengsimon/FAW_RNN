"""Standalone shuffle-only feedback-ablation figure (single poster-style panel).

Shows the digit- and sector-readout test accuracy for three conditions -- GaWF baseline,
Shuffle digit, Shuffle sector -- with no shuffle_all. Bars are grouped by readout (the three
condition bars for each readout sit flush together), each bar carrying the across-seed mean, a
sample-SD error bar, and one dot per seed.

The baseline bar heights come from ``best_acc_test_mean_std.csv`` (the canonical multiseed test
accuracy used by the best-model figure) so the baseline matches that panel exactly; the two
shuffle bars come from the per-seed ``ablation_metrics.json`` files. PNG only.
"""
from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# (data key, legend label, colour) for the two readouts; order here is left-to-right group
# order (sector group on the left, digit group on the right). Colours align with
# core_objects_aggregate_2x2 (digit #E76F51, sector #264653).
READOUTS = (("sector_acc", "Sector", "#264653"), ("char_acc", "Digit", "#E76F51"))
# Displayed conditions in order, with their pretty x labels. baseline is drawn from the
# canonical test CSV; the two shuffle conditions from the ablation metrics.
CONDITIONS = (
    ("baseline", "GaWF\nbaseline"),
    ("shuffle_digit", "Shuffle\ndigit"),
    ("shuffle_sector", "Shuffle\nsector"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation_dir",
        type=str,
        default=str(
            _ANAL_PROJECT_ROOT
            + "/results/data/anal_data/G_behaviour/feedback_ablation_multiseed"
        ),
        help="Parent dir with per-seed gawf-seed*/ablation_metrics.json.",
    )
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default=str(
            _ANAL_PROJECT_ROOT
            + "/results/data/anal_data/G_behaviour/clutter_multiseed_best_acc_bars"
            + "/clutter_best6_multiseed_40h_ep150/best_acc_test_mean_std.csv"
        ),
        help="Per-seed canonical test accuracy CSV (source,model,seed,char_acc,sector_acc).",
    )
    parser.add_argument("--model", type=str, default="gawf")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(output_dir("G_behaviour", "viz_feedback_ablation", "figs")),
    )
    parser.add_argument("--out_name", type=str, default="fig_ablation_shuffle_standalone.png")
    return parser.parse_args()


def _baseline_from_csv(path: str, model: str) -> Dict[str, np.ndarray]:
    """Return per-seed baseline arrays for the requested model from the canonical test CSV."""

    char: List[float] = []
    sector: List[float] = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == model:
                char.append(float(row["char_acc"]))
                sector.append(float(row["sector_acc"]))
    if not char:
        raise ValueError(f"No rows for model={model!r} in {path}")
    return {"char_acc": np.asarray(char), "sector_acc": np.asarray(sector)}


def _shuffle_from_ablation(ablation_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Return per-seed shuffle-condition arrays from the ablation metrics."""

    files = sorted(glob.glob(os.path.join(ablation_dir, "gawf-seed*", "ablation_metrics.json")))
    if not files:
        raise FileNotFoundError(f"No gawf-seed*/ablation_metrics.json under {ablation_dir}")
    collected: Dict[str, Dict[str, List[float]]] = {}
    for path in files:
        conditions = json.load(open(path))["conditions"]
        for cond in ("shuffle_digit", "shuffle_sector"):
            collected.setdefault(cond, {"char_acc": [], "sector_acc": []})
            for key in ("char_acc", "sector_acc"):
                collected[cond][key].append(float(conditions[cond][key]))
    return {c: {k: np.asarray(v) for k, v in d.items()} for c, d in collected.items()}


def main() -> None:
    args = parse_args()
    baseline = _baseline_from_csv(args.baseline_csv, args.model)
    shuffle = _shuffle_from_ablation(args.ablation_dir)
    data = {"baseline": baseline, **shuffle}
    n_seeds = baseline["char_acc"].size

    conds = [c for c, _ in CONDITIONS]
    labels = {c: lbl for c, lbl in CONDITIONS}
    n = len(conds)
    width = 1.0            # bars within a readout group touch (no gap)
    group_gap = 1.5        # 1.5 bar widths between the two groups, matching core_objects_2x2
    # Left-to-right group order follows READOUTS (sector group first, then digit).
    group_base = {
        key: i * (n * width + group_gap) for i, (key, _lbl, _c) in enumerate(READOUTS)
    }
    rng = np.random.default_rng(0)

    with plt.rc_context(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    ):
        # Match one row of core_objects_aggregate_2x2 (5.05in wide x 8.2in / 2 rows tall) so the
        # panel height aligns with that figure's first row and the bars share its tall-narrow
        # width-to-height ratio, while staying a single axis.
        fig, axis = plt.subplots(figsize=(5.05, 4.1))
        xticks: List[float] = []
        xticklabels: List[str] = []
        for key, label, color in READOUTS:
            base = group_base[key]
            for j, cond in enumerate(conds):
                xpos = base + j * width
                values = data[cond][key]
                sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
                axis.bar(
                    xpos,
                    float(values.mean()),
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
                    xpos + jitter, values, s=16, color="#222222", alpha=0.7, zorder=3,
                    linewidths=0,
                )
                xticks.append(xpos)
                xticklabels.append(labels[cond])

        axis.set_ylabel("Test accuracy (%)")
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels, fontsize=9)
        axis.set_xlim(-0.7, max(group_base.values()) + (n - 1) * width + 0.7)
        axis.set_ylim(70.0, 95.0)
        axis.set_yticks(np.arange(70.0, 95.1, 5.0))
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        handles, legend_labels = axis.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=2, frameon=False)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, args.out_name)
        # Publication candidate: high-res PNG plus a vector PDF with the same basename.
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    print(f"Saved figure: {out_path}  (baseline n={n_seeds} seeds)")
    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
