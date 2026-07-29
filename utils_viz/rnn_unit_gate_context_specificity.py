"""Plot GaWF, LSTM, and GRU unit-level gate context variance decompositions.

Input: ``unit_gate_context_variance.json`` from the matching analysis module, or the cross-seed
``unit_gate_context_variance_multiseed.json`` from the multi-seed driver.  Output: one
Figure-03-style PNG for every available GaWF/LSTM/GRU report, plus a compact poster-oriented PNG
containing their condition-mean marginalization panels.  When the report stores per-seed fraction
arrays, the 1-by-3 panel draws cross-seed mean +/- sample sd instead of single-seed point
estimates.  Its official PDF is written to the configured publication-figure directory.
"""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from utils.publication_paths import publication_figures_dir


SAVE_DATA_REPORT = (
    Path(_ANAL_PROJECT_ROOT)
    / "results"
    / "save_data"
    / "fig5"
    / "unit_gate_context_variance_multiseed.json"
)


GATE_LABELS = {
    "gawf": {
        "input_mean": "Input",
        "recurrent_mean": "Recurrent",
    },
    "lstm": {"input": "Input", "forget": "Forget", "output": "Output"},
    "gru": {"reset": "Reset", "update": "Update"},
}

MODEL_ORDER = ("gawf", "lstm", "gru")
MODEL_TITLES = {
    "gawf": "GaWF afferent gates",
    "lstm": "LSTM gates",
    "gru": "GRU gates",
}
# Gate identities use a palette distinct from the cross-model blue/C1-C5 mapping.  Colors are
# unique across this result panel so that no hue changes meaning between its three subplots.
GATE_COLORS = {
    ("gawf", "input_mean"): "#D95F02",
    ("gawf", "recurrent_mean"): "#1B9E77",
    ("lstm", "input"): "#7570B3",
    ("lstm", "forget"): "#E7298A",
    ("lstm", "output"): "#66A61E",
    ("gru", "reset"): "#E6AB02",
    ("gru", "update"): "#4D4D4D",
}


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default=str(SAVE_DATA_REPORT),
    )
    parser.add_argument(
        "--fig_dir",
        default=str(
            output_dir(
                "D_variance_decomposition",
                "rnn_unit_gate_context_specificity",
                "figs",
            )
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--publication_fig_dir",
        type=Path,
        default=None,
        help=(
            "Opt-in extra PDF destination. When omitted, the PDF is written only next to its "
            "PNG in the local results tree (no automatic 6-Writing sync)."
        ),
    )
    return parser.parse_args()


def _annotate_bars(ax: plt.Axes) -> None:
    """Add compact percentage labels above visible bars."""

    for container in ax.containers:
        labels = [
            f"{bar.get_height():.1f}" if bar.get_height() >= 0.08 else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, padding=2, fontsize=9, rotation=0)


def _fraction_mean_sd(value: object) -> tuple[float, float | None]:
    """Reduce one fraction entry to ``(mean, sample sd)``.

    A scalar (single-seed report) yields no spread; an array (cross-seed report) yields the mean
    and the sample sd (ddof=1), the same convention as the best-model-accuracy figure.
    """

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array), None
    if array.size <= 1:
        return float(array.mean()), None
    return float(array.mean()), float(np.std(array, ddof=1))


def _is_multiseed_report(report: dict) -> bool:
    """Return True when any gate fraction is stored as a per-seed array."""

    for model_block in report.get("models", {}).values():
        for gate_block in model_block.get("gates", {}).values():
            fractions = gate_block["equal_cell_condition_mean"]["fractions"]
            if any(isinstance(value, (list, tuple)) for value in fractions.values()):
                return True
    return False


def plot_model(report: dict, model_type: str, fig_dir: str, dpi: int) -> str:
    """Write one two-panel Figure-03-style decomposition for a recurrent model."""

    model_report = report["models"][model_type]
    gate_names = tuple(GATE_LABELS[model_type])
    factors = ("sector", "digit", "interaction")
    trial_factors = factors + ("residual",)
    width = 0.8 / len(gate_names)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    colors = plt.get_cmap("tab10").colors
    for gate_index, gate_name in enumerate(gate_names):
        gate = model_report["gates"][gate_name]
        condition = gate["equal_cell_condition_mean"]["fractions"]
        trial = gate["equal_cell_trial_total"]["percent"]
        offset = (gate_index - (len(gate_names) - 1) / 2) * width
        axes[0].bar(
            np.arange(len(factors)) + offset,
            [100.0 * condition[factor] for factor in factors],
            width,
            color=colors[gate_index],
            label=GATE_LABELS[model_type][gate_name],
        )
        axes[1].bar(
            np.arange(len(trial_factors)) + offset,
            [trial[factor] for factor in trial_factors],
            width,
            color=colors[gate_index],
            label=GATE_LABELS[model_type][gate_name],
        )
    axes[0].set_xticks(np.arange(len(factors)), [name.title() for name in factors])
    axes[0].set_ylabel("Aggregate variance (%)")
    axes[0].set_title("Hidden-state-compatible marginalization")
    axes[1].set_xticks(
        np.arange(len(trial_factors)), [name.title() for name in trial_factors]
    )
    axes[1].set_ylabel("Aggregate variance (%)")
    axes[1].set_title("Balanced trial-level ANOVA")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        legend_title = (
            "Destination-unit projection" if model_type == "gawf" else "Unit-level gate"
        )
        ax.legend(title=legend_title, frameon=False)
        _annotate_bars(ax)
        ax.margins(y=0.15)
    level = (
        "destination-unit means of incoming synapse gates"
        if model_type == "gawf"
        else "unit-level gates"
    )
    fig.suptitle(f"{model_type.upper()} context variance decomposition ({level})")
    fig.tight_layout()
    path = os.path.join(
        fig_dir, f"03_{model_type}_unit_gate_variance_decomposition.png"
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_marginalization_summary(
    report: dict,
    fig_dir: str,
    dpi: int,
    publication_fig_dir: Path | None,
) -> tuple[str, str | None]:
    """Write the poster-oriented 1-by-3 condition-mean gate marginalization figure."""

    missing = [model for model in MODEL_ORDER if model not in report["models"]]
    if missing:
        raise KeyError(f"Marginalization summary requires all models; missing {missing}")

    factors = ("sector", "digit", "interaction")
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
        # Narrower panels but more vertical plotting space than the existing 2-by-3 summary.
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 5.2), sharey=True)
        x = np.arange(len(factors), dtype=np.float64)
        for axis, model_type in zip(axes, MODEL_ORDER):
            gate_names = tuple(GATE_LABELS[model_type])
            width = 0.78 / len(gate_names)
            for gate_index, gate_name in enumerate(gate_names):
                fractions = report["models"][model_type]["gates"][gate_name][
                    "equal_cell_condition_mean"
                ]["fractions"]
                offset = (gate_index - (len(gate_names) - 1) / 2) * width
                stats = [_fraction_mean_sd(fractions[factor]) for factor in factors]
                heights = [100.0 * mean for mean, _sd in stats]
                sds = [sd for _mean, sd in stats]
                bar_kwargs: dict = {}
                if any(sd is not None for sd in sds):
                    bar_kwargs.update(
                        yerr=[100.0 * (sd if sd is not None else 0.0) for sd in sds],
                        capsize=2.6,
                        ecolor="#252525",
                        error_kw={"linewidth": 1.0, "capthick": 1.0},
                    )
                axis.bar(
                    x + offset,
                    heights,
                    width,
                    color=GATE_COLORS[(model_type, gate_name)],
                    edgecolor="none",
                    label=GATE_LABELS[model_type][gate_name],
                    **bar_kwargs,
                )

            axis.set_xticks(x, [factor.title() for factor in factors])
            axis.set_ylim(0.0, 105.0)
            axis.set_yticks(np.arange(0.0, 100.1, 20.0))
            axis.set_title(MODEL_TITLES[model_type], pad=47)
            axis.grid(axis="y", alpha=0.25, linewidth=0.7)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            handles = [
                Patch(
                    facecolor=GATE_COLORS[(model_type, gate_name)],
                    edgecolor="none",
                    label=GATE_LABELS[model_type][gate_name],
                )
                for gate_name in gate_names
            ]
            axis.legend(
                handles=handles,
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.015),
                ncol=len(gate_names),
                handlelength=1.15,
                handleheight=0.85,
                columnspacing=0.75,
                borderaxespad=0.0,
            )

        fig.subplots_adjust(left=0.085, right=0.995, bottom=0.14, top=0.72, wspace=0.20)
        row_center = np.mean(
            [axis.get_position().y0 + axis.get_position().height / 2 for axis in axes]
        )
        fig.text(
            0.024,
            row_center,
            "Explained variance (%)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=16,
        )
        if _is_multiseed_report(report):
            seed_counts = {
                model: len(block.get("seeds", []))
                for model, block in report["models"].items()
            }
            counts = sorted(set(seed_counts.values()))
            seed_note = (
                f"{counts[0]} seeds" if len(counts) == 1 else f"{counts[0]}-{counts[-1]} seeds"
            )
            fig.text(
                0.54,
                0.012,
                f"Bars: cross-seed mean; error bars: \u00b1 sample SD ({seed_note} per model)",
                ha="center",
                va="bottom",
                fontsize=11.5,
                color="#404040",
            )

        os.makedirs(fig_dir, exist_ok=True)
        base_path = os.path.join(fig_dir, "03_unit_gate_marginalization_1x3")
        png_path = f"{base_path}.png"
        fig.savefig(png_path, dpi=max(dpi, 180), bbox_inches="tight", pad_inches=0.04)
        # Workflow policy: the PDF always sits next to the PNG in the local results tree.
        pdf_path = f"{base_path}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
        # ``publication_fig_dir`` is an opt-in extra copy only (no automatic 6-Writing sync).
        publication_pdf = None
        if publication_fig_dir is not None:
            publication_pdf = str(publication_fig_dir / "03_unit_gate_marginalization_1x3.pdf")
            fig.savefig(publication_pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    if publication_pdf is not None:
        print(f"Saved {publication_pdf}")
    return png_path, pdf_path


def main() -> None:
    """Load the analysis report and plot every available model."""

    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    publication_dir = (
        publication_figures_dir(args.publication_fig_dir, create=True)
        if args.publication_fig_dir is not None
        else None
    )
    with open(args.report, "r", encoding="utf-8") as file_obj:
        report = json.load(file_obj)
    # A cross-seed report stores per-seed fraction arrays and carries no trial-total ANOVA, so
    # only the 1-by-3 marginalization summary (drawn as mean +/- sd) is produced from it.
    if not _is_multiseed_report(report):
        for model_type in MODEL_ORDER:
            if model_type in report["models"]:
                plot_model(report, model_type, args.fig_dir, args.dpi)
    plot_marginalization_summary(report, args.fig_dir, args.dpi, publication_dir)


if __name__ == "__main__":
    main()
