"""Regenerate the poster-style four-core-object aggregate figure from saved NPZ summaries.

Inputs are the four unified variance-decomposition NPZ files.  The development PNG remains in
``results/anal_figs/D_variance_decomposition/`` and the official PDF is written to the configured
publication-figure directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.publication_paths import publication_figures_dir
from utils_anal.anal_paths import output_dir
from utils_anal.run_unified_variance_decomposition import (
    _plot_compact_aggregate,
    _plot_compact_aggregate_1x4,
)
from utils_anal.variance_decomposition import CM_FACTORS, RepeatedDecomposition


OBJECTS = ("input_gate", "recurrent_gate", "encoder_activation", "hidden_state")
SAVE_DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "save_data"
    / "fig4"
    / "unified_variance_decomposition"
)


def parse_args() -> argparse.Namespace:
    """Parse saved-data and figure destinations.

    ``--data_dir``/``--figure_dir`` default to ``None`` here and are resolved lazily in
    ``main()`` so that ``output_dir(...)`` (which creates its directory as a side effect) only
    runs when the caller actually left that argument unset.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument(
        "--seed_dirs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Cross-seed mode: one per-seed unified data dir each holding the four "
            "{object}_per_unit_distributions.npz. When given, each seed contributes one point "
            "estimate (its within-seed aggregate mean) and the figure draws cross-seed mean +/- "
            "sd, matching the best-model-accuracy figure. Overrides --data_dir."
        ),
    )
    parser.add_argument("--figure_dir", type=Path, default=None)
    parser.add_argument("--publication_fig_dir", type=Path, default=None)
    return parser.parse_args()


def load_saved_results(data_dir: Path) -> dict[str, RepeatedDecomposition]:
    """Load the repeated condition-mean aggregate fractions needed by the compact figure."""

    results: dict[str, RepeatedDecomposition] = {}
    for object_name in OBJECTS:
        path = data_dir / f"{object_name}_per_unit_distributions.npz"
        with np.load(path, allow_pickle=False) as arrays:
            aggregate_cm = {
                factor: np.asarray(arrays[f"aggregate_cm_{factor}"], dtype=np.float64)
                for factor in CM_FACTORS
            }
        results[object_name] = RepeatedDecomposition(
            aggregate_cm=aggregate_cm,
            aggregate_trial={},
            per_unit_cm={},
            per_unit_trial={},
            unweighted_per_unit_mean_cm={},
            unweighted_per_unit_mean_trial={},
            consistency={},
        )
    return results


def load_multiseed_results(seed_dirs: list[Path]) -> dict[str, RepeatedDecomposition]:
    """Collapse each seed to one point estimate per object/factor for cross-seed error bars.

    For every seed we take the within-seed aggregate mean of ``aggregate_cm[factor]`` (the same
    scalar the single-seed bar height uses), then stack those across seeds so ``aggregate_cm`` maps
    each factor to a length-``n_seeds`` array. Passing ``error_mode="sd"`` to the plotters turns
    that into cross-seed mean +/- sd, the same spread convention as the best-model-accuracy figure.
    """

    per_seed = [load_saved_results(seed_dir) for seed_dir in seed_dirs]
    results: dict[str, RepeatedDecomposition] = {}
    for object_name in OBJECTS:
        aggregate_cm = {
            factor: np.asarray(
                [float(seed[object_name].aggregate_cm[factor].mean()) for seed in per_seed],
                dtype=np.float64,
            )
            for factor in CM_FACTORS
        }
        results[object_name] = RepeatedDecomposition(
            aggregate_cm=aggregate_cm,
            aggregate_trial={},
            per_unit_cm={},
            per_unit_trial={},
            unweighted_per_unit_mean_cm={},
            unweighted_per_unit_mean_trial={},
            consistency={},
        )
    return results


def main() -> None:
    """Load the saved summaries and regenerate the official compact aggregate figure."""

    args = parse_args()
    if args.figure_dir is None:
        args.figure_dir = output_dir("D_variance_decomposition", "unified", "figs")
    # Cross-seed mode (--seed_dirs) draws mean +/- sd across seeds; otherwise the single-model
    # within-subsample quantile band.
    if args.seed_dirs is not None:
        saved_results = load_multiseed_results(args.seed_dirs)
        error_mode = "sd"
        print(f"Cross-seed mode: {len(args.seed_dirs)} seeds -> mean +/- sd error bars")
    else:
        if args.data_dir is None:
            args.data_dir = SAVE_DATA_DIR
        saved_results = load_saved_results(args.data_dir)
        error_mode = "ci"
    # Workflow policy: the PDF sits next to the PNG in the local results tree. The
    # ``--publication_fig_dir`` copy is an opt-in extra only (no automatic 6-Writing sync).
    publication_dir = (
        publication_figures_dir(args.publication_fig_dir, create=True)
        if args.publication_fig_dir is not None
        else None
    )
    destination = _plot_compact_aggregate(
        args.figure_dir, saved_results, publication_dir, error_mode=error_mode
    )
    print(f"Saved {destination}")
    print(f"Saved {destination.with_suffix('.pdf')}")
    if publication_dir is not None:
        print(f"Saved {publication_dir / 'core_objects_aggregate_2x2.pdf'}")

    destination_1x4 = _plot_compact_aggregate_1x4(
        args.figure_dir, saved_results, publication_dir, error_mode=error_mode
    )
    print(f"Saved {destination_1x4}")
    print(f"Saved {destination_1x4.with_suffix('.pdf')}")
    if publication_dir is not None:
        print(f"Saved {publication_dir / 'core_objects_aggregate_1x4.pdf'}")


if __name__ == "__main__":
    main()
