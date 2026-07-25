"""Tests for the afferent top-10% vs remaining destination-gate distribution analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils_anal.gawf_afferent_top10_gate_distributions import (
    CELL_SPECS,
    GROUP_NAMES,
    parse_args,
)
from utils_anal.gawf_recurrent_sector_relevance_distributions import (
    accumulate_context_group_distributions,
    summarize_group_moments,
)
from utils_anal.gawf_symmetric_stats import (
    bootstrap_d,
    cosine_alignment,
    relevance_label_null,
    relevance_masks,
    trial_relevance_moments,
)
import utils_viz.gawf_afferent_top10_gate_distributions as afferent_top10_viz


def test_cell_specs_cover_input_and_recurrent_across_sector_and_digit() -> None:
    """The afferent 2-group analysis must mirror the efferent cell layout."""

    assert set(CELL_SPECS) == {
        "input_sector",
        "input_digit",
        "recurrent_sector",
        "recurrent_digit",
    }
    for cell, (gate_name, factor, levels, label_column) in CELL_SPECS.items():
        assert gate_name in ("input", "recurrent")
        assert factor in ("sector", "digit")
        expected_levels = 9 if factor == "sector" else 10
        assert levels == expected_levels, cell
        expected_column = 1 if factor == "sector" else 0
        assert label_column == expected_column, cell
    assert GROUP_NAMES == ("top10", "remaining")


def _synthetic_destination_view(
    frames: int,
    hidden_size: int,
    top_mask: np.ndarray,
    rem_mask: np.ndarray,
    top_value: float,
    rem_value: float,
    outside_value: float,
) -> np.ndarray:
    """Build a ``(frames, hidden_size)`` per-destination gate view with constant group values."""

    view = np.full((frames, hidden_size), outside_value, dtype=np.float32)
    view[:, top_mask] = top_value
    view[:, rem_mask] = rem_value
    return view


def test_afferent_accumulator_partitions_hidden_destinations_by_context() -> None:
    hidden_size = 12
    top_mask = np.zeros(hidden_size, dtype=bool)
    top_mask[[0, 1, 2]] = True
    eligible = np.ones(hidden_size, dtype=bool)
    eligible[[10, 11]] = False  # Two ineligible destinations must be excluded from both groups.
    rem_mask = eligible & ~top_mask

    frames = 4
    top_value = 0.85
    rem_value = 0.20
    outside_value = 0.99  # Should not enter any group.
    view = _synthetic_destination_view(
        frames, hidden_size, top_mask, rem_mask, top_value, rem_value, outside_value
    )
    contexts = np.zeros(frames, dtype=np.int64)
    relevant_masks = top_mask[None, :]
    bin_edges = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    hist, sums, sums_sq, counts = accumulate_context_group_distributions(
        ((0, frames, view),),
        contexts,
        relevant_masks,
        eligible,
        bin_edges,
    )
    assert hist.shape == (1, 2, bin_edges.size - 1)
    top_count = int(top_mask.sum())
    rem_count = int(rem_mask.sum())
    np.testing.assert_array_equal(counts[0], [frames * top_count, frames * rem_count])
    means, _stds, context_d, global_d = summarize_group_moments(sums, sums_sq, counts)
    np.testing.assert_allclose(means[0], [top_value, rem_value], atol=1e-6)
    # Group values are constant, so within-group SD is 0 => Cohen's d degenerates. summarize
    # should return a finite value when there is any variance; here we simply verify sign.
    assert context_d[0] > 0.0 or np.isinf(context_d[0])


def test_afferent_accumulator_ignores_ineligible_destinations() -> None:
    hidden_size = 8
    top_mask = np.zeros(hidden_size, dtype=bool)
    top_mask[[0, 1]] = True
    eligible = np.ones(hidden_size, dtype=bool)
    eligible[[6, 7]] = False
    rem_mask = eligible & ~top_mask
    frames = 3
    view = _synthetic_destination_view(
        frames, hidden_size, top_mask, rem_mask, 0.9, 0.1, outside_value=999.0
    )
    contexts = np.zeros(frames, dtype=np.int64)
    bin_edges = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    _hist, _sums, _sums_sq, counts = accumulate_context_group_distributions(
        ((0, frames, view),),
        contexts,
        top_mask[None, :],
        eligible,
        bin_edges,
    )
    # If ineligible destinations leaked into either group, counts would include them and the
    # constant ``999.0`` value would appear in later means. Verify exact partition sizes.
    assert counts[0, 0] == frames * int(top_mask.sum())
    assert counts[0, 1] == frames * int(rem_mask.sum())


def test_afferent_grid_specs_match_efferent_layout() -> None:
    """Afferent grid layouts must line up with the four efferent cells."""

    assert set(afferent_top10_viz.GRID_SPECS) == set(CELL_SPECS)
    for cell, (nrows, ncols, gate_name, factor, levels, _label, _src_dim) in (
        afferent_top10_viz.GRID_SPECS.items()
    ):
        expected_gate, expected_factor, expected_levels, _label_col = CELL_SPECS[cell]
        assert gate_name == expected_gate
        assert factor == expected_factor
        assert levels == expected_levels
        # Layouts follow the same convention as the existing efferent combined grids.
        expected_grid = (3, 3) if levels == 9 else (2, 5)
        assert (nrows, ncols) == expected_grid, cell


def test_afferent_panel_renders_two_stairs_and_two_mean_lines() -> None:
    figure, axis = afferent_top10_viz.plt.subplots()
    afferent_top10_viz.plot_afferent_top10_panel(
        axis,
        "Digit",
        3,
        "input",
        "1152 encoder features",
        np.linspace(0.0, 1.0, 6, dtype=np.float64),
        np.asarray([[2, 4, 6, 8, 10], [1, 2, 3, 4, 5]], dtype=np.int64),
        np.asarray([0.72, 0.35], dtype=np.float64),
        np.asarray([30, 200], dtype=np.int64),
        0.87,
        30,
        200,
        density_limit=6.0,
    )
    assert axis.get_xlim() == (0.0, 1.0)
    assert "Digit 3" in axis.get_title()
    assert "AFFERENT" in axis.get_title()
    axvline_lines = [line for line in axis.get_lines() if line.get_linestyle() == "--"]
    assert len(axvline_lines) == 2
    afferent_top10_viz.plt.close(figure)


def test_afferent_render_grid_writes_expected_filenames(tmp_path: Path) -> None:
    bin_edges = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    for cell, (_nrows, _ncols, _gate, _factor, levels, _label, _src) in (
        afferent_top10_viz.GRID_SPECS.items()
    ):
        hist_counts = np.ones((levels, 2, 5), dtype=np.int64) * 4
        group_mean = np.full((levels, 2), 0.4, dtype=np.float64)
        group_count = np.full((levels, 2), 20, dtype=np.int64)
        context_d = np.zeros(levels, dtype=np.float64)
        relevant_mask = np.zeros((levels, 24), dtype=bool)
        relevant_mask[:, :3] = True
        eligible_mask = np.ones(24, dtype=bool)
        destination = afferent_top10_viz.render_grid(
            cell,
            bin_edges,
            hist_counts,
            group_mean,
            group_count,
            context_d,
            relevant_mask,
            eligible_mask,
            tmp_path,
            100,
        )
        assert destination.name == (
            f"{cell}_top10_vs_remaining_afferent_distribution_grid.png"
        )
        assert destination.exists()


def test_parse_args_exposes_aggregate_resamples_seed_and_top_percent(monkeypatch) -> None:
    """CLI defaults for the aggregate Part-2 style knobs must match the Amarel runner."""

    monkeypatch.setattr("sys.argv", ["gawf_afferent_top10_gate_distributions"])
    args = parse_args()
    assert args.resamples == 1000
    assert args.seed == 260718
    assert args.top_percent == [10]


def test_parse_args_accepts_multiple_top_percent_thresholds(monkeypatch) -> None:
    """``--top_percent`` must accept a whitespace-separated list of integer percentages."""

    monkeypatch.setattr(
        "sys.argv",
        [
            "gawf_afferent_top10_gate_distributions",
            "--top_percent", "5", "10", "20",
            "--resamples", "32",
            "--seed", "7",
        ],
    )
    args = parse_args()
    assert args.top_percent == [5, 10, 20]
    assert args.resamples == 32
    assert args.seed == 7


def test_aggregate_pipeline_uses_shared_primitives_with_synthetic_gates() -> None:
    """Reproduce the aggregate cell block in miniature to lock in the primitive contract.

    The full ``main()`` requires a real GaWF checkpoint. Instead we exercise the same sequence
    of ``relevance_masks`` -> ``trial_relevance_moments`` -> ``bootstrap_d`` ->
    ``relevance_label_null`` -> ``cosine_alignment`` calls on synthetic tuning and gate columns
    so any future refactor that breaks the shared contract fails here.
    """

    rng = np.random.default_rng(0)
    levels = 9  # ``cosine_alignment`` hardcodes NUM_SECTORS=9 for factor="sector".
    hidden = 24
    trials = 900
    eligible = np.ones(hidden, dtype=bool)
    eligible[-2:] = False  # Ineligible destinations must never enter any group.
    contexts = rng.integers(0, levels, size=trials).astype(np.int64)
    digit_labels = rng.integers(0, 10, size=trials).astype(np.int64)
    labels = np.stack([digit_labels, contexts], axis=1)  # factor=="sector" reads column 1
    tuning = rng.standard_normal((levels, hidden)).astype(np.float64)
    # Inject a large positive bias for a rotating triplet of eligible destinations per context.
    for level in range(levels):
        start = (level * 2) % (hidden - 4)
        tuning[level, start : start + 3] += 5.0
    gates = np.full((trials, hidden), 0.2, dtype=np.float64)
    for level in range(levels):
        top_ids = np.argsort(tuning[level, :])[-3:]
        trial_indices = np.flatnonzero(contexts == level)
        # Broadcast per-context relevance signal directly onto the target trial/column pairs.
        gates[np.ix_(trial_indices, top_ids)] += 0.6

    masks = relevance_masks(tuning, eligible, 0.25)
    assert masks.shape == (levels, hidden)
    assert masks[:, eligible].sum(axis=1).min() >= 1
    assert not masks[:, ~eligible].any()

    moments = trial_relevance_moments(gates, contexts, masks, eligible)
    assert moments.shape == (trials, 6)
    point, draws = bootstrap_d(moments, resamples=64, seed=1)
    assert draws.shape == (64,)
    assert point > 0.5  # Injected relevance signal must produce a large positive d.

    null = relevance_label_null(
        gates, labels, "sector", tuning, eligible, 0.25, resamples=64, seed=2,
    )
    assert null.shape == (64,)
    p_value = float((1 + np.count_nonzero(null >= point)) / (64 + 1))
    assert 0.0 <= p_value <= 1.0

    alignment = cosine_alignment(
        tuning, gates, labels, "sector", eligible, resamples=64, seed=3,
    )
    matrix = alignment["matrix"]
    assert matrix.shape == (levels, levels)
    # Injected structure aligns activation tuning with the gate columns per context.
    assert alignment["diagonal_minus_off_diagonal"] > 0.0
    assert 0.0 <= alignment["permutation_p_value"] <= 1.0
    assert alignment["permutation_null"].shape == (64,)
    assert alignment["permutation_alternative"] == "two-sided"
