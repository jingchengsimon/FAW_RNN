"""Tests for the recurrent source/destination four-group gate distribution analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils_anal.gawf_recurrent_group_gate_distributions import (
    GROUP_NAMES,
    accumulate_group_split_distributions,
    summarize_group_moments,
    within_between_cohens_d,
)
import utils_viz.gawf_recurrent_group_gate_distributions as group_viz


def _synthetic_frame_gate(hidden_size: int, top_mask: np.ndarray, rem_mask: np.ndarray,
                          within_top: float, top_to_rem: float,
                          rem_to_top: float, within_rem: float) -> np.ndarray:
    """Build a full ``(hidden, hidden)`` gate matrix with 4 constant group cells."""

    gate = np.zeros((hidden_size, hidden_size), dtype=np.float32)
    # gate[h, i] with h = destination, i = source
    dest_top = np.flatnonzero(top_mask)
    dest_rem = np.flatnonzero(rem_mask)
    src_top = dest_top  # same partition
    src_rem = dest_rem
    gate[np.ix_(dest_top, src_top)] = within_top
    gate[np.ix_(dest_rem, src_top)] = top_to_rem
    gate[np.ix_(dest_top, src_rem)] = rem_to_top
    gate[np.ix_(dest_rem, src_rem)] = within_rem
    return gate


def _per_source_means_from_gate(
    gate: np.ndarray, top_mask: np.ndarray, rem_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-source mean over top-10% and remaining destinations for one frame."""

    top_count = float(top_mask.sum())
    rem_count = float(rem_mask.sum())
    m_top = gate[top_mask].sum(axis=0) / top_count
    m_rem = gate[rem_mask].sum(axis=0) / rem_count
    return m_top.astype(np.float32), m_rem.astype(np.float32)


def test_accumulator_partitions_source_and_destination_groups() -> None:
    hidden_size = 8
    top_mask = np.zeros(hidden_size, dtype=bool)
    top_mask[[0, 1]] = True
    eligible = np.ones(hidden_size, dtype=bool)
    rem_mask = eligible & ~top_mask
    values_within_top = 0.9
    values_top_to_rem = 0.6
    values_rem_to_top = 0.4
    values_within_rem = 0.15
    gate = _synthetic_frame_gate(
        hidden_size,
        top_mask,
        rem_mask,
        values_within_top,
        values_top_to_rem,
        values_rem_to_top,
        values_within_rem,
    )
    m_top, m_rem = _per_source_means_from_gate(gate, top_mask, rem_mask)
    frames_top = np.stack([m_top, m_top], axis=0)
    frames_rem = np.stack([m_rem, m_rem], axis=0)
    contexts = np.array([0, 0], dtype=np.int64)
    relevant_masks = top_mask[None, :]
    bin_edges = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    hist, sums, sums_sq, counts = accumulate_group_split_distributions(
        frames_top, frames_rem, contexts, relevant_masks, eligible, bin_edges
    )
    assert list(GROUP_NAMES) == [
        "within_top",
        "top_to_rem",
        "rem_to_top",
        "within_rem",
    ]
    assert hist.shape == (1, 4, bin_edges.size - 1)
    top_count = int(top_mask.sum())
    rem_count = int(rem_mask.sum())
    np.testing.assert_array_equal(
        counts[0], np.asarray([2 * top_count, 2 * top_count, 2 * rem_count, 2 * rem_count])
    )
    means, _stds = summarize_group_moments(sums, sums_sq, counts)
    np.testing.assert_allclose(
        means[0],
        [values_within_top, values_top_to_rem, values_rem_to_top, values_within_rem],
        atol=1e-6,
    )


def test_accumulator_rejects_shape_mismatch() -> None:
    frames_top = np.zeros((3, 4), dtype=np.float32)
    frames_rem = np.zeros((3, 5), dtype=np.float32)
    try:
        accumulate_group_split_distributions(
            frames_top,
            frames_rem,
            np.zeros(3, dtype=np.int64),
            np.zeros((1, 4), dtype=bool),
            np.ones(4, dtype=bool),
            np.linspace(0.0, 1.0, 3),
        )
    except ValueError as error:
        assert "share shape" in str(error)
        return
    raise AssertionError("Expected ValueError on shape mismatch")


def test_within_between_cohens_d_signs_and_pairing() -> None:
    counts = np.full((2, 4), 32, dtype=np.int64)
    means = np.array(
        [
            [0.90, 0.30, 0.20, 0.10],  # within_top >> top_to_rem, within_rem << rem_to_top
            [0.10, 0.20, 0.30, 0.90],  # within_top << top_to_rem, within_rem >> rem_to_top
        ],
        dtype=np.float64,
    )
    sums = means * counts
    variances = np.full((2, 4), 0.01, dtype=np.float64)
    sums_sq = variances * (counts - 1) + sums * means
    result = within_between_cohens_d(sums, sums_sq, counts)
    assert result.shape == (2, 2)
    assert result[0, 0] > 0  # within_top (0.90) > top_to_rem (0.30)
    assert result[0, 1] < 0  # within_rem (0.10) < rem_to_top (0.20)
    assert result[1, 0] < 0  # within_top (0.10) < top_to_rem (0.20)
    assert result[1, 1] > 0  # within_rem (0.90) > rem_to_top (0.30)


def test_group_panel_renders_four_stairs_and_two_mean_lines(
    tmp_path: Path,
) -> None:
    figure, axis = group_viz.plt.subplots()
    group_viz.plot_group_panel(
        axis,
        "Digit",
        3,
        np.linspace(0.0, 1.0, 6),
        np.asarray([[2, 4, 6, 8, 10], [1, 2, 3, 4, 5], [4, 3, 2, 1, 0], [0, 1, 2, 3, 4]]),
        np.asarray([0.72, 0.5, 0.4, 0.2]),
        np.asarray([30, 30, 270, 270]),
        np.asarray([0.7, 0.4]),
        density_limit=8.0,
    )
    assert axis.get_xlim() == (0.0, 1.0)
    assert "Digit 3" in axis.get_title()
    assert "within_top vs top" in axis.get_title() or "within_top" in axis.get_title().lower() or "d(within" in axis.get_title()
    lines = axis.get_lines()
    # Four axvline mean lines plus stairs use lines for edges
    axvline_lines = [line for line in lines if line.get_linestyle() == ":"]
    assert len(axvline_lines) == 4
    group_viz.plt.close(figure)


def test_render_grid_writes_expected_filename(tmp_path: Path) -> None:
    bin_edges = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    hist_counts = np.ones((9, 4, 5), dtype=np.int64) * 4
    group_mean = np.full((9, 4), 0.4, dtype=np.float64)
    group_count = np.full((9, 4), 20, dtype=np.int64)
    within_between = np.zeros((9, 2), dtype=np.float64)
    destination = group_viz.render_grid(
        "recurrent_sector",
        bin_edges,
        hist_counts,
        group_mean,
        group_count,
        within_between,
        tmp_path,
        100,
    )
    assert destination.name == "recurrent_sector_source_dest_group_split_distribution_grid.png"
    assert destination.exists()
