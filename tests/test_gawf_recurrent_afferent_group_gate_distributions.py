"""Tests for the recurrent afferent four-group gate distribution analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils_anal.gawf_recurrent_afferent_group_gate_distributions import (
    GROUP_NAMES,
    accumulate_afferent_group_distributions,
    summarize_group_moments,
    within_between_cohens_d,
)
import utils_viz.gawf_recurrent_afferent_group_gate_distributions as afferent_viz


def _synthetic_frame_gate(
    hidden_size: int,
    top_mask: np.ndarray,
    rem_mask: np.ndarray,
    within_top: float,
    top_to_rem: float,
    rem_to_top: float,
    within_rem: float,
) -> np.ndarray:
    """Build a full ``(hidden, hidden)`` gate matrix with 4 constant group cells.

    Convention: ``gate[h, i]`` where ``h`` is destination and ``i`` is source.
    ``top_to_rem`` means source in top10 and destination in remaining;
    ``rem_to_top`` means source in remaining and destination in top10.
    """

    gate = np.zeros((hidden_size, hidden_size), dtype=np.float32)
    dest_top = np.flatnonzero(top_mask)
    dest_rem = np.flatnonzero(rem_mask)
    src_top = dest_top
    src_rem = dest_rem
    gate[np.ix_(dest_top, src_top)] = within_top
    gate[np.ix_(dest_rem, src_top)] = top_to_rem
    gate[np.ix_(dest_top, src_rem)] = rem_to_top
    gate[np.ix_(dest_rem, src_rem)] = within_rem
    return gate


def _per_destination_means_from_gate(
    gate: np.ndarray, top_mask: np.ndarray, rem_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-destination mean over top-10% and remaining SOURCES for one frame."""

    top_count = float(top_mask.sum())
    rem_count = float(rem_mask.sum())
    # gate shape (dest, source). Mean over sources: gate[:, mask].mean(axis=1).
    m_from_top = gate[:, top_mask].sum(axis=1) / top_count
    m_from_rem = gate[:, rem_mask].sum(axis=1) / rem_count
    return m_from_top.astype(np.float32), m_from_rem.astype(np.float32)


def test_accumulator_partitions_afferent_destination_and_source_groups() -> None:
    hidden_size = 8
    top_mask = np.zeros(hidden_size, dtype=bool)
    top_mask[[0, 1]] = True
    eligible = np.ones(hidden_size, dtype=bool)
    rem_mask = eligible & ~top_mask
    values_within_top = 0.9
    values_top_to_rem = 0.6  # source top -> destination rem
    values_rem_to_top = 0.4  # source rem -> destination top
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
    m_from_top, m_from_rem = _per_destination_means_from_gate(gate, top_mask, rem_mask)
    frames_from_top = np.stack([m_from_top, m_from_top], axis=0)
    frames_from_rem = np.stack([m_from_rem, m_from_rem], axis=0)
    contexts = np.array([0, 0], dtype=np.int64)
    relevant_masks = top_mask[None, :]
    bin_edges = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    hist, sums, sums_sq, counts = accumulate_afferent_group_distributions(
        frames_from_top,
        frames_from_rem,
        contexts,
        relevant_masks,
        eligible,
        bin_edges,
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
    # Afferent counts per group:
    # within_top: destination in top, from top sources -> 2 frames * top_count values
    # top_to_rem: destination in rem, from top sources -> 2 frames * rem_count values
    # rem_to_top: destination in top, from rem sources -> 2 frames * top_count values
    # within_rem: destination in rem, from rem sources -> 2 frames * rem_count values
    np.testing.assert_array_equal(
        counts[0], np.asarray([2 * top_count, 2 * rem_count, 2 * top_count, 2 * rem_count])
    )
    means, _stds = summarize_group_moments(sums, sums_sq, counts)
    np.testing.assert_allclose(
        means[0],
        [values_within_top, values_top_to_rem, values_rem_to_top, values_within_rem],
        atol=1e-6,
    )


def test_accumulator_rejects_afferent_shape_mismatch() -> None:
    frames_from_top = np.zeros((3, 4), dtype=np.float32)
    frames_from_rem = np.zeros((3, 5), dtype=np.float32)
    try:
        accumulate_afferent_group_distributions(
            frames_from_top,
            frames_from_rem,
            np.zeros(3, dtype=np.int64),
            np.zeros((1, 4), dtype=bool),
            np.ones(4, dtype=bool),
            np.linspace(0.0, 1.0, 3),
        )
    except ValueError as error:
        assert "share shape" in str(error)
        return
    raise AssertionError("Expected ValueError on shape mismatch")


def test_afferent_within_between_cohens_d_pairs_by_destination_group() -> None:
    counts = np.full((2, 4), 32, dtype=np.int64)
    means = np.array(
        [
            # row 0: within_top >> rem_to_top (both at top dest); within_rem << top_to_rem (both at rem dest)
            [0.90, 0.30, 0.20, 0.10],
            # row 1: within_top << rem_to_top; within_rem >> top_to_rem
            [0.10, 0.20, 0.30, 0.90],
        ],
        dtype=np.float64,
    )
    sums = means * counts
    variances = np.full((2, 4), 0.01, dtype=np.float64)
    sums_sq = variances * (counts - 1) + sums * means
    result = within_between_cohens_d(sums, sums_sq, counts)
    assert result.shape == (2, 2)
    # Column 0 compares within_top (idx 0) vs rem_to_top (idx 2) at top destinations.
    # Column 1 compares within_rem (idx 3) vs top_to_rem (idx 1) at remaining destinations.
    assert result[0, 0] > 0  # within_top (0.90) > rem_to_top (0.20)
    assert result[0, 1] < 0  # within_rem (0.10) < top_to_rem (0.30)
    assert result[1, 0] < 0  # within_top (0.10) < rem_to_top (0.30)
    assert result[1, 1] > 0  # within_rem (0.90) > top_to_rem (0.20)


def test_afferent_panel_renders_four_stairs_and_four_mean_lines() -> None:
    figure, axis = afferent_viz.plt.subplots()
    afferent_viz.plot_afferent_group_panel(
        axis,
        "Digit",
        3,
        np.linspace(0.0, 1.0, 6),
        np.asarray([[2, 4, 6, 8, 10], [1, 2, 3, 4, 5], [4, 3, 2, 1, 0], [0, 1, 2, 3, 4]]),
        np.asarray([0.72, 0.5, 0.4, 0.2]),
        np.asarray([30, 270, 30, 270]),
        np.asarray([0.7, 0.4]),
        density_limit=8.0,
    )
    assert axis.get_xlim() == (0.0, 1.0)
    assert "Digit 3" in axis.get_title()
    assert "AFFERENT" in axis.get_title()
    assert "d(within_top" in axis.get_title() and "d(within_rem" in axis.get_title()
    axvline_lines = [line for line in axis.get_lines() if line.get_linestyle() == ":"]
    assert len(axvline_lines) == 4
    afferent_viz.plt.close(figure)


def test_afferent_render_grid_writes_expected_filenames(tmp_path: Path) -> None:
    bin_edges = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    for cell, (_nrows, _ncols, levels, _factor, _label) in afferent_viz.GRID_SPECS.items():
        hist_counts = np.ones((levels, 4, 5), dtype=np.int64) * 4
        group_mean = np.full((levels, 4), 0.4, dtype=np.float64)
        group_count = np.full((levels, 4), 20, dtype=np.int64)
        within_between = np.zeros((levels, 2), dtype=np.float64)
        destination = afferent_viz.render_grid(
            cell,
            bin_edges,
            hist_counts,
            group_mean,
            group_count,
            within_between,
            tmp_path,
            100,
        )
        assert destination.name == f"{cell}_afferent_group_split_distribution_grid.png"
        assert destination.exists()
