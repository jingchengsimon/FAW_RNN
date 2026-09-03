"""Regression checks for Figure 6 input-gate weight-sign maps."""

from __future__ import annotations

import numpy as np

from utils.analysis.clutter.fig6_sector_gate_weight_sign import _sign_maps


def test_sign_maps_keep_all_sectors_and_both_weight_groups() -> None:
    """One selected frame per sector produces finite positive and negative 6-by-6 maps."""

    rng = np.random.default_rng(7)
    feedback = np.vstack(
        (np.zeros((1, 19), dtype=np.float32), rng.normal(size=(9, 19)).astype(np.float32))
    )
    labels = np.vstack(
        (
            np.zeros((1, 2), dtype=np.int64),
            np.column_stack((np.zeros(9, dtype=np.int64), np.arange(9, dtype=np.int64))),
        )
    )
    u = rng.normal(size=(2, 19)).astype(np.float32)
    v = rng.normal(size=(19, 1408)).astype(np.float32)
    weight_ih = rng.normal(size=(2, 1152)).astype(np.float32)

    maps, original_counts, target = _sign_maps(
        feedback,
        labels,
        u,
        v,
        weight_ih,
        selection_seed=7,
        gate_tau=0.5,
        gate_chunk_size=3,
        point_tolerance=1e-6,
        device="cpu",
    )

    assert maps.shape == (2, 9, 6, 6)
    assert np.isfinite(maps).all()
    np.testing.assert_array_equal(original_counts, np.ones(9, dtype=np.int64))
    assert int(target) == 1
