"""Regression checks for the Sector-0 input-connection partition."""

from __future__ import annotations

import numpy as np

from utils.analysis.clutter.supple2_input_gate_sign_magnitude_sector import (
    ALL_SECTORS_RESULT_NAME,
    _collect_connection_means,
    _connection_level_stats_by_seed,
    _load_all_sector_by_seed,
    _seed_level_stats,
)


def test_sector0_connection_collection_keeps_matching_and_other_sources() -> None:
    """One equal-n frame per sector yields finite source-partitioned connection means."""

    rng = np.random.default_rng(11)
    values, original_counts, target = _collect_connection_means(
        np.vstack(
            (np.zeros((1, 19), dtype=np.float32), rng.normal(size=(9, 19)).astype(np.float32))
        ),
        np.vstack(
            (
                np.zeros((1, 2), dtype=np.int64),
                np.column_stack((np.zeros(9, dtype=np.int64), np.arange(9, dtype=np.int64))),
            )
        ),
        rng.normal(size=(2, 19)).astype(np.float32),
        rng.normal(size=(19, 1408)).astype(np.float32),
        rng.normal(size=(2, 1152)).astype(np.float32),
        selection_seed=11,
        gate_tau=0.5,
        gate_chunk_size=3,
        device="cpu",
    )

    assert int(target) == 1
    np.testing.assert_array_equal(original_counts, np.ones(9, dtype=np.int64))
    assert values["sector0_sources_weight"].shape == (256,)
    assert values["other_sources_weight"].shape == (2048,)
    assert all(np.isfinite(value).all() for value in values.values())


def test_nine_sector_summary_keeps_seed_as_inference_unit(tmp_path) -> None:
    """Ten compact seeds produce finite matching/other seed-level statistics."""

    weights = np.linspace(-2.0, 2.0, 2 * 1152, dtype=np.float32).reshape(2, 1152)
    magnitudes = np.abs(weights)
    sector_scale = np.linspace(-0.2, 0.2, 9, dtype=np.float32)[:, None, None]
    means = 0.5 + sector_scale * magnitudes[None, :, :]
    for seed in range(1, 11):
        output = tmp_path / f"seed{seed:02d}"
        output.mkdir()
        np.savez_compressed(
            output / ALL_SECTORS_RESULT_NAME,
            weight=weights,
            sector_gate_mean=means,
        )

    by_seed = _load_all_sector_by_seed(tmp_path)
    assert all(len(frames) == 10 for frames in by_seed.values())
    assert len(by_seed["sector0_sources"][0]) == 9 * 2 * 128
    assert len(by_seed["other_sources"][0]) == 9 * 2 * 1024
    stats = _seed_level_stats(by_seed)
    for group in stats.values():
        assert all(np.isfinite(metric["mean"]) for metric in group.values())

    diagnostics = _connection_level_stats_by_seed(by_seed)
    assert set(diagnostics) == {"matching", "other"}
    for group in diagnostics.values():
        assert set(group) == {
            *(f"seed{seed:02d}" for seed in range(1, 11)),
            "all_seeds_pooled",
        }
        for result in group.values():
            assert np.isfinite(result["gap"])
            assert 0.0 <= result["gap_p_value"] <= 1.0
            assert np.isfinite(result["ols_slope"])
            assert 0.0 <= result["ols_slope_p_value"] <= 1.0
