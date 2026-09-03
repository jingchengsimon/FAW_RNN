"""Regression checks for ten-seed recovery aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.analysis.clutter.clutter_multiseed_summary import (
    _recovery_mean_sem,
    load_recovery_curves,
    load_test_metrics,
)
from utils.analysis.clutter.supple1_feedback_ablation import _load_metrics, _mean_sem


def test_recovery_helpers_compute_offsetwise_sem() -> None:
    """Both figure paths use training-seed mean and SEM at every offset."""

    values = np.arange(30, dtype=np.float64).reshape(10, 3)
    expected_mean = values.mean(axis=0)
    expected_half = values.std(axis=0, ddof=1) / np.sqrt(10.0)
    for helper in (_recovery_mean_sem, _mean_sem):
        mean, half = helper(values)
        np.testing.assert_allclose(mean, expected_mean)
        np.testing.assert_allclose(half, expected_half)


def test_recovery_loaders_find_ten_seed_subdirectories(tmp_path: Path) -> None:
    """Both loaders preserve ten independent seed curves from nested result leaves."""

    offsets = np.asarray([-1, 1], dtype=np.int64)
    for seed in range(1, 11):
        fig1 = tmp_path / "fig1" / f"gawf-seed{seed:02d}"
        fig1.mkdir(parents=True)
        np.savez(
            fig1 / "fg_switch_offset_acc_gawf_sector_acc_h256.npz",
            offset_order=offsets,
            char_acc=np.asarray([seed, seed + 1], dtype=np.float32),
            sector_acc=np.asarray([seed + 2, seed + 3], dtype=np.float32),
        )
        supple1 = tmp_path / "supple1" / f"gawf-seed{seed:02d}"
        supple1.mkdir(parents=True)
        (supple1 / "ablation_metrics.json").write_text(
            json.dumps({"conditions": {"baseline": {"switch_offsets": offsets.tolist()}}}),
            encoding="utf-8",
        )

    loaded_offsets, curves = load_recovery_curves(tmp_path / "fig1")
    np.testing.assert_array_equal(loaded_offsets, offsets)
    assert curves["gawf"]["char"].shape == (10, 2)
    assert len(_load_metrics(str(tmp_path / "supple1"))) == 10


def test_test_metrics_rejects_incomplete_seed_set(tmp_path: Path) -> None:
    """Formal test bars must not silently render fewer than ten training seeds."""

    path = tmp_path / "metrics.csv"
    rows = ["model,seed,char_acc,sector_acc"]
    rows.extend(f"gawf,{seed},{80 + seed},{90 + seed}" for seed in range(1, 10))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly ten"):
        load_test_metrics(path)
