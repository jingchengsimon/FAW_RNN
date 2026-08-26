"""Checks for compact ten-seed recurrent-gate analysis."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from utils.analysis.clutter.fig3_gate_distribution import _gate_tensors
from utils.analysis.clutter.fig7_recurrent_gate_multiseed import (
    RESULT_NAME,
    _compact_paths,
    _recurrent_gate_chunks,
)


def test_compact_paths_use_canonical_subdirectory(tmp_path) -> None:
    expected = []
    for seed in (1, 2):
        path = tmp_path / f"seed{seed:02d}" / "compact" / RESULT_NAME
        path.parent.mkdir(parents=True)
        path.touch()
        expected.append(path)
    assert _compact_paths(tmp_path) == expected


def test_recurrent_only_chunks_match_full_gate_reconstruction() -> None:
    rng = np.random.default_rng(4)
    feedback = rng.normal(size=(2, 3, 2)).astype(np.float32)
    u = rng.normal(size=(3, 2)).astype(np.float32)
    v = rng.normal(size=(2, 7)).astype(np.float32)
    reconstructed = np.concatenate(
        [gate for _start, _end, gate in _recurrent_gate_chunks(
            feedback, u, v, 4, 0.5, 2, "cpu"
        )]
    )
    with torch.no_grad():
        _input_gate, expected = _gate_tensors(
            torch.from_numpy(feedback.reshape(-1, 2)),
            torch.from_numpy(u),
            torch.from_numpy(v),
            4,
            0.5,
        )
    np.testing.assert_allclose(reconstructed, expected.numpy(), rtol=0.0, atol=0.0)
