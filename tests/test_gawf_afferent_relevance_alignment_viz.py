"""Tests for the aggregate afferent Part-2 alignment / bootstrap-d visualisation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import utils_viz.gawf_afferent_relevance_alignment as afferent_alignment_viz


CELLS = (
    ("input", "sector"),
    ("input", "digit"),
    ("recurrent", "sector"),
    ("recurrent", "digit"),
)


def _build_alignment_matrix(levels: int, diagonal_boost: float, rng: np.random.Generator) -> np.ndarray:
    """Return a `(levels, levels)` matrix that looks like cosine similarity output."""

    matrix = 0.1 * rng.standard_normal((levels, levels)).astype(np.float32)
    np.fill_diagonal(matrix, diagonal_boost)
    return matrix


def _build_synthetic_npz(destination: Path, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Persist a synthetic NPZ mirroring the aggregate keys produced by the anal script."""

    arrays: dict[str, np.ndarray] = {
        "bin_edges": np.linspace(0.0, 1.0, 6, dtype=np.float32),
    }
    for gate, factor in CELLS:
        cell = f"{gate}_{factor}"
        levels = 9 if factor == "sector" else 10
        arrays[f"{cell}_alignment_matrix"] = _build_alignment_matrix(levels, 0.42, rng)
        arrays[f"{cell}_alignment_null"] = 0.05 * rng.standard_normal(64).astype(np.float32)
        arrays[f"{cell}_top10_bootstrap_d"] = (
            0.4 + 0.05 * rng.standard_normal(64)
        ).astype(np.float32)
        arrays[f"{cell}_top10_relevance_null_d"] = (
            0.01 * rng.standard_normal(64)
        ).astype(np.float32)
    np.savez_compressed(destination, **arrays)
    return arrays


def _build_synthetic_metadata(destination: Path) -> dict[str, object]:
    """Persist a synthetic metadata JSON matching the aggregate report layout."""

    cells: dict[str, object] = {}
    for gate, factor in CELLS:
        cell = f"{gate}_{factor}"
        levels = 9 if factor == "sector" else 10
        cells[cell] = {
            "gate": f"{gate} DESTINATION gate",
            "factor": factor,
            "context_levels": levels,
            "continuous_alignment": {
                "diagonal_minus_off_diagonal": 0.31 + (0.05 if gate == "recurrent" else 0.0),
                "permutation_p_value": 0.001,
                "permutation_alternative": "two-sided",
            },
            "top_percent": {
                "10": {
                    "cohens_d": 0.42 if gate == "input" else 0.55,
                    "bootstrap_ci95": [0.35, 0.62],
                    "relevant_units_per_level": [3] * levels,
                    "relevance_shuffle_p_value": 0.001,
                }
            },
        }
    metadata: dict[str, object] = {
        "resamples": 64,
        "seed": 260718,
        "top_percent": [10],
        "aggregate_measures": {
            "bootstrap_d": "synthetic",
            "continuous_alignment": "synthetic",
        },
        "cells": cells,
    }
    with destination.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj)
    return metadata


def test_plot_alignment_grid_writes_paired_png_and_pdf(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    npz_path = tmp_path / "afferent_top10_gate_distributions.npz"
    metadata_path = tmp_path / "afferent_top10_gate_distributions_meta.json"
    _build_synthetic_npz(npz_path, rng)
    metadata = _build_synthetic_metadata(metadata_path)
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    with np.load(npz_path, allow_pickle=False) as data:
        png_path = afferent_alignment_viz.plot_alignment_grid(data, metadata, fig_dir, dpi=100)
    assert png_path.name == "part2_afferent_continuous_alignment.png"
    assert png_path.exists()
    assert (fig_dir / "part2_afferent_continuous_alignment.pdf").exists()


def test_plot_top_percent_bars_writes_expected_filename(tmp_path: Path) -> None:
    metadata_path = tmp_path / "afferent_top10_gate_distributions_meta.json"
    metadata = _build_synthetic_metadata(metadata_path)
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    destination = afferent_alignment_viz.plot_top_percent_bars(
        metadata, fig_dir, top_percent=10, dpi=100
    )
    assert destination.name == "part2_afferent_relevance_effects_top10.png"
    assert destination.exists()


def test_plot_top_percent_bars_raises_when_percent_missing(tmp_path: Path) -> None:
    metadata_path = tmp_path / "afferent_top10_gate_distributions_meta.json"
    metadata = _build_synthetic_metadata(metadata_path)
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    try:
        afferent_alignment_viz.plot_top_percent_bars(
            metadata, fig_dir, top_percent=25, dpi=100
        )
    except KeyError as error:
        assert "25" in str(error)
    else:
        raise AssertionError("expected KeyError for missing top_percent bucket")


def test_plot_alignment_grid_rejects_non_square_matrix(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    npz_path = tmp_path / "afferent_top10_gate_distributions.npz"
    metadata_path = tmp_path / "afferent_top10_gate_distributions_meta.json"
    _build_synthetic_npz(npz_path, rng)
    metadata = _build_synthetic_metadata(metadata_path)
    # Corrupt one alignment matrix to be non-square so validation must fail.
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["input_sector_alignment_matrix"] = np.zeros((9, 4), dtype=np.float32)
    corrupted = tmp_path / "corrupted.npz"
    np.savez_compressed(corrupted, **arrays)
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    with np.load(corrupted, allow_pickle=False) as data:
        try:
            afferent_alignment_viz.plot_alignment_grid(data, metadata, fig_dir, dpi=100)
        except ValueError as error:
            assert "square" in str(error)
        else:
            raise AssertionError("expected ValueError for non-square alignment matrix")
