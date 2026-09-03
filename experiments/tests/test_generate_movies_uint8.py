"""Check the production defaults used by Clutter stimulus generation."""

from __future__ import annotations

import sys

import numpy as np

from source.clutter.generate_movies import (
    StimulusConfig,
    generate_stimulus_video,
    parse_args,
)


def test_generate_movies_defaults_to_reproducible_uint8(monkeypatch) -> None:
    """The public CLI keeps uint8 storage and an explicit deterministic seed by default."""

    monkeypatch.setattr(sys, "argv", ["generate_movies.py"])
    args = parse_args()
    assert args.storage_dtype == "uint8"
    assert args.seed == 42
    assert args.output_mode == "simple"


def test_simple_generation_writes_uint8_memmap(tmp_path) -> None:
    """A minimal generated stimulus has the promised storage dtype and shape."""

    config = StimulusConfig(
        width=96,
        height=96,
        duration_seconds=1,
        fps=1,
        fg_speeds=[1.0],
        bg_char_counts=[1],
        bg_mean_speeds=[1.0],
        output_dir=str(tmp_path),
        output_mode="simple",
        suffix="reg-train-test-uint8",
        storage_dtype="uint8",
    )
    digits = {digit: [np.full((28, 28), 255, dtype=np.uint8)] for digit in range(10)}
    np.random.seed(42)
    generate_stimulus_video(config, digits)
    generated = np.load(tmp_path / "stimulus_reg-train-test-uint8.npy", mmap_mode="r")
    assert generated.dtype == np.uint8
    assert generated.shape == (1, 96, 96)
    assert int(generated.max()) == 255
