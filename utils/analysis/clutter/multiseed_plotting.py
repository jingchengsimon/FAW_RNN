"""Shared visual convention for bar charts summarized across training seeds.

The helper overlays one neutral-gray, jittered point for every seed on a bar chart.  It is used
only when the bar heights and error bars are computed from independent training-seed values.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


SEED_POINT_COLOR = "#333333"
SEED_POINT_ALPHA = 0.52
SEED_POINT_SIZE = 10


def add_seed_points(
    axis: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    *,
    bar_width: float,
    show: bool = True,
    rng: np.random.Generator | None = None,
) -> None:
    """Overlay gray, deterministically jittered seed values on one or more bars.

    Args:
        axis: Bar-chart axis.
        positions: One x position per bar, shape ``(bars,)``.
        values: Independent seed values, shape ``(seeds, bars)``.
        bar_width: Width of each associated bar in axis units.
        show: Whether to render the points; the default implements the delivery convention.
        rng: Optional generator for deterministic jitter across a multi-panel figure.
    """

    if not show:
        return
    positions = np.asarray(positions, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != positions.size:
        raise ValueError(
            "Seed-point values must have shape (seeds, bars) matching the number of positions."
        )
    if values.shape[0] < 2:
        return
    generator = np.random.default_rng(0) if rng is None else rng
    for index, position in enumerate(positions):
        jitter = generator.uniform(-0.22 * bar_width, 0.22 * bar_width, size=values.shape[0])
        axis.scatter(
            position + jitter,
            values[:, index],
            s=SEED_POINT_SIZE,
            color=SEED_POINT_COLOR,
            alpha=SEED_POINT_ALPHA,
            linewidths=0,
            zorder=3,
        )
