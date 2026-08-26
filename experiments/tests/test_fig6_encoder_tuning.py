"""Regression checks for the compact Figure 6 channel-wise tuning summary."""

from __future__ import annotations

import numpy as np

from utils.analysis.clutter.fig6_encoder_tuning import ENCODER_SHAPE, _tuning_profiles


def test_tuning_profiles_average_the_correct_encoder_axes() -> None:
    """Channel and spatial profiles preserve the requested axis after draw averaging."""

    unit_count = int(np.prod(ENCODER_SHAPE))
    source = np.arange(2 * 3 * unit_count, dtype=np.float64).reshape(2, 3, unit_count)
    channel_values, spatial_values = _tuning_profiles(
        {"encoder_sector": source, "encoder_digit": source + 1.0}
    )
    shaped = source.reshape(2, 3, *ENCODER_SHAPE)
    expected_channel = shaped.mean(axis=(1, 3, 4))
    expected_spatial = shaped.mean(axis=(1, 2)).reshape(2, -1)

    assert channel_values["sector"].shape == (2, ENCODER_SHAPE[0])
    assert spatial_values["sector"].shape == (2, int(np.prod(ENCODER_SHAPE[1:])))
    np.testing.assert_allclose(channel_values["sector"], expected_channel)
    np.testing.assert_allclose(channel_values["digit"], expected_channel + 1.0)
    np.testing.assert_allclose(spatial_values["sector"], expected_spatial)
    np.testing.assert_allclose(spatial_values["digit"], expected_spatial + 1.0)
