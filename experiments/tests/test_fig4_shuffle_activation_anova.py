"""Regression check for the compact CUDA aggregate-moment calculation."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from utils.analysis.clutter.fig4_shuffle_activation_anova import RepeatedCudaMoments
from utils.analysis.variance_decomposition import StreamingMoments


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_aggregate_matches_canonical_balanced_decomposition() -> None:
    """The compact collector must agree with the existing balanced ANOVA algebra."""

    labels = np.asarray(
        [[digit, sector] for sector in range(9) for digit in range(10) for _ in range(2)],
        dtype=np.int64,
    )
    values = np.column_stack(
        [labels[:, 0] + 0.2 * labels[:, 1], labels[:, 0] * labels[:, 1] + 1.0]
    ).astype(np.float32)
    canonical = StreamingMoments(num_units=2)
    canonical.update(values, labels)
    expected = canonical.finalize()

    device = torch.device("cuda")
    collector = RepeatedCudaMoments(repeats=1, num_units=2, device=device)
    collector.update(
        torch.from_numpy(values).to(device),
        torch.from_numpy(labels).to(device),
        torch.ones((1, labels.shape[0]), dtype=torch.bool, device=device),
    )
    observed = collector.finalize()

    for factor in ("sector", "digit", "interaction"):
        np.testing.assert_allclose(observed[factor], 100.0 * expected.aggregate_cm[factor])
    np.testing.assert_allclose(
        observed["between_condition_var"],
        expected.sum_squares["total_cm"].sum() / expected.total_trials,
    )
    np.testing.assert_allclose(observed["residual_frac"], expected.aggregate_trial["residual"])
