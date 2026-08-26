"""Checks for Fig7 paired seed-level inference."""

from __future__ import annotations

import pytest

from utils.analysis.clutter.fig7_recurrent_gate_disinhibition import seed_level_gap_p
from utils.analysis.clutter.fig7_recurrent_gate_disinhibition_delta import _holm_adjust


def _record(gap: float) -> dict[str, dict]:
    return {"TT": {"+": {"mean": gap}, "-": {"mean": 0.0}}}


def test_seed_level_gap_p_uses_paired_seed_differences() -> None:
    records = [_record(1.0), _record(1.0), _record(1.0)]
    assert seed_level_gap_p(records, "TT") == pytest.approx(0.25)


def test_seed_level_gap_p_is_one_for_zero_observed_mean() -> None:
    records = [_record(1.0), _record(-1.0)]
    assert seed_level_gap_p(records, "TT") == pytest.approx(1.0)


def test_holm_adjusts_all_preplanned_tests_together() -> None:
    raw = {("digit", "TT", "+"): 0.01, ("digit", "TT", "-"): 0.03,
           ("digit", "TT", "gap"): 0.04}
    adjusted = _holm_adjust(raw)
    expected = {("digit", "TT", "+"): 0.03, ("digit", "TT", "-"): 0.06,
                ("digit", "TT", "gap"): 0.06}
    assert adjusted == pytest.approx(expected)
