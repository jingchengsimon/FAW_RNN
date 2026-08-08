"""Compare repeated Atari runs against the project's acceleration variance criterion.

Inputs are four completed run directories: two matched baseline runs and two
matched accelerated runs.  The tool compares finite per-log-step loss, Q-value,
and return series.  It accepts an acceleration only when its repeated-run
dispersion and every accelerated-to-baseline distance stay within the baseline
cross-GPU/run envelope.  This is an end-to-end RL acceptance check rather than
a bitwise-forward equivalence test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = ("loss", "q_values_mean", "episodic_return_100")


def _parse_args() -> argparse.Namespace:
    """Parse the four completed Atari result directories and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-a", type=Path, required=True)
    parser.add_argument("--baseline-b", type=Path, required=True)
    parser.add_argument("--accelerated-a", type=Path, required=True)
    parser.add_argument("--accelerated-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_history(run_dir: Path) -> dict[int, dict[str, float]]:
    """Read finite logged Atari metrics keyed by their global environment step."""
    history_path = run_dir / "metrics_history.jsonl"
    metrics_path = run_dir / "metrics.json"
    if not history_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Expected completed metrics and history under {run_dir}")
    rows: dict[int, dict[str, float]] = {}
    for raw_line in history_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw_line)
        step = int(row["global_step"])
        values: dict[str, float] = {}
        for metric in METRICS:
            value = float(row.get(metric, float("nan")))
            if math.isfinite(value):
                values[metric] = value
        rows[step] = values
    if not rows:
        raise ValueError(f"No history rows found in {history_path}")
    return rows


def _distance(
    left: dict[int, dict[str, float]],
    right: dict[int, dict[str, float]],
) -> dict[str, dict[str, float | int]]:
    """Return aligned finite-series mean/max absolute distances for each metric."""
    shared_steps = sorted(set(left).intersection(right))
    result: dict[str, dict[str, float | int]] = {}
    for metric in METRICS:
        differences = [
            abs(left[step][metric] - right[step][metric])
            for step in shared_steps
            if metric in left[step] and metric in right[step]
        ]
        if differences:
            result[metric] = {
                "n_steps": len(differences),
                "mean_abs": float(sum(differences) / len(differences)),
                "max_abs": float(max(differences)),
            }
    return result


def _within_envelope(
    candidate: dict[str, dict[str, float | int]],
    envelope: dict[str, dict[str, float | int]],
) -> tuple[bool, dict[str, bool]]:
    """Check whether candidate distances stay within the baseline envelope."""
    decisions: dict[str, bool] = {}
    for metric in METRICS:
        if metric not in envelope:
            continue
        if metric not in candidate:
            decisions[metric] = False
            continue
        decisions[metric] = (
            float(candidate[metric]["mean_abs"]) <= float(envelope[metric]["mean_abs"])
            and float(candidate[metric]["max_abs"]) <= float(envelope[metric]["max_abs"])
        )
    return bool(decisions) and all(decisions.values()), decisions


def main() -> None:
    """Evaluate the repeated-run acceleration acceptance criterion and write JSON."""
    args = _parse_args()
    runs = {
        "baseline_a": _read_history(args.baseline_a),
        "baseline_b": _read_history(args.baseline_b),
        "accelerated_a": _read_history(args.accelerated_a),
        "accelerated_b": _read_history(args.accelerated_b),
    }
    baseline_envelope = _distance(runs["baseline_a"], runs["baseline_b"])
    accelerated_dispersion = _distance(runs["accelerated_a"], runs["accelerated_b"])
    cross_distances = {
        "baseline_a_to_accelerated_a": _distance(runs["baseline_a"], runs["accelerated_a"]),
        "baseline_b_to_accelerated_b": _distance(runs["baseline_b"], runs["accelerated_b"]),
    }
    dispersion_ok, dispersion_by_metric = _within_envelope(
        accelerated_dispersion, baseline_envelope
    )
    cross_checks = {
        label: _within_envelope(distance, baseline_envelope)
        for label, distance in cross_distances.items()
    }
    cross_ok = all(check[0] for check in cross_checks.values())
    report: dict[str, Any] = {
        "protocol": "matched repeated Atari acceleration variance acceptance",
        "metrics": list(METRICS),
        "baseline_run_to_run_envelope": baseline_envelope,
        "accelerated_run_to_run_dispersion": accelerated_dispersion,
        "accelerated_dispersion_within_baseline": dispersion_ok,
        "accelerated_dispersion_by_metric": dispersion_by_metric,
        "accelerated_to_baseline_distances": cross_distances,
        "accelerated_to_baseline_within_envelope": cross_ok,
        "accelerated_to_baseline_by_metric": {
            label: decision for label, (_, decision) in cross_checks.items()
        },
        "accepted": dispersion_ok and cross_ok,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit("acceleration variance acceptance failed")


if __name__ == "__main__":
    main()
