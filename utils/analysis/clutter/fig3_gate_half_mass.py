"""Audit the exact 0.5 point mass in saved Figure 3 GaWF gate trajectories.

The input is one directory containing ``seedXX/gawf_gate_trajectory.npz`` files. Gates are
reconstructed with the original eager float32 computation, counted without saving dense gate
tensors, and summarized as per-seed values plus 10-seed mean ± SEM in JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from utils.analysis.clutter.fig3_gate_distribution import _gate_tensors


def parse_args() -> argparse.Namespace:
    """Parse trajectory-audit arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--input_size", type=int, default=1152)
    parser.add_argument("--gate_tau", type=float, default=0.5)
    parser.add_argument("--half_tolerance", type=float, default=1e-6)
    parser.add_argument("--inactive_u_tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _mean_sem(values: list[float]) -> dict[str, float]:
    """Return the arithmetic mean and cross-seed SEM."""

    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return {"mean": mean, "sem": sem}


def audit_seed(
    trajectory_path: Path,
    device: torch.device,
    chunk_size: int,
    input_size: int,
    gate_tau: float,
    half_tolerance: float,
    inactive_u_tolerance: float,
) -> dict[str, object]:
    """Count half-mass and middle-interval observations for one saved trajectory."""

    with np.load(trajectory_path) as trajectory:
        feedback = trajectory["feedback"].astype(np.float32, copy=False)
        u = trajectory["U"].astype(np.float32, copy=False)
        v = trajectory["V"].astype(np.float32, copy=False)

    flat_feedback = feedback.reshape(-1, feedback.shape[-1])
    reset_mask = np.all(flat_feedback == 0.0, axis=1)
    hidden_size = int(u.shape[0])
    source_sizes = {"input": input_size, "recurrent": hidden_size}
    counts = {
        kind: {"total": 0, "half": 0, "half_nonreset": 0, "middle": 0}
        for kind in source_sizes
    }
    u_tensor = torch.from_numpy(u).to(device)
    v_tensor = torch.from_numpy(v).to(device)

    for start in range(0, flat_feedback.shape[0], chunk_size):
        end = min(start + chunk_size, flat_feedback.shape[0])
        feedback_tensor = torch.from_numpy(flat_feedback[start:end]).to(device)
        with torch.no_grad():
            gate_input, gate_recurrent = _gate_tensors(
                feedback_tensor, u_tensor, v_tensor, input_size, gate_tau
            )
            for kind, gate in (("input", gate_input), ("recurrent", gate_recurrent)):
                half = torch.abs(gate - 0.5) < half_tolerance
                middle = (gate >= 0.1) & (gate <= 0.9)
                counts[kind]["total"] += int(gate.numel())
                counts[kind]["half"] += int(half.sum().item())
                counts[kind]["middle"] += int(middle.sum().item())
                nonreset = torch.from_numpy(~reset_mask[start:end]).to(device)
                counts[kind]["half_nonreset"] += int(half[nonreset].sum().item())

    result: dict[str, object] = {
        "trajectory": str(trajectory_path.resolve()),
        "n_frames": int(flat_feedback.shape[0]),
        "n_reset_frames": int(reset_mask.sum()),
        "reset_frame_fraction": float(reset_mask.mean()),
        "inactive_feedback_hidden_units": int(
            np.count_nonzero(np.max(np.abs(u), axis=1) < inactive_u_tolerance)
        ),
        "u_row_max_abs_quantiles": {
            str(q): float(value)
            for q, value in zip(
                (0.0, 0.25, 0.5, 0.75, 1.0),
                np.quantile(np.max(np.abs(u), axis=1), (0.0, 0.25, 0.5, 0.75, 1.0)),
            )
        },
    }
    for kind in source_sizes:
        population = counts[kind]
        nonhalf = population["total"] - population["half"]
        result[kind] = {
            **population,
            "half_fraction": population["half"] / population["total"],
            "half_nonreset_fraction": population["half_nonreset"] / population["total"],
            "middle_fraction": population["middle"] / population["total"],
            "middle_fraction_after_half_exclusion": (
                (population["middle"] - population["half"]) / nonhalf
            ),
        }
    return result


def aggregate(seed_results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate the requested seed-level quantities with seed as the sampling unit."""

    summary: dict[str, object] = {
        "n_seeds": len(seed_results),
        "inactive_feedback_hidden_units": _mean_sem(
            [float(seed["inactive_feedback_hidden_units"]) for seed in seed_results]
        ),
    }
    for kind in ("input", "recurrent"):
        summary[kind] = {}
        for metric in (
            "half_fraction",
            "half_nonreset_fraction",
            "middle_fraction",
            "middle_fraction_after_half_exclusion",
        ):
            summary[kind][metric] = _mean_sem(
                [float(seed[kind][metric]) for seed in seed_results]  # type: ignore[index]
            )
    return summary


def main() -> None:
    """Run all requested seeds and save one compact JSON result."""

    args = parse_args()
    device = torch.device(args.device)
    seed_results = []
    for seed in args.seeds:
        trajectory_path = (
            args.trajectory_root / f"seed{seed:02d}" / "gawf_gate_trajectory.npz"
        )
        if not trajectory_path.is_file():
            raise FileNotFoundError(trajectory_path)
        result = audit_seed(
            trajectory_path,
            device,
            args.chunk_size,
            args.input_size,
            args.gate_tau,
            args.half_tolerance,
            args.inactive_u_tolerance,
        )
        result["seed"] = seed
        seed_results.append(result)
        print(f"seed {seed:02d} complete", flush=True)

    payload = {
        "definition": {
            "half": f"abs(g - 0.5) < {args.half_tolerance:g}",
            "middle": "0.1 <= g <= 0.9",
            "inactive_feedback_row": (
                f"max_r abs(U[j, r]) < {args.inactive_u_tolerance:g}"
            ),
            "aggregation": "seed-level arithmetic mean +/- SEM",
            "context_variance_for_reset_point_mass": {
                "definition": "variance across digit-sector labels at reset frames",
                "quantiles": {"min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0},
            },
        },
        "seeds": seed_results,
        "aggregate": aggregate(seed_results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
