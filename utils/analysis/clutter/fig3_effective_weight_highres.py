"""Rebuild reset-excluded Figure 3 input/recurrent weight histograms at high resolution.

The script reconstructs both ``G\odot W`` distributions from the saved ten-seed trajectories,
accumulating exact fixed-edge 5,000-bin counts on the selected device.  It writes only the four
weight-row histograms required to replace the Figure 3 2-by-2 weight panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.analysis.clutter.fig3_gate_distribution import (
    _gate_tensors,
    exclude_zero_feedback_reset_frames,
)


def parse_args() -> argparse.Namespace:
    """Parse trajectory directories and high-resolution output settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--bins", type=int, default=5000)
    return parser.parse_args()


def _histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Count values in a fixed-edge histogram, retaining both endpoint bins."""

    bins = edges.size - 1
    indices = np.floor((values.reshape(-1) - edges[0]) * bins / (edges[-1] - edges[0]))
    return np.bincount(np.clip(indices.astype(np.int64), 0, bins - 1), minlength=bins)


def _gpu_histogram(values: torch.Tensor, max_abs: float, bins: int) -> torch.Tensor:
    """Accumulate exact integer histogram counts without transferring gate tensors to CPU."""

    indices = torch.floor((values.reshape(-1) + max_abs) * bins / (2.0 * max_abs))
    indices = indices.to(torch.int64).clamp_(0, bins - 1)
    return torch.bincount(indices, minlength=bins)


def main() -> None:
    """Write pooled 5,000-bin static/effective histograms for both Figure 3 weight panels."""

    args = parse_args()
    if args.chunk_size <= 0 or args.bins <= 500:
        raise ValueError("chunk_size must be positive and bins must exceed 500")
    directories = [Path(value) for value in args.seed_dirs]
    if len(directories) < 2:
        raise ValueError("At least two independent seed directories are required")
    records = []
    for directory in directories:
        with np.load(directory / "gawf_gate_trajectory.npz", allow_pickle=False) as data:
            feedback, _labels, reset_frames = exclude_zero_feedback_reset_frames(
                data["feedback"], data["labels"]
            )
            records.append(
                {
                    "feedback": feedback,
                    "u": data["U"],
                    "v": data["V"],
                    "weight_input": data["weight_ih"],
                    "weight_recurrent": data["weight_hh"],
                    "input_size": int(data["weight_ih"].shape[1]),
                    "reset_frames": reset_frames,
                }
            )
    max_abs = {
        kind: max(float(np.max(np.abs(record[f"weight_{kind}"]))) for record in records)
        for kind in ("input", "recurrent")
    }
    edges = {
        kind: np.linspace(-max_abs[kind], max_abs[kind], args.bins + 1, dtype=np.float64)
        for kind in max_abs
    }
    static_counts = {kind: np.zeros(args.bins, dtype=np.int64) for kind in max_abs}
    device = torch.device(args.device)
    effective_counts = {kind: torch.zeros(args.bins, dtype=torch.int64, device=device) for kind in max_abs}
    for directory, record in zip(directories, records):
        u_tensor = torch.from_numpy(record["u"]).to(device)
        v_tensor = torch.from_numpy(record["v"]).to(device)
        weights = {
            kind: torch.from_numpy(record[f"weight_{kind}"]).to(device) for kind in max_abs
        }
        for kind in max_abs:
            static_counts[kind] += _histogram(record[f"weight_{kind}"], edges[kind])
        feedback = record["feedback"]
        for start in range(0, feedback.shape[0], args.chunk_size):
            feedback_tensor = torch.from_numpy(feedback[start : start + args.chunk_size]).to(device)
            with torch.no_grad():
                gate_input, gate_recurrent = _gate_tensors(
                    feedback_tensor, u_tensor, v_tensor, record["input_size"], 0.5
                )
                effective_counts["input"] += _gpu_histogram(
                    gate_input * weights["input"], max_abs["input"], args.bins
                )
                effective_counts["recurrent"] += _gpu_histogram(
                    gate_recurrent * weights["recurrent"], max_abs["recurrent"], args.bins
                )
        print(f"{directory.name}: reset_frames_excluded={record['reset_frames']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for kind in max_abs:
        arrays[f"effective_edges_{kind}"] = edges[kind].astype(np.float32)
        arrays[f"hist_weight_{kind}"] = static_counts[kind]
        arrays[f"hist_effective_{kind}"] = effective_counts[kind].cpu().numpy()
    np.savez_compressed(args.output_dir / "fig3_effective_weight_5000bins.npz", **arrays)
    metadata = {
        "n_seeds": len(records),
        "bins": args.bins,
        "device": args.device,
        "reset_frames_excluded_per_seed": [int(record["reset_frames"]) for record in records],
        "ranges": {kind: [-max_abs[kind], max_abs[kind]] for kind in max_abs},
    }
    (args.output_dir / "fig3_effective_weight_5000bins.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
