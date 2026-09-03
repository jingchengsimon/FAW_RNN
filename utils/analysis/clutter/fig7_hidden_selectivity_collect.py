"""Collect reset-excluded hidden or encoder selectivity masks for GaWF analyses.

Inputs are one GaWF checkpoint and the Clutter validation split. The output is a compact
``part1_selectivity.npz`` containing requested primary-population tuning, FDR masks, and
interaction masks; no test gates, timing events, or connection-level arrays are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.analysis.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device
from utils.analysis.clutter.fig7_relevance_stats import (
    interaction_dominant,
    permutation_selectivity,
    two_way_decomposition,
)
from utils.analysis.clutter.fig7_relevance_timing import collect_split, _selectivity_payload


POPULATIONS = ("hidden", "encoder")


def parse_args() -> argparse.Namespace:
    """Parse one-checkpoint compact selectivity collection arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--data_suffix", default="40h-uint8")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--permutation_batch_size", type=int, default=10)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--populations",
        nargs="+",
        choices=POPULATIONS,
        default=("hidden",),
        help="Activation populations to test; the historical default preserves hidden-only use.",
    )
    return parser.parse_args()


def _reset_excluded_mask(frame_count: int, frame_num: int) -> np.ndarray:
    """Return the flattened held-out-frame mask after dropping each recurrent reset step."""

    if frame_count <= 0 or frame_num <= 1 or frame_count % frame_num != 0:
        raise ValueError("Validation frames must contain complete recurrent windows.")
    return np.arange(frame_count, dtype=np.int64) % frame_num != 0


def main() -> None:
    """Run reset-excluded validation selectivity and save the requested compact payload."""

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    device = resolve_device(args.device, require_cuda_if_requested=True)
    dataset_args = argparse.Namespace(
        data_dir=str(args.data_dir), data_suffix=args.data_suffix, use_mmap=True,
        chan_num=2, use_sector_mode=True, predict_all_chars=False,
    )
    validation, num_pos = build_eval_dataset(dataset_args, "validation")
    model = build_model_from_ckpt(str(args.ckpt), num_pos, device, chan_num=2)
    collected = collect_split(
        validation, model, device, args.batch_size, args.num_workers, record_gates=False
    )
    raw_labels = collected["labels"].astype(np.int64, copy=False)
    keep = _reset_excluded_mask(raw_labels.shape[0], int(validation.frame_num))
    labels = raw_labels[keep]
    payload: dict[str, np.ndarray] = {}
    population_metadata: dict[str, object] = {}
    for population_index, population in enumerate(args.populations):
        activations = collected[population].astype(np.float32, copy=False)[keep]
        result = two_way_decomposition(activations, labels)
        inference = permutation_selectivity(
            activations,
            labels,
            result,
            resamples=args.resamples,
            seed=args.seed + 10_000 + population_index * 10_000,
            device=device,
            permutation_batch_size=args.permutation_batch_size,
            fdr_alpha=args.fdr_alpha,
        )
        payload.update(
            {
                f"primary_{population}_{key}": value
                for key, value in _selectivity_payload(result, inference).items()
            }
        )
        population_metadata[population] = {
            "units": int(activations.shape[1]),
            "interaction_dominant": int(interaction_dominant(result).sum()),
        }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "part1_selectivity.npz", **payload)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.ckpt.resolve()),
                "seed": args.seed,
                "validation_frames_before_reset_exclusion": int(raw_labels.shape[0]),
                "reset_frames_excluded": int((~keep).sum()),
                "validation_frames": int(labels.shape[0]),
                "resamples": args.resamples,
                "fdr_alpha": args.fdr_alpha,
                "populations": population_metadata,
                "scope": "reset-excluded primary selectivity",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
