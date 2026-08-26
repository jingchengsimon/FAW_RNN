"""Collect only the hidden-unit selectivity masks needed by ten-seed Figure 7.

Inputs are one GaWF checkpoint and the Clutter validation split. The output is a compact
``part1_selectivity.npz`` containing the primary hidden tuning, FDR masks, and interaction mask;
no test gates, timing events, or connection-level arrays are retained.
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
    return parser.parse_args()


def main() -> None:
    """Run validation-hidden selectivity and save the compact primary payload."""

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
    labels = collected["labels"].astype(np.int64, copy=False)
    hidden = collected["hidden"].astype(np.float32, copy=False)
    result = two_way_decomposition(hidden, labels)
    inference = permutation_selectivity(
        hidden,
        labels,
        result,
        resamples=args.resamples,
        seed=args.seed + 10_000,
        device=device,
        permutation_batch_size=args.permutation_batch_size,
        fdr_alpha=args.fdr_alpha,
    )
    payload = {
        f"primary_hidden_{key}": value
        for key, value in _selectivity_payload(result, inference).items()
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "part1_selectivity.npz", **payload)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.ckpt.resolve()),
                "seed": args.seed,
                "validation_frames": int(labels.shape[0]),
                "hidden_units": int(hidden.shape[1]),
                "resamples": args.resamples,
                "fdr_alpha": args.fdr_alpha,
                "interaction_dominant": int(interaction_dominant(result).sum()),
                "scope": "primary hidden selectivity only",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
