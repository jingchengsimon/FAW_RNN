"""Export afferent top-10% versus remaining DESTINATION-gate distributions.

Symmetric counterpart of the efferent SOURCE-gate top-10% analyses. For every hidden destination
``h``, the destination-side gate value is the mean of the raw post-sigmoid gate over its source
axis: 1152 encoder features for the input gate, 256 hidden units for the recurrent gate. Hidden
destinations are then filtered by the standard FDR + interaction-dominant hidden eligibility and
split into top-10% versus remaining per context using the same hidden tuning masks that drive the
recurrent efferent analysis. Four cells are exported, matching the naming used by the efferent
side (``gawf_remaining_relevance_distributions``): ``input_sector``, ``input_digit``,
``recurrent_sector``, ``recurrent_digit``.

In addition to the per-context histogram distributions, the same NPZ carries two aggregate
afferent Part-2 style measures mirroring ``gawf_symmetric_relevance_timing``:

* Bootstrap Cohen's d contrasting the top-``k%`` afferent gate columns against the remaining
  eligible afferent gate columns, plus the relevance-shuffle label null used to derive a
  one-sided p-value.
* Continuous cosine alignment between hidden activation tuning and afferent gate tuning across
  contexts, together with the permutation null on the diagonal-minus-off-diagonal contrast.

Outputs are one compressed NPZ plus JSON metadata under
``E_relevance_alignment/gawf_afferent_top10_gate_distributions/data/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir
from utils_anal.gawf_recurrent_sector_relevance_distributions import (
    accumulate_context_group_distributions,
    summarize_group_moments,
)


CELL_SPECS = {
    "input_sector":     ("input",     "sector", 9,  1),
    "input_digit":      ("input",     "digit",  10, 0),
    "recurrent_sector": ("recurrent", "sector", 9,  1),
    "recurrent_digit":  ("recurrent", "digit",  10, 0),
}
GROUP_NAMES = ("top10", "remaining")


def parse_args() -> argparse.Namespace:
    """Parse analysis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        default=(
            "./results/train_data/clutter/best_6model_param_matched_40h/"
            "gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
        ),
    )
    parser.add_argument("--data_dir", default="./stimuli")
    parser.add_argument("--data_suffix", default="40h-uint8")
    parser.add_argument(
        "--selectivity",
        default=str(
            output_dir("D_variance_decomposition", "gawf_symmetric_relevance_timing", "data")
            / "part1_selectivity.npz"
        ),
    )
    parser.add_argument(
        "--split_report",
        default=str(
            output_dir("H_controls", "gawf_symmetric_relevance_timing", "data")
            / "part0_splits.json"
        ),
    )
    parser.add_argument(
        "--save_dir",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_afferent_top10_gate_distributions",
                "data",
            )
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--num_workers", type=int, default=int(os.environ.get("AIM3_NUM_WORKERS", "0"))
    )
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hist_bins", type=int, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--resamples",
        type=int,
        default=1000,
        help="Bootstrap and permutation resample count for aggregate afferent statistics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=260718,
        help="Master seed shared with the symmetric relevance-timing Part-2 outputs.",
    )
    parser.add_argument(
        "--top_percent",
        nargs="+",
        type=int,
        default=[10],
        help="Top-k%% relevance thresholds for the aggregate bootstrap Cohen's d statistics.",
    )
    return parser.parse_args()


def collect_test_destination_gates(
    dataset: object,
    model: object,
    device: object,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run reset trajectories and return per-destination gate views plus labels.

    Reproduces the exact Part-2 reset trajectory computation. For every step the raw gate
    tensor is averaged over its source axis (last dim), yielding one value per hidden
    destination unit. Both the input gate (source axis size = encoder feature count) and the
    recurrent gate (source axis size = hidden size) are returned as flat ``(total_frames,
    hidden_size)`` arrays.
    """

    import torch
    from torch.utils.data import DataLoader

    from utils_anal.gawf_symmetric_relevance_timing import _gate_tensors

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    hidden_size = int(model.rnn.hidden_size)
    input_gate_batches: list[np.ndarray] = []
    recurrent_gate_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32)
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            batch_size_actual, frame_num, input_size = encoded.shape
            hidden = torch.zeros(
                batch_size_actual,
                hidden_size,
                dtype=encoded.dtype,
                device=encoded.device,
            )
            feedback = torch.zeros(
                batch_size_actual,
                model.feedback_dim,
                dtype=torch.float32,
                device=encoded.device,
            )
            input_gate_time: list[torch.Tensor] = []
            recurrent_gate_time: list[torch.Tensor] = []
            for time_idx in range(frame_num):
                gate_input, gate_recurrent = _gate_tensors(feedback, model, input_size)
                # Afferent view: mean over source axis, per destination hidden unit.
                input_gate_time.append(gate_input.mean(dim=-1))
                recurrent_gate_time.append(gate_recurrent.mean(dim=-1))
                input_term = torch.einsum(
                    "bi,bhi,hi->bh",
                    encoded[:, time_idx],
                    gate_input,
                    model.rnn.weight_ih_l0,
                )
                recurrent_term = torch.einsum(
                    "bi,bhi,hi->bh", hidden, gate_recurrent, model.rnn.weight_hh_l0
                )
                preactivation = input_term + recurrent_term
                if model.rnn.bias_ih_l0 is not None:
                    preactivation = preactivation + model.rnn.bias_ih_l0.unsqueeze(0)
                if model.rnn.bias_hh_l0 is not None:
                    preactivation = preactivation + model.rnn.bias_hh_l0.unsqueeze(0)
                hidden = torch.relu(model.LNormRNN(torch.tanh(preactivation)))
                char_logits, sector_logits = model.classifier(hidden)
                feedback = torch.cat([char_logits, sector_logits], dim=-1).to(torch.float32)
            input_gate_batches.append(
                torch.stack(input_gate_time, dim=1)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            recurrent_gate_batches.append(
                torch.stack(recurrent_gate_time, dim=1)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            label_batches.append(labels.numpy().astype(np.int64, copy=False))
            samples_done = min((batch_index + 1) * batch_size, len(dataset))
            if samples_done % 200 < batch_size or batch_index + 1 == len(loader):
                print(
                    f"  collected {samples_done}/{len(dataset)} test sequences | "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    input_gates = np.concatenate(input_gate_batches, axis=0)
    recurrent_gates = np.concatenate(recurrent_gate_batches, axis=0)
    labels_all = np.concatenate(label_batches, axis=0)
    gates = {
        "input": input_gates.reshape(-1, input_gates.shape[-1]),
        "recurrent": recurrent_gates.reshape(-1, recurrent_gates.shape[-1]),
    }
    return gates, labels_all.reshape(-1, labels_all.shape[-1])


def main() -> None:
    """Collect exact test destination gates and export the four afferent distributions."""

    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.hist_bins <= 1:
        raise ValueError(
            "batch_size and hist_bins must be positive; num_workers cannot be negative"
        )
    if args.resamples <= 0:
        raise ValueError("resamples must be positive")
    if not args.top_percent or any(not (0 < p < 100) for p in args.top_percent):
        raise ValueError("top_percent entries must lie in the open interval (0, 100)")
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    selectivity_path = Path(args.selectivity).expanduser().resolve()
    split_report_path = Path(args.split_report).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    from utils_anal.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device
    from utils_anal.gawf_symmetric_stats import (
        bootstrap_d,
        cosine_alignment,
        relevance_label_null,
        relevance_masks,
        trial_relevance_moments,
    )

    args.use_sector_mode = True
    args.predict_all_chars = False
    device = resolve_device(args.device, require_cuda_if_requested=True)
    test_dataset, num_pos = build_eval_dataset(args, "test")
    model = build_model_from_ckpt(str(checkpoint_path), num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("This analysis requires a single-layer GaWF checkpoint")
    hidden_size = int(model.rnn.hidden_size)

    # Destination population is always hidden (256). Selection masks are shared across cells
    # sharing the same factor: sector cells use hidden sector tuning, digit cells use hidden
    # digit tuning. The full tuning array is retained so aggregate bootstrap-d and cosine
    # alignment can reuse the exact z-scored context profiles.
    hidden_selections: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with np.load(selectivity_path, allow_pickle=False) as selectivity:
        dominant = np.asarray(selectivity["primary_hidden_interaction_dominant"], dtype=bool)
        for factor in ("sector", "digit"):
            tuning = np.asarray(
                selectivity[f"primary_hidden_tuning_{factor}"], dtype=np.float64
            )
            passed = np.asarray(
                selectivity[f"primary_hidden_passed_{factor}"], dtype=bool
            )
            eligible = passed & ~dominant
            relevant = relevance_masks(tuning, eligible, 0.10)
            expected_levels = 9 if factor == "sector" else 10
            if relevant.shape != (expected_levels, hidden_size):
                raise RuntimeError(
                    f"Hidden relevance masks for factor {factor!r} have shape {relevant.shape}, "
                    f"expected {(expected_levels, hidden_size)}"
                )
            if eligible.size != hidden_size:
                raise RuntimeError(
                    f"Hidden eligibility for factor {factor!r} has {eligible.size} units, "
                    f"expected {hidden_size}"
                )
            if tuning.shape != (expected_levels, hidden_size):
                raise RuntimeError(
                    f"Hidden tuning for factor {factor!r} has shape {tuning.shape}, "
                    f"expected {(expected_levels, hidden_size)}"
                )
            hidden_selections[factor] = (eligible, relevant, tuning)

    gates, labels = collect_test_destination_gates(
        test_dataset,
        model,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    with split_report_path.open(encoding="utf-8") as file_obj:
        split_report = json.load(file_obj)["test"]
    observed_counts = {
        "sector": np.bincount(labels[:, 1], minlength=9).astype(np.int64),
        "digit": np.bincount(labels[:, 0], minlength=10).astype(np.int64),
    }
    for factor, counts_observed in observed_counts.items():
        expected = np.asarray(split_report[f"{factor}_counts"], dtype=np.int64)
        if not np.array_equal(counts_observed, expected):
            raise RuntimeError(
                f"Local test {factor} counts {counts_observed.tolist()} do not match "
                f"Part-2 counts {expected.tolist()}"
            )

    bin_edges = np.linspace(0.0, 1.0, args.hist_bins + 1, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {"bin_edges": bin_edges.astype(np.float32)}
    cell_metadata: dict[str, dict[str, object]] = {}
    gate_order = {"input": 0, "recurrent": 1}
    factor_order = {"sector": 0, "digit": 1}
    for cell, (gate_name, factor, levels, label_column) in CELL_SPECS.items():
        eligible, relevant, tuning = hidden_selections[factor]
        destination_view = gates[gate_name]
        if destination_view.shape[1] != hidden_size:
            raise RuntimeError(
                f"{cell} destination-view width {destination_view.shape[1]} != hidden size "
                f"{hidden_size}"
            )
        contexts = labels[:, label_column]
        hist, sums, sums_sq, counts = accumulate_context_group_distributions(
            ((0, destination_view.shape[0], destination_view),),
            contexts,
            relevant,
            eligible,
            bin_edges,
        )
        means, stds, context_d, global_d = summarize_group_moments(sums, sums_sq, counts)
        arrays.update(
            {
                f"{cell}_hist_counts": hist.astype(np.int64),
                f"{cell}_group_mean": means.astype(np.float32),
                f"{cell}_group_std": stds.astype(np.float32),
                f"{cell}_group_count": counts.astype(np.int64),
                f"{cell}_context_cohens_d": context_d.astype(np.float32),
                f"{cell}_relevant_mask": relevant.astype(np.uint8),
                f"{cell}_eligible_mask": eligible.astype(np.uint8),
            }
        )
        top_count = relevant.sum(axis=1).astype(int)
        source_dim = "1152 encoder features" if gate_name == "input" else "256 hidden units"
        cell_meta: dict[str, object] = {
            "gate": f"{gate_name} DESTINATION gate averaged over {source_dim}",
            "source_axis_size": 1152 if gate_name == "input" else hidden_size,
            "destination_population": "hidden",
            "factor": factor,
            "context_levels": levels,
            "selection": (
                f"{factor} FDR-selective hidden units, interaction-dominant excluded, "
                "top 10% independently per context on the DESTINATION side"
            ),
            "eligible_hidden_destinations": int(eligible.sum()),
            "top10_units_per_context": top_count.tolist(),
            "remaining_units_per_context": (eligible.sum() - top_count).astype(int).tolist(),
            "frames_per_context": observed_counts[factor].astype(int).tolist(),
            "context_cohens_d": context_d.tolist(),
            "global_cohens_d": float(global_d),
        }

        # Aggregate Part-2 style bootstrap Cohen's d comparing the top-k%% afferent gate columns
        # against the remaining eligible afferent columns. Seed layout is disjoint from the
        # symmetric-relevance-timing Part-2 seeds by using an independent 500000 offset for the
        # bootstrap and label null draws.
        cell_idx = gate_order[gate_name] * 2 + factor_order[factor]
        top_percent_report: dict[str, dict[str, object]] = {}
        for percent_idx, percent in enumerate(args.top_percent):
            fraction = percent / 100.0
            masks = relevance_masks(tuning, eligible, fraction)
            moments = trial_relevance_moments(destination_view, contexts, masks, eligible)
            point, draws = bootstrap_d(
                moments,
                resamples=args.resamples,
                seed=args.seed + 200000 + cell_idx * 100 + percent_idx,
            )
            null = relevance_label_null(
                destination_view,
                labels,
                factor,
                tuning,
                eligible,
                fraction,
                resamples=args.resamples,
                seed=args.seed + 300000 + cell_idx * 100 + percent_idx,
            )
            p_value = float((1 + np.count_nonzero(null >= point)) / (args.resamples + 1))
            arrays[f"{cell}_top{percent}_bootstrap_d"] = draws.astype(np.float32)
            arrays[f"{cell}_top{percent}_relevance_null_d"] = null.astype(np.float32)
            top_percent_report[str(percent)] = {
                "cohens_d": float(point),
                "bootstrap_ci95": [
                    float(x) for x in np.quantile(draws, [0.025, 0.975])
                ],
                "relevant_units_per_level": masks.sum(axis=1).astype(int).tolist(),
                "relevance_shuffle_p_value": p_value,
            }
        cell_meta["top_percent"] = top_percent_report

        # Continuous cosine alignment between hidden activation tuning and afferent gate tuning
        # per context, plus a two-sided permutation test on the diagonal contrast.
        alignment = cosine_alignment(
            tuning,
            destination_view,
            labels,
            factor,
            eligible,
            resamples=args.resamples,
            seed=args.seed + 500000 + cell_idx,
        )
        arrays[f"{cell}_alignment_matrix"] = alignment["matrix"].astype(np.float32)
        arrays[f"{cell}_alignment_null"] = alignment["permutation_null"].astype(np.float32)
        cell_meta["continuous_alignment"] = {
            "diagonal_minus_off_diagonal": alignment["diagonal_minus_off_diagonal"],
            "permutation_p_value": alignment["permutation_p_value"],
            "permutation_alternative": alignment["permutation_alternative"],
        }

        cell_metadata[cell] = cell_meta

    arrays_path = save_dir / "afferent_top10_gate_distributions.npz"
    metadata_path = save_dir / "afferent_top10_gate_distributions_meta.json"
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "data_suffix": args.data_suffix,
        "selectivity": str(selectivity_path),
        "split_report": str(split_report_path),
        "trajectory": "reset sequential trajectory, identical to Part-2 test collection",
        "gate_view": "afferent DESTINATION gate; per-destination mean over source axis",
        "gate_values": "raw post-sigmoid; 0.5 point mass included",
        "groups": list(GROUP_NAMES),
        "hist_bins": args.hist_bins,
        "gate_tau": float(model.gate_tau),
        "resamples": int(args.resamples),
        "seed": int(args.seed),
        "top_percent": [int(p) for p in args.top_percent],
        "aggregate_measures": {
            "bootstrap_d": (
                "top-k%% afferent gate columns vs remaining eligible afferent columns; "
                "pooled Cohen's d over frames with trial bootstrap and relevance-label null"
            ),
            "continuous_alignment": (
                "cosine similarity between hidden activation tuning and per-context afferent "
                "gate tuning, z-scored per row; two-sided permutation on diagonal contrast"
            ),
        },
        "cells": cell_metadata,
    }
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)
    print(f"Saved {arrays_path}", flush=True)
    print(f"Saved {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
