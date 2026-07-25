"""Export recurrent SOURCE-gate distributions split by source AND destination group.

This complements ``gawf_recurrent_sector_relevance_distributions`` and the recurrent-digit
branch of ``gawf_remaining_relevance_distributions``. Those analyses reduce each frame's
``(hidden, hidden)`` recurrent gate matrix to one value per source unit (``mean_h G[h, i]``)
and then compare the top-10% source group against the remaining eligible source group per
context. This module keeps the same source-oriented convention but splits each source unit's
destinations into the same two groups, so every source unit contributes TWO per-frame values
instead of one:

- ``m_top[i] = mean_{h in top10(context)} G[h, i]``
- ``m_rem[i] = mean_{h in remaining_eligible(context)} G[h, i]``

Grouping the sources themselves by the top-10% relevance mask yields four distributions per
context (per frame values pooled across frames):

- ``within_top``    : source in top10, destination in top10.
- ``top_to_rem``    : source in top10, destination in remaining eligible.
- ``rem_to_top``    : source in remaining eligible, destination in top10.
- ``within_rem``    : source in remaining eligible, destination in remaining eligible.

Outputs are one compressed NPZ plus JSON metadata for the recurrent sector (9 levels) and
recurrent digit (10 levels) contexts.
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


CELL_SPECS = {
    "recurrent_sector": ("sector", 9, 1),
    "recurrent_digit": ("digit", 10, 0),
}
GROUP_NAMES = ("within_top", "top_to_rem", "rem_to_top", "within_rem")
GROUP_KIND = {
    "within_top": "within",
    "top_to_rem": "between",
    "rem_to_top": "between",
    "within_rem": "within",
}


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
                "gawf_recurrent_group_gate_distributions",
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
    return parser.parse_args()


def accumulate_group_split_distributions(
    per_source_top_mean: np.ndarray,
    per_source_rem_mean: np.ndarray,
    contexts: np.ndarray,
    relevant_masks: np.ndarray,
    eligible: np.ndarray,
    bin_edges: np.ndarray,
    *,
    hist: np.ndarray | None = None,
    sums: np.ndarray | None = None,
    sums_sq: np.ndarray | None = None,
    counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update histogram counts and moments for the four (source, destination) groups.

    ``per_source_top_mean`` and ``per_source_rem_mean`` share shape ``(frames, H)`` and hold,
    for every frame, the mean of that frame's recurrent gate over the top-10% destinations
    and remaining eligible destinations respectively. ``contexts`` gives the per-frame
    context index; ``relevant_masks`` has shape ``(num_contexts, H)``; ``eligible`` is a
    ``(H,)`` boolean mask over hidden units. Returns the updated histograms, first moment,
    second moment, and observation counts, all shaped ``(num_contexts, 4, ...)``.
    """

    per_source_top_mean = np.asarray(per_source_top_mean, dtype=np.float32)
    per_source_rem_mean = np.asarray(per_source_rem_mean, dtype=np.float32)
    if per_source_top_mean.shape != per_source_rem_mean.shape:
        raise ValueError("top and remaining per-source arrays must share shape")
    if per_source_top_mean.ndim != 2:
        raise ValueError("per-source arrays must be two-dimensional (frames, H)")
    contexts = np.asarray(contexts, dtype=np.int64).reshape(-1)
    if contexts.size != per_source_top_mean.shape[0]:
        raise ValueError("contexts must align with the frame dimension")
    relevant_masks = np.asarray(relevant_masks, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    if relevant_masks.ndim != 2 or relevant_masks.shape[1] != eligible.size:
        raise ValueError(
            f"relevant_masks must have shape (contexts, {eligible.size}), "
            f"got {relevant_masks.shape}"
        )
    if per_source_top_mean.shape[1] != eligible.size:
        raise ValueError("per-source arrays must have H columns matching eligible")
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("bin_edges must be a one-dimensional sequence with at least two edges")
    num_contexts = relevant_masks.shape[0]
    if contexts.size and (contexts.min() < 0 or contexts.max() >= num_contexts):
        raise ValueError("contexts must be non-empty indices covered by relevant_masks")
    bin_count = bin_edges.size - 1
    if hist is None:
        hist = np.zeros((num_contexts, len(GROUP_NAMES), bin_count), dtype=np.int64)
    if sums is None:
        sums = np.zeros((num_contexts, len(GROUP_NAMES)), dtype=np.float64)
    if sums_sq is None:
        sums_sq = np.zeros_like(sums)
    if counts is None:
        counts = np.zeros((num_contexts, len(GROUP_NAMES)), dtype=np.int64)
    if hist.shape != (num_contexts, len(GROUP_NAMES), bin_count):
        raise ValueError("hist buffer has an unexpected shape")
    if sums.shape != (num_contexts, len(GROUP_NAMES)) or counts.shape != sums.shape:
        raise ValueError("moment buffers have an unexpected shape")
    for context in np.unique(contexts):
        frame_selector = contexts == context
        top_mask = relevant_masks[context]
        rem_mask = eligible & ~top_mask
        top_source_top_vals = per_source_top_mean[frame_selector][:, top_mask].reshape(-1)
        top_source_rem_vals = per_source_rem_mean[frame_selector][:, top_mask].reshape(-1)
        rem_source_top_vals = per_source_top_mean[frame_selector][:, rem_mask].reshape(-1)
        rem_source_rem_vals = per_source_rem_mean[frame_selector][:, rem_mask].reshape(-1)
        group_values = (
            top_source_top_vals,
            top_source_rem_vals,
            rem_source_top_vals,
            rem_source_rem_vals,
        )
        for group_index, values in enumerate(group_values):
            hist[context, group_index] += np.histogram(values, bins=bin_edges)[0]
            sums[context, group_index] += float(values.sum(dtype=np.float64))
            sums_sq[context, group_index] += float(
                np.square(values, dtype=np.float64).sum(dtype=np.float64)
            )
            counts[context, group_index] += int(values.size)
    return hist, sums, sums_sq, counts


def summarize_group_moments(
    sums: np.ndarray,
    sums_sq: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-group means and sample standard deviations."""

    sums = np.asarray(sums, dtype=np.float64)
    sums_sq = np.asarray(sums_sq, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(counts <= 1):
        raise RuntimeError("Every context and group requires at least two observations")
    means = sums / counts
    variances = (sums_sq - np.square(sums) / counts) / (counts - 1)
    np.maximum(variances, 0.0, out=variances)
    stds = np.sqrt(variances)
    return means, stds


def within_between_cohens_d(
    sums: np.ndarray,
    sums_sq: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Compute Cohen's d comparing within-group against the paired between-group per context.

    Returns a ``(num_contexts, 2)`` array where column 0 is
    ``within_top`` minus ``top_to_rem`` (both share source in top10) and column 1 is
    ``within_rem`` minus ``rem_to_top`` (both share source in remaining). A positive value
    means the within-group afferent gate is stronger than the between-group afferent gate.
    """

    sums = np.asarray(sums, dtype=np.float64)
    sums_sq = np.asarray(sums_sq, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if sums.ndim != 2 or sums.shape[1] != len(GROUP_NAMES):
        raise ValueError("moment arrays must have shape (num_contexts, 4)")
    pairs = ((0, 1), (3, 2))
    result = np.zeros((sums.shape[0], 2), dtype=np.float64)
    for column, (within_idx, between_idx) in enumerate(pairs):
        n_within = counts[:, within_idx]
        n_between = counts[:, between_idx]
        if np.any(n_within <= 1) or np.any(n_between <= 1):
            raise RuntimeError("Cohen's d requires more than one observation per group")
        mean_within = sums[:, within_idx] / n_within
        mean_between = sums[:, between_idx] / n_between
        var_within = np.maximum(
            (sums_sq[:, within_idx] - np.square(sums[:, within_idx]) / n_within)
            / (n_within - 1),
            0.0,
        )
        var_between = np.maximum(
            (sums_sq[:, between_idx] - np.square(sums[:, between_idx]) / n_between)
            / (n_between - 1),
            0.0,
        )
        pooled = np.sqrt(
            ((n_within - 1) * var_within + (n_between - 1) * var_between)
            / (n_within + n_between - 2)
        )
        result[:, column] = np.where(
            pooled > 0, (mean_within - mean_between) / pooled, np.nan
        )
    return result


def collect_and_accumulate(
    dataset: object,
    model: object,
    device: object,
    selections: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    batch_size: int,
    num_workers: int,
    hist_bins: int,
    label_columns: dict[str, int],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], np.ndarray]:
    """Run reset trajectories once and accumulate every cell's four-group histograms.

    Returns a dictionary mapping cell name to ``{"hist": ..., "sums": ..., "sums_sq": ...,
    "counts": ...}``, the overall context frame counts per cell, and the shared bin edges.
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
    input_size = int(model.rnn.input_size)
    bin_edges = np.linspace(0.0, 1.0, hist_bins + 1, dtype=np.float64)

    cell_buffers: dict[str, dict[str, np.ndarray]] = {}
    cell_masks: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for cell, (_factor, levels, _label_col) in CELL_SPECS.items():
        eligible, relevant = selections[cell]
        if eligible.size != hidden_size or relevant.shape != (levels, hidden_size):
            raise RuntimeError(f"{cell} masks do not align with hidden size {hidden_size}")
        cell_buffers[cell] = {
            "hist": np.zeros(
                (levels, len(GROUP_NAMES), bin_edges.size - 1), dtype=np.int64
            ),
            "sums": np.zeros((levels, len(GROUP_NAMES)), dtype=np.float64),
            "sums_sq": np.zeros((levels, len(GROUP_NAMES)), dtype=np.float64),
            "counts": np.zeros((levels, len(GROUP_NAMES)), dtype=np.int64),
            "frames_per_context": np.zeros((levels,), dtype=np.int64),
        }
        relevant_tensor = torch.from_numpy(np.ascontiguousarray(relevant)).to(
            device=device, dtype=torch.bool
        )
        eligible_tensor = torch.from_numpy(np.ascontiguousarray(eligible)).to(
            device=device, dtype=torch.bool
        )
        remaining_tensor = eligible_tensor.unsqueeze(0) & ~relevant_tensor
        cell_masks[cell] = (relevant_tensor, remaining_tensor, eligible_tensor)

    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32)
            labels = labels.to(dtype=torch.int64)
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            batch_size_actual, frame_num, _ = encoded.shape
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
            for time_idx in range(frame_num):
                gate_input, gate_recurrent = _gate_tensors(feedback, model, input_size)
                # gate_recurrent shape: (B, H_dest, H_source); dim=1 is destination.
                _accumulate_batch_frame(
                    gate_recurrent,
                    labels[:, time_idx, :],
                    cell_buffers,
                    cell_masks,
                    bin_edges,
                )
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
            samples_done = min((batch_index + 1) * batch_size, len(dataset))
            if samples_done % 200 < batch_size or batch_index + 1 == len(loader):
                print(
                    f"  collected {samples_done}/{len(dataset)} test sequences | "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    return cell_buffers, {cell: cell_buffers[cell]["frames_per_context"] for cell in cell_buffers}, bin_edges


def _accumulate_batch_frame(
    gate_recurrent,
    labels_frame,
    cell_buffers: dict[str, dict[str, np.ndarray]],
    cell_masks,
    bin_edges: np.ndarray,
) -> None:
    """Update every cell's four-group buffers with one batch-frame recurrent gate."""

    import torch

    for cell, (factor, levels, label_col) in CELL_SPECS.items():
        relevant_tensor, remaining_tensor, _eligible = cell_masks[cell]
        contexts_tensor = labels_frame[:, label_col]
        unique_contexts = torch.unique(contexts_tensor)
        buffers = cell_buffers[cell]
        for context_value in unique_contexts.tolist():
            selector = contexts_tensor == context_value
            sub_gate = gate_recurrent[selector]
            if sub_gate.shape[0] == 0:
                continue
            top_mask_c = relevant_tensor[context_value]
            rem_mask_c = remaining_tensor[context_value]
            top_count = float(top_mask_c.sum())
            rem_count = float(rem_mask_c.sum())
            if top_count <= 0 or rem_count <= 0:
                raise RuntimeError(
                    f"{cell} context {context_value} has empty top or remaining group"
                )
            # Per-source unit mean over top-10% destinations and remaining destinations.
            m_top = torch.einsum(
                "h,bhi->bi", top_mask_c.to(sub_gate.dtype), sub_gate
            ) / top_count
            m_rem = torch.einsum(
                "h,bhi->bi", rem_mask_c.to(sub_gate.dtype), sub_gate
            ) / rem_count
            top_source_top_vals = m_top[:, top_mask_c].reshape(-1).detach().cpu().numpy()
            top_source_rem_vals = m_rem[:, top_mask_c].reshape(-1).detach().cpu().numpy()
            rem_source_top_vals = m_top[:, rem_mask_c].reshape(-1).detach().cpu().numpy()
            rem_source_rem_vals = m_rem[:, rem_mask_c].reshape(-1).detach().cpu().numpy()
            group_values = (
                top_source_top_vals,
                top_source_rem_vals,
                rem_source_top_vals,
                rem_source_rem_vals,
            )
            for group_index, values in enumerate(group_values):
                values64 = values.astype(np.float64, copy=False)
                buffers["hist"][context_value, group_index] += np.histogram(
                    values, bins=bin_edges
                )[0]
                buffers["sums"][context_value, group_index] += float(values64.sum())
                buffers["sums_sq"][context_value, group_index] += float(
                    np.square(values64).sum()
                )
                buffers["counts"][context_value, group_index] += int(values.size)
            buffers["frames_per_context"][context_value] += int(sub_gate.shape[0])


def main() -> None:
    """Collect exact test gates and export the source/destination four-group distributions."""

    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.hist_bins <= 1:
        raise ValueError(
            "batch_size and hist_bins must be positive; num_workers cannot be negative"
        )
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    selectivity_path = Path(args.selectivity).expanduser().resolve()
    split_report_path = Path(args.split_report).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    from utils_anal.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device
    from utils_anal.gawf_symmetric_stats import relevance_masks

    args.use_sector_mode = True
    args.predict_all_chars = False
    device = resolve_device(args.device, require_cuda_if_requested=True)
    test_dataset, num_pos = build_eval_dataset(args, "test")
    model = build_model_from_ckpt(str(checkpoint_path), num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("This analysis requires a single-layer GaWF checkpoint")
    hidden_size = int(model.rnn.hidden_size)

    selections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with np.load(selectivity_path, allow_pickle=False) as selectivity:
        for cell, (factor, levels, _label_col) in CELL_SPECS.items():
            tuning = np.asarray(
                selectivity[f"primary_hidden_tuning_{factor}"], dtype=np.float64
            )
            passed = np.asarray(
                selectivity[f"primary_hidden_passed_{factor}"], dtype=bool
            )
            dominant = np.asarray(
                selectivity["primary_hidden_interaction_dominant"], dtype=bool
            )
            eligible = passed & ~dominant
            if eligible.size != hidden_size:
                raise RuntimeError(
                    f"{cell} eligibility does not align with hidden size {hidden_size}"
                )
            relevant = relevance_masks(tuning, eligible, 0.10)
            if relevant.shape != (levels, hidden_size):
                raise RuntimeError(f"{cell} masks have unexpected shape {relevant.shape}")
            selections[cell] = (eligible, relevant)

    label_columns = {cell: spec[2] for cell, spec in CELL_SPECS.items()}
    cell_buffers, frames_per_context, bin_edges = collect_and_accumulate(
        test_dataset,
        model,
        device,
        selections,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hist_bins=args.hist_bins,
        label_columns=label_columns,
    )

    with split_report_path.open(encoding="utf-8") as file_obj:
        split_report = json.load(file_obj)["test"]
    expected_counts_by_factor = {
        "sector": np.asarray(split_report["sector_counts"], dtype=np.int64),
        "digit": np.asarray(split_report["digit_counts"], dtype=np.int64),
    }
    for cell, (factor, _levels, _label_col) in CELL_SPECS.items():
        observed = cell_buffers[cell]["frames_per_context"]
        expected = expected_counts_by_factor[factor]
        if not np.array_equal(observed, expected):
            raise RuntimeError(
                f"{cell} frame counts {observed.tolist()} do not match split report "
                f"{expected.tolist()}"
            )

    arrays: dict[str, np.ndarray] = {"bin_edges": bin_edges.astype(np.float32)}
    cell_metadata: dict[str, dict[str, object]] = {}
    for cell, (factor, _levels, _label_col) in CELL_SPECS.items():
        eligible, relevant = selections[cell]
        buffers = cell_buffers[cell]
        means, stds = summarize_group_moments(
            buffers["sums"], buffers["sums_sq"], buffers["counts"]
        )
        cohens_d = within_between_cohens_d(
            buffers["sums"], buffers["sums_sq"], buffers["counts"]
        )
        arrays.update(
            {
                f"{cell}_hist_counts": buffers["hist"].astype(np.int64),
                f"{cell}_group_mean": means.astype(np.float32),
                f"{cell}_group_std": stds.astype(np.float32),
                f"{cell}_group_count": buffers["counts"].astype(np.int64),
                f"{cell}_within_between_cohens_d": cohens_d.astype(np.float32),
                f"{cell}_relevant_mask": relevant.astype(np.uint8),
                f"{cell}_eligible_mask": eligible.astype(np.uint8),
                f"{cell}_frames_per_context": buffers["frames_per_context"].astype(np.int64),
            }
        )
        top_count = relevant.sum(axis=1).astype(int)
        cell_metadata[cell] = {
            "gate": "recurrent gate, source-oriented unit mean over destination subsets",
            "factor": factor,
            "context_levels": relevant.shape[0],
            "selection": (
                f"{factor} FDR-selective, interaction-dominant excluded, "
                "top 10% independently per context"
            ),
            "eligible_units": int(eligible.sum()),
            "top10_units_per_context": top_count.tolist(),
            "remaining_units_per_context": (eligible.sum() - top_count).astype(int).tolist(),
            "frames_per_context": buffers["frames_per_context"].astype(int).tolist(),
            "group_names": list(GROUP_NAMES),
            "group_kind": {name: GROUP_KIND[name] for name in GROUP_NAMES},
            "within_top_cohens_d_vs_top_to_rem": cohens_d[:, 0].tolist(),
            "within_rem_cohens_d_vs_rem_to_top": cohens_d[:, 1].tolist(),
        }

    arrays_path = save_dir / "recurrent_group_gate_distributions.npz"
    metadata_path = save_dir / "recurrent_group_gate_distributions_meta.json"
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "data_suffix": args.data_suffix,
        "selectivity": str(selectivity_path),
        "split_report": str(split_report_path),
        "trajectory": "reset sequential trajectory, identical to Part-2 test collection",
        "gate_values": "raw post-sigmoid; source-oriented per-unit means",
        "gate": (
            "for each source unit i in a frame, two values are stored: "
            "mean over destinations in top10(context) and mean over destinations in "
            "remaining eligible units; sources are then split by the same top10 mask"
        ),
        "hist_bins": args.hist_bins,
        "gate_tau": float(model.gate_tau),
        "cells": cell_metadata,
    }
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)
    print(f"Saved {arrays_path}", flush=True)
    print(f"Saved {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
