"""Recompute reset-excluded Figure 4 GaWF synapse-gate ANOVA and core-object PDFs.

``collect`` reads one retained feedback trajectory and its matching checkpoint, excludes
zero-feedback reset frames, reconstructs raw input/recurrent sigmoid gates on CUDA, and keeps
only twenty balanced-draw aggregate sufficient statistics.  ``aggregate`` combines ten such
seed outputs with the retained reset-excluded GaWF activation summaries and renders the Figure 4
core-object PDFs with seed points.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from utils.analysis.anal_helpers import build_model_from_ckpt, resolve_device
from utils.analysis.clutter.fig3_gate_distribution import exclude_zero_feedback_reset_frames
from utils.analysis.clutter.fig4_shuffle_activation_anova import RepeatedCudaMoments
from utils.analysis.clutter.fig4_variance_decomposition import (
    _plot_compact_aggregate,
    _plot_compact_aggregate_1x4,
)
from utils.analysis.clutter.fig4_variance_sources import _gate_values
from utils.analysis.variance_decomposition import (
    CM_FACTORS,
    RepeatedDecomposition,
    balanced_subsample_indices,
)


RESULT_NAME = "gate_synapse_anova.npz"
GATE_OBJECTS = ("input_gate", "recurrent_gate")
ACTIVATION_OBJECTS = ("encoder_activation", "hidden_state")


def parse_args() -> argparse.Namespace:
    """Parse per-seed collection or ten-seed aggregation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--ckpt", required=True, type=Path)
    collect.add_argument("--trajectory", required=True, type=Path)
    collect.add_argument("--output_dir", required=True, type=Path)
    collect.add_argument("--seed", required=True, type=int)
    collect.add_argument("--balance_seed", type=int, default=20260719)
    collect.add_argument("--repeats", type=int, default=20)
    collect.add_argument("--frame_batch_size", type=int, default=64)
    collect.add_argument("--device", default="cuda")

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--gate_data_root", required=True, type=Path)
    aggregate.add_argument("--activation_data_root", required=True, type=Path)
    aggregate.add_argument("--figure_dir", required=True, type=Path)
    aggregate.add_argument("--summary_dir", required=True, type=Path)
    aggregate.add_argument("--expected_seeds", type=int, default=10)
    return parser.parse_args()


def _gate_seed_files(root: Path, expected_seeds: int) -> list[Path]:
    """Return exactly one compact gate result for each expected training seed."""

    files = [root / f"seed{seed:02d}" / RESULT_NAME for seed in range(1, expected_seeds + 1)]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("Missing gate ANOVA seed outputs: " + ", ".join(missing))
    return files


def _activation_seed_files(root: Path, expected_seeds: int) -> list[Path]:
    """Return exactly one retained reset-excluded GaWF activation output per seed."""

    files = [
        root / f"gawf-seed{seed:02d}" / "activation_anova.npz"
        for seed in range(1, expected_seeds + 1)
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("Missing reset-excluded activation seed outputs: " + ", ".join(missing))
    return files


def collect(args: argparse.Namespace) -> Path:
    """Save reset-excluded repeated aggregate ANOVA values for one GaWF seed."""

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    if args.repeats != 20 or args.frame_batch_size <= 0:
        raise ValueError("This formal analysis requires 20 draws and a positive frame batch size.")
    with np.load(args.trajectory, allow_pickle=False) as arrays:
        required = {"feedback", "labels", "weight_ih", "weight_hh"}
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"Trajectory is missing arrays: {missing}")
        feedback, labels, reset_frames = exclude_zero_feedback_reset_frames(
            np.asarray(arrays["feedback"], dtype=np.float32),
            np.asarray(arrays["labels"], dtype=np.int64),
        )
        weight_ih_shape = tuple(int(value) for value in arrays["weight_ih"].shape)
        weight_hh_shape = tuple(int(value) for value in arrays["weight_hh"].shape)
    device = resolve_device(args.device, require_cuda_if_requested=True)
    model = build_model_from_ckpt(str(args.ckpt), 9, device, chan_num=2)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("Figure 4 synapse analysis requires a single-layer GaWF checkpoint.")
    hidden_size, input_size = weight_ih_shape
    if weight_hh_shape != (hidden_size, hidden_size):
        raise RuntimeError(f"Unexpected recurrent shape: {weight_hh_shape}")
    if (hidden_size, input_size) != (model.rnn.hidden_size, model.encoder_flatten_size):
        raise RuntimeError("Trajectory and checkpoint gate dimensions disagree.")
    draws, balance = balanced_subsample_indices(
        labels, repeats=args.repeats, seed=args.balance_seed
    )
    membership = torch.zeros((args.repeats, labels.shape[0]), dtype=torch.bool, device=device)
    for repeat, indices in enumerate(draws):
        membership[repeat, torch.from_numpy(indices).to(device)] = True
    moments = {
        "input_gate": RepeatedCudaMoments(args.repeats, hidden_size * input_size, device),
        "recurrent_gate": RepeatedCudaMoments(args.repeats, hidden_size * hidden_size, device),
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, labels.shape[0], args.frame_batch_size):
            stop = min(start + args.frame_batch_size, labels.shape[0])
            current_feedback = torch.from_numpy(feedback[start:stop]).to(device)
            current_labels = torch.from_numpy(labels[start:stop]).to(device)
            gate_ih, gate_hh = _gate_values(model, current_feedback, input_size)
            current_membership = membership[:, start:stop]
            moments["input_gate"].update(
                gate_ih.flatten(1), current_labels, current_membership
            )
            moments["recurrent_gate"].update(
                gate_hh.flatten(1), current_labels, current_membership
            )
            if stop == labels.shape[0] or stop % (args.frame_batch_size * 100) == 0:
                print(f"seed{args.seed:02d}: frames={stop}/{labels.shape[0]}", flush=True)
    result = {object_name: state.finalize() for object_name, state in moments.items()}
    payload = {
        f"{object_name}_{metric}": values.astype(np.float64)
        for object_name, metrics in result.items()
        for metric, values in metrics.items()
    }
    args.output_dir.mkdir(parents=True)
    destination = args.output_dir / RESULT_NAME
    np.savez_compressed(destination, **payload)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "analysis": "Figure 4 raw-synapse gate ANOVA",
                "checkpoint": str(args.ckpt.resolve()),
                "trajectory": str(args.trajectory.resolve()),
                "seed": args.seed,
                "reset_excluded": True,
                "reset_frames_excluded": reset_frames,
                "analysis_n_frames": int(labels.shape[0]),
                "gate_level": "raw sigmoid synapse",
                "objects": list(GATE_OBJECTS),
                "repeats": args.repeats,
                "balance": balance.__dict__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _seed_repeated(values: list[float]) -> np.ndarray:
    """Return the ten per-seed mean estimates used for the cross-seed SEM."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (10,):
        raise RuntimeError(f"Expected ten seed means, got {array.shape}.")
    return array


def _npz_mean(path: Path, key: str) -> float:
    """Return one repeated-draw mean while closing its compact archive."""

    with np.load(path, allow_pickle=False) as arrays:
        return float(np.asarray(arrays[key], dtype=np.float64).mean())


def _load_core_results(
    gate_root: Path,
    activation_root: Path,
    expected_seeds: int,
) -> dict[str, RepeatedDecomposition]:
    """Merge new gate summaries with already reset-excluded GaWF activation summaries."""

    gate_files = _gate_seed_files(gate_root, expected_seeds)
    activation_files = _activation_seed_files(activation_root, expected_seeds)
    if expected_seeds != 10:
        raise ValueError("The formal core-object figure requires exactly ten training seeds.")
    results: dict[str, RepeatedDecomposition] = {}
    for object_name in GATE_OBJECTS:
        condition = {
            factor: _seed_repeated(
                [
                    _npz_mean(path, f"{object_name}_{factor}") / 100.0
                    for path in gate_files
                ]
            )
            for factor in CM_FACTORS
        }
        residual = _seed_repeated(
            [
                _npz_mean(path, f"{object_name}_residual_frac")
                for path in gate_files
            ]
        )
        results[object_name] = RepeatedDecomposition(
            aggregate_cm=condition,
            aggregate_trial={"residual": residual},
            per_unit_cm={},
            per_unit_trial={},
            unweighted_per_unit_mean_cm={},
            unweighted_per_unit_mean_trial={},
            consistency={},
        )
    for object_name, source_name in zip(
        ACTIVATION_OBJECTS, ("input_activation", "hidden_activation")
    ):
        condition = {
            factor: _seed_repeated(
                [
                    _npz_mean(path, f"{source_name}_{factor}")
                    for path in activation_files
                ]
            )
            for factor in CM_FACTORS
        }
        residual = _seed_repeated(
            [
                _npz_mean(path, f"{source_name}_residual")
                for path in activation_files
            ]
        )
        results[object_name] = RepeatedDecomposition(
            aggregate_cm=condition,
            aggregate_trial={"residual": residual},
            per_unit_cm={},
            per_unit_trial={},
            unweighted_per_unit_mean_cm={},
            unweighted_per_unit_mean_trial={},
            consistency={},
        )
    return results


def aggregate(args: argparse.Namespace) -> tuple[Path, ...]:
    """Render all core-object Figure 4 variants and save a ten-seed summary CSV."""

    results = _load_core_results(
        args.gate_data_root, args.activation_data_root, args.expected_seeds
    )
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for object_name, result in results.items():
        for factor in (*CM_FACTORS, "residual"):
            values = (
                result.aggregate_trial["residual"]
                if factor == "residual"
                else result.aggregate_cm[factor]
            )
            rows.append(
                {
                    "object": object_name,
                    "factor": factor,
                    "mean_percent": 100.0 * float(values.mean()),
                    "sem_percent": 100.0 * float(values.std(ddof=1) / np.sqrt(values.size)),
                }
            )
    summary = args.summary_dir / "Fig4_core_objects_aggregate_resetexcluded_10seed_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs = []
    for include_residual in (False, True):
        outputs.append(
            _plot_compact_aggregate(
                args.figure_dir,
                results,
                error_mode="sem",
                show_seed_points=True,
                include_residual=include_residual,
            ).with_suffix(".pdf")
        )
        outputs.append(
            _plot_compact_aggregate_1x4(
                args.figure_dir,
                results,
                error_mode="sem",
                show_seed_points=True,
                include_residual=include_residual,
            ).with_suffix(".pdf")
        )
    return (summary, *outputs)


def main() -> None:
    """Dispatch collection or aggregation and report exact artifacts."""

    args = parse_args()
    if args.command == "collect":
        print(f"Saved {collect(args)}")
    else:
        for output in aggregate(args):
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
