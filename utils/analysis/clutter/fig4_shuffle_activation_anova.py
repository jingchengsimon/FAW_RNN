"""Run Figure 4 variance decomposition during the Figure 2 feedback shuffles.

``collect`` evaluates one frozen GaWF checkpoint in the identical 512-frame baseline,
``shuffle_digit``, and ``shuffle_sector`` rollouts used by Figure 2.  It excludes every reset
frame, streams balanced 9-sector x 10-digit sufficient statistics for encoder/hidden activations
and both gate matrices on CUDA, and writes only compact repeated aggregate statistics.
``aggregate`` combines the ten per-seed outputs into the requested long table, SEM summary,
and a Figure-4-prefixed 1-by-3 activation panel.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils.analysis.clutter.fig2_feedback_ablation import (
    DIGIT_SLICE,
    _run_baseline_feedback_schedule,
    _shuffled_schedule_slice,
)
from utils.analysis.clutter.fig4_variance_sources import _gate_values
from utils.analysis.clutter.multiseed_plotting import add_seed_points
from utils.analysis.variance_decomposition import CM_FACTORS, NUM_CELLS, balanced_subsample_indices


CONDITIONS = ("baseline", "shuffle_digit", "shuffle_sector")
OBJECTS = ("encoder_activation", "hidden_activation", "input_gate", "recurrent_gate")
RESULT_NAME = "shuffle_activation_anova.npz"
FACTOR_COLORS = {
    "sector": "#264653",
    "digit": "#E76F51",
    "interaction": "#E9C46A",
    "residual_frac": "#8D99AE",
}


def parse_args() -> argparse.Namespace:
    """Parse collection and cross-seed aggregation commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Collect one checkpoint's three-condition data.")
    collect.add_argument("--ckpt", required=True, type=Path)
    collect.add_argument("--data_dir", required=True, type=Path)
    collect.add_argument("--output_dir", required=True, type=Path)
    collect.add_argument("--seed", required=True, type=int)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--batch_size", type=int, default=16)
    collect.add_argument("--num_workers", type=int, default=2)
    collect.add_argument("--data_suffix", default="40h-uint8")
    collect.add_argument("--sequence_length", type=int, default=512)
    collect.add_argument("--chan_num", type=int, default=2)
    collect.add_argument("--repeats", type=int, default=20)

    aggregate = commands.add_parser("aggregate", help="Summarize ten seed outputs and plot Fig4.")
    aggregate.add_argument("--data_root", required=True, type=Path)
    aggregate.add_argument("--figure_dir", required=True, type=Path)
    aggregate.add_argument("--expected_seeds", type=int, default=10)
    aggregate.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay neutral-gray points for the ten training seeds.",
    )
    return parser.parse_args()


class RepeatedCudaMoments:
    """GPU-resident aggregate sufficient statistics for repeated balanced draws.

    Only sums, squared sums, and 90 condition-cell sums are retained.  This deliberately omits
    the per-unit distributions used by the original Figure 4 source workflow: this extension
    requires its aggregate quantities only and must not persist dense gate arrays.
    """

    def __init__(self, repeats: int, num_units: int, device: torch.device) -> None:
        self.repeats = repeats
        self.num_units = num_units
        self.device = device
        self.cell_sum = torch.zeros(
            (repeats, NUM_CELLS, num_units), device=device, dtype=torch.float64
        )
        self.total_sum = torch.zeros((repeats, num_units), device=device, dtype=torch.float64)
        self.total_sum_sq = torch.zeros((repeats, num_units), device=device, dtype=torch.float64)
        self.cell_count = torch.zeros((repeats, NUM_CELLS), device=device, dtype=torch.int64)

    @property
    def bytes_allocated(self) -> int:
        return (
            self.cell_sum.numel() + self.total_sum.numel() + self.total_sum_sq.numel()
        ) * torch.finfo(torch.float64).bits // 8 + self.cell_count.numel() * 8

    def update(
        self,
        values: torch.Tensor,
        labels: torch.Tensor,
        membership: torch.Tensor,
    ) -> None:
        """Add one non-reset timestep's ``(batch, units)`` values to every selected draw."""

        if values.ndim != 2 or values.shape[1] != self.num_units:
            raise ValueError(f"Expected values (_, {self.num_units}), got {tuple(values.shape)}")
        if labels.shape != (values.shape[0], 2):
            raise ValueError("labels must align with values as (batch, 2)")
        if membership.shape != (self.repeats, values.shape[0]):
            raise ValueError("membership must have shape (repeats, batch)")
        weights = membership.to(device=self.device, dtype=torch.float64)
        values64 = values.to(dtype=torch.float64)
        self.total_sum.add_(weights @ values64)
        self.total_sum_sq.add_(weights @ values64.square())
        codes = labels[:, 1] * 10 + labels[:, 0]
        for code in torch.unique(codes).tolist():
            mask = codes == int(code)
            selected_weights = weights[:, mask]
            self.cell_sum[:, int(code)].add_(selected_weights @ values64[mask])
            self.cell_count[:, int(code)].add_(selected_weights.sum(dim=1).to(torch.int64))

    def finalize(self) -> dict[str, np.ndarray]:
        """Return repeated aggregate components, absolute variance, and residual fraction."""

        counts = self.cell_count
        if torch.any(counts <= 0) or not torch.equal(counts, counts[:, :1].expand_as(counts)):
            minimum = int(counts.min().item())
            maximum = int(counts.max().item())
            raise RuntimeError(
                f"Balanced draw counts differ across cells: min={minimum}, max={maximum}"
            )
        n_per_cell = counts[:, 0].to(torch.float64)
        total_trials = n_per_cell * NUM_CELLS
        totals = {
            name: torch.zeros(self.repeats, device=self.device, dtype=torch.float64)
            for name in CM_FACTORS
        }
        for start in range(0, self.num_units, 4096):
            stop = min(start + 4096, self.num_units)
            means = (self.cell_sum[:, :, start:stop] / n_per_cell[:, None, None]).reshape(
                self.repeats, 9, 10, stop - start
            )
            grand = means.mean(dim=(1, 2), keepdim=True)
            sector = means.mean(dim=2, keepdim=True) - grand
            digit = means.mean(dim=1, keepdim=True) - grand
            interaction = means - grand - sector - digit
            totals["sector"].add_(n_per_cell * 10.0 * sector.square().sum(dim=(1, 2, 3)))
            totals["digit"].add_(n_per_cell * 9.0 * digit.square().sum(dim=(1, 2, 3)))
            totals["interaction"].add_(n_per_cell * interaction.square().sum(dim=(1, 2, 3)))
        total_cm = sum(totals.values())
        total_trial = (self.total_sum_sq - self.total_sum.square() / total_trials[:, None]).sum(
            dim=1
        )
        residual = torch.clamp(total_trial - total_cm, min=0.0)
        if torch.any(total_cm <= 0) or torch.any(total_trial <= 0):
            raise RuntimeError("Condition-mean and trial-level variance must both be positive")
        result = {
            factor: (100.0 * totals[factor] / total_cm).detach().cpu().numpy()
            for factor in CM_FACTORS
        }
        # This is the trace of the balanced condition-mean covariance, averaged over trials;
        # unlike the percentages above it remains on the representation's absolute scale.
        result["between_condition_var"] = (total_cm / total_trials).detach().cpu().numpy()
        result["residual_frac"] = (residual / total_trial).detach().cpu().numpy()
        return {name: np.asarray(values, dtype=np.float64) for name, values in result.items()}


def _cuda_required_bytes(repeats: int, input_units: int, hidden_units: int) -> int:
    """Return compact-moment bytes for three conditions and all four requested objects."""

    object_units = (
        input_units,
        hidden_units,
        hidden_units * input_units,
        hidden_units * hidden_units,
    )
    per_object = sum((NUM_CELLS + 2) * repeats * units * 8 for units in object_units)
    return 3 * per_object


def _labels_and_membership(
    dataset: object,
    chan_num: int,
    sequence_length: int,
    repeats: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, dict[str, int]]:
    """Build the shared reset-excluded labels and fixed balanced-draw membership matrix."""

    total_frames = len(dataset) * sequence_length  # type: ignore[arg-type]
    raw_labels = np.asarray(
        dataset.labels_sector[chan_num : chan_num + total_frames],  # type: ignore[attr-defined]
        dtype=np.int64,
    ).reshape(
        len(dataset), sequence_length, 2
    )  # type: ignore[arg-type]
    labels = raw_labels[:, 1:, :].reshape(-1, 2)
    draws, balance = balanced_subsample_indices(labels, repeats=repeats, seed=seed)
    membership = np.zeros((repeats, labels.shape[0]), dtype=bool)
    for repeat, indices in enumerate(draws):
        membership[repeat, indices] = True
    return labels, torch.from_numpy(membership).to(device), balance.__dict__


def _condition_rollout(
    model: torch.nn.Module,
    encoded: torch.Tensor,
    labels: torch.Tensor,
    membership: torch.Tensor,
    moments: dict[str, RepeatedCudaMoments],
    *,
    condition: str,
    num_pos: int,
    rng: np.random.Generator,
    valid_indices: torch.Tensor,
) -> tuple[int, int, int]:
    """Run one Figure-2 condition and update all requested reset-excluded objects."""

    batch_size, frame_num, input_size = encoded.shape
    schedule = None
    if condition != "baseline":
        schedule = _shuffled_schedule_slice(
            _run_baseline_feedback_schedule(model, encoded, num_pos=num_pos, device=encoded.device),
            condition,
            rng,
            num_pos,
        )
    hidden = model.core.initial_state(batch_size, encoded.device, encoded.dtype)
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError("The shuffle ANOVA supports single-layer GaWF only.")
    feedback = torch.zeros(
        batch_size, model.feedback_dim, device=encoded.device, dtype=torch.float32
    )
    char_correct = sector_correct = n_frames = 0
    for time_idx in range(frame_num):
        gate_ih, gate_hh = _gate_values(model, feedback, input_size)
        hidden = model.core.step(encoded[:, time_idx], hidden, feedback)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Unexpected non-tensor hidden state.")
        char_logits, sector_logits = model.classifier(hidden)
        if time_idx > 0:
            current_labels = labels[:, time_idx]
            current_membership = membership[:, valid_indices[:, time_idx - 1]]
            moments["encoder_activation"].update(
                encoded[:, time_idx], current_labels, current_membership
            )
            moments["hidden_activation"].update(hidden, current_labels, current_membership)
            moments["input_gate"].update(gate_ih.flatten(1), current_labels, current_membership)
            moments["recurrent_gate"].update(gate_hh.flatten(1), current_labels, current_membership)
            char_correct += int((char_logits.argmax(dim=1) == current_labels[:, 0]).sum().item())
            sector_correct += int(
                (sector_logits.argmax(dim=1) == current_labels[:, 1]).sum().item()
            )
            n_frames += batch_size
        feedback = model._compute_feedback(char_logits, sector_logits).to(dtype=torch.float32)
        if schedule is not None:
            feedback = feedback.clone()
            if condition == "shuffle_digit":
                feedback[:, DIGIT_SLICE] = schedule[:, time_idx, DIGIT_SLICE]
            else:
                feedback[:, 10 : 10 + num_pos] = schedule[:, time_idx, 10 : 10 + num_pos]
    return char_correct, sector_correct, n_frames


def collect(args: argparse.Namespace) -> Path:
    """Collect the three-condition compact summary for one GaWF training seed."""

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    if args.batch_size <= 0 or args.num_workers < 0 or args.repeats <= 0:
        raise ValueError("batch_size/repeats must be positive and num_workers must be nonnegative")
    device = resolve_device(args.device, require_cuda_if_requested=True)
    dataset_args = argparse.Namespace(
        data_dir=str(args.data_dir),
        data_suffix=args.data_suffix,
        use_mmap=True,
        use_sector_mode=True,
        predict_all_chars=False,
        chan_num=args.chan_num,
        sequence_length=args.sequence_length,
    )
    dataset, num_pos = build_test_dataset(dataset_args)
    if num_pos != 9:
        raise RuntimeError(f"Expected 9 sector labels, got {num_pos}")
    model = build_model_from_ckpt(str(args.ckpt), num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("The shuffle ANOVA requires a single-layer GaWF checkpoint.")
    if getattr(model, "proj_out", None) is not None or int(model.feedback_dim) != 19:
        raise RuntimeError("The Figure 2 shuffle protocol requires direct 19-dimensional feedback.")
    if int(dataset.frame_num) != args.sequence_length:
        raise RuntimeError("Dataset frame length does not match the requested shuffle window.")
    required_bytes = _cuda_required_bytes(
        args.repeats, model.encoder_flatten_size, model.rnn.hidden_size
    )
    available_bytes = torch.cuda.get_device_properties(device).total_memory
    if required_bytes + 6 * 1024**3 > available_bytes:
        raise MemoryError(
            f"Need at least {(required_bytes + 6 * 1024**3) / 1024**3:.1f} GiB CUDA memory for "
            f"three-condition moments; device has {available_bytes / 1024**3:.1f} GiB."
        )
    analysis_labels, membership, balance = _labels_and_membership(
        dataset, args.chan_num, args.sequence_length, args.repeats, args.seed, device
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    states = {
        condition: {
            "encoder_activation": RepeatedCudaMoments(
                args.repeats, model.encoder_flatten_size, device
            ),
            "hidden_activation": RepeatedCudaMoments(args.repeats, model.rnn.hidden_size, device),
            "input_gate": RepeatedCudaMoments(
                args.repeats, model.rnn.hidden_size * model.encoder_flatten_size, device
            ),
            "recurrent_gate": RepeatedCudaMoments(
                args.repeats, model.rnn.hidden_size * model.rnn.hidden_size, device
            ),
        }
        for condition in CONDITIONS
    }
    condition_counts = {condition: [0, 0, 0] for condition in CONDITIONS}
    rng = np.random.default_rng(args.seed)
    sequence_offset = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            frames, batch_labels = batch[0].to(
                device=device, dtype=torch.float32, non_blocking=True
            ), batch[1].to(device)
            batch_size = int(frames.shape[0])
            encoded = model.encode_frames(frames)
            sequence_ids = torch.arange(
                sequence_offset, sequence_offset + batch_size, device=device
            )
            valid_indices = sequence_ids[:, None] * (args.sequence_length - 1) + torch.arange(
                args.sequence_length - 1, device=device
            )
            for condition in CONDITIONS:
                counts = _condition_rollout(
                    model,
                    encoded,
                    batch_labels,
                    membership,
                    states[condition],
                    condition=condition,
                    num_pos=num_pos,
                    rng=rng,
                    valid_indices=valid_indices,
                )
                condition_counts[condition] = [
                    prior + update for prior, update in zip(condition_counts[condition], counts)
                ]
            sequence_offset += batch_size
            if (batch_index + 1) % 10 == 0 or sequence_offset == len(dataset):
                print(f"processed {sequence_offset}/{len(dataset)} sequences", flush=True)
    if sequence_offset != len(dataset):
        raise RuntimeError(f"Collection stopped at {sequence_offset}/{len(dataset)} sequences")
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for condition in CONDITIONS:
        result[condition] = {name: state.finalize() for name, state in states[condition].items()}
    for object_name in OBJECTS:
        baseline = result["baseline"][object_name]
        if object_name == "encoder_activation":
            for condition in CONDITIONS[1:]:
                if not all(
                    np.array_equal(baseline[key], result[condition][object_name][key])
                    for key in baseline
                ):
                    raise RuntimeError("Encoder activation differs across feedback conditions.")
    payload: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        for object_name in OBJECTS:
            for metric, values in result[condition][object_name].items():
                payload[f"{condition}__{object_name}__{metric}"] = values.astype(np.float64)
    args.output_dir.mkdir(parents=True)
    destination = args.output_dir / RESULT_NAME
    np.savez_compressed(destination, **payload)
    metrics = {
        condition: {
            "digit_acc": 100.0 * counts[0] / counts[2],
            "sector_acc": 100.0 * counts[1] / counts[2],
            "n_frames": counts[2],
        }
        for condition, counts in condition_counts.items()
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.ckpt.resolve()),
                "seed": args.seed,
                "data_suffix": args.data_suffix,
                "sequence_length": args.sequence_length,
                "conditions": list(CONDITIONS),
                "objects": list(OBJECTS),
                "factors": list(CM_FACTORS),
                "reset_excluded": True,
                "eligible_frames": int(analysis_labels.shape[0]),
                "frames_per_draw": balance["trials_retained_per_draw"],
                "balance": balance,
                "accuracy_on_reset_excluded_frames": metrics,
                "encoder_identical_across_conditions": True,
                "between_condition_var": "sum_u SS_total_cm[u] / n_balanced_trials",
                "residual_frac": "SS_residual / SS_total_trial",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {destination}", flush=True)
    return destination


def _seed_files(data_root: Path, expected_seeds: int) -> list[Path]:
    """Return the exact ten per-seed compact result files."""

    files = [data_root / f"seed{seed:02d}" / RESULT_NAME for seed in range(1, expected_seeds + 1)]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("Missing shuffle ANOVA summaries: " + ", ".join(missing))
    return files


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    """Return the seed-level mean and SEM for the registered ten-seed protocol."""

    values = np.asarray(values, dtype=np.float64)
    if values.size != 10:
        raise ValueError(f"Expected 10 training seeds, got {values.size}")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean, sem


def aggregate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Write the requested long/summary tables and render the 1-by-2 Figure 4 panel."""

    files = _seed_files(args.data_root, args.expected_seeds)
    if args.expected_seeds != 10:
        raise ValueError("This registered protocol requires seeds 1-10.")
    manifests = [
        json.loads((path.parent / "manifest.json").read_text(encoding="utf-8")) for path in files
    ]
    long_rows: list[dict[str, float | int | str]] = []
    values: dict[tuple[str, str, str], list[float]] = {}
    for seed, (path, manifest) in enumerate(zip(files, manifests), start=1):
        with np.load(path, allow_pickle=False) as archive:
            for condition in CONDITIONS:
                metrics = manifest["accuracy_on_reset_excluded_frames"][condition]
                for object_name in OBJECTS:
                    row: dict[str, float | int | str] = {
                        "object": object_name,
                        "condition": condition,
                        "seed": seed,
                        "n_frames": int(metrics["n_frames"]),
                        "n_draws": int(manifest["balance"]["repeats"]),
                        "frames_per_draw": int(manifest["frames_per_draw"]),
                        "digit_acc": float(metrics["digit_acc"]),
                        "sector_acc": float(metrics["sector_acc"]),
                    }
                    for metric in (*CM_FACTORS, "between_condition_var", "residual_frac"):
                        value = float(
                            np.asarray(
                                archive[f"{condition}__{object_name}__{metric}"], dtype=np.float64
                            ).mean()
                        )
                        row[f"{metric}_pct" if metric in CM_FACTORS else metric] = (
                            100.0 * value if metric == "residual_frac" else value
                        )
                        values.setdefault((condition, object_name, metric), []).append(value)
                    retained_trial_fraction = 1.0 - float(row["residual_frac"]) / 100.0
                    for factor in CM_FACTORS:
                        row[f"{factor}_trial_pct"] = (
                            float(row[f"{factor}_pct"]) * retained_trial_fraction
                        )
                    long_rows.append(row)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.figure_dir / "Fig4_shuffle_activation_anova_long_10seed.csv"
    fieldnames = [
        "object",
        "condition",
        "seed",
        "sector_pct",
        "digit_pct",
        "interaction_pct",
        "between_condition_var",
        "residual_frac",
        "sector_trial_pct",
        "digit_trial_pct",
        "interaction_trial_pct",
        "n_frames",
        "n_draws",
        "frames_per_draw",
        "digit_acc",
        "sector_acc",
    ]
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)
    summary_rows: list[dict[str, float | str]] = []
    for condition in CONDITIONS:
        for object_name in OBJECTS:
            row: dict[str, float | str] = {"object": object_name, "condition": condition}
            for metric in (*CM_FACTORS, "between_condition_var", "residual_frac"):
                mean, sem = _mean_sem(np.asarray(values[(condition, object_name, metric)]))
                suffix = "_pct" if metric in CM_FACTORS or metric == "residual_frac" else ""
                scale = 100.0 if metric == "residual_frac" else 1.0
                row[f"{metric}{suffix}_mean"] = scale * mean
                row[f"{metric}{suffix}_sem"] = scale * sem
            for factor in CM_FACTORS:
                seed_values = np.asarray(
                    [
                        float(item[f"{factor}_trial_pct"])
                        for item in long_rows
                        if item["object"] == object_name and item["condition"] == condition
                    ]
                )
                mean, sem = _mean_sem(seed_values)
                row[f"{factor}_trial_pct_mean"] = mean
                row[f"{factor}_trial_pct_sem"] = sem
            for metric in ("digit_acc", "sector_acc"):
                seed_values = np.asarray(
                    [
                        float(item[metric])
                        for item in long_rows
                        if item["object"] == object_name and item["condition"] == condition
                    ]
                )
                mean, sem = _mean_sem(seed_values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sem"] = sem
            summary_rows.append(row)
    summary_path = args.figure_dir / "Fig4_shuffle_activation_anova_summary_10seed.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    png, pdf = _plot(args.figure_dir, values, args.show_seed_points)
    (args.figure_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8"
    )
    return long_path, summary_path, pdf


def _plot(
    figure_dir: Path,
    values: dict[tuple[str, str, str], list[float]],
    show_seed_points: bool,
) -> tuple[Path, Path]:
    """Render baseline and shuffled versions of Fig4's activation subpanel."""

    components = (*CM_FACTORS, "residual_frac")
    component_labels = {"residual_frac": "Residual"}
    category_center = np.array([0.0])
    bar_width = 0.12
    rng = np.random.default_rng(0)
    with plt.rc_context(
        {"font.size": 13, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13}
    ):
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 5.2), sharey=True)
        for axis, condition, title in zip(
            axes,
            ("baseline", "shuffle_digit", "shuffle_sector"),
            ("Default GaWF", "Shuffle digit feedback", "Shuffle sector feedback"),
        ):
            for component_index, component in enumerate(components):
                residual = np.asarray(
                    values[(condition, "hidden_activation", "residual_frac")], dtype=np.float64
                )
                seed_values = np.asarray(
                    values[(condition, "hidden_activation", component)], dtype=np.float64
                ).reshape(-1, 1)
                if component == "residual_frac":
                    seed_values *= 100.0
                else:
                    seed_values *= 1.0 - residual[:, None]
                means = seed_values.mean(axis=0)
                errors = seed_values.std(axis=0, ddof=1) / np.sqrt(seed_values.shape[0])
                positions = category_center + (component_index - 1.5) * bar_width
                axis.bar(
                    positions,
                    means,
                    width=bar_width,
                    color=FACTOR_COLORS[component],
                    edgecolor="none",
                    yerr=errors,
                    capsize=2.5,
                    error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
                    label=component_labels.get(component, component.title()),
                    zorder=2,
                )
                add_seed_points(
                    axis,
                    positions,
                    seed_values,
                    bar_width=bar_width,
                    show=show_seed_points,
                    rng=rng,
                )
            axis.set_title(title, fontsize=15, pad=40)
            axis.set_xticks(category_center, ("Hidden\nactivation",))
            axis.set_xlim(-0.35, 0.35)
            axis.set_ylim(0.0, 100.0)
            axis.set_yticks(np.arange(0.0, 101.0, 25.0))
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        axes[0].set_ylabel("Variance component (%)")
        for axis in axes[1:]:
            axis.tick_params(axis="y", labelleft=False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995)
        )
        fig.subplots_adjust(left=0.085, right=0.995, bottom=0.14, top=0.72, wspace=0.20)
        png = figure_dir / "Fig4_shuffle_activation_anova_1x3_10seed.png"
        pdf = figure_dir / "Fig4_shuffle_activation_anova_1x3_10seed.pdf"
        fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    return png, pdf


def main() -> None:
    """Dispatch collection or aggregation and print saved output paths."""

    args = parse_args()
    if args.command == "collect":
        print(f"Saved {collect(args)}")
        return
    long_path, summary_path, pdf = aggregate(args)
    print(f"Saved {long_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
