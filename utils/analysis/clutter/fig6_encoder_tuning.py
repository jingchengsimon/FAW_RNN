"""Validate encoder tuning topology before the Figure 6 input-gate spatial maps.

For each of ten single-layer GaWF seeds, ``collect`` streams validation encoder activations and
test input-gate columns through balanced two-factor decompositions.  It writes per-unit Sector
and Digit condition-mean eta-squared draws, spatial/channel structure fractions, and direct
encoder-tuning-to-gate-modulation correlations.  ``plot`` aggregates those compact files into
the Figure 6 preliminary panel; no raw activation or gate arrays are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from utils.analysis.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device
from utils.analysis.clutter.multiseed_plotting import add_seed_points
from utils.analysis.variance_decomposition import (
    CM_FACTORS,
    StreamingMoments,
    balanced_subsample_indices,
)
from utils.training.recurrent_cores.gawf import _compute_gawf_transforms


ENCODER_SHAPE = (32, 6, 6)
RESULT_NAME = "encoder_tuning.npz"
AXIS_NAMES = ("Spatial position (6×6)", "Feature channel (32)", "Interaction")
AXIS_COLORS = ("#4C78A8", "#F58518", "#BAB0AC")
TUNING_FACTORS = ("sector", "digit")
TUNING_COLORS = ("#264653", "#E76F51")


def parse_args() -> argparse.Namespace:
    """Parse the single-seed collector and ten-seed plotter commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--ckpt", required=True, type=Path)
    collect_parser.add_argument("--data_dir", required=True, type=Path)
    collect_parser.add_argument("--output_dir", required=True, type=Path)
    collect_parser.add_argument("--seed", required=True, type=int)
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--batch_size", type=int, default=16)
    collect_parser.add_argument("--num_workers", type=int, default=2)
    collect_parser.add_argument("--data_suffix", default="40h-uint8")
    collect_parser.add_argument("--chan_num", type=int, default=2)
    collect_parser.add_argument("--repeats", type=int, default=20)

    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("--data_root", required=True, type=Path)
    plot_parser.add_argument("--figure_dir", required=True, type=Path)
    plot_parser.add_argument("--stem", default="encoder_tuning_spatiality_1x5_10seed")
    plot_parser.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay one neutral-gray point per training seed on each bar.",
    )
    return parser.parse_args()


def _frame_labels(dataset: object) -> np.ndarray:
    """Return exactly the frame labels emitted by the sequential evaluation dataset."""

    frame_num = int(dataset.frame_num)
    start = int(dataset.chan_num)
    stop = start + len(dataset) * frame_num
    return np.asarray(dataset.labels_sector[start:stop], dtype=np.int64)


def _input_gate_columns(model: torch.nn.Module, encoded: torch.Tensor) -> torch.Tensor:
    """Replay reset GaWF feedback and return hidden-mean input gates for each encoder unit."""

    batch_size, frame_num, input_size = encoded.shape
    hidden = model.core.initial_state(batch_size, encoded.device, encoded.dtype)
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError("Figure 6 encoder tuning requires a single-layer GaWF checkpoint.")
    feedback = torch.zeros(
        batch_size, model.feedback_dim, device=encoded.device, dtype=torch.float32
    )
    columns: list[torch.Tensor] = []
    for time_idx in range(frame_num):
        transform_ih, _transform_hh = _compute_gawf_transforms(
            model.U, feedback.clamp(-10, 10).unsqueeze(2), model.V, input_size
        )
        columns.append(torch.sigmoid(transform_ih / model.gate_tau).mean(dim=1))
        hidden = model.core.step(encoded[:, time_idx], hidden, feedback)
        char_logits, sector_logits = model.classifier(hidden)
        feedback = model._compute_feedback(char_logits, sector_logits).to(dtype=torch.float32)
    return torch.stack(columns, dim=1)


def _balanced_draws(
    labels: np.ndarray, repeats: int, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Return boolean draw membership plus auditable equal-cell balance metadata."""

    draws, report = balanced_subsample_indices(labels, repeats=repeats, seed=seed)
    membership = np.zeros((repeats, labels.shape[0]), dtype=bool)
    for draw_index, indices in enumerate(draws):
        membership[draw_index, indices] = True
    return membership, report.__dict__


def _stream_selectivity(
    dataset: object,
    model: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    repeats: int,
    seed: int,
    source: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Stream balanced condition-mean selectivity for encoder activations or gate columns."""

    labels = _frame_labels(dataset)
    membership, balance = _balanced_draws(labels, repeats, seed)
    accumulators: list[StreamingMoments] | None = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    frame_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            frames, batch_labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32, non_blocking=True)
            encoded = model.encode_frames(frames)
            values = encoded if source == "encoder" else _input_gate_columns(model, encoded)
            flat_values = values.detach().cpu().numpy().reshape(-1, values.shape[-1])
            flat_labels = np.asarray(batch_labels, dtype=np.int64).reshape(-1, 2)
            frame_count = flat_labels.shape[0]
            if not np.array_equal(flat_labels, labels[frame_offset : frame_offset + frame_count]):
                raise RuntimeError("DataLoader label order differs from balanced-draw indexing.")
            if accumulators is None:
                accumulators = [StreamingMoments(flat_values.shape[1]) for _ in range(repeats)]
            active = membership[:, frame_offset : frame_offset + frame_count]
            for draw_index, selected in enumerate(active):
                if np.any(selected):
                    accumulators[draw_index].update(
                        flat_values[selected], flat_labels[selected]
                    )
            frame_offset += frame_count
    if frame_offset != labels.shape[0] or accumulators is None:
        raise RuntimeError(
            f"{source} collection stopped at {frame_offset}/{labels.shape[0]} frames."
        )
    decompositions = [accumulator.finalize() for accumulator in accumulators]
    values_by_factor = {
        factor: np.stack(
            [decomposition.per_unit_cm[factor] for decomposition in decompositions], axis=0
        ).astype(np.float32)
        for factor in CM_FACTORS
    }
    return values_by_factor, balance


def _axis_fractions(eta: np.ndarray) -> np.ndarray:
    """Partition a 32×6×6 unit-tuning map into spatial, channel, and interaction structure."""

    shaped = np.asarray(eta, dtype=np.float64).reshape(ENCODER_SHAPE)
    grand = shaped.mean()
    spatial = shaped.mean(axis=0)
    channel = shaped.mean(axis=(1, 2))
    interaction = shaped - spatial[None, :, :] - channel[:, None, None] + grand
    sums = np.asarray(
        (
            ENCODER_SHAPE[0] * np.square(spatial - grand).sum(),
            ENCODER_SHAPE[1] * ENCODER_SHAPE[2] * np.square(channel - grand).sum(),
            np.square(interaction).sum(),
        ),
        dtype=np.float64,
    )
    total = float(sums.sum())
    if total <= 0:
        raise RuntimeError("Encoder tuning does not vary across spatial positions or channels.")
    return (sums / total).astype(np.float32)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Return a deterministic rank correlation without adding a new statistics dependency."""

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        output = np.empty(values.size, dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and sorted_values[stop] == sorted_values[start]:
                stop += 1
            output[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return output

    left_rank, right_rank = ranks(np.asarray(left)), ranks(np.asarray(right))
    denominator = float(
        np.linalg.norm(left_rank - left_rank.mean())
        * np.linalg.norm(right_rank - right_rank.mean())
    )
    return float(np.dot(left_rank - left_rank.mean(), right_rank - right_rank.mean()) / denominator)


def collect(args: argparse.Namespace) -> Path:
    """Collect compact validation encoder tuning and test gate-modulation evidence for one seed."""

    if args.batch_size <= 0 or args.num_workers < 0 or args.repeats <= 0:
        raise ValueError("batch_size/repeats must be positive and num_workers nonnegative.")
    destination = args.output_dir / RESULT_NAME
    if destination.exists() or args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    device = resolve_device(args.device, require_cuda_if_requested=True)
    dataset_args = argparse.Namespace(
        data_dir=str(args.data_dir),
        data_suffix=args.data_suffix,
        use_mmap=True,
        use_sector_mode=True,
        predict_all_chars=False,
        chan_num=args.chan_num,
    )
    validation, num_pos = build_eval_dataset(dataset_args, "validation")
    test, test_num_pos = build_eval_dataset(dataset_args, "test")
    if num_pos != test_num_pos:
        raise RuntimeError("Validation/test sector-head dimensions differ.")
    model = build_model_from_ckpt(str(args.ckpt), num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise ValueError("Figure 6 encoder tuning requires a single-layer GaWF checkpoint.")
    encoder, validation_balance = _stream_selectivity(
        validation,
        model,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        repeats=args.repeats,
        seed=args.seed,
        source="encoder",
    )
    gate, test_balance = _stream_selectivity(
        test,
        model,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        repeats=args.repeats,
        seed=args.seed + 10_000,
        source="gate",
    )
    if encoder["sector"].shape[1] != int(np.prod(ENCODER_SHAPE)):
        raise RuntimeError("Encoder topology must be exactly 32×6×6.")
    axis_sector = np.stack([_axis_fractions(item) for item in encoder["sector"]])
    axis_digit = np.stack([_axis_fractions(item) for item in encoder["digit"]])
    alignment = np.asarray(
        [
            (
                _spearman(encoder["sector"][index], gate["sector"][index]),
                _spearman(encoder["digit"][index], gate["sector"][index]),
            )
            for index in range(args.repeats)
        ],
        dtype=np.float32,
    )
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        destination,
        encoder_sector=encoder["sector"],
        encoder_digit=encoder["digit"],
        gate_sector=gate["sector"],
        axis_sector=axis_sector,
        axis_digit=axis_digit,
        alignment=alignment,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.ckpt),
                "seed": args.seed,
                "repeats": args.repeats,
                "encoder_split": "validation",
                "gate_split": "test",
                "encoder_activation": "CNN output before the input gate",
                "gate_measure": "mean input gate over hidden destinations, reset feedback",
                "axis_partition": list(AXIS_NAMES),
                "validation_balance": validation_balance,
                "test_balance": test_balance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _load_seeds(data_root: Path) -> dict[str, np.ndarray]:
    """Load exactly ten compact GaWF outputs and stack seed then repeated-draw dimensions."""

    paths = sorted(data_root.glob(f"gawf-seed*/{RESULT_NAME}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten gawf-seed outputs in {data_root}, found {len(paths)}.")
    keys = ("encoder_sector", "encoder_digit", "axis_sector", "axis_digit", "alignment")
    values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for key in keys:
                values[key].append(np.asarray(arrays[key], dtype=np.float64))
    return {key: np.stack(items, axis=0) for key, items in values.items()}


def _bar_with_seeds(
    axis: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    colors: tuple[str, ...],
    *,
    show_seed_points: bool,
) -> None:
    """Draw cross-seed mean ± SEM bars with individual seed points."""

    rng = np.random.default_rng(0)
    means = values.mean(axis=0)
    errors = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    axis.bar(
        positions,
        100.0 * means,
        color=colors,
        yerr=100.0 * errors,
        capsize=2.5,
        error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
        zorder=2,
    )
    add_seed_points(
        axis,
        positions,
        100.0 * values,
        bar_width=0.72,
        show=show_seed_points,
        rng=rng,
    )


def _tuning_profiles(
    data: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return seed-wise channel and flattened-spatial eta-squared profiles for each factor."""

    channel_values: dict[str, np.ndarray] = {}
    spatial_values: dict[str, np.ndarray] = {}
    unit_count = int(np.prod(ENCODER_SHAPE))
    for factor in TUNING_FACTORS:
        values = np.asarray(data[f"encoder_{factor}"], dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] != unit_count:
            raise RuntimeError(
                f"Expected seed-by-draw-by-{unit_count} encoder {factor} tuning values, "
                f"received {values.shape}."
            )
        channel_values[factor] = values.reshape(
            values.shape[0], values.shape[1], *ENCODER_SHAPE
        ).mean(axis=(1, 3, 4))
        spatial_values[factor] = values.reshape(
            values.shape[0], values.shape[1], *ENCODER_SHAPE
        ).mean(axis=(1, 2)).reshape(values.shape[0], -1)
    return channel_values, spatial_values


def _plot_tuning_profile(
    axis: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    color: str,
    label: str,
    rng: np.random.Generator,
) -> float:
    """Draw a cross-seed mean, SEM band, and seed points for one tuning profile."""

    means = values.mean(axis=0)
    errors = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    axis.fill_between(
        positions, means - errors, means + errors, color=color, alpha=0.18, linewidth=0
    )
    axis.plot(positions, means, color=color, linewidth=1.4, label=label, zorder=2)
    axis.scatter(
        positions[None, :] + rng.uniform(-0.13, 0.13, size=values.shape),
        values,
        color=color,
        s=7,
        alpha=0.25,
        linewidths=0,
        zorder=3,
    )
    return float(np.max(means + errors))


def plot(args: argparse.Namespace) -> tuple[Path, Path]:
    """Render the 1-by-4 preliminary panel and save its numeric cross-seed summary."""

    data = _load_seeds(args.data_root)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    maps = {
        "sector": data["encoder_sector"].mean(axis=(0, 1)).reshape(ENCODER_SHAPE).mean(axis=0),
        "digit": data["encoder_digit"].mean(axis=(0, 1)).reshape(ENCODER_SHAPE).mean(axis=0),
    }
    axis_values = {
        "sector": data["axis_sector"].mean(axis=1),
        "digit": data["axis_digit"].mean(axis=1),
    }
    channel_values, spatial_values = _tuning_profiles(data)
    alignment = data["alignment"].mean(axis=1)
    np.savez_compressed(
        args.figure_dir / f"{args.stem}_summary.npz",
        sector_map=maps["sector"].astype(np.float32),
        digit_map=maps["digit"].astype(np.float32),
        axis_sector=axis_values["sector"].astype(np.float32),
        axis_digit=axis_values["digit"].astype(np.float32),
        channel_sector=channel_values["sector"].astype(np.float32),
        channel_digit=channel_values["digit"].astype(np.float32),
        spatial_sector=spatial_values["sector"].astype(np.float32),
        spatial_digit=spatial_values["digit"].astype(np.float32),
        alignment=alignment.astype(np.float32),
    )
    with plt.rc_context({"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12}):
        fig = plt.figure(figsize=(15.0, 4.0))
        grid = fig.add_gridspec(
            1,
            4,
            width_ratios=(1.48, 1.48, 1.46, 1.25),
            wspace=0.50,
        )
        axes = [fig.add_subplot(grid[0, index]) for index in range(4)]

        spatial_positions = np.arange(1, int(np.prod(ENCODER_SHAPE[1:])) + 1)
        spatial_rng = np.random.default_rng(3)
        spatial_top = 0.0
        for factor, color in zip(TUNING_FACTORS, TUNING_COLORS):
            spatial_top = max(
                spatial_top,
                _plot_tuning_profile(
                    axes[0],
                    spatial_positions,
                    spatial_values[factor],
                    color,
                    factor.title(),
                    spatial_rng,
                ),
            )
        axes[0].set_title("Spatial-position $\\eta^2$ distribution")
        axes[0].set_xlabel("Spatial position")
        axes[0].set_ylabel("Encoder $\\eta^2$")
        axes[0].set_xlim(0.5, spatial_positions[-1] + 0.5)
        axes[0].set_xticks((1, 9, 18, 27, 36))
        axes[0].grid(axis="y", alpha=0.25, linewidth=0.7)
        axes[0].set_axisbelow(True)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].legend(frameon=False, fontsize=9, loc="upper right")

        positions = np.arange(6, dtype=np.float64)
        _bar_with_seeds(
            axes[2], positions[:3], axis_values["sector"], AXIS_COLORS,
            show_seed_points=args.show_seed_points,
        )
        _bar_with_seeds(
            axes[2], positions[3:], axis_values["digit"], AXIS_COLORS,
            show_seed_points=args.show_seed_points,
        )
        axes[2].set_xticks((1, 4), ("Sector tuning", "Digit tuning"))
        axes[2].set_ylabel("Tuning-structure variance (%)")
        axes[2].set_ylim(0.0, 100.0)
        axes[2].set_yticks(np.arange(0.0, 101.0, 20.0))
        axes[2].grid(axis="y", alpha=0.25, linewidth=0.7)
        axes[2].set_axisbelow(True)
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["right"].set_visible(False)
        axes[2].legend(
            [Line2D([0], [0], color=color, linewidth=5) for color in AXIS_COLORS],
            AXIS_NAMES,
            frameon=False,
            fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.31),
        )

        channels = np.arange(1, ENCODER_SHAPE[0] + 1)
        channel_axis = axes[1]
        channel_rng = np.random.default_rng(2)
        channel_top = 0.0
        for factor, color in zip(TUNING_FACTORS, TUNING_COLORS):
            channel_top = max(
                channel_top,
                _plot_tuning_profile(
                    channel_axis,
                    channels,
                    channel_values[factor],
                    color,
                    factor.title(),
                    channel_rng,
                ),
            )
        channel_axis.set_title("Channel-wise $\\eta^2$ distribution")
        channel_axis.set_xlabel("Feature channel")
        channel_axis.set_ylabel("Encoder $\\eta^2$")
        channel_axis.set_xlim(0.5, ENCODER_SHAPE[0] + 0.5)
        channel_axis.set_xticks((1, 8, 16, 24, 32))
        profile_top = max(spatial_top, channel_top)
        for profile_axis in (axes[0], channel_axis):
            profile_axis.set_ylim(0.0, max(0.05, 1.08 * profile_top))
        channel_axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        channel_axis.set_axisbelow(True)
        channel_axis.spines["top"].set_visible(False)
        channel_axis.spines["right"].set_visible(False)
        channel_axis.legend(frameon=False, fontsize=9, loc="upper right")

        labels = ("Sector tuning", "Digit tuning")
        means = alignment.mean(axis=0)
        errors = alignment.std(axis=0, ddof=1) / np.sqrt(alignment.shape[0])
        x = np.arange(2, dtype=np.float64)
        axes[3].bar(x, means, color=TUNING_COLORS, yerr=errors, capsize=2.5, zorder=2)
        add_seed_points(
            axes[3],
            x,
            alignment,
            bar_width=0.72,
            show=args.show_seed_points,
            rng=np.random.default_rng(1),
        )
        axes[3].axhline(0.0, color="0.35", linewidth=0.8, zorder=1)
        axes[3].set_xticks(x, labels, rotation=20, ha="right")
        axes[3].set_ylabel("Spearman $\\rho$ with\ninput-gate Sector modulation")
        axes[3].set_ylim(-1.0, 1.0)
        axes[3].set_yticks(np.arange(-1.0, 1.01, 0.5))
        axes[3].grid(axis="y", alpha=0.25, linewidth=0.7)
        axes[3].set_axisbelow(True)
        axes[3].spines["top"].set_visible(False)
        axes[3].spines["right"].set_visible(False)
        fig.subplots_adjust(left=0.045, right=0.995, bottom=0.22, top=0.80)
        png = args.figure_dir / f"{args.stem}.png"
        pdf = args.figure_dir / f"{args.stem}.pdf"
        fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    return png, pdf


def main() -> None:
    """Dispatch one-seed streaming collection or ten-seed figure aggregation."""

    args = parse_args()
    if args.command == "collect":
        print(f"Saved {collect(args)}")
    else:
        png, pdf = plot(args)
        print(f"Saved {png}")
        print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
