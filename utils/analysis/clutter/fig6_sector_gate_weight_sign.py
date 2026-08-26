"""Split Figure 6 sequential input-gate maps by their corresponding input-weight sign.

``collect`` reconstructs exact sequential gates from one compact trajectory, uses the original
equal-n sector selection, and writes positive- and negative-weight point-excluded maps. ``plot``
averages ten seed files and renders both 3-by-3 sector grids in one 1-by-2 figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402

from utils.analysis.clutter.fig3_gate_distribution import (
    exclude_zero_feedback_reset_frames,
    iter_gate_chunks,
)
from utils.analysis.clutter.fig6_sector_gate_sequential import (
    ENCODER_SHAPE,
    NUM_SECTORS,
    equal_n_sector_mask,
)


RESULT_NAME = "sector_gate_mean_sequential_equal_n_weight_sign.npz"
SIGNS = ("positive", "negative")


def parse_args() -> argparse.Namespace:
    """Parse per-seed gate reconstruction and ten-seed plotting commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--trajectory", required=True, type=Path)
    collect_parser.add_argument("--output_dir", required=True, type=Path)
    collect_parser.add_argument("--seed", required=True, type=int)
    collect_parser.add_argument("--gate_tau", type=float, default=0.5)
    collect_parser.add_argument("--gate_chunk_size", type=int, default=16)
    collect_parser.add_argument("--point_tolerance", type=float, default=1e-6)
    collect_parser.add_argument("--device", default="cpu")

    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("--data_root", required=True, type=Path)
    plot_parser.add_argument("--figure_dir", required=True, type=Path)
    plot_parser.add_argument(
        "--stem",
        default="Fig6_sector_gate_mean_sequential_equal_n_point_excluded_weight_sign_10seed",
    )
    plot_parser.add_argument("--delta", action="store_true")
    plot_parser.add_argument("--pdf_only", action="store_true")
    return parser.parse_args()


def _sign_maps(
    feedback: np.ndarray,
    labels: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    weight_ih: np.ndarray,
    *,
    selection_seed: int,
    gate_tau: float,
    gate_chunk_size: int,
    point_tolerance: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-n, point-excluded spatial maps for ``weight_ih > 0`` and ``weight_ih < 0``."""

    if weight_ih.ndim != 2 or weight_ih.shape[1] != int(np.prod(ENCODER_SHAPE)):
        raise ValueError(f"Expected input weights (hidden, 1152), got {weight_ih.shape}.")
    feedback, labels, _reset_frames = exclude_zero_feedback_reset_frames(feedback, labels)
    sectors = np.asarray(labels, dtype=np.int64).reshape(-1, 2)[:, 1]
    selected, target, original_counts = equal_n_sector_mask(sectors, selection_seed)
    sign_masks = np.stack((weight_ih > 0.0, weight_ih < 0.0), axis=0)
    if not np.all(sign_masks.reshape(2, -1).sum(axis=1) > 0):
        raise RuntimeError("Both input-weight sign groups must contain at least one synapse.")
    sums = np.zeros((2, NUM_SECTORS, *weight_ih.shape), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    for start, end, gate_input, _gate_recurrent in iter_gate_chunks(
        feedback, u, v, weight_ih.shape[1], gate_tau, gate_chunk_size, device=device
    ):
        chunk_selected = selected[start:end]
        chunk_sectors = sectors[start:end]
        for sector in np.unique(chunk_sectors[chunk_selected]):
            use = chunk_selected & (chunk_sectors == sector)
            values = gate_input[use]
            valid = np.abs(values - 0.5) >= point_tolerance
            for sign_index, sign_mask in enumerate(sign_masks):
                keep = valid & sign_mask[None, :, :]
                sums[sign_index, sector] += np.where(keep, values, 0.0).sum(
                    axis=0, dtype=np.float64
                )
                counts[sign_index, sector] += keep.sum(axis=0, dtype=np.int64)
    for sign_index, sign_mask in enumerate(sign_masks):
        if np.any(counts[sign_index, :, sign_mask] == 0):
            raise RuntimeError("A selected sign-weight synapse had no point-excluded observations.")
    connection_means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    maps = np.empty((2, NUM_SECTORS, ENCODER_SHAPE[1], ENCODER_SHAPE[2]), dtype=np.float64)
    for sign_index, sign_mask in enumerate(sign_masks):
        reshaped_mask = sign_mask.reshape(weight_ih.shape[0], *ENCODER_SHAPE)
        reshaped_means = connection_means[sign_index].reshape(NUM_SECTORS, *reshaped_mask.shape)
        numerator = (reshaped_means * reshaped_mask[None, :, :, :]).sum(axis=(1, 2))
        denominator = reshaped_mask.sum(axis=(0, 1))
        maps[sign_index] = numerator / denominator[None, :, :]
    return maps.astype(np.float32), original_counts, np.asarray(target, dtype=np.int64)


def collect(args: argparse.Namespace) -> Path:
    """Reconstruct and save compact sign-split Figure 6 maps for one seed."""

    if args.gate_tau <= 0 or args.gate_chunk_size <= 0 or args.point_tolerance < 0:
        raise ValueError("Invalid gate_tau, gate_chunk_size, or point_tolerance.")
    destination = args.output_dir / RESULT_NAME
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    with np.load(args.trajectory, allow_pickle=False) as arrays:
        required = ("feedback", "labels", "U", "V", "weight_ih")
        if any(key not in arrays for key in required):
            raise RuntimeError(f"Trajectory is missing one of {required}: {args.trajectory}")
        maps, original_counts, target = _sign_maps(
            arrays["feedback"].astype(np.float32, copy=False),
            arrays["labels"].astype(np.int64, copy=False),
            arrays["U"].astype(np.float32, copy=False),
            arrays["V"].astype(np.float32, copy=False),
            arrays["weight_ih"].astype(np.float32, copy=False),
            selection_seed=args.seed,
            gate_tau=args.gate_tau,
            gate_chunk_size=args.gate_chunk_size,
            point_tolerance=args.point_tolerance,
            device=args.device,
        )
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(destination, positive=maps[0], negative=maps[1])
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "trajectory": str(args.trajectory),
                "selection": "original equal-n sector selection",
                "selection_seed": args.seed,
                "selected_frames_per_sector": int(target),
                "original_frames_by_sector": original_counts.astype(int).tolist(),
                "gate_measure": "point-excluded sequential input gate",
                "positive_definition": "corresponding static input weight_ih > 0",
                "negative_definition": "corresponding static input weight_ih < 0",
                "aggregation": "per-synapse frame mean, then mean over hidden units by 6-by-6",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _load_maps(data_root: Path) -> dict[str, np.ndarray]:
    """Load exactly ten independent sign-split map files and average seeds."""

    paths = sorted(data_root.glob(f"seed*/{RESULT_NAME}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten seed files in {data_root}, found {len(paths)}.")
    result: dict[str, list[np.ndarray]] = {sign: [] for sign in SIGNS}
    expected_shape = (NUM_SECTORS, ENCODER_SHAPE[1], ENCODER_SHAPE[2])
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for sign in SIGNS:
                value = np.asarray(arrays[sign], dtype=np.float64)
                if value.shape != expected_shape:
                    raise RuntimeError(f"Unexpected {sign} map shape in {path}: {value.shape}.")
                result[sign].append(value)
    return {sign: np.mean(values, axis=0) for sign, values in result.items()}


def plot(args: argparse.Namespace) -> tuple[Path | None, Path, Path]:
    """Render the requested 1-by-2 sign-split Figure 6 grid and numeric summary."""

    maps = _load_maps(args.data_root)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    if args.delta:
        maps = {sign: value - value.mean(axis=0, keepdims=True) for sign, value in maps.items()}
        limit = max(float(np.abs(value).max()) for value in maps.values())
        if limit <= 0.0:
            raise RuntimeError("Delta gate maps have zero range.")
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        colorbar_label = "Delta input gate"
    else:
        norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
        colorbar_label = "Mean input gate"
    fig = plt.figure(figsize=(14.2, 6.6), constrained_layout=True)
    subfigures = fig.subfigures(1, 2, wspace=0.035)
    for subfigure, sign, title in zip(subfigures, SIGNS, ("Input weight > 0", "Input weight < 0")):
        axes = subfigure.subplots(3, 3)
        subfigure.suptitle(f"{title}{' (delta gate)' if args.delta else ''}", fontsize=17)
        image = None
        for sector, axis in enumerate(axes.flat):
            image = axis.pcolormesh(
                maps[sign][sector],
                cmap="RdBu_r",
                norm=norm,
                shading="flat",
                edgecolors="face",
                linewidth=0.01,
                antialiased=False,
                rasterized=False,
                snap=True,
            )
            axis.set_xlim(0, 6)
            axis.set_ylim(6, 0)
            axis.set_aspect("equal")
            axis.set_title(f"Sector {sector}", fontsize=13)
            axis.set_xticks([])
            axis.set_yticks([])
        assert image is not None
        subfigure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.76, label=colorbar_label)
    summary = args.figure_dir / f"{args.stem}_summary.npz"
    np.savez_compressed(
        summary,
        positive=maps["positive"].astype(np.float32),
        negative=maps["negative"].astype(np.float32),
    )
    png, pdf = (args.figure_dir / f"{args.stem}.png", args.figure_dir / f"{args.stem}.pdf")
    if not args.pdf_only:
        fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return (None if args.pdf_only else png), pdf, summary


def main() -> None:
    """Dispatch per-seed reconstruction or the ten-seed Figure 6 aggregation."""

    args = parse_args()
    if args.command == "collect":
        print(f"Saved {collect(args)}")
    else:
        for path in plot(args):
            if path is not None:
                print(f"Saved {path}")


if __name__ == "__main__":
    main()
