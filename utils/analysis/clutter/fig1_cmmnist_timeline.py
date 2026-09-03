"""Render a six-frame CM-MNIST target-switch timeline for Figure 1.

Inputs are standard MNIST IDX image/label files. The movie frames and switch metadata are
generated with ``source.clutter.generate_movies``. The output is one publication-ready PDF.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import struct
import tempfile
from pathlib import Path
from typing import BinaryIO

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from source.clutter.generate_movies import StimulusConfig, generate_stimulus_video


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/save/Fig1_cmmnist_target_switch_timeline.pdf"
FRAME_OFFSETS = (-10, -1, 0, 1, 2, 9)
FRAME_LABELS = ("pre10", "pre1", "post1", "post2", "post3", "post10")
TARGET_COLOR = "#D55E00"


def _open_idx(path: Path) -> BinaryIO:
    """Open an uncompressed or gzip-compressed IDX file."""

    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _load_mnist(images_path: Path, labels_path: Path, limit: int) -> dict[int, list[np.ndarray]]:
    """Load the first ``limit`` standard MNIST samples into digit-indexed lists."""

    with _open_idx(images_path) as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic number: {magic}")
        selected = min(limit, count)
        images = np.frombuffer(stream.read(selected * rows * columns), dtype=np.uint8)
        images = images.reshape(selected, rows, columns)

    with _open_idx(labels_path) as stream:
        magic, label_count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic number: {magic}")
        labels = np.frombuffer(stream.read(selected), dtype=np.uint8)

    if label_count != count or labels.size != selected:
        raise ValueError("MNIST image and label counts do not match")

    digits = {digit: [] for digit in range(10)}
    for image, label in zip(images, labels):
        digits[int(label)].append(image)
    missing = [digit for digit, samples in digits.items() if not samples]
    if missing:
        raise ValueError(f"Selected MNIST prefix is missing digits: {missing}")
    return digits


def _read_metadata(path: Path) -> list[dict[str, str]]:
    """Read frame metadata emitted by ``generate_stimulus_video``."""

    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _background_count(row: dict[str, str]) -> int:
    """Return the number of background digits in one metadata row."""

    value = row["bg_char_ids"]
    return len(value.split(",")) if value else 0


def _select_target_switch(rows: list[dict[str, str]]) -> int | None:
    """Choose an isolated, visually legible target switch with ten frames on each side."""

    switch_flags = np.array(
        [int(row["fg_switch"]) or int(row["bg_switch"]) for row in rows],
        dtype=bool,
    )
    for switch_frame, row in enumerate(rows):
        if int(row["fg_switch"]) != 1:
            continue
        if switch_frame < 10 or switch_frame + 9 >= len(rows):
            continue
        window = slice(switch_frame - 10, switch_frame + 10)
        if np.count_nonzero(switch_flags[window]) != 1:
            continue
        if rows[switch_frame - 1]["fg_char_id"] == row["fg_char_id"]:
            continue
        counts = [_background_count(rows[switch_frame + offset]) for offset in FRAME_OFFSETS]
        if not all(4 <= count <= 8 for count in counts):
            continue
        return switch_frame
    return None


def _generate_example(
    mnist_data: dict[int, list[np.ndarray]],
    work_dir: Path,
    seed_start: int,
    max_seeds: int,
) -> tuple[np.ndarray, list[dict[str, str]], int, int]:
    """Generate deterministic candidates until an isolated target switch is available."""

    config = StimulusConfig(
        width=96,
        height=96,
        duration_seconds=12,
        fps=24,
        fg_speeds=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        bg_char_counts=[1, 2, 4, 8, 12],
        bg_mean_speeds=[1.0, 2.0, 4.0, 6.0, 8.0],
        mean_switch_interval_seconds=1.0,
        switch_mode="exclusive",
        output_dir=str(work_dir),
        suffix="fig1-cmmnist-timeline",
        output_mode="simple",
        storage_dtype="uint8",
    )
    for seed in range(seed_start, seed_start + max_seeds):
        np.random.seed(seed)
        generate_stimulus_video(config, mnist_data)
        rows = _read_metadata(work_dir / f"stimulus_{config.suffix}.tsv")
        switch_frame = _select_target_switch(rows)
        if switch_frame is not None:
            movie = np.load(
                work_dir / f"stimulus_{config.suffix}.npy",
                mmap_mode="r",
            )
            return movie, rows, switch_frame, seed
    raise RuntimeError(f"No suitable target switch found in {max_seeds} deterministic seeds")


def _add_group_bar(fig: plt.Figure, left: float, right: float, label: str) -> None:
    """Draw one compact clutter-span label in figure coordinates."""

    y = 0.93
    fig.add_artist(
        Line2D([left, right], [y, y], color="#555555", linewidth=1.2, transform=fig.transFigure)
    )
    fig.text((left + right) / 2, y + 0.018, label, ha="center", va="bottom", fontsize=10)


def _render(
    movie: np.ndarray,
    rows: list[dict[str, str]],
    switch_frame: int,
    seed: int,
    output: Path,
) -> None:
    """Render and save the five selected frames with one continuous time axis."""

    fig = plt.figure(figsize=(8.2, 2.60))
    grid = fig.add_gridspec(
        1,
        9,
        width_ratios=(1, 0.22, 1, 0.34, 1, 1, 1, 0.22, 1),
        left=0.035,
        right=0.985,
        bottom=0.32,
        top=0.75,
        wspace=0.10,
    )
    frame_columns = (0, 2, 4, 5, 6, 8)
    axes = [fig.add_subplot(grid[0, column]) for column in frame_columns]
    for column in (1, 7):
        gap_axis = fig.add_subplot(grid[0, column])
        gap_axis.axis("off")
        gap_axis.text(0.5, 0.5, r"$\cdots$", ha="center", va="center", fontsize=15)

    selected_indices = [switch_frame + offset for offset in FRAME_OFFSETS]
    for axis, frame_idx, label in zip(axes, selected_indices, FRAME_LABELS):
        row = rows[frame_idx]
        axis.imshow(movie[frame_idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.add_patch(
            Circle(
                (float(row["fg_char_x"]), float(row["fg_char_y"])),
                radius=15,
                fill=False,
                edgecolor=TARGET_COLOR,
                linewidth=1.5,
            )
        )
        axis.set_title(label, fontsize=9, pad=3)
        axis.text(
            0.5,
            -0.12,
            f"target {row['fg_char_id']}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        axis.set_axis_off()

    fig.canvas.draw()
    positions = [axis.get_position() for axis in axes]
    _add_group_bar(fig, positions[0].x0, positions[1].x1, r"Clutter $k$")
    _add_group_bar(fig, positions[2].x0, positions[5].x1, r"Clutter $k+1$")

    switch_x = (positions[1].x1 + positions[2].x0) / 2
    fig.add_artist(
        Line2D(
            [switch_x, switch_x],
            [0.25, 0.79],
            color=TARGET_COLOR,
            linewidth=1.2,
            linestyle="--",
            transform=fig.transFigure,
        )
    )
    fig.text(
        switch_x,
        0.80,
        "Target switch",
        color=TARGET_COLOR,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    axis_y = 0.14
    start_x = positions[0].x0
    end_x = positions[-1].x1
    plt.annotate(
        "",
        xy=(end_x, axis_y),
        xytext=(start_x, axis_y),
        xycoords=fig.transFigure,
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.0},
    )
    fig.text((start_x + end_x) / 2, 0.105, "Time", ha="center", va="top", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Title": "CM-MNIST target-switch timeline",
            "Subject": f"Generated with seed {seed}; switch frame {switch_frame}",
        },
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-images", type=Path, required=True)
    parser.add_argument("--mnist-labels", type=Path, required=True)
    parser.add_argument("--mnist-limit", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Generate the stimulus example and render the Figure 1 timeline PDF."""

    args = parse_args()
    if args.mnist_limit <= 0 or args.max_seeds <= 0:
        raise ValueError("--mnist-limit and --max-seeds must be positive")
    mnist_data = _load_mnist(args.mnist_images, args.mnist_labels, args.mnist_limit)
    with tempfile.TemporaryDirectory(prefix="fig1-cmmnist-") as temp_dir:
        movie, rows, switch_frame, seed = _generate_example(
            mnist_data,
            Path(temp_dir),
            args.seed_start,
            args.max_seeds,
        )
        selected = [switch_frame + offset for offset in FRAME_OFFSETS]
        expected = [
            switch_frame - 10,
            switch_frame - 1,
            switch_frame,
            switch_frame + 1,
            switch_frame + 2,
            switch_frame + 9,
        ]
        if selected != expected:
            raise AssertionError("Unexpected recovery-frame mapping")
        _render(movie, rows, switch_frame, seed, args.output)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError(f"PDF was not created: {args.output}")
    print(f"seed={seed} switch_frame={switch_frame} output={args.output}")


if __name__ == "__main__":
    main()
