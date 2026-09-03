"""Render a Fig3 companion panel from ten saved gate histograms and feedback trajectories.

The panel removes each sequence's zero-feedback reset mass before plotting input and recurrent
gate distributions, then shows the per-seed middle-interval mass. It writes one curated PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEED_DIRS = tuple(
    PROJECT_ROOT / "results" / "save_data" / "fig3" / f"seed{seed:02d}"
    for seed in range(1, 11)
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "save" / "Supple_Fig3_reset_excluded_gate_panel.png"
GATE_KINDS = (("input", "Input gate", "#2b6cb0"), ("recurrent", "Recurrent gate", "#d95f02"))


def parse_args() -> argparse.Namespace:
    """Parse paths for the saved ten-seed Fig3 inputs and companion output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed_dirs", type=Path, nargs="+", default=DEFAULT_SEED_DIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _reset_frames(trajectory_path: Path) -> int:
    """Return the number of frames whose saved pre-step feedback is exactly zero."""

    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        feedback = trajectory["feedback"].reshape(-1, trajectory["feedback"].shape[-1])
    return int(np.all(feedback == 0.0, axis=1).sum())


def _edge_index(edges: np.ndarray, value: float) -> int:
    """Return the saved float32 histogram edge nearest an exact gate threshold."""

    index = int(np.argmin(np.abs(edges - value)))
    if not np.isclose(edges[index], value, atol=1e-6):
        raise ValueError(f"Histogram does not contain the required edge {value}")
    return index


def _seed_data(seed_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Load one seed's reset-excluded gate probabilities and middle masses."""

    stats_path = seed_dir / "gawf_gate_distribution_stats.npz"
    trajectory_path = seed_dir / "gawf_gate_trajectory.npz"
    with np.load(stats_path, allow_pickle=False) as stats:
        edges = np.asarray(stats["gate_edges"], dtype=np.float64)
        counts = {
            kind: np.asarray(stats[f"hist_{kind}_all"], dtype=np.float64)
            for kind, *_ in GATE_KINDS
        }
    reset_frames = _reset_frames(trajectory_path)
    low = _edge_index(edges, 0.1)
    high = _edge_index(edges, 0.9)
    half_bin = _edge_index(edges, 0.5)
    probabilities: dict[str, np.ndarray] = {}
    middle_mass: dict[str, float] = {}
    for kind, _title, _color in GATE_KINDS:
        total = int(counts[kind].sum())
        source_size = 1152 if kind == "input" else 256
        hidden_size = 256
        reset_count = reset_frames * hidden_size * source_size
        if reset_count >= total or counts[kind][half_bin] < reset_count:
            raise ValueError(f"Invalid reset mass in {seed_dir} for {kind} gate")
        reset_excluded = counts[kind].copy()
        reset_excluded[half_bin] -= reset_count
        denominator = reset_excluded.sum()
        probabilities[kind] = 100.0 * reset_excluded / denominator
        middle_mass[kind] = float(100.0 * reset_excluded[low:high].sum() / denominator)
    return edges, probabilities, middle_mass


def _load_seeds(
    seed_dirs: list[Path],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return binwise and per-seed reset-excluded summaries for exactly ten seeds."""

    if len(seed_dirs) != 10:
        raise ValueError(f"Expected ten seed directories, found {len(seed_dirs)}")
    all_probabilities = {kind: [] for kind, *_ in GATE_KINDS}
    all_middle_mass = {kind: [] for kind, *_ in GATE_KINDS}
    reference_edges: np.ndarray | None = None
    for seed_dir in seed_dirs:
        edges, probabilities, middle_mass = _seed_data(seed_dir)
        if reference_edges is None:
            reference_edges = edges
        elif not np.array_equal(reference_edges, edges):
            raise ValueError("All Fig3 seeds must share gate histogram edges")
        for kind, *_ in GATE_KINDS:
            all_probabilities[kind].append(probabilities[kind])
            all_middle_mass[kind].append(middle_mass[kind])
    if reference_edges is None:
        raise RuntimeError("No Fig3 seed summaries loaded")
    return (
        reference_edges,
        {kind: np.stack(values) for kind, values in all_probabilities.items()},
        {kind: np.asarray(values) for kind, values in all_middle_mass.items()},
    )


def _style(axis: plt.Axes) -> None:
    """Apply the shared uncluttered axis style."""

    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    """Render and save the reset-excluded Fig3 companion panel."""

    args = parse_args()
    edges, probabilities, middle_mass = _load_seeds(list(args.seed_dirs))
    centers = (edges[:-1] + edges[1:]) / 2.0
    figure = plt.figure(figsize=(8.2, 4.9))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.55, 1.0), hspace=0.22, wspace=0.42)
    distribution_axes = [figure.add_subplot(grid[row, 0]) for row in range(2)]
    mass_axis = figure.add_subplot(grid[:, 1])
    for axis, (kind, title, color) in zip(distribution_axes, GATE_KINDS):
        values = probabilities[kind]
        axis.axvspan(0.1, 0.9, color="#f4a261", alpha=0.13, zorder=0)
        axis.plot(centers, values.mean(axis=0), color=color, linewidth=1.6)
        axis.set(title=title, xlim=(0.0, 1.0), ylabel="Probability (%)")
        axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
        _style(axis)
    distribution_axes[0].tick_params(axis="x", labelbottom=False)
    distribution_axes[1].set_xlabel("Gate value (zero-feedback reset removed)")
    distribution_axes[0].text(
        0.5,
        0.92,
        "shaded: intermediate interval [0.1, 0.9]",
        transform=distribution_axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    rng = np.random.default_rng(0)
    for index, (kind, title, color) in enumerate(GATE_KINDS):
        values = middle_mass[kind]
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(values.size))
        mass_axis.bar(index, mean, width=0.68, color=color, alpha=0.86, yerr=sem, capsize=3)
        mass_axis.scatter(
            index + rng.uniform(-0.16, 0.16, values.size),
            values,
            color="#222222",
            alpha=0.7,
            s=19,
            linewidths=0,
            zorder=3,
        )
        mass_axis.text(index, mean + sem + 1.1, f"{mean:.1f}%", ha="center", fontsize=11)
    mass_axis.set(
        title="Remaining intermediate mass",
        xticks=(0, 1),
        xticklabels=("Input", "Recurrent"),
        ylim=(0.0, 40.0),
        ylabel="[0.1, 0.9] gates (%)",
    )
    _style(mass_axis)
    figure.text(0.01, 0.98, "Fig3 companion | mean ± SEM across 10 training seeds", va="top")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
