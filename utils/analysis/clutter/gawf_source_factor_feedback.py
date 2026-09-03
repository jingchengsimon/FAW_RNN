"""Measure learned source-side GaWF feedback factors on held-out test trajectories.

Inputs are the ten retained Figure-3 ``gawf_gate_trajectory.npz`` files.  Each archive
contains the original GaWF factors ``U_saved`` (destination by feedback) and ``V_saved``
(feedback by source), plus a reset-feedback test trajectory.  This script uses the algebraically
equivalent source-side convention ``U_source = V_saved.T`` and
``V_destination = U_saved.T``.  It estimates the reset-excluded feedback covariance per training
seed, decomposes each gate-logit variance into Digit, Sector, and cross terms, and writes a
seed-level long CSV, a ten-seed summary JSON, and feedback/destination attribution figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from utils.analysis.anal_paths import output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[3]
N_DIGIT = 10
N_SEEDS = 10
INITIALIZER_STD = 0.01
NULL_RNG_OFFSET = 20260823


def parse_args() -> argparse.Namespace:
    """Parse retained-trajectory inputs and canonical output locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory_root",
        type=Path,
        default=PROJECT_ROOT / "results" / "save_data" / "fig3",
        help="Directory containing seed01 through seed10 trajectory archives.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, N_SEEDS + 1)))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for the long CSV and summary JSON.",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=None,
        help="Directory for the reset-excluded feedback-attribution figures.",
    )
    parser.add_argument("--self_check", action="store_true")
    parser.add_argument(
        "--plot_colwise_normalized_only",
        action="store_true",
        help="Write only the clipped, column-wise normalized ten-seed mean |U| PDF.",
    )
    return parser.parse_args()


def _sem(values: np.ndarray) -> float:
    """Return the training-seed SEM using the project-wide ddof=1 convention."""

    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(values.size))


def _share_statistics(
    source_u: np.ndarray,
    destination_v: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sector share, cross ratio, and signed 19-d contribution shares.

    The first two arrays have shape ``(sources, destinations)``.  The final array has shape
    ``(sources, feedback, destinations)`` and sums to one across its feedback axis.
    """

    feedback_dim = source_u.shape[1]
    if feedback_dim != destination_v.shape[0] or covariance.shape != (feedback_dim, feedback_dim):
        raise ValueError("Source/destination factors and feedback covariance are incompatible.")
    if feedback_dim != 19:
        raise ValueError(
            f"Expected 19 feedback features (10 Digit + 9 Sector), got {feedback_dim}."
        )

    coefficients = source_u[:, :, None] * destination_v[None, :, :]
    digit = coefficients[:, :N_DIGIT, :]
    sector = coefficients[:, N_DIGIT:, :]
    c_dd = covariance[:N_DIGIT, :N_DIGIT]
    c_ss = covariance[N_DIGIT:, N_DIGIT:]
    c_ds = covariance[:N_DIGIT, N_DIGIT:]
    digit_variance = np.einsum("ikj,kl,ilj->ij", digit, c_dd, digit, optimize=True)
    sector_variance = np.einsum("ikj,kl,ilj->ij", sector, c_ss, sector, optimize=True)
    cross = 2.0 * np.einsum("ikj,kl,ilj->ij", digit, c_ds, sector, optimize=True)
    total = digit_variance + sector_variance + cross
    if np.any(total <= 0.0):
        raise RuntimeError("A gate-logit total variance was non-positive after reset exclusion.")
    additive = digit_variance + sector_variance
    if np.any(additive <= 0.0):
        raise RuntimeError("A gate-logit Digit+Sector variance was non-positive.")
    sector_share = sector_variance / additive
    cross_ratio = np.abs(cross) / total
    per_dimension = coefficients * np.einsum("kl,ilj->ikj", covariance, coefficients)
    return sector_share, cross_ratio, per_dimension / total[:, None, :]


def _seed_result(seed: int, trajectory_path: Path) -> dict[str, Any]:
    """Compute all observed and null factor statistics for one training seed."""

    with np.load(trajectory_path, allow_pickle=False) as archive:
        feedback = np.asarray(archive["feedback"], dtype=np.float64)
        saved_u = np.asarray(archive["U"], dtype=np.float64)
        saved_v = np.asarray(archive["V"], dtype=np.float64)
    if feedback.ndim != 3:
        raise ValueError(f"Expected (sequence, time, feedback) feedback, got {feedback.shape}.")
    if saved_u.ndim != 2 or saved_v.ndim != 2:
        raise ValueError("Saved GaWF factors must each be matrices.")

    # The saved model uses destination-by-feedback U and feedback-by-source V.
    source_u = saved_v.T
    destination_v = saved_u.T
    n_sources, feedback_dim = source_u.shape
    if destination_v.shape != (feedback_dim, saved_u.shape[0]):
        raise ValueError("Saved GaWF factor shapes do not form a source-side factorization.")
    n_hidden = destination_v.shape[1]
    n_input = n_sources - n_hidden
    if n_input <= 0:
        raise ValueError("Could not infer a positive encoder-source count from GaWF factor shapes.")
    reset_mask = np.all(feedback == 0.0, axis=-1)
    feedback_frames = feedback[~reset_mask]
    if feedback_frames.shape[0] < 2:
        raise RuntimeError("Need at least two reset-excluded feedback frames for a covariance.")
    covariance = np.cov(feedback_frames, rowvar=False, ddof=1)
    observed_share, observed_cross, observed_contribution = _share_statistics(
        source_u, destination_v, covariance
    )

    rng = np.random.default_rng(NULL_RNG_OFFSET + seed)
    initialized_u = rng.normal(0.0, INITIALIZER_STD, size=source_u.shape)
    initialized_share, _, _ = _share_statistics(initialized_u, destination_v, covariance)
    shuffled_u = source_u[rng.permutation(n_sources)]
    shuffled_share, _, _ = _share_statistics(shuffled_u, destination_v, covariance)
    blocks = {"input": slice(0, n_input), "recurrent": slice(n_input, n_sources)}
    input_energy = np.square(source_u[:n_input]).mean(axis=0)
    recurrent_energy = np.square(source_u[n_input:]).mean(axis=0)
    input_energy_share = input_energy / (input_energy + recurrent_energy)
    input_recurrent_log_energy_ratio = np.log(input_energy / recurrent_energy)
    destination_v_weights = np.square(destination_v)
    destination_v_weights /= destination_v_weights.sum(axis=1, keepdims=True)
    digit_target_mean = destination_v_weights[:N_DIGIT].mean(axis=0)
    sector_target_mean = destination_v_weights[N_DIGIT:].mean(axis=0)
    destination_v_selectivity = (sector_target_mean - digit_target_mean) / (
        sector_target_mean + digit_target_mean
    )
    block_statistics = {
        block: {
            "observed_sector_share": float(observed_share[rows].mean()),
            "observed_abs_cross_over_total_variance": float(observed_cross[rows].mean()),
            "initializer_u_sector_share": float(initialized_share[rows].mean()),
            "row_permuted_u_sector_share": float(shuffled_share[rows].mean()),
        }
        for block, rows in blocks.items()
    }
    return {
        "seed": seed,
        "trajectory_path": str(trajectory_path.resolve()),
        "n_x": n_input,
        "n_h": n_hidden,
        "n_f": feedback_dim,
        "raw_n_frames": int(feedback.shape[0] * feedback.shape[1]),
        "reset_frames_excluded": int(reset_mask.sum()),
        "analysis_n_frames": int(feedback_frames.shape[0]),
        "covariance": covariance,
        "source_u": source_u,
        "source_u_input_energy_share": input_energy_share,
        "source_u_input_recurrent_log_energy_ratio": input_recurrent_log_energy_ratio,
        "destination_v_weights": destination_v_weights,
        "destination_v_selectivity": destination_v_selectivity,
        "block_statistics": block_statistics,
        "feedback_dimension_contribution_shares": {
            block: observed_contribution[rows].mean(axis=(0, 2)) for block, rows in blocks.items()
        },
        "destination_source_block_shares": {
            block: observed_share[rows].mean(axis=0) for block, rows in blocks.items()
        },
    }


def _long_rows(seed_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one seed result to the requested long CSV schema."""

    rows: list[dict[str, Any]] = []
    for block, statistics in seed_result["block_statistics"].items():
        for statistic, value in statistics.items():
            rows.append(
                {
                    "seed": seed_result["seed"],
                    "analysis": "source_factor",
                    "block": block,
                    "statistic": statistic,
                    "unit": "",
                    "value": value,
                }
            )
    for statistic, values in (
        ("input_energy_share", seed_result["source_u_input_energy_share"]),
        (
            "log_input_over_recurrent_energy",
            seed_result["source_u_input_recurrent_log_energy_ratio"],
        ),
    ):
        for unit, value in enumerate(values):
            prefix = "digit" if unit < N_DIGIT else "sector"
            index = unit if unit < N_DIGIT else unit - N_DIGIT
            rows.append(
                {
                    "seed": seed_result["seed"],
                    "analysis": "source_u_block_energy",
                    "block": "input_vs_recurrent",
                    "statistic": statistic,
                    "unit": f"{prefix}_{index}",
                    "value": float(value),
                }
            )
    for unit, value in enumerate(seed_result["destination_v_selectivity"]):
        rows.append(
            {
                "seed": seed_result["seed"],
                "analysis": "destination_v_static",
                "block": "target_recurrent_unit",
                "statistic": "sector_minus_digit_target_selectivity",
                "unit": unit,
                "value": float(value),
            }
        )
    for block, values in seed_result["feedback_dimension_contribution_shares"].items():
        for unit, value in enumerate(values):
            prefix = "digit" if unit < N_DIGIT else "sector"
            index = unit if unit < N_DIGIT else unit - N_DIGIT
            rows.append(
                {
                    "seed": seed_result["seed"],
                    "analysis": "feedback_dimension",
                    "block": block,
                    "statistic": "signed_variance_contribution_share",
                    "unit": f"{prefix}_{index}",
                    "value": float(value),
                }
            )
    for block, values in seed_result["destination_source_block_shares"].items():
        for unit, value in enumerate(values):
            rows.append(
                {
                    "seed": seed_result["seed"],
                    "analysis": "destination_connection",
                    "block": block,
                    "statistic": "sector_share",
                    "unit": unit,
                    "value": float(value),
                }
            )
    return rows


def _plot_feedback_dimension_contributions(
    seed_results: list[dict[str, Any]], destination: Path
) -> None:
    """Draw signed contributions for each original Digit and Sector feedback dimension."""

    figure, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    labels = [f"D{index}" for index in range(N_DIGIT)]
    labels.extend(f"S{index}" for index in range(19 - N_DIGIT))
    positions = np.arange(len(labels))
    colors = ["#2b6cb0"] * N_DIGIT + ["#c05621"] * (19 - N_DIGIT)
    for axis, block, title in zip(
        axes, ("input", "recurrent"), ("Input-source block", "Recurrent-source block")
    ):
        values = np.asarray(
            [result["feedback_dimension_contribution_shares"][block] for result in seed_results],
            dtype=np.float64,
        )
        axis.bar(
            positions,
            100.0 * values.mean(axis=0),
            yerr=100.0 * values.std(axis=0, ddof=1) / math.sqrt(values.shape[0]),
            color=colors,
            width=0.78,
            capsize=2.5,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(title=title, ylabel="Variance contribution (%)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[-1].set(
        xticks=positions,
        xticklabels=labels,
        xlabel="Feedback dimension (D: digit; S: sector)",
    )
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_destination_source_block_pairs(
    seed_results: list[dict[str, Any]], destination: Path
) -> None:
    """Draw paired source-block shares for each destination hidden unit and seed."""

    figure, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharex=True, sharey=True)
    for axis, result in zip(axes.flat, seed_results):
        shares = result["destination_source_block_shares"]
        axis.scatter(shares["input"], shares["recurrent"], s=10, alpha=0.52, color="#5a67d8")
        axis.plot((0.0, 1.0), (0.0, 1.0), color="black", linewidth=0.8, linestyle="--")
        axis.set(title=f"Seed {result['seed']:02d}", xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        axis.set_aspect("equal", adjustable="box")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Recurrent-source sector share")
    axes[1, 0].set_ylabel("Recurrent-source sector share")
    for axis in axes[1, :]:
        axis.set_xlabel("Input-source sector share")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_destination_v_target_allocations(
    seed_results: list[dict[str, Any]], destination: Path
) -> None:
    """Draw frozen V target allocations after normalizing each feedback row across targets."""

    all_values = np.concatenate(
        [result["destination_v_weights"].ravel() for result in seed_results]
    )
    color_max = float(np.quantile(all_values, 0.995))
    labels = [f"D{index}" for index in range(N_DIGIT)]
    labels.extend(f"S{index}" for index in range(19 - N_DIGIT))
    figure, axes = plt.subplots(2, 5, figsize=(15.5, 7.4), sharex=True, sharey=True)
    image = None
    for axis, result in zip(axes.flat, seed_results):
        order = np.argsort(result["destination_v_selectivity"])
        image = axis.imshow(
            result["destination_v_weights"][:, order],
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=color_max,
        )
        axis.axhline(N_DIGIT - 0.5, color="white", linewidth=0.9)
        axis.set_title(f"Seed {result['seed']:02d}")
        axis.set_xticks((0, 127.5, 255), ("D-pref", "mixed", "S-pref"), fontsize=7)
    for axis in axes[:, 0]:
        axis.set_yticks(np.arange(19), labels, fontsize=7)
    figure.suptitle("Frozen destination-side V: target allocation by feedback component", y=0.995)
    figure.subplots_adjust(left=0.065, right=0.88, bottom=0.08, top=0.88, wspace=0.10, hspace=0.15)
    color_axis = figure.add_axes((0.91, 0.27, 0.015, 0.45))
    figure.colorbar(image, cax=color_axis, label="V² / sum target V²")
    figure.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_destination_v_selectivity_ranks(
    seed_results: list[dict[str, Any]], destination: Path
) -> None:
    """Draw the static Digit-versus-Sector target selectivity rank curve for each seed."""

    figure, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharex=True, sharey=True)
    ranks = np.linspace(0.0, 1.0, 256)
    for axis, result in zip(axes.flat, seed_results):
        axis.plot(ranks, np.sort(result["destination_v_selectivity"]), color="#5a67d8")
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set(title=f"Seed {result['seed']:02d}", xlim=(0.0, 1.0), ylim=(-1.0, 1.0))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Sector − Digit\ntarget selectivity")
    axes[1, 0].set_ylabel("Sector − Digit\ntarget selectivity")
    for axis in axes[1, :]:
        axis.set_xlabel("Target rank: Digit → Sector")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_source_u_matrix_blocks(seed_results: list[dict[str, Any]], destination: Path) -> None:
    """Draw all ten raw source-side absolute-U matrices with semantic and row-block boundaries."""

    all_values = np.concatenate([np.abs(result["source_u"]).ravel() for result in seed_results])
    color_max = float(np.quantile(all_values, 0.995))
    figure, axes = plt.subplots(2, 5, figsize=(15.5, 8.2), sharex=True, sharey=True)
    image = None
    for axis, result in zip(axes.flat, seed_results):
        image = axis.imshow(
            np.abs(result["source_u"]),
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=color_max,
        )
        axis.axhline(result["n_x"] - 0.5, color="white", linewidth=0.9)
        axis.axvline(N_DIGIT - 0.5, color="white", linewidth=0.9)
        axis.set_title(f"Seed {result['seed']:02d}")
        axis.set_xticks((4.5, 14.0), ("Digit 0–9", "Sector 0–8"), fontsize=8)
    for axis in axes[:, 0]:
        axis.set_yticks((575.5, 1279.5), ("Input\n1152", "Recurrent\n256"), fontsize=8)
    figure.suptitle("Raw source-side |U|: row and feedback-component blocks", y=0.995)
    figure.subplots_adjust(left=0.065, right=0.88, bottom=0.07, top=0.89, wspace=0.10, hspace=0.13)
    color_axis = figure.add_axes((0.91, 0.27, 0.015, 0.45))
    figure.colorbar(image, cax=color_axis, label="|U_source|")
    figure.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_source_u_matrix_blocks_colwise_normalized(
    seed_results: list[dict[str, Any]], destination: Path
) -> None:
    """Draw the clipped, column-wise normalized source-|U| mean across ten seeds."""

    magnitudes = np.stack([np.abs(result["source_u"]) for result in seed_results])
    clip_threshold = float(np.quantile(magnitudes, 0.995))
    clipped = np.minimum(magnitudes, clip_threshold)
    column_minimum = clipped.min(axis=1, keepdims=True)
    column_span = clipped.max(axis=1, keepdims=True) - column_minimum
    if np.any(column_span <= 0.0):
        raise RuntimeError("A source-|U| feedback column was constant after clipping.")
    mean_normalized = ((clipped - column_minimum) / column_span).mean(axis=0)
    n_input = seed_results[0]["n_x"]
    n_hidden = seed_results[0]["n_h"]
    figure, axis = plt.subplots(figsize=(7.4, 8.4))
    image = axis.imshow(
        mean_normalized,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("white_to_red", ("white", "#d73027")),
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    axis.axhline(n_input - 0.5, color="black", linewidth=0.9)
    axis.axvline(N_DIGIT - 0.5, color="black", linewidth=0.9)
    axis.set_xticks((4.5, 14.0), ("Digit 0–9", "Sector 0–8"))
    axis.set_yticks(
        (n_input / 2 - 0.5, n_input + n_hidden / 2 - 0.5),
        (f"Input\n{n_input}", f"Recurrent\n{n_hidden}"),
    )
    axis.set_title("10-seed mean clipped, column-wise normalized |U_source|")
    figure.subplots_adjust(left=0.18, right=0.82, bottom=0.10, top=0.91)
    color_axis = figure.add_axes((0.86, 0.26, 0.025, 0.42))
    figure.colorbar(image, cax=color_axis, ticks=(0.0, 0.5, 1.0), label="Normalized |U_source|")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_source_u_block_energy(seed_results: list[dict[str, Any]], destination: Path) -> None:
    """Draw source-block energy share and log-ratio for all 19 semantic feedback columns."""

    figure, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    labels = [f"D{index}" for index in range(N_DIGIT)]
    labels.extend(f"S{index}" for index in range(19 - N_DIGIT))
    positions = np.arange(len(labels))
    colors = ["#2b6cb0"] * N_DIGIT + ["#c05621"] * (19 - N_DIGIT)
    specifications = (
        ("source_u_input_energy_share", "Input energy share", 0.5),
        ("source_u_input_recurrent_log_energy_ratio", "log(Input / recurrent energy)", 0.0),
    )
    for axis, (key, ylabel, reference) in zip(axes, specifications):
        values = np.asarray([result[key] for result in seed_results], dtype=np.float64)
        axis.bar(
            positions,
            values.mean(axis=0),
            yerr=values.std(axis=0, ddof=1) / math.sqrt(values.shape[0]),
            color=colors,
            width=0.78,
            capsize=2.5,
        )
        axis.axhline(reference, color="black", linewidth=0.8, linestyle="--")
        axis.set(ylabel=ylabel)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_title("Source-side U block allocation (10-seed mean ± SEM)")
    axes[-1].set(
        xticks=positions,
        xticklabels=labels,
        xlabel="Feedback dimension (D: digit; S: sector)",
    )
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _summary(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a record-style ten-training-seed summary without pooled connection inference."""

    statistics = tuple(seed_results[0]["block_statistics"]["input"])
    seed_level = {
        block: {
            statistic: [
                float(result["block_statistics"][block][statistic]) for result in seed_results
            ]
            for statistic in statistics
        }
        for block in ("input", "recurrent")
    }
    cross_seed = {
        block: {
            statistic: {
                "mean": float(np.mean(values)),
                "sem": _sem(np.asarray(values, dtype=np.float64)),
                "n": len(values),
            }
            for statistic, values in block_statistics.items()
        }
        for block, block_statistics in seed_level.items()
    }
    destination_seed_descriptives = []
    for result in seed_results:
        input_values = np.asarray(
            result["destination_source_block_shares"]["input"], dtype=np.float64
        )
        recurrent_values = np.asarray(
            result["destination_source_block_shares"]["recurrent"], dtype=np.float64
        )
        destination_seed_descriptives.append(
            {
                "seed": result["seed"],
                "input_source_mean": float(input_values.mean()),
                "recurrent_source_mean": float(recurrent_values.mean()),
                "mean_gap_input_minus_recurrent": float(
                    input_values.mean() - recurrent_values.mean()
                ),
            }
        )
    dimension_seed_level = {
        block: [
            np.asarray(result["feedback_dimension_contribution_shares"][block]).tolist()
            for result in seed_results
        ]
        for block in ("input", "recurrent")
    }
    return {
        "analysis": "GaWF source-side feedback-factor variance decomposition",
        "training_seeds": [result["seed"] for result in seed_results],
        "inference_unit": "training seed; no pooled connection-level hypothesis test",
        "spread_convention": "cross-seed mean +/- SEM (ddof=1)",
        "factor_orientation": {
            "saved": "U_saved[destination, feedback] * V_saved[feedback, source]",
            "analyzed": "U_source[source, feedback] = V_saved.T; "
            "V_destination[feedback, destination] = U_saved.T",
            "gate_logit": "M[source, destination, t] = sum_k "
            "U_source[source,k] * V_destination[k,destination] * feedback[t,k]",
        },
        "dimensions_by_seed": [
            {
                "seed": result["seed"],
                "N_x": result["n_x"],
                "N_h": result["n_h"],
                "N_f": result["n_f"],
            }
            for result in seed_results
        ],
        "test_feedback_trajectories": [
            {
                "seed": result["seed"],
                "test_split_path": result["trajectory_path"],
                "raw_n_frames": result["raw_n_frames"],
                "reset_frames_excluded": result["reset_frames_excluded"],
                "analysis_n_frames": result["analysis_n_frames"],
            }
            for result in seed_results
        ],
        "feedback_covariance_by_seed": [
            {
                "seed": result["seed"],
                "digit_indices": list(range(N_DIGIT)),
                "sector_indices": list(range(N_DIGIT, result["n_f"])),
                "matrix": np.asarray(result["covariance"], dtype=np.float64).tolist(),
            }
            for result in seed_results
        ],
        "reset_exclusion": "exclude frames whose retained pre-step feedback is all zero",
        "variance_definition": {
            "sector_share": (
                "Var_sector / (Var_digit + Var_sector); cross excluded from denominator"
            ),
            "cross_ratio": "abs(2 * c_D^T C_DS c_S) / Var(M)",
            "dimension_contribution": "c_k * (C c)_k / Var(M); sums to one over 19 dimensions",
            "covariance": "per-seed unbiased Cov(feedback) on reset-excluded test frames",
        },
        "null_models": {
            "initializer_u": {
                "definition": "U_source ~ Normal(0, 0.01), matching the GaWF factor initializer; "
                "true V_destination and covariance retained",
                "rng_seed": "20260823 + training_seed",
            },
            "row_permuted_u": {
                "definition": "one global permutation of true U_source rows across input and "
                "recurrent blocks; preserves its marginal row distribution",
                "rng_seed": "same generator after initializer draw",
            },
        },
        "source_factor_seed_level": seed_level,
        "source_factor_cross_seed_summary": cross_seed,
        "source_u_block_energy_seed_level": {
            "input_energy_share": [
                np.asarray(result["source_u_input_energy_share"]).tolist()
                for result in seed_results
            ],
            "log_input_over_recurrent_energy": [
                np.asarray(result["source_u_input_recurrent_log_energy_ratio"]).tolist()
                for result in seed_results
            ],
        },
        "feedback_dimension_contribution_seed_level": dimension_seed_level,
        "destination_source_block_pairs": {
            "definition": "each destination unit has one input-source and one recurrent-source "
            "sector share; neither is a V-only measure",
            "n_destination_units": int(seed_results[0]["n_h"]),
            "per_seed_descriptives": destination_seed_descriptives,
        },
        "destination_v_static": {
            "definition": "V[k,j]^2 normalized within each feedback row across 256 targets; "
            "no feedback trajectory or covariance is used",
            "target_selectivity_by_seed": [
                np.asarray(result["destination_v_selectivity"]).tolist() for result in seed_results
            ],
        },
    }


def _self_check() -> None:
    """Verify the decomposition against direct variance of a small synthetic gate logit."""

    rng = np.random.default_rng(7)
    source_u = rng.normal(size=(3, 19))
    destination_v = rng.normal(size=(19, 2))
    feedback = rng.normal(size=(5000, 19))
    covariance = np.cov(feedback, rowvar=False, ddof=1)
    share, cross_ratio, dimension_share = _share_statistics(source_u, destination_v, covariance)
    coefficients = source_u[:, :, None] * destination_v[None, :, :]
    direct = np.var(np.einsum("tk,ikj->tij", feedback, coefficients), axis=0, ddof=1)
    digit = coefficients[:, :N_DIGIT, :]
    sector = coefficients[:, N_DIGIT:, :]
    digit_var = np.einsum("ikj,kl,ilj->ij", digit, covariance[:10, :10], digit)
    sector_var = np.einsum("ikj,kl,ilj->ij", sector, covariance[10:, 10:], sector)
    assert np.all(np.isfinite(share)) and np.all(np.isfinite(cross_ratio))
    np.testing.assert_allclose(dimension_share.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)
    # The direct equality includes the cross term; calculate it separately to keep the check exact.
    cross = 2.0 * np.einsum("ikj,kl,ilj->ij", digit, covariance[:10, 10:], sector)
    np.testing.assert_allclose(direct, digit_var + sector_var + cross, rtol=1e-11, atol=1e-11)


def main() -> None:
    """Run the ten-seed source-factor analysis and write its reproducible artifacts."""

    args = parse_args()
    if args.self_check:
        _self_check()
        print("Synthetic variance-decomposition self-check passed.")
        return
    if sorted(args.seeds) != list(range(1, N_SEEDS + 1)):
        raise ValueError("The formal analysis requires exactly training seeds 1 through 10.")
    if args.output_dir is None:
        args.output_dir = output_dir(
            "D_variance_decomposition", "gawf_source_factor_feedback", "data"
        )
    if args.figure_dir is None:
        args.figure_dir = output_dir(
            "D_variance_decomposition", "gawf_source_factor_feedback", "figs"
        )
    trajectory_paths = [
        args.trajectory_root / f"seed{seed:02d}" / "gawf_gate_trajectory.npz" for seed in args.seeds
    ]
    missing = [str(path) for path in trajectory_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing retained test trajectories: " + ", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    seed_results = [_seed_result(seed, path) for seed, path in zip(args.seeds, trajectory_paths)]
    dimensions = {(item["n_x"], item["n_h"], item["n_f"]) for item in seed_results}
    if len(dimensions) != 1:
        raise RuntimeError(f"Seed factor dimensions differ: {sorted(dimensions)}")
    if args.plot_colwise_normalized_only:
        destination = args.figure_dir / "gawf_source_u_matrix_blocks_colwise_normalized_10seed"
        _plot_source_u_matrix_blocks_colwise_normalized(seed_results, destination)
        print(f"Saved {destination.with_suffix('.pdf')}")
        return
    long_path = args.output_dir / "gawf_source_factor_feedback_long_10seed.csv"
    rows = [row for result in seed_results for row in _long_rows(result)]
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "analysis", "block", "statistic", "unit", "value"),
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "gawf_source_factor_feedback_summary_10seed.json"
    summary_path.write_text(json.dumps(_summary(seed_results), indent=2) + "\n", encoding="utf-8")
    _plot_feedback_dimension_contributions(
        seed_results, args.figure_dir / "gawf_feedback_dimension_contributions_10seed"
    )
    _plot_destination_source_block_pairs(
        seed_results,
        args.figure_dir / "gawf_destination_source_block_sector_share_pairs_10seed",
    )
    _plot_destination_v_target_allocations(
        seed_results, args.figure_dir / "gawf_destination_v_target_allocations_10seed"
    )
    _plot_destination_v_selectivity_ranks(
        seed_results, args.figure_dir / "gawf_destination_v_selectivity_ranks_10seed"
    )
    _plot_source_u_matrix_blocks(
        seed_results, args.figure_dir / "gawf_source_u_matrix_blocks_10seed"
    )
    _plot_source_u_matrix_blocks_colwise_normalized(
        seed_results, args.figure_dir / "gawf_source_u_matrix_blocks_colwise_normalized_10seed"
    )
    _plot_source_u_block_energy(
        seed_results, args.figure_dir / "gawf_source_u_block_energy_19dim_10seed"
    )
    for stem in (
        "gawf_source_factor_row_share_histograms_10seed",
        "gawf_destination_v_sector_share_distribution_10seed",
    ):
        for suffix in (".pdf", ".png"):
            (args.figure_dir / stem).with_suffix(suffix).unlink(missing_ok=True)
    print(f"Saved {long_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
