"""Plot all twelve completed Breakout GaWF LR-decay runs without aggregation.

Inputs are final ``metrics.json`` and ``metrics_history.jsonl`` files for the L3/L4 replay
conditions. The script validates the shared 3M-step protocol and writes one PNG: color encodes
the ``(layer, replay)`` condition and line style encodes the seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from utils.analysis.rl.atari.atari_breakout_gawf_lr_buffer_completed11_mean_std import (
    EXPECTED_STEPS,
    load_curve,
    smooth_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CONDITIONS = (
    (3, "1m", "L3 · 1M replay + LR decay"),
    (3, "2m", "L3 · 2M replay + LR decay"),
    (4, "1m", "L4 · 1M replay + LR decay"),
    (4, "2m", "L4 · 2M replay + LR decay"),
)
SEED_STYLES = {1: "-", 2: "--", 3: ":"}


def parse_args() -> argparse.Namespace:
    """Parse structured-result roots and output settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--diagnostic-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=190.0)
    return parser.parse_args()


def run_path(data_root: Path, diagnostic_data_root: Path, layer: int, buffer_tag: str,
             seed: int) -> Path:
    """Return the exact result directory for a completed LR-decay condition and seed."""
    if (layer, buffer_tag, seed) == (3, "1m", 2):
        return diagnostic_data_root / "atari_dqn_breakout_fs4_stack4_l3diag_lr_decay_gawf_seed2"
    suffix = f"atari_dqn_breakout_fs4_stack4_l{layer}_lrdecay1m_buf{buffer_tag}_gawf_seed{seed}"
    return data_root / suffix


def main() -> None:
    """Render all twelve learning curves with condition color and seed line style."""
    args = parse_args()
    if args.y_min >= args.y_max:
        raise ValueError("--y-min must be below --y-max")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    for color_index, (layer, buffer_tag, condition_label) in enumerate(CONDITIONS):
        for seed in (1, 2, 3):
            steps, returns = load_curve(
                run_path(args.data_root, args.diagnostic_data_root, layer, buffer_tag, seed), layer
            )
            diagnostic_note = " · diagnostic" if (layer, buffer_tag, seed) == (3, "1m", 2) else ""
            ax.plot(
                steps / 1_000_000.0,
                smooth_curve(returns, args.smooth),
                color=f"C{color_index}",
                linestyle=SEED_STYLES[seed],
                linewidth=1.7,
                label=f"{condition_label} · seed {seed}{diagnostic_note}",
            )

    ax.set_title("Strict 4-action Breakout · fs4/stack4 · GaWF · LR-decay conditions")
    ax.set_xlabel("environment steps (×10⁶)")
    ax.set_ylabel("episodic return (last 100 episodes)")
    ax.set_xlim(0.0, EXPECTED_STEPS / 1_000_000.0)
    ax.set_ylim(args.y_min, args.y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(
        ncol=2,
        title="color: layer + replay · line style: seed (1 solid, 2 dashed, 3 dotted)",
        fontsize=8.0,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
