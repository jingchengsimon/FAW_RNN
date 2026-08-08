"""Plot sequential-feedback, equal-n sector input-gate spatial means.

Input is ``sector_gate_mean_sequential_equal_n.npz`` from the matching analysis module. Outputs
are point-included and point-excluded sector-only 3x3 grids in both PNG and PDF formats.
"""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__))))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils.analysis.anal_paths import output_dir

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


SAVE_DATA_FILE = os.path.join(
    _ANAL_PROJECT_ROOT,
    "results",
    "save_data",
    "fig6",
    "sector_gate_mean_sequential_equal_n.npz",
)


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=SAVE_DATA_FILE,
    )
    parser.add_argument(
        "--fig_dir",
        default=str(output_dir("B_gate_by_context", "sector_sigmoid_gate_sequential", "figs")),
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--stem",
        default="fig2_sector_gate_mean_sequential_equal_n",
        help="Output filename stem; point-included/excluded suffix is appended.",
    )
    parser.add_argument(
        "--seed_data",
        type=str,
        nargs="+",
        default=None,
        help="Ten per-seed Fig6 NPZ files; their point maps are averaged before plotting.",
    )
    return parser.parse_args()


def plot_sector_grid(
    maps: np.ndarray, point_key: str, fig_dir: str, dpi: int, stem_prefix: str
) -> tuple[str, str]:
    """Write one sector-only 3x3 raw gate-mean grid as PNG and PDF."""

    values = np.asarray(maps, dtype=np.float32)
    if values.shape != (9, 6, 6):
        raise ValueError(f"Expected maps with shape (9, 6, 6), got {values.shape}")
    if point_key not in ("point_included", "point_excluded"):
        raise ValueError("point_key must be point_included or point_excluded")
    suffix = "included" if point_key == "point_included" else "excluded"
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.8), constrained_layout=True)
    image = None
    for sector, axis in enumerate(axes.flat):
        # Match Supple2's vector-cell rendering so PDF and PNG share the same discrete 6-by-6
        # geometry without interpolation.  Fig6 retains its own absolute gate-mean values.
        image = axis.pcolormesh(
            values[sector],
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
        axis.set_title(f"Sector {sector}", fontsize=16)
        axis.set_xticks([])
        axis.set_yticks([])
    assert image is not None
    fig.suptitle("Sequential input-gate mean (equal-n sectors)\n" f"0.5 point mass {suffix}")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
    stem = f"{stem_prefix}_{point_key}"
    png_path = os.path.join(fig_dir, f"{stem}.png")
    pdf_path = os.path.join(fig_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    return png_path, pdf_path


def main() -> None:
    """Load both gate-mean definitions and render their sector grids."""

    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    if args.seed_data is None:
        with np.load(args.data, allow_pickle=False) as loaded:
            maps = {key: np.asarray(loaded[key], dtype=np.float32) for key in loaded.files}
    else:
        if len(args.seed_data) < 2:
            raise ValueError("--seed_data requires at least two independent seed files")
        per_seed = []
        for path in args.seed_data:
            with np.load(path, allow_pickle=False) as loaded:
                per_seed.append({key: np.asarray(loaded[key], dtype=np.float32) for key in loaded.files})
        required = ("point_included", "point_excluded")
        if any(any(key not in item for key in required) for item in per_seed):
            raise ValueError("Every --seed_data file must contain both point maps")
        maps = {
            key: np.mean(np.stack([item[key] for item in per_seed], axis=0), axis=0, dtype=np.float64)
            .astype(np.float32)
            for key in required
        }
    for point_key in ("point_included", "point_excluded"):
        plot_sector_grid(maps[point_key], point_key, args.fig_dir, args.dpi, args.stem)


if __name__ == "__main__":
    main()
