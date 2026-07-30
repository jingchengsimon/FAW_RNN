"""Recurrent gate sign-vs-magnitude ("disinhibition") test, pooled across sectors.

Sector companion to gawf_recurrent_gate_sign_vs_magnitude_disinhibition.py (the digit version,
left untouched): same TT/TR/RT/RR groups, same per-connection open-fraction pooling, same
overlap-band diagnostics, same OLS regression (of ~ signpos + absW + C(context)), same 2x2
scatter+binned-curve figure (full-range and zoomed variants) -- only the pooling axis changes
from digit to sector. Reuses every reusable piece from the digit module via ``analyze()``; this
file only supplies its own SECTORS list, CATEGORY/SCRIPT_NAME, and output filenames.

Convention: gate[..., i, j] = src j -> dst i (row=dst, col=src), same for W[i, j]. T is each
sector's cache T_old (FDR-selective top-10% among eligible units, using the recurrent-sector
selectivity from part1_selectivity.npz), matching gawf_recurrent_gate_sector_collect.py.

Reuses the real per-sector gate/act/W caches (sector{s}_gate_act_cache.npz, s=0..8) written by
gawf_recurrent_gate_sector_collect.py. Falls back to synthetic multi-sector data end-to-end
when no real cache is found on disk, exactly like the digit module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir
from utils_anal.clutter.fig7_recurrent_gate_sign_magnitude import (
    DELTA_TITLE,
    DELTA_Y_LABEL,
    analyze,
    render_figure,
    render_figure_zoomed,
)

CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem
SECTORS = tuple(range(9))  # matches gawf_recurrent_gate_sector_collect.py's full 0-8 range
KIND = "sector"


def main() -> None:
    pooled, _meta, overlap_stats, _regressions_full, _regressions_clean = analyze(SECTORS, KIND)

    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    render_figure(
        pooled, overlap_stats,
        fig_dir / "recurrent_gate_sign_vs_magnitude_disinhibition_sector.png",
    )
    render_figure_zoomed(
        pooled, overlap_stats,
        fig_dir / "recurrent_gate_sign_vs_magnitude_disinhibition_sector_zoom.png",
    )
    render_figure_zoomed(
        pooled, overlap_stats,
        fig_dir / "recurrent_gate_sign_vs_magnitude_disinhibition_sector_delta_zoom.png",
        y_col="delta_of", y_label=DELTA_Y_LABEL.format(kind=KIND), title=DELTA_TITLE.format(kind=KIND),
    )


if __name__ == "__main__":
    main()
