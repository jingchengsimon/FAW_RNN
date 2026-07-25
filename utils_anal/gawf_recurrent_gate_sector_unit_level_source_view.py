"""Run the source/efferent unit-level grid for recurrent-sector contexts 1-4.

Same rendering logic as gawf_recurrent_gate_unit_level_all_digits.py, but reads the sector
caches produced by gawf_recurrent_gate_sector_collect.py (kind="sector") instead of the digit
caches, and only covers sectors 1-4 (not all 9) per request.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir
from utils_anal.gawf_recurrent_gate_raw_group_sign_grid import load_data, prepare
from utils_anal.gawf_recurrent_gate_unit_level_group_sign_grid import (
    CATEGORY,
    SCRIPT_NAME,
    render_unit_level_grid,
)

SECTORS = (1, 2, 3, 4)


def main() -> None:
    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    for sector in SECTORS:
        print(f"\n{'=' * 20} sector {sector} {'=' * 20}")
        data = load_data(sector, kind="sector")
        prepared = prepare(data["W"], data["gate_raw"], data["act"], data["T"],
                            digit=sector, kind="sector")
        render_unit_level_grid(
            prepared, fig_dir / f"sector{sector}_unit_level_gate_W_gW_group_sign_grid.png"
        )


if __name__ == "__main__":
    main()
