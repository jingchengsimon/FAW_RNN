"""Run the source/efferent unit-level grid (gawf_recurrent_gate_unit_level_group_sign_grid.py)
for every digit 0-9. Digit 0 uses the exhaustive n=6437 cache; digits 1-9 use the capped
(n=1200) cache from gawf_recurrent_gate_multi_digit_collect.py. Each digit gets its own PNG,
named identically to the existing digit-0 figure but with that digit's index -- nothing is
overwritten.
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

DIGITS = tuple(range(10))


def main() -> None:
    fig_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    for digit in DIGITS:
        print(f"\n{'=' * 20} digit {digit} {'=' * 20}")
        data = load_data(digit)
        prepared = prepare(data["W"], data["gate_raw"], data["act"], data["T"], digit=digit)
        render_unit_level_grid(
            prepared, fig_dir / f"digit{digit}_unit_level_gate_W_gW_group_sign_grid.png"
        )


if __name__ == "__main__":
    main()
