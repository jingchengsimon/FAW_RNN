"""Run one AIM3 training task through the unified public command-line interface.

The first positional argument selects a task-specific implementation in
``utils.training``. All remaining arguments are passed through unchanged, so
each task retains its documented CLI and result contracts.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from collections.abc import Sequence


TASK_MODULES = {
    "clutter": "utils.training.train_scripts.clutter",
    "atari-a2c": "utils.training.train_scripts.atari_a2c",
    "atari-dqn": "utils.training.train_scripts.atari_dqn",
    "minigrid-dqn": "utils.training.train_scripts.minigrid_dqn",
    "minigrid-ppo": "utils.training.train_scripts.minigrid_ppo",
    "minigrid-ppo-paper": "utils.training.train_scripts.minigrid_ppo_paper",
    "minigrid-ppo-paper-align": "utils.training.train_scripts.minigrid_ppo_paper_align",
    "imdb": "utils.training.train_scripts.imdb",
    "sentihood": "utils.training.train_scripts.sentihood",
}


def _usage() -> str:
    tasks = "\n".join(f"  {task}" for task in TASK_MODULES)
    return (
        "Usage: python run_task.py <task> [task options]\n\n"
        "Tasks:\n"
        f"{tasks}\n\n"
        "Use `python run_task.py <task> --help` for task-specific options."
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one task while preserving its original argument parser."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_usage())
        return

    task, *task_args = arguments
    module_name = TASK_MODULES.get(task)
    if module_name is None:
        raise SystemExit(f"Unknown task {task!r}.\n\n{_usage()}")

    sys.argv = [f"run_task.py {task}", *task_args]
    # The established Clutter trainer executes its parser at module scope under
    # ``__main__``. Keep that mature entry point intact while the other task
    # modules expose an explicit ``main()``.
    if task == "clutter":
        runpy.run_module(module_name, run_name="__main__")
        return

    module = importlib.import_module(module_name)
    entry_point = getattr(module, "main", None)
    if not callable(entry_point):
        raise RuntimeError(f"Training module {module_name} does not expose main().")

    entry_point()


if __name__ == "__main__":
    main()
