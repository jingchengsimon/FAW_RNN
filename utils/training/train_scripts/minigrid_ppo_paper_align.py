"""Compatibility entry point for the isolated paper-alignment trainer.

The implementation lives in :mod:`utils.training.train_scripts.minigrid_ppo_paper`; this alias is
kept so older launch notes do not accidentally invoke the accelerated pilot.
"""

from utils.training.train_scripts.minigrid_ppo_paper import main


if __name__ == "__main__":
    main()
