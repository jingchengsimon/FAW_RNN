# Experiment layout

`experiments/` only contains task-specific protocols and Amarel launchers. Training entry
points and reusable implementations remain at the repository root and in `utils/`.

| Directory | Role |
|---|---|
| `clutter/` | Final Clutter protocol and curation notes; no active launcher is retained. |
| `text/` | IMDB/SentiHood task definitions and `amarel/` launchers. |
| `rl/atari/` | Atari definitions and `amarel/` launchers. |
| `rl/minigrid/` | MiniGrid definitions and `amarel/` launchers. |

Cross-host synchronization and run manifests live under `remote/`, not this directory.
