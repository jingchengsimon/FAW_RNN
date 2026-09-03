# Remote Workflow Wrappers

These wrappers synchronize committed code, update the configured remote checkout, run commands,
and fetch result files. General SSH and environment rules are in
`docs/operations/REMOTE_EXECUTION.md`.

## Local setup

```bash
cp experiments/remote/config.example.sh experiments/remote/config.sh
```

Fill in the ignored `experiments/remote/config.sh` with the SSH target, remote project root, branch, result
paths, and optional activation command. Do not put real endpoints in the tracked example.

The wrapper never runs `git add` or `git commit`. It stops if local edits make synchronization
unsafe and prints the required manual action.

## Synchronize code

```bash
./experiments/remote/sync_code.sh --push
```

## Run commands

Run in the foreground and fetch files created after the run began:

```bash
./experiments/remote/run.sh --push -- python run_task.py clutter --help
```

Run a long command in remote tmux:

```bash
./experiments/remote/run.sh --push --detach my_session -- \
  bash experiments/<task>/amarel/run_hparam_full_grid_2gpu.sh --scale 4
```

## SJC Atari L3 GRU comparison

`run_sjc_atari_multitask_l3_gru.sh` is run through `run.sh`, which supplies the remote
environment activation. It keeps the two-task Pong/Breakout run and the single-task comparisons
under `results/data/rl/atari/multitask_18action/`, with a distinct result leaf for each phase.
Run the isolated 25k smoke first, then use the same mode without `--smoke` for the formal run:

```bash
./experiments/remote/run.sh --push --detach sjc_breakout_l3_smoke -- \
  bash experiments/remote/run_sjc_atari_multitask_l3_gru.sh \
  --mode breakout --cuda-device 1 --smoke

./experiments/remote/run.sh --push --detach sjc_breakout_l3_formal -- \
  bash experiments/remote/run_sjc_atari_multitask_l3_gru.sh \
  --mode breakout --cuda-device 1
```

The launcher uses GRU L3 h458, seed 42, `fs4/stack4`, full18 actions, independent 1M mmap
replay per task, 50k-step checkpoints, and per-task 1M LR decay. It resumes only from
`checkpoint.pth`; an existing history without that checkpoint is rejected to prevent a spliced
trajectory. Use `--dry-run` to inspect the resolved command and result leaf without training.

## SJC Skiing stall/actionfix adaptation

`run_sjc_atari_skiing_warmstart_l3.sh` runs a seed-1, weights-only single-Skiing adaptation from
a five-task model `state_dict`. It fixes the model sizes to LSTM h373, GRU h458, or GaWF h604,
exposes full18, applies `skiing-stall-actionfix-v1`, and always writes a unique leaf
below `results/data/rl/atari/5task_18action/formal_20m_4mpertask_raw_seeds/`. The default formal
budget is 1M environment steps; `--smoke` is fixed at 25k and renders a three-episode video plus
metadata in its smoke leaf. By default, pass the exact completed 20M final checkpoint and its
`metrics.json`; the launcher validates their model/protocol identity before either dry-run or
training. Diagnostic runs may explicitly pass `--allow-incomplete-source` with a pure model
`state_dict` extracted from a stable copied checkpoint and metadata recording its positive exact
source step. Such a run remains a fresh adaptation, not a resume, and its leaf records that step.

For the cumulative-2M diagnostic, `--extend-from-skiing-1m` accepts a completed 1M
`skiing-stall-actionfix-v1` final state and runs one fresh additional 1M phase with a new replay,
optimizer, and phase-local global step. It keeps the terminal schedule values throughout that
phase: epsilon `0.01`, LR `1e-5`, and no further LR decay. An active resumable run may instead
increase only its target with `--total-timesteps 2000000
--allow-total-timesteps-extension`; all other resume protocol fields remain strict, and the
checkpoint records the original target.

For cumulative 4M exposure, `--extend-from-skiing-2m` accepts only the registered completed-2M
source leaf for the selected model and starts a fresh additional 2M phase. Finalization has already
removed every 2M source's optimizer/replay checkpoint, so all three models use weights-only
initialization with fresh optimizer, replay, RNG, and phase-local global step. The terminal
schedule remains epsilon `0.01`, LR `1e-5`, and no further LR decay. The new leaf is named
`stallactionfix_v1_extend2mto4m_2m_<model>_seed1`; cumulative plotting offsets this phase by 2M.

After a detached launch succeeds, record its run ID, tmux session, remote root, exact logs,
results, and validity conditions with the project-local registry in
`experiments/monitoring/README.md`. This makes the same run discoverable from Mac and Mac mini
without relying on a separate task service.

The wrapper prints a marker for detached runs. Fetch only files newer than that marker:

```bash
./experiments/remote/fetch_results.sh --since <marker-file>
```

Fetch one result subdirectory:

```bash
./experiments/remote/fetch_results.sh train_data/<result-suffix>
```

Use `--all` only when the full remote history is intentionally required.

## Connections

The example enables SSH ControlMaster so one authenticated connection can be reused briefly by
sync, run, and fetch operations. Configure SSH keys and aliases outside the repository. Keep real
usernames, addresses, and project paths in `experiments/remote/config.sh` and `.agents/local.md`.
