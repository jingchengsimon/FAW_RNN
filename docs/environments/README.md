# Environment profiles

The two YAML files in this directory are the only maintained environment definitions. They are
portable contracts, not machine exports. Record the Git commit and `pip freeze` for each formal
remote run.

- `aim3_rnn-linux-cuda.yml` is the common training baseline for Amarel and sjc-remote. Both
  hosts currently use Python 3.11, PyTorch 2.3.x, CUDA 12.1, Gymnasium 1.3.0, ALE 0.12.0,
  Mamba-SSM 2.2.4, and S5 0.2.1. Minor NumPy/Pandas/Matplotlib build versions differ, so the
  profile specifies tested compatibility ranges rather than pretending the hosts are identical.
- `aim3_rnn-macos.yml` describes the current lightweight macOS development environment. It
  intentionally excludes PyTorch, CUDA, Atari, Mamba, and S5: it is sufficient for documentation
  work and the shell-only Amarel submitter safety test, but cannot run training or model imports.

Create a new environment instead of changing an active remote training environment in place:

```bash
conda env create -n aim3_rnn_next -f docs/environments/aim3_rnn-linux-cuda.yml
```

Linux Mamba builds require a compiler-compatible CUDA host. If a new host cannot install
`mamba-ssm` from the YAML directly, install the pinned package in the activated environment with
`pip install --no-build-isolation mamba-ssm==2.2.4`, then run the project environment verifier on
a compute node.
