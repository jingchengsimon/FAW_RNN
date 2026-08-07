"""Compare gate-only and full-scan ``torch.compile`` for L3 Atari GaWF.

This benchmark uses deterministic synthetic Atari sequences only.  It compares
the current gate-only compiled path against an opt-in full-graph
``AtariQNetwork.forward_sequence`` path, validates Q-values, recurrent state,
and parameter gradients, then writes a compact JSON report.  It does not run
an environment, optimizer step, replay buffer, or training experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch

from utils.training.atari.atari_dqn_models import AtariQNetwork, AtariQNetworkState
from utils.training.recurrent_cores.gawf import configure_gawf_feedback_acceleration


def parse_args() -> argparse.Namespace:
    """Parse deterministic benchmark settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def _autocast(dtype_name: str) -> torch.autocast | torch.autocast_mode.autocast:
    """Return the benchmark's CUDA autocast context."""
    if dtype_name == "none":
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _flatten_state(state: AtariQNetworkState | None) -> list[torch.Tensor]:
    """Convert the public recurrent state container into tensors for comparisons."""
    if state is None:
        return []
    recurrent = state.recurrent
    recurrent_parts = list(recurrent) if isinstance(recurrent, list) else [recurrent]
    return [*recurrent_parts, state.prev_q]


def _run_with_grad(
    forward: Callable[..., tuple[torch.Tensor, AtariQNetworkState | None]],
    model: AtariQNetwork,
    obs: torch.Tensor,
    prev_dones: torch.Tensor,
    amp_dtype: str,
) -> tuple[torch.Tensor, AtariQNetworkState | None, dict[str, torch.Tensor]]:
    """Run one deterministic forward/backward pass and clone parameter gradients."""
    model.zero_grad(set_to_none=True)
    with _autocast(amp_dtype):
        q_values, state = forward(obs, prev_dones, None, False)
        loss = q_values.float().square().mean()
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return q_values.detach().float(), state, gradients


def _timed_iteration(
    forward: Callable[..., tuple[torch.Tensor, AtariQNetworkState | None]],
    model: AtariQNetwork,
    obs: torch.Tensor,
    prev_dones: torch.Tensor,
    amp_dtype: str,
) -> None:
    """Run one non-updating forward/backward timing iteration."""
    _run_with_grad(forward, model, obs, prev_dones, amp_dtype)


def _median_ms(
    forward: Callable[..., tuple[torch.Tensor, AtariQNetworkState | None]],
    model: AtariQNetwork,
    obs: torch.Tensor,
    prev_dones: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    """Measure synchronized median forward/backward latency after warmup."""
    for _ in range(args.warmup):
        _timed_iteration(forward, model, obs, prev_dones, args.amp_dtype)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        _timed_iteration(forward, model, obs, prev_dones, args.amp_dtype)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1_000.0)
    return float(torch.tensor(samples).median().item())


def _assert_close(reference: torch.Tensor, candidate: torch.Tensor, label: str) -> float:
    """Validate numerical equivalence and return the maximum absolute difference."""
    difference = float((reference - candidate).abs().max().item())
    torch.testing.assert_close(reference, candidate, rtol=2e-2, atol=2e-2, msg=label)
    return difference


def main() -> None:
    """Run the deterministic full-scan compile comparison on one CUDA GPU."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA")
    torch.manual_seed(20260807)
    torch.cuda.manual_seed_all(20260807)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    obs = torch.randint(
        0,
        256,
        (args.batch_size, args.seq_len, 4, 84, 84),
        device=device,
        dtype=torch.uint8,
    )
    prev_dones = torch.zeros(args.batch_size, args.seq_len, device=device)

    gate_only = AtariQNetwork(
        num_actions=18,
        input_channels=4,
        model_type="gawf",
        hidden_size=512,
        feedback_mode="qvalues",
        num_layers=3,
    ).to(device)
    full_scan = AtariQNetwork(
        num_actions=18,
        input_channels=4,
        model_type="gawf",
        hidden_size=512,
        feedback_mode="qvalues",
        num_layers=3,
    ).to(device)
    full_scan.load_state_dict(gate_only.state_dict())
    gate_only.train()
    full_scan.train()

    configure_gawf_feedback_acceleration(gate_only, enabled=True, compile_mode="reduce-overhead")
    gate_forward = gate_only.forward_sequence
    full_forward = torch.compile(
        full_scan.forward_sequence,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )

    gate_q, gate_state, gate_grads = _run_with_grad(
        gate_forward, gate_only, obs, prev_dones, args.amp_dtype
    )
    full_q, full_state, full_grads = _run_with_grad(
        full_forward, full_scan, obs, prev_dones, args.amp_dtype
    )
    q_max_abs_diff = _assert_close(gate_q, full_q, "Q-values differ")
    state_max_abs_diff = max(
        (_assert_close(left.float(), right.float(), "recurrent state differs")
         for left, right in zip(_flatten_state(gate_state), _flatten_state(full_state))),
        default=0.0,
    )
    gradient_max_abs_diff = max(
        (_assert_close(gate_grads[name], full_grads[name], f"gradient differs: {name}")
         for name in gate_grads),
        default=0.0,
    )

    repeat_q, repeat_state, _ = _run_with_grad(
        full_forward, full_scan, obs, prev_dones, args.amp_dtype
    )
    repeat_q_max_abs_diff = _assert_close(full_q, repeat_q, "full-scan output is non-deterministic")
    repeat_state_max_abs_diff = max(
        (_assert_close(left.float(), right.float(), "full-scan state is non-deterministic")
         for left, right in zip(_flatten_state(full_state), _flatten_state(repeat_state))),
        default=0.0,
    )

    gate_only_ms = _median_ms(gate_forward, gate_only, obs, prev_dones, args)
    full_scan_ms = _median_ms(full_forward, full_scan, obs, prev_dones, args)
    report = {
        "protocol": "L3 Atari GaWF qvalues, synthetic deterministic sequence",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "amp_dtype": args.amp_dtype,
        "gate_only_compile_median_forward_backward_ms": gate_only_ms,
        "full_scan_compile_median_forward_backward_ms": full_scan_ms,
        "full_scan_speedup": gate_only_ms / full_scan_ms,
        "q_max_abs_diff": q_max_abs_diff,
        "state_max_abs_diff": state_max_abs_diff,
        "gradient_max_abs_diff": gradient_max_abs_diff,
        "repeat_q_max_abs_diff": repeat_q_max_abs_diff,
        "repeat_state_max_abs_diff": repeat_state_max_abs_diff,
        "extra_randomness_detected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
