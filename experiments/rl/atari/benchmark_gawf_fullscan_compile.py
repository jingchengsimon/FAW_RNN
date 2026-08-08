"""Isolate L3 Atari GaWF full-scan compilation numerical differences.

This benchmark uses deterministic synthetic Atari sequences only.  It compares
four paths: the public eager scan, an eager tensor-state static scan, the
current gate-only compiled scan, and the proposed full-scan compiled path.
This separates a tensor-state refactor difference from compilation/precision
effects. It does not run an environment, optimizer step, replay buffer, or
training experiment.
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

SequenceForward = Callable[
    [torch.Tensor, torch.Tensor, AtariQNetworkState | None, bool],
    tuple[torch.Tensor, AtariQNetworkState | None],
]


def parse_args() -> argparse.Namespace:
    """Parse deterministic benchmark settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow TF32 matmul kernels for the FP32 diagnostic condition.",
    )
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


def _clone_state(state: AtariQNetworkState | None) -> AtariQNetworkState | None:
    """Clone CUDAGraph-backed outputs before a later invocation can reuse them."""
    if state is None:
        return None
    recurrent = state.recurrent
    if not isinstance(recurrent, list):
        raise TypeError("The L3 GaWF benchmark expects a list-valued recurrent state")
    return AtariQNetworkState(
        recurrent=[part.detach().float().clone() for part in recurrent],
        prev_q=state.prev_q.detach().float().clone(),
    )


def _run_with_grad(
    forward: SequenceForward,
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
    return q_values.detach().float().clone(), _clone_state(state), gradients


def _timed_iteration(
    forward: SequenceForward,
    model: AtariQNetwork,
    obs: torch.Tensor,
    prev_dones: torch.Tensor,
    amp_dtype: str,
) -> None:
    """Run one non-updating forward/backward timing iteration."""
    _run_with_grad(forward, model, obs, prev_dones, amp_dtype)


def _median_ms(
    forward: SequenceForward,
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


def _max_abs_difference(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return the maximum absolute difference without discarding diagnostic evidence."""
    difference = float((reference - candidate).abs().max().item())
    return difference


def _difference_report(
    reference: tuple[torch.Tensor, AtariQNetworkState | None, dict[str, torch.Tensor]],
    candidate: tuple[torch.Tensor, AtariQNetworkState | None, dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Report maximum Q, state, and gradient differences for two fixed inputs."""
    reference_q, reference_state, reference_grads = reference
    candidate_q, candidate_state, candidate_grads = candidate
    if reference_grads.keys() != candidate_grads.keys():
        raise RuntimeError("Compared paths produced different parameter-gradient keys")
    return {
        "q_max_abs_diff": _max_abs_difference(reference_q, candidate_q),
        "state_max_abs_diff": max(
            (
                _max_abs_difference(left.float(), right.float())
                for left, right in zip(
                    _flatten_state(reference_state), _flatten_state(candidate_state)
                )
            ),
            default=0.0,
        ),
        "gradient_max_abs_diff": max(
            (
                _max_abs_difference(reference_grads[name], candidate_grads[name])
                for name in reference_grads
            ),
            default=0.0,
        ),
    }


def _make_static_forward(
    model: AtariQNetwork,
    compiled: bool,
) -> SequenceForward:
    """Adapt the tensor-only static scan to the public recurrent-state interface."""
    scan = model.forward_sequence_gawf_l3_static
    if compiled:
        scan = torch.compile(scan, mode="reduce-overhead", fullgraph=True, dynamic=False)

    def forward(
        observations: torch.Tensor,
        dones: torch.Tensor,
        state: AtariQNetworkState | None,
        _has_internal_reset: bool,
    ) -> tuple[torch.Tensor, AtariQNetworkState]:
        if state is None:
            q_values, state0, state1, state2, next_q = scan(
                observations, dones, None, None, None, None
            )
        else:
            if not isinstance(state.recurrent, list) or len(state.recurrent) != 3:
                raise TypeError("L3 GaWF static scan requires three recurrent state tensors")
            q_values, state0, state1, state2, next_q = scan(
                observations,
                dones,
                state.recurrent[0],
                state.recurrent[1],
                state.recurrent[2],
                state.prev_q,
            )
        # Match the training wrapper: compiled CUDAGraph outputs need stable
        # storage before recurrent state crosses the next invocation boundary.
        return q_values.clone(), AtariQNetworkState(
            [state0.clone(), state1.clone(), state2.clone()], next_q.clone()
        )

    return forward


def main() -> None:
    """Run the deterministic full-scan compile comparison on one CUDA GPU."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA")
    torch.manual_seed(20260807)
    torch.cuda.manual_seed_all(20260807)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
    device = torch.device("cuda")
    obs = torch.randint(
        0,
        256,
        (args.batch_size, args.seq_len, 4, 84, 84),
        device=device,
        dtype=torch.uint8,
    )
    prev_dones = torch.zeros(args.batch_size, args.seq_len, device=device)

    eager = AtariQNetwork(
        num_actions=18,
        input_channels=4,
        model_type="gawf",
        hidden_size=512,
        feedback_mode="qvalues",
        num_layers=3,
    ).to(device)
    static_eager = AtariQNetwork(
        num_actions=18,
        input_channels=4,
        model_type="gawf",
        hidden_size=512,
        feedback_mode="qvalues",
        num_layers=3,
    ).to(device)
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
    static_eager.load_state_dict(eager.state_dict())
    gate_only.load_state_dict(eager.state_dict())
    full_scan.load_state_dict(eager.state_dict())
    for model in (eager, static_eager, gate_only, full_scan):
        model.train()

    configure_gawf_feedback_acceleration(gate_only, enabled=True, compile_mode="reduce-overhead")
    gate_forward = gate_only.forward_sequence
    static_eager_forward = _make_static_forward(static_eager, compiled=False)
    full_forward = _make_static_forward(full_scan, compiled=True)

    eager_result = _run_with_grad(eager.forward_sequence, eager, obs, prev_dones, args.amp_dtype)
    static_eager_result = _run_with_grad(
        static_eager_forward, static_eager, obs, prev_dones, args.amp_dtype
    )
    gate_result = _run_with_grad(gate_forward, gate_only, obs, prev_dones, args.amp_dtype)
    full_result = _run_with_grad(full_forward, full_scan, obs, prev_dones, args.amp_dtype)
    static_refactor = _difference_report(eager_result, static_eager_result)
    compile_effect = _difference_report(static_eager_result, full_result)
    gate_compile_effect = _difference_report(eager_result, gate_result)
    full_effect = _difference_report(eager_result, full_result)

    repeat_result = _run_with_grad(full_forward, full_scan, obs, prev_dones, args.amp_dtype)
    repeat_effect = _difference_report(full_result, repeat_result)

    gate_only_ms = _median_ms(gate_forward, gate_only, obs, prev_dones, args)
    full_scan_ms = _median_ms(full_forward, full_scan, obs, prev_dones, args)
    report = {
        "protocol": "L3 Atari GaWF qvalues, synthetic deterministic sequence",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "amp_dtype": args.amp_dtype,
        "allow_tf32": args.allow_tf32,
        "gate_only_compile_median_forward_backward_ms": gate_only_ms,
        "full_scan_compile_median_forward_backward_ms": full_scan_ms,
        "full_scan_speedup": gate_only_ms / full_scan_ms,
        "eager_vs_static_eager": static_refactor,
        "static_eager_vs_full_scan_compile": compile_effect,
        "eager_vs_gate_only_compile": gate_compile_effect,
        "eager_vs_full_scan_compile": full_effect,
        "full_scan_compile_repeat": repeat_effect,
        "q_max_abs_diff": full_effect["q_max_abs_diff"],
        "state_max_abs_diff": full_effect["state_max_abs_diff"],
        "gradient_max_abs_diff": full_effect["gradient_max_abs_diff"],
        "repeat_q_max_abs_diff": repeat_effect["q_max_abs_diff"],
        "repeat_state_max_abs_diff": repeat_effect["state_max_abs_diff"],
        "equivalence_tolerance": 0.001,
        "extra_randomness_detected": max(repeat_effect.values()) > 0.001,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
