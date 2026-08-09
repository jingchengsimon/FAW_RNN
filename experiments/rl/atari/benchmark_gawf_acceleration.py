"""Benchmark opt-in GaWF feedback execution strategies on the real Atari DRQN update path.

The script builds the production ``AtariQNetwork`` and calls the production
``_drqn_sequence_loss`` helper with fixed synthetic replay windows.  It writes
JSON summaries for current eager, BF16 cast-cache, and combined-transform
strategies; no launcher default or checkpoint format is changed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch

from utils.training.atari.atari_dqn_models import AtariQNetwork, AtariQNetworkState
from utils.training.atari.atari_replay import SequenceBatch
from utils.training.recurrent_cores import configure_gawf_feedback_execution
from utils.training.train_scripts.atari_dqn import _build_atari_optimizer, _drqn_sequence_loss


STRATEGIES = ("eager", "cast_cache", "combined_transform")


class SyntheticSequenceBuffer:
    """Replay-shaped immutable data source used by the formal DRQN loss helper."""

    sampling_mode = "global_uniform"
    num_tasks = 1

    def __init__(self, batch: SequenceBatch) -> None:
        self._batch = batch

    def sample_sequences(self, batch_size: int, seq_len: int) -> SequenceBatch:
        """Return the fixed formal ``(B, L+1)`` sequence batch after checking geometry."""

        if batch_size != self._batch.obs.shape[0] or seq_len + 1 != self._batch.obs.shape[1]:
            raise ValueError("Synthetic replay request differs from benchmark geometry")
        return self._batch


class TimedForward:
    """Attach CUDA event boundaries to one real model forward callable."""

    def __init__(self, forward: Callable[..., Any]) -> None:
        self.forward = forward
        self.starts: list[torch.cuda.Event] = []
        self.ends: list[torch.cuda.Event] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = self.forward(*args, **kwargs)
        end.record()
        self.starts.append(start)
        self.ends.append(end)
        return result

    def elapsed_ms(self) -> float:
        return sum(start.elapsed_time(end) for start, end in zip(self.starts, self.ends))


def parse_args() -> argparse.Namespace:
    """Parse the controlled A6000 benchmark configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--measured-updates", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixed-updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def _make_batch(device: torch.device, seed: int) -> SequenceBatch:
    """Create one deterministic production-geometry Atari replay window."""

    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (8, 17, 4, 84, 84)
    obs = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator, device=device)
    return SequenceBatch(
        obs=obs,
        actions=torch.randint(0, 18, (8, 17), generator=generator, device=device),
        rewards=torch.randn(8, 17, generator=generator, device=device),
        dones=torch.zeros(8, 17, device=device),
        prev_dones=torch.zeros(8, 17, device=device),
        loss_mask=torch.ones(8, 17, device=device),
        task_ids=torch.zeros(8, 17, dtype=torch.long, device=device),
        has_internal_reset=False,
    )


def _make_network(
    device: torch.device,
    state: dict[str, torch.Tensor] | None,
    strategy: str,
) -> AtariQNetwork:
    """Build an L3 Q-value-feedback GaWF network from an identical initial state."""

    network = AtariQNetwork(
        num_actions=18,
        input_channels=4,
        model_type="gawf",
        hidden_size=604,
        feedback_mode="qvalues",
        num_layers=3,
        core_dropout=0.0,
    ).to(device)
    if state is not None:
        network.load_state_dict(state)
    configure_gawf_feedback_execution(network, strategy)
    return network


def _make_optimizer(model: AtariQNetwork) -> torch.optim.Optimizer:
    """Use the formal fused Adam grouping, including GaWF's zero-decay U/V group."""

    return _build_atari_optimizer(
        model,
        model_type="gawf",
        learning_rate=1e-4,
        gawf_feedback_lr_scale=1.0,
        use_fused_optimizer=True,
    )


def _loss_args() -> argparse.Namespace:
    """Return exactly the loss arguments used by the DRQN update helper."""

    return argparse.Namespace(sequences_per_batch=8, seq_len=16, gamma=0.99, double_dqn=False)


def _one_update(
    model: AtariQNetwork,
    target: AtariQNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: SyntheticSequenceBuffer,
    *,
    timed: bool,
) -> tuple[float, dict[str, float]]:
    """Run the formal BF16 DRQN loss/backward/clip/fused-Adam update once."""

    model_forward: Callable[..., Any] = model.forward_sequence
    target_forward: Callable[..., Any] = target.forward_sequence
    online = TimedForward(model_forward) if timed else None
    target_timed = TimedForward(target_forward) if timed else None
    if online is not None and target_timed is not None:
        model_forward, target_forward = online, target_timed
    total_start = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _q_mean = _drqn_sequence_loss(
            model_forward,
            target_forward,
            buffer,
            _loss_args(),
            next(model.parameters()).device,
        )
    optimizer.zero_grad(set_to_none=True)
    backward_start.record()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    backward_end.record()
    optimizer_start.record()
    optimizer.step()
    optimizer_end.record()
    total_end.record()
    torch.cuda.synchronize()
    times = {
        "complete_update_ms": total_start.elapsed_time(total_end),
        "online_forward_ms": online.elapsed_ms() if online is not None else float("nan"),
        "target_forward_ms": (
            target_timed.elapsed_ms() if target_timed is not None else float("nan")
        ),
        "backward_ms": backward_start.elapsed_time(backward_end),
        "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
    }
    return float(loss.detach().cpu()), times


def _summary(values: list[float]) -> dict[str, float]:
    """Return requested latency statistics for one measured series."""

    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p25": ordered[round((len(ordered) - 1) * 0.25)],
        "p75": ordered[round((len(ordered) - 1) * 0.75)],
        "mean": statistics.fmean(ordered),
    }


def _run_repeat(
    state: dict[str, torch.Tensor],
    batch: SequenceBatch,
    strategy: str,
    warmup_updates: int,
    measured_updates: int,
) -> dict[str, Any]:
    """Measure one warmup-excluded repeat with CUDA events and peak memory."""

    device = batch.obs.device
    model = _make_network(device, state, strategy)
    target = _make_network(device, state, strategy)
    target.requires_grad_(False)
    target.eval()
    optimizer = _make_optimizer(model)
    buffer = SyntheticSequenceBuffer(batch)
    for _ in range(warmup_updates):
        _one_update(model, target, optimizer, buffer, timed=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(measured_updates):
        _loss, times = _one_update(model, target, optimizer, buffer, timed=True)
        for name, value in times.items():
            samples[name].append(value)
    return {
        "latency_ms": {name: _summary(values) for name, values in samples.items()},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _tensor_difference(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare identically keyed tensors without masking numerical differences."""

    max_abs = -1.0
    mean_abs_sum = 0.0
    numel = 0
    max_rel = 0.0
    worst = ""
    bitwise = True
    for name, tensor in reference.items():
        other = candidate[name]
        if not torch.equal(tensor, other):
            bitwise = False
        diff = (tensor.float() - other.float()).abs()
        current_max = float(diff.max().cpu())
        if current_max > max_abs:
            max_abs, worst = current_max, name
        mean_abs_sum += float(diff.sum().cpu())
        numel += diff.numel()
        max_rel = max(max_rel, float((diff / tensor.float().abs().clamp_min(1e-12)).max().cpu()))
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs_sum / max(numel, 1),
        "max_relative": max_rel,
        "bitwise_equal": bitwise,
        "worst_tensor": worst,
    }


def _outputs(model: AtariQNetwork, batch: SequenceBatch) -> dict[str, torch.Tensor]:
    """Extract fixed-input Q output and every final recurrent-state component."""

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        q_values, state = model.forward_sequence(batch.obs, batch.prev_dones, None, False)
    assert state is not None
    result = {"q_values": q_values.detach().clone(), "prev_q": state.prev_q.detach().clone()}
    recurrent = state.recurrent
    if isinstance(recurrent, list):
        result.update(
            {
                f"state_layer_{index}": value.detach().clone()
                for index, value in enumerate(recurrent)
            }
        )
    else:
        result["state"] = recurrent.detach().clone()
    return result


def _fixed_input_checks(
    initial: dict[str, torch.Tensor], batch: SequenceBatch, strategy: str, updates: int
) -> dict[str, Any]:
    """Compare candidate and eager after 1, 10, and N identical formal updates."""

    device = batch.obs.device
    baseline = _make_network(device, initial, "eager")
    candidate = _make_network(device, initial, strategy)
    baseline_target = _make_network(device, initial, "eager")
    candidate_target = _make_network(device, initial, strategy)
    for target in (baseline_target, candidate_target):
        target.requires_grad_(False)
        target.eval()
    base_opt, cand_opt = _make_optimizer(baseline), _make_optimizer(candidate)
    buffer = SyntheticSequenceBuffer(batch)
    checkpoints = {1, 10, updates}
    report: dict[str, Any] = {}
    for update in range(1, updates + 1):
        base_loss, _ = _one_update(baseline, baseline_target, base_opt, buffer, timed=False)
        cand_loss, _ = _one_update(candidate, candidate_target, cand_opt, buffer, timed=False)
        if update not in checkpoints:
            continue
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in baseline.named_parameters()
            if parameter.grad is not None
        }
        candidate_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in candidate.named_parameters()
            if parameter.grad is not None
        }
        report[str(update)] = {
            "loss": {
                "baseline": base_loss,
                "candidate": cand_loss,
                "absolute_difference": abs(base_loss - cand_loss),
            },
            "q_and_state": _tensor_difference(
                _outputs(baseline, batch), _outputs(candidate, batch)
            ),
            "gradients": _tensor_difference(gradients, candidate_gradients),
            "parameters": _tensor_difference(
                dict(baseline.named_parameters()), dict(candidate.named_parameters())
            ),
        }
    return report


def _profile(
    initial: dict[str, torch.Tensor],
    batch: SequenceBatch,
    strategy: str,
) -> dict[str, Any]:
    """Collect CUDA kernel, copy/cast, and layer gate-transform profiler evidence."""

    device = batch.obs.device
    model = _make_network(device, initial, strategy)
    target = _make_network(device, initial, strategy)
    configure_gawf_feedback_execution(model, strategy, profile_gate_transforms=True)
    configure_gawf_feedback_execution(target, strategy, profile_gate_transforms=True)
    target.requires_grad_(False)
    target.eval()
    optimizer, buffer = _make_optimizer(model), SyntheticSequenceBuffer(batch)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        _one_update(model, target, optimizer, buffer, timed=False)
    rows = []
    gate_times: dict[str, float] = {}
    copy_count = 0
    copy_time = 0.0
    for event in profiler.key_averages():
        cuda_us = float(event.self_cuda_time_total)
        if cuda_us:
            rows.append({"name": event.key, "count": event.count, "self_cuda_ms": cuda_us / 1000.0})
        if event.key.startswith("gawf_gate_transform_layer_"):
            gate_times[event.key] = float(event.cuda_time_total) / 1000.0
        if "copy" in event.key or "to" in event.key:
            copy_count += int(event.count)
            copy_time += cuda_us / 1000.0
    rows.sort(key=lambda row: row["self_cuda_ms"], reverse=True)
    return {
        "top_cuda_kernels": rows[:20],
        "gate_transform_cuda_ms": gate_times,
        "cast_copy": {"count": copy_count, "self_cuda_ms": copy_time},
    }


def main() -> None:
    """Execute the complete controlled benchmark and save machine-readable evidence."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark requires BF16 support")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_model = _make_network(device, None, "eager")
    initial = deepcopy(seed_model.state_dict())
    del seed_model
    batch = _make_batch(device, args.seed + 1)
    summary: dict[str, Any] = {
        "metadata": {
            "geometry": {
                "model_type": "gawf",
                "feedback_mode": "qvalues",
                "num_layers": 3,
                "hidden_size": 604,
                "num_actions": 18,
                "batch": 8,
                "seq_len": 16,
                "timesteps": 17,
            },
            "amp_dtype": "bfloat16",
            "allow_tf32": True,
            "fused_adam": True,
            "core_dropout": 0.0,
            "seed": args.seed, "torch": torch.__version__, "cuda": torch.version.cuda,
        },
        "strategies": {},
    }
    for strategy in STRATEGIES:
        repeat_results = [
            _run_repeat(initial, batch, strategy, args.warmup_updates, args.measured_updates)
            for _ in range(args.repeats)
        ]
        summary["strategies"][strategy] = {
            "repeats": repeat_results,
            "fixed_input": _fixed_input_checks(initial, batch, strategy, args.fixed_updates),
            "profiler": _profile(initial, batch, strategy),
        }
    baseline = summary["strategies"]["eager"]["repeats"]
    baseline_median = statistics.median(
        row["latency_ms"]["complete_update_ms"]["median"] for row in baseline
    )
    for strategy, result in summary["strategies"].items():
        median = statistics.median(
            row["latency_ms"]["complete_update_ms"]["median"] for row in result["repeats"]
        )
        result["complete_update_median_ms_across_repeats"] = median
        result["speedup_vs_eager"] = baseline_median / median
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
