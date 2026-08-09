"""Unit tests for opt-in, checkpoint-neutral GaWF execution strategies."""

from __future__ import annotations

from copy import deepcopy

import torch

from utils.training.recurrent_cores.gawf import (
    GaWFCore,
    configure_gawf_feedback_execution,
)


def _run(
    strategy: str,
    state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one deterministic multi-layer feedback step and expose parameter gradients."""

    core = GaWFCore(
        input_size=4,
        hidden_size=5,
        feedback_dim=3,
        num_layers=3,
        layer_feedback_dims=[5, 5, 3],
        dropout=0.0,
    )
    core.load_state_dict(state)
    configure_gawf_feedback_execution(core, strategy)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10.0
    recurrent = core.initial_state(2, "cpu", torch.float32)
    assert isinstance(recurrent, list)
    feedback = [recurrent[1].detach(), recurrent[2].detach(), torch.ones(2, 3)]
    with core.feedback_sequence():
        output, _next_state = core.step(x, recurrent, feedback)
    output.square().mean().backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in core.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), gradients


def test_feedback_execution_strategies_preserve_cpu_reference_and_checkpoint_keys() -> None:
    """CPU fallback keeps eager numerics while strategy state stays non-persistent."""

    torch.manual_seed(19)
    reference_core = GaWFCore(
        input_size=4,
        hidden_size=5,
        feedback_dim=3,
        num_layers=3,
        layer_feedback_dims=[5, 5, 3],
        dropout=0.0,
    )
    state = deepcopy(reference_core.state_dict())
    eager_output, eager_gradients = _run("eager", state)
    assert "_feedback_execution" not in state
    for strategy in ("cast_cache", "combined_transform"):
        output, gradients = _run(strategy, state)
        assert torch.equal(eager_output, output)
        assert gradients.keys() == eager_gradients.keys()
        assert all(torch.equal(eager_gradients[name], gradients[name]) for name in gradients)


def test_feedback_execution_rejects_unknown_strategy() -> None:
    """Only explicit benchmark strategies may alter the runtime path."""

    core = GaWFCore(4, 5, feedback_dim=3)
    try:
        core.set_feedback_execution("compiled_everything")
    except ValueError as exc:
        assert "Unsupported GaWF" in str(exc)
    else:
        raise AssertionError("Unknown GaWF feedback strategy was accepted")
