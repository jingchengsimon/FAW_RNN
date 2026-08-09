"""Tests for lower-layer projected feedback in multi-layer Atari GaWF."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from utils.training.atari.atari_dqn_models import AtariQNetwork
from utils.training.train_scripts.atari_dqn import _build_atari_optimizer


class _TinyEncoder(nn.Module):
    """Small observation encoder used to isolate recurrent behavior in tests."""

    output_size = 6

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.float().reshape(obs.shape[0], -1)[:, : self.output_size]


def _projected_model() -> AtariQNetwork:
    return AtariQNetwork(
        num_actions=4,
        input_channels=1,
        model_type="gawf",
        hidden_size=8,
        num_layers=3,
        feedback_mode="qvalues",
        lower_feedback_dim=3,
        encoder_factory=_TinyEncoder,
    )


def test_projected_feedback_shapes_and_gradients() -> None:
    model = _projected_model()
    assert model.core.layer_feedback_dims == [3, 3, 4]
    assert len(model.lower_feedback_projectors) == 2

    obs = torch.randn(2, 4, 1, 2, 3)
    prev_dones = torch.zeros(2, 4)
    q_values, _ = model.forward_sequence(obs, prev_dones)
    q_values.square().mean().backward()

    assert q_values.shape == (2, 4, 4)
    for projector in model.lower_feedback_projectors:
        assert projector.weight.grad is not None
        assert torch.isfinite(projector.weight.grad).all()


def test_projector_input_detaches_upper_hidden_state() -> None:
    model = _projected_model()
    state = model.initial_state(2, "cpu")
    assert state is not None and isinstance(state.recurrent, list)
    recurrent = [part.requires_grad_() for part in state.recurrent]
    seen_requires_grad: list[bool] = []

    def capture_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen_requires_grad.append(inputs[0].requires_grad)

    handles = [
        projector.register_forward_pre_hook(capture_input)
        for projector in model.lower_feedback_projectors
    ]
    try:
        model._core_step(
            torch.randn(2, _TinyEncoder.output_size),
            recurrent,
            torch.zeros(2, 4),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert seen_requires_grad == [False, False]


def test_projectors_use_feedback_optimizer_group() -> None:
    model = _projected_model()
    optimizer = _build_atari_optimizer(
        model,
        model_type="gawf",
        learning_rate=1e-3,
        gawf_feedback_lr_scale=0.25,
        use_fused_optimizer=False,
    )
    projector_ids = {
        id(parameter)
        for projector in model.lower_feedback_projectors
        for parameter in projector.parameters()
    }
    feedback_group_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(2.5e-4)
    assert projector_ids <= feedback_group_ids
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_type": "rnn"}, "only valid for model_type='gawf'"),
        ({"num_layers": 1}, "requires num_layers >= 2"),
        ({"feedback_mode": "none"}, "requires qvalues feedback"),
    ],
)
def test_projected_feedback_rejects_invalid_protocol(kwargs: dict, message: str) -> None:
    model_kwargs = {
        "num_actions": 4,
        "model_type": "gawf",
        "hidden_size": 8,
        "num_layers": 3,
        "feedback_mode": "qvalues",
        "lower_feedback_dim": 3,
        "encoder_factory": _TinyEncoder,
    }
    model_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        AtariQNetwork(**model_kwargs)
