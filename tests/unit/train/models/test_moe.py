import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.models.layers.activations import ActivationDispatch
from prime_rl.trainer.models.layers.moe import (
    GroupedExperts,
    MoE,
    MoEArgs,
    _run_grouped_experts_impl,
)


def _grouped_mm_reference(x: torch.Tensor, weights: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
    outputs = []
    start = 0
    for expert, end in enumerate(offs.tolist()):
        outputs.append(x[start:end] @ weights[expert])
        start = end
    return torch.cat(outputs)


@pytest.mark.parametrize(
    ("expert_type", "activation"),
    [
        ("gated", "silu"),
        ("gated", "relu2"),
        ("non_gated", "silu"),
        ("non_gated", "relu2"),
    ],
)
def test_expert_type_and_activation_are_independent(expert_type, activation):
    experts = GroupedExperts(
        dim=4,
        hidden_dim=8,
        num_experts=2,
        expert_type=expert_type,
        activation=activation,
        bias=True,
        grouped_mm_fn=_grouped_mm_reference,
    )
    experts.init_weights(0.02)

    has_gate = expert_type == "gated"
    assert (experts.gate_proj is not None) == has_gate
    assert (experts.gate_proj_bias is not None) == has_gate
    assert ("gate_proj" in experts.state_dict()) == has_gate

    x = torch.randn(3, 4)
    counts = torch.tensor([2, 1])
    actual = _run_grouped_experts_impl(
        experts.up_proj.transpose(-2, -1),
        experts.down_proj.transpose(-2, -1),
        x,
        counts,
        gate_proj=experts.gate_proj.transpose(-2, -1) if experts.gate_proj is not None else None,
        grouped_mm_fn=_grouped_mm_reference,
        activation=experts.activation,
        gate_proj_bias=experts.gate_proj_bias,
        up_proj_bias=experts.up_proj_bias,
        down_proj_bias=experts.down_proj_bias,
    )

    expected = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        expert_input = x[start : start + count].bfloat16()
        up = F.linear(expert_input, experts.up_proj[expert].bfloat16(), experts.up_proj_bias[expert].bfloat16())
        activation_input = up
        if has_gate:
            activation_input = F.linear(
                expert_input,
                experts.gate_proj[expert].bfloat16(),
                experts.gate_proj_bias[expert].bfloat16(),
            )
        hidden = F.silu(activation_input) if activation == "silu" else F.relu(activation_input).square()
        if has_gate:
            hidden = hidden * up
        expected.append(
            F.linear(hidden, experts.down_proj[expert].bfloat16(), experts.down_proj_bias[expert].bfloat16())
        )
        start += count

    torch.testing.assert_close(actual, torch.cat(expected).float())

    with torch.device("meta"):
        moe = MoE.from_args(
            MoEArgs(
                num_experts=2,
                num_shared_experts=1,
                expert_type=expert_type,
                activation=activation,
                load_balance_coeff=None,
            ),
            dim=4,
            hidden_dim=8,
        )
    assert (moe.experts.gate_proj is not None) == has_gate
    assert moe.experts.activation is ActivationDispatch[activation]
    assert (moe.shared_expert.gate_proj is not None) == has_gate
    assert moe.shared_expert.activation is ActivationDispatch[activation]
