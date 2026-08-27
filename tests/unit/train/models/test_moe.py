import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.distributed.token_dispatcher import LocalTokenDispatcher
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


class ReferenceGroupedGemm:
    token_group_alignment = 1

    def __call__(self, x: torch.Tensor, weights: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
        return _grouped_mm_reference(x, weights, offs=offs)


def _scaled_square_experts(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(x)
    scales = torch.repeat_interleave(torch.arange(1, len(counts) + 1, dtype=x.dtype), counts)
    output[: len(scales)] = x[: len(scales)].square() * scales.unsqueeze(1)
    return output


@pytest.mark.parametrize("score_before_experts", [True, False])
def test_local_token_dispatcher(score_before_experts):
    dispatcher = LocalTokenDispatcher(num_experts=3, top_k=2, token_group_alignment=1)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    scores = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]], requires_grad=True)
    selected = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 0]])

    actual = dispatcher.run(
        x,
        scores,
        selected,
        _scaled_square_experts,
        score_before_experts=score_before_experts,
    )

    expected = torch.zeros_like(x)
    for token in range(len(x)):
        for route in range(selected.shape[1]):
            score = scores[token, route]
            expert_scale = selected[token, route] + 1
            expert_input = x[token] * score if score_before_experts else x[token]
            contribution = expert_input.square() * expert_scale
            if not score_before_experts:
                contribution = contribution * score
            expected[token] = expected[token] + contribution

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None
    assert scores.grad is not None


def test_local_token_dispatcher_handles_empty_input_and_unused_experts():
    dispatcher = LocalTokenDispatcher(num_experts=3, top_k=1, token_group_alignment=1)
    x = torch.randn(2, 4)
    scores = torch.ones(2, 1)
    selected = torch.tensor([[0], [2]])

    output = dispatcher.run(x, scores, selected, _scaled_square_experts, score_before_experts=False)
    torch.testing.assert_close(output[0], x[0].square())
    torch.testing.assert_close(output[1], x[1].square() * 3)

    empty = dispatcher.run(
        x.new_empty((0, 4)),
        scores.new_empty((0, 1)),
        selected.new_empty((0, 1)),
        _scaled_square_experts,
        score_before_experts=False,
    )
    assert empty.shape == (0, 4)


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
        grouped_gemm=ReferenceGroupedGemm(),
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
        grouped_gemm=ReferenceGroupedGemm(),
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
