import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.distributed.token_dispatcher import LocalTokenDispatcher
from prime_rl.trainer.models.layers.activations import ActivationDispatch
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import (
    GroupedExperts,
    MoE,
    MoEArgs,
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
    actual = experts(x, counts)

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
        shared_expert = FeedForward(
            dim=4,
            hidden_dim=8,
            expert_type=expert_type,
            activation=activation,
        )
        moe = MoE.from_args(
            MoEArgs(
                num_experts=2,
                expert_type=expert_type,
                activation=activation,
                load_balance_coeff=None,
            ),
            dim=4,
            hidden_dim=8,
            shared_expert=shared_expert,
        )
    assert moe.shared_expert is shared_expert
    assert (moe.experts.gate_proj is not None) == has_gate
    assert moe.experts.activation is ActivationDispatch[activation]
    assert (moe.shared_expert.gate_proj is not None) == has_gate
    assert moe.shared_expert.activation is ActivationDispatch[activation]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="overlap path only runs on CUDA")
def test_overlapped_shared_expert_matches_sequential():
    def build_moe() -> MoE:
        torch.manual_seed(42)
        moe = MoE.from_args(
            MoEArgs(num_experts=4, top_k=2, load_balance_coeff=None),
            dim=16,
            hidden_dim=32,
            shared_expert=FeedForward(dim=16, hidden_dim=32),
        ).to("cuda")
        moe.experts.init_weights(0.02)
        moe.router.init_weights(0.02)
        moe.shared_expert.init_weights(0.02)
        moe.set_token_dispatcher(LocalTokenDispatcher(num_experts=4, top_k=2, token_group_alignment=1))
        return moe

    x = torch.randn(3, 8, 16, device="cuda", requires_grad=True)

    moe_sequential = build_moe()
    out_sequential = moe_sequential(x)
    out_sequential.sum().backward()
    grad_sequential = x.grad.clone()
    x.grad = None

    moe_overlapped = build_moe()
    moe_overlapped.set_overlap_shared_expert(True)
    out_overlapped = moe_overlapped(x)
    out_overlapped.sum().backward()
    grad_overlapped = x.grad.clone()

    torch.testing.assert_close(out_sequential, out_overlapped)
    torch.testing.assert_close(grad_sequential, grad_overlapped)
    for (name, p_sequential), (_, p_overlapped) in zip(
        moe_sequential.named_parameters(), moe_overlapped.named_parameters()
    ):
        torch.testing.assert_close(p_sequential.grad, p_overlapped.grad, msg=f"param grad mismatch: {name}")
