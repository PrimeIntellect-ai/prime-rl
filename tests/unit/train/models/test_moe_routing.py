import pytest
import torch

from prime_rl.trainer.models.layers.moe import MoE, MoEArgs


def test_moe_forced_routing_bypasses_router():
    moe_args = MoEArgs(
        num_experts=3,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=False,
        route_scale=1.0,
        score_before_experts=False,
        top_k=2,
        use_grouped_mm=False,
        load_balance_coeff=None,
    )
    moe = MoE(moe_args, dim=4, hidden_dim=8)

    def _router_forward(*args, **kwargs):
        raise AssertionError("router.forward should not be called when forcing routing")

    moe.router.forward = _router_forward

    x = torch.randn(1, 3, 4)
    forced_indices = torch.tensor([[[0, 1], [1, 2], [2, 2]]], dtype=torch.long)
    forced_probs = torch.ones_like(forced_indices, dtype=torch.float32)

    output = moe(x, forced_expert_indices=forced_indices, forced_expert_probs=forced_probs)

    assert output.shape == x.shape
    expected_counts = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(moe.tokens_per_expert.cpu(), expected_counts)


def test_moe_forced_routing_reports_logprobs_without_router_forward():
    moe_args = MoEArgs(
        num_experts=4,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=False,
        route_scale=1.0,
        score_before_experts=False,
        top_k=2,
        use_grouped_mm=False,
        load_balance_coeff=None,
    )
    moe = MoE(moe_args, dim=2, hidden_dim=4)

    def _router_forward(*args, **kwargs):
        raise AssertionError("router.forward should not be called when forcing routing")

    moe.router.forward = _router_forward
    torch.nn.init.zeros_(moe.router.gate.weight)

    x = torch.randn(1, 2, 2)
    forced_indices = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
    forced_probs = torch.ones_like(forced_indices, dtype=torch.float32)

    output = moe(
        x,
        forced_expert_indices=forced_indices,
        forced_expert_probs=forced_probs,
        return_router_logprobs=True,
    )

    assert output.shape == x.shape
    assert moe.last_router_logprobs is not None
    expected = torch.full((1, 2), 2 * torch.log(torch.tensor(1.0 / moe_args.num_experts)))
    assert torch.allclose(moe.last_router_logprobs.cpu(), expected)


if __name__ == "__main__":
    pytest.main([__file__])
