import pytest
import torch
from torch import nn

import prime_rl.inference.vllm.worker.weight_transfer as weight_transfer
from prime_rl.inference.vllm.worker.weight_transfer import load_weights_kernel


def _moe_model() -> nn.Module:
    """Module tree mirroring vLLM 0.24's nested MoERunner -> RoutedExperts params."""
    model = nn.Module()
    model.embed = nn.Module()
    model.embed.weight = nn.Parameter(torch.zeros(4))
    model.mlp = nn.Module()
    model.mlp.experts = nn.Module()
    model.mlp.experts.routed_experts = nn.Module()
    model.mlp.experts.routed_experts.w13_weight = nn.Parameter(torch.zeros(2, 3))
    model.mlp.experts.routed_experts.w2_weight = nn.Parameter(torch.zeros(2, 3))
    return model


def test_load_weights_kernel_remaps_flat_expert_names():
    """The trainer emits pre-0.24 flat kernel names (``...experts.w13_weight``);
    they must land on the nested ``...experts.routed_experts.*`` params."""
    model = _moe_model()
    state = [
        ("mlp.experts.w13_weight", torch.ones(2, 3)),
        ("mlp.experts.w2_weight", torch.full((2, 3), 2.0)),
        ("embed.weight", torch.arange(4.0)),
    ]

    load_weights_kernel(model, iter(state))

    assert model.mlp.experts.routed_experts.w13_weight.eq(1.0).all()
    assert model.mlp.experts.routed_experts.w2_weight.eq(2.0).all()
    assert model.embed.weight.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_load_weights_kernel_slices_experts_with_expert_map(monkeypatch):
    """Expert-map keys are MoERunner module names; they must prefix-match the
    remapped nested param names so EP workers slice their local experts."""
    model = _moe_model()
    monkeypatch.setattr(
        weight_transfer,
        "build_expert_map",
        lambda _model: {"mlp.experts": torch.tensor([2, 0])},
    )
    full_w13 = torch.arange(4.0)[:, None].expand(4, 3).contiguous()

    load_weights_kernel(model, iter([("mlp.experts.w13_weight", full_w13)]))

    torch.testing.assert_close(model.mlp.experts.routed_experts.w13_weight.data, full_w13[[2, 0]])


def test_load_weights_kernel_raises_on_unknown_names():
    model = _moe_model()
    with pytest.raises(ValueError, match="skipped"):
        load_weights_kernel(model, iter([("nonexistent.weight", torch.zeros(1))]))
