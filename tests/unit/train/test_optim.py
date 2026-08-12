import torch
import torch.nn as nn

from prime_rl.configs.trainer import AdamWConfig
from prime_rl.trainer.models.layers.moe import NonGatedGroupedExperts
from prime_rl.trainer.optim import _create_optimizer


class _DummyW3Model(nn.Module):
    """A model with a zero-element placeholder param, like the dummy expert w3
    that NonGatedGroupedExperts registers for the @expert_parallel signature."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.w3 = nn.Parameter(torch.empty(0))


def _optimizer_params(optimizer) -> list[torch.Tensor]:
    return [p for group in optimizer.param_groups for p in group["params"]]


def test_zero_numel_params_stay_out_of_the_optimizer():
    """A param that can never step must not enter param groups: the checkpoint
    save has no optimizer state for it, while DCP load materializes state for
    every optimizer param, so resume fails with a missing key."""
    model = _DummyW3Model()
    optimizer = _create_optimizer(AdamWConfig(), list(model.named_parameters()), parallel_dims=None)
    params = _optimizer_params(optimizer)
    assert model.linear.weight in set(params)
    assert not any(p.numel() == 0 for p in params)


def test_frozen_params_stay_out_of_the_optimizer():
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    model[0].weight.requires_grad_(False)
    optimizer = _create_optimizer(AdamWConfig(), list(model.named_parameters()), parallel_dims=None)
    assert _optimizer_params(optimizer) == [model[1].weight]


def test_nongated_experts_dummy_w3_is_frozen():
    """The dummy w3 itself must be untrainable, so every optimizer path (incl.
    Muon's internal grouping) excludes it without special-casing."""
    experts = NonGatedGroupedExperts(input_dim=8, intermediate_dim=16, num_experts=2, use_grouped_mm=False)
    assert experts.w3.numel() == 0
    assert not experts.w3.requires_grad
