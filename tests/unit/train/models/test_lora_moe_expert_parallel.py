"""LoRA-wrapped grouped experts under expert parallelism.

The token dispatcher permutes and pads tokens per local expert before calling the
experts, and un-permutes afterwards. The LoRA wrappers must consume that input as-is,
exactly like ``GroupedGemmExpertCompute`` does, whether or not the expert weights are
sharded ``DTensor``s. Regression for #3936.
"""

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from prime_rl.trainer.distributed.expert_parallel import ExpertWeightParallel
from prime_rl.trainer.distributed.token_dispatcher import permute_for_grouped_gemm, unpermute_from_grouped_gemm
from prime_rl.trainer.models.layers.lora.base import set_lora_num_tokens, set_multilora_scaling
from prime_rl.trainer.models.layers.lora.multi_moe import (
    MultiLoRAGptOssGroupedExperts,
    MultiLoRAGroupedExperts,
    MultiLoRANonGatedGroupedExperts,
)
from prime_rl.trainer.models.layers.moe import GroupedExperts

DIM = 16
HIDDEN_DIM = 32
NUM_EXPERTS = 4
RANK = 8


@pytest.fixture
def ep_mesh():
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
    finally:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def single_adapter():
    set_lora_num_tokens(torch.tensor([1]), reset_reference=True)
    set_multilora_scaling(torch.tensor([1.0]), reset_reference=True)


def _build(wrapper: type[nn.Module], expert_type: str, bias: bool, activation: str) -> nn.Module:
    experts = GroupedExperts(
        dim=DIM,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        expert_type=expert_type,
        activation=activation,
        bias=bias,
    )
    experts.init_weights(0.02)
    if bias:
        for name in ("gate_proj_bias", "up_proj_bias", "down_proj_bias"):
            if getattr(experts, name) is not None:
                nn.init.normal_(getattr(experts, name), std=0.02)
    lora = wrapper(base_layer=experts, rank=RANK, n_adapters=1, alpha=32.0)
    lora.reset_parameters()
    # reset_parameters zeroes every B; perturb them so the adapter path contributes.
    for name, parameter in lora.named_parameters_for_adapter(0):
        if name.endswith("_B"):
            nn.init.normal_(parameter, std=0.02)
    return lora


@pytest.mark.parametrize(
    ("wrapper", "expert_type", "bias", "activation"),
    [
        (MultiLoRAGroupedExperts, "gated", False, "silu"),
        (MultiLoRANonGatedGroupedExperts, "non_gated", False, "relu2"),
        (MultiLoRAGptOssGroupedExperts, "gated", True, "clamped_swiglu"),
    ],
    ids=["gated", "non_gated", "gpt_oss"],
)
def test_lora_experts_sharded_match_unsharded(ep_mesh, wrapper, expert_type, bias, activation):
    torch.manual_seed(0)
    reference = _build(wrapper, expert_type, bias, activation)
    sharded = copy.deepcopy(reference)
    parallelize_module(sharded, device_mesh=ep_mesh, parallelize_plan=ExpertWeightParallel())
    assert isinstance(sharded.base_layer.up_proj, DTensor)

    # Tokens exactly as the dispatcher hands them to the experts: expert-sorted and padded.
    x = torch.randn(11, DIM)
    num_tokens_per_expert = torch.tensor([3, 0, 6, 2])
    routed, padded_counts, permutation = permute_for_grouped_gemm(
        x,
        num_tokens_per_expert,
        experts_per_rank=NUM_EXPERTS,
        num_ranks=1,
        alignment=reference.base_layer.token_group_alignment,
    )

    expected = unpermute_from_grouped_gemm(reference(routed, padded_counts), permutation)
    actual = unpermute_from_grouped_gemm(sharded(routed, padded_counts), permutation)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert expected.abs().sum() > 0
