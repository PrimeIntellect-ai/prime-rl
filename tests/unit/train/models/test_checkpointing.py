from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import CheckpointPolicy, SelectiveCheckpointContext

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.activation_checkpointing import (
    _full_checkpoint_policy,
    _selective_checkpoint_policy,
    get_activation_checkpoint_wrapper,
)
from prime_rl.trainer.models.layers.moe import MoE, TokenChoiceTopKRouter


class DropoutBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(x)).square()


class IdentityExperts(nn.Module):
    num_experts = 2
    token_group_alignment = 1

    def forward(self, x: torch.Tensor, _token_counts: torch.Tensor) -> torch.Tensor:
        return x


def test_whole_block_checkpoint_preserves_rng_and_gradients():
    reference = DropoutBlock()
    checkpointed = deepcopy(reference)
    state_dict_keys = tuple(checkpointed.state_dict())
    checkpointed = get_activation_checkpoint_wrapper(
        ActivationCheckpointConfig(mode="selective", preserve_rng_state=True)
    )(checkpointed)
    assert tuple(checkpointed.state_dict()) == state_dict_keys
    reference_input = torch.randn(4, 8, requires_grad=True)
    checkpointed_input = reference_input.detach().clone().requires_grad_()

    torch.manual_seed(1234)
    reference_output = reference(reference_input)
    reference_output.sum().backward()
    torch.manual_seed(1234)
    checkpointed_output = checkpointed(checkpointed_input)
    checkpointed_output.sum().backward()

    torch.testing.assert_close(checkpointed_output, reference_output)
    torch.testing.assert_close(checkpointed_input.grad, reference_input.grad)
    for expected, actual in zip(reference.parameters(), checkpointed.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_selective_policy_saves_routing_and_expensive_ops():
    context = SelectiveCheckpointContext(is_recompute=False)

    assert _selective_checkpoint_policy(context, torch.ops.aten.topk.default) is CheckpointPolicy.MUST_SAVE
    assert (
        _selective_checkpoint_policy(context, torch.ops.aten._scaled_dot_product_flash_attention.default)
        is CheckpointPolicy.MUST_SAVE
    )
    assert _selective_checkpoint_policy(context, torch.ops.aten.mm.default) is CheckpointPolicy.MUST_SAVE
    assert _selective_checkpoint_policy(context, torch.ops.aten.addmm.default) is CheckpointPolicy.MUST_SAVE
    assert _selective_checkpoint_policy(context, torch.ops.aten.bmm.default) is CheckpointPolicy.MUST_SAVE
    assert _selective_checkpoint_policy(context, torch.ops.aten._grouped_mm.default) is CheckpointPolicy.MUST_SAVE
    assert (
        _selective_checkpoint_policy(context, torch.ops.prime_rl_collectives.all_to_all_single_equal.default)
        is CheckpointPolicy.MUST_SAVE
    )
    assert (
        _selective_checkpoint_policy(context, torch.ops.prime_rl_collectives.mxfp8_all_to_all.default)
        is CheckpointPolicy.MUST_SAVE
    )
    assert _selective_checkpoint_policy(context, torch.ops.aten.silu.default) is CheckpointPolicy.PREFER_RECOMPUTE


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_checkpoint_records_moe_routing_once(mode):
    moe = MoE(
        router=TokenChoiceTopKRouter(
            dim=4,
            num_experts=2,
            top_k=1,
            score_func="softmax",
            route_norm=False,
            route_scale=1.0,
        ),
        experts=IdentityExperts(),
        shared_expert=None,
        score_before_experts=True,
        load_balance_coeff=0.1,
    )
    checkpointed = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode=mode))(moe)
    hidden_states = torch.randn(1, 3, 4, requires_grad=True)

    output = checkpointed(hidden_states)
    tokens_after_forward = moe.tokens_per_expert.clone()
    confidence_after_forward = moe.routing_confidence_sum.clone()
    output.sum().backward()

    assert tokens_after_forward.sum() == 3
    torch.testing.assert_close(moe.tokens_per_expert, tokens_after_forward)
    torch.testing.assert_close(moe.routing_confidence_sum, confidence_after_forward)
    assert hidden_states.grad is not None


def test_full_policy_only_retains_non_replayable_ops():
    context = SelectiveCheckpointContext(is_recompute=False)

    assert _full_checkpoint_policy(context, torch.ops.aten.topk.default) is CheckpointPolicy.MUST_SAVE
    assert _full_checkpoint_policy(context, torch.ops.prime_rl.record_moe_routing.default) is CheckpointPolicy.MUST_SAVE
    assert (
        _full_checkpoint_policy(context, torch.ops.prime_rl_collectives.all_to_all_single_equal.default)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )
    assert (
        _full_checkpoint_policy(context, torch.ops.prime_rl_collectives.mxfp8_all_to_all.default)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )
    assert (
        _full_checkpoint_policy(context, torch.ops.aten._scaled_dot_product_flash_attention.default)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )
