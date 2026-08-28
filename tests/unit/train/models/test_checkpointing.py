import torch
from torch import nn
from torch.utils.checkpoint import CheckpointPolicy, SelectiveCheckpointContext

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.activation_checkpointing import (
    _mandatory_checkpoint_policy,
    _selective_checkpoint_policy,
    get_activation_checkpoint_wrapper,
)
from prime_rl.trainer.models.layers.moe import MoE, TokenChoiceTopKRouter


class IdentityExperts(nn.Module):
    num_experts = 2
    token_group_alignment = 1

    def forward(self, x: torch.Tensor, _token_counts: torch.Tensor) -> torch.Tensor:
        return x


def test_selective_policy_saves_default_and_custom_targets():
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

    custom_targets = frozenset({"aten::silu", "prime_rl_collectives"})
    assert (
        _selective_checkpoint_policy(context, torch.ops.aten.silu.default, targets=custom_targets)
        is CheckpointPolicy.MUST_SAVE
    )
    assert (
        _selective_checkpoint_policy(
            context,
            torch.ops.prime_rl_collectives.all_to_all_single_equal.default,
            targets=custom_targets,
        )
        is CheckpointPolicy.MUST_SAVE
    )
    assert (
        _selective_checkpoint_policy(context, torch.ops.aten.mm.default, targets=custom_targets)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )
    assert (
        _selective_checkpoint_policy(context, torch.ops.aten.topk.default, targets=custom_targets)
        is CheckpointPolicy.MUST_SAVE
    )


def test_selective_checkpoint_records_moe_routing_once():
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
    checkpointed = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode="selective"))(moe)
    hidden_states = torch.randn(1, 3, 4, requires_grad=True)

    output = checkpointed(hidden_states)
    tokens_after_forward = moe.tokens_per_expert.clone()
    confidence_after_forward = moe.routing_confidence_sum.clone()
    output.sum().backward()

    assert tokens_after_forward.sum() == 3
    torch.testing.assert_close(moe.tokens_per_expert, tokens_after_forward)
    torch.testing.assert_close(moe.routing_confidence_sum, confidence_after_forward)
    assert hidden_states.grad is not None


def test_mandatory_policy_only_retains_non_replayable_ops():
    context = SelectiveCheckpointContext(is_recompute=False)

    assert _mandatory_checkpoint_policy(context, torch.ops.aten.topk.default) is CheckpointPolicy.MUST_SAVE
    assert (
        _mandatory_checkpoint_policy(context, torch.ops.prime_rl.record_moe_routing_statistics.default)
        is CheckpointPolicy.MUST_SAVE
    )
    assert (
        _mandatory_checkpoint_policy(context, torch.ops.prime_rl_collectives.all_to_all_single_equal.default)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )
    assert (
        _mandatory_checkpoint_policy(context, torch.ops.prime_rl_collectives.mxfp8_all_to_all.default)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )
    assert (
        _mandatory_checkpoint_policy(context, torch.ops.aten._scaled_dot_product_flash_attention.default)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )
