import pytest
import torch
from torch import nn
from torch.utils.checkpoint import CheckpointPolicy, SelectiveCheckpointContext

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.activation_checkpointing import (
    TokenChunkedCheckpointWrapper,
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


class TokenLocalBlock(nn.Module):
    is_token_local = True

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.embed = nn.Embedding(4, 8)

    def forward(self, hidden_states: torch.Tensor, scale: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return hidden_states + scale * torch.tanh(self.proj(hidden_states)) + self.embed(token_ids)


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
    assert (
        _selective_checkpoint_policy(context, torch.ops.prime_rl.fp8_indexer.default, targets=custom_targets)
        is CheckpointPolicy.MUST_SAVE
    )


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


def test_mandatory_policy_only_retains_non_replayable_ops():
    context = SelectiveCheckpointContext(is_recompute=False)

    assert _mandatory_checkpoint_policy(context, torch.ops.aten.topk.default) is CheckpointPolicy.MUST_SAVE
    assert _mandatory_checkpoint_policy(context, torch.ops.prime_rl.fp8_indexer.default) is CheckpointPolicy.MUST_SAVE
    assert (
        _mandatory_checkpoint_policy(context, torch.ops.prime_rl.record_moe_routing_statistics.default)
        is CheckpointPolicy.MUST_SAVE
    )
    assert (
        _mandatory_checkpoint_policy(context, torch.ops.prime_rl_collectives.all_to_all_counts.default)
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


@pytest.mark.parametrize("seq_len", [3, 4, 5, 8, 9, 10, 13, 16, 17, 31])
def test_token_chunked_checkpoint_matches_whole_block(seq_len):
    torch.manual_seed(0)
    block = TokenLocalBlock().double()
    whole = get_activation_checkpoint_wrapper(ActivationCheckpointConfig())(block)
    chunked = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(token_chunk_size=4))(block)
    assert isinstance(chunked, TokenChunkedCheckpointWrapper)
    assert not isinstance(whole, TokenChunkedCheckpointWrapper)

    hidden_states = torch.randn(2, seq_len, 8, dtype=torch.float64)
    scale = torch.randn(1, dtype=torch.float64)
    token_ids = torch.randint(0, 4, (2, seq_len))

    def run(wrapper):
        inputs = hidden_states.clone().requires_grad_()
        output = wrapper(inputs, scale, token_ids=token_ids)
        output.square().sum().backward()
        param_grads = [param.grad.clone() for param in block.parameters()]
        block.zero_grad()
        return output.detach(), inputs.grad, param_grads

    output, input_grad, param_grads = run(whole)
    chunked_output, chunked_input_grad, chunked_param_grads = run(chunked)

    torch.testing.assert_close(chunked_output, output)
    torch.testing.assert_close(chunked_input_grad, input_grad)
    for chunked_grad, grad in zip(chunked_param_grads, param_grads, strict=True):
        torch.testing.assert_close(chunked_grad, grad)
