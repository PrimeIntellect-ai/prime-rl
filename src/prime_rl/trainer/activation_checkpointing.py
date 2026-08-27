"""Whole-block activation checkpointing with PrimeRL's fixed operator policy."""

from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.utils.checkpoint import (
    CheckpointPolicy,
    SelectiveCheckpointContext,
    create_selective_checkpoint_contexts,
)

from prime_rl.configs.trainer import ActivationCheckpointConfig

# Adapted from TorchTitan's whole-block selective activation checkpointing policy.
_NON_REPLAYABLE_SAVE_NAMESPACES = frozenset({"deepep"})
_SELECTIVE_SAVE_NAMESPACES = frozenset(
    {
        "_c10d_functional",
        "flash_attn",
        "flash_attn_3",
        "prime_rl_collectives",
        "prime_rl_ring",
    }
)
_ALWAYS_SAVE_OPERATIONS = frozenset(
    {
        "aten::topk",
        "prime_rl::record_moe_routing",
    }
)
_SELECTIVE_SAVE_OPERATIONS = frozenset(
    {
        "aten::_efficient_attention_forward",
        "aten::_flash_attention_forward",
        "aten::_scaled_dot_product_attention_math",
        "aten::_scaled_dot_product_cudnn_attention",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_flash_attention_for_cpu",
        "aten::_scaled_dot_product_fused_attention_overrideable",
        "aten::_scaled_grouped_mm",
        "aten::_scaled_mm",
        "aten::_scaled_mm_v2",
        "aten::_grouped_mm",
        "aten::addmm",
        "aten::bmm",
        "aten::convolution",
        "aten::linear",
        "aten::mm",
        "prime_rl::fp8_blockwise_mm",
        "prime_rl::grouped_fp8_gemm",
        "prime_rl::sparse_mla",
    }
)


def _full_checkpoint_policy(
    _context: SelectiveCheckpointContext,
    operation: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
    *args,
    **kwargs,
) -> CheckpointPolicy:
    if operation.namespace in _NON_REPLAYABLE_SAVE_NAMESPACES or operation.name() in _ALWAYS_SAVE_OPERATIONS:
        return CheckpointPolicy.MUST_SAVE

    if operation.name() == "aten::_to_copy":
        device = kwargs.get("device")
        if isinstance(device, torch.device) and device.type == "cpu":
            return CheckpointPolicy.MUST_SAVE

    return CheckpointPolicy.PREFER_RECOMPUTE


def _selective_checkpoint_policy(
    context: SelectiveCheckpointContext,
    operation: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
    *args,
    **kwargs,
) -> CheckpointPolicy:
    runtime_policy = _full_checkpoint_policy(context, operation, *args, **kwargs)
    if runtime_policy is CheckpointPolicy.MUST_SAVE:
        return runtime_policy
    if operation.namespace in _SELECTIVE_SAVE_NAMESPACES or operation.name() in _SELECTIVE_SAVE_OPERATIONS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def get_activation_checkpoint_wrapper(config: ActivationCheckpointConfig) -> Callable[[nn.Module], nn.Module]:
    policy = _selective_checkpoint_policy if config.mode == "selective" else _full_checkpoint_policy
    return partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=config.preserve_rng_state,
        context_fn=partial(create_selective_checkpoint_contexts, policy),
    )


__all__ = ["get_activation_checkpoint_wrapper"]
