"""Whole-block activation checkpointing with an operator-based policy."""

from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.utils.checkpoint import (
    SAC_IGNORED_OPS,
    CheckpointPolicy,
    SelectiveCheckpointContext,
    create_selective_checkpoint_contexts,
)

from prime_rl.configs.trainer import ActivationCheckpointConfig

# FSDP's replay-specific hooks emit a different number of profiler markers.
SAC_IGNORED_OPS.update(
    {
        torch.ops.profiler._record_function_enter.default,
        torch.ops.profiler._record_function_enter_new.default,
        torch.ops.profiler._record_function_exit.default,
        torch.ops.profiler._record_function_exit._RecordFunction,
    }
)

# Adapted from TorchTitan's whole-block selective activation checkpointing policy.
# These targets and the CUDA-to-CPU copy rule are correctness requirements. They
# must remain active when custom targets replace the default selective targets.
MANDATORY_SAVE_NAMESPACES = frozenset({"deepep"})
MANDATORY_SAVE_OPERATIONS = frozenset(
    {
        "aten::topk",
        "prime_rl::fp8_indexer",
        "prime_rl::record_moe_routing_statistics",
    }
)

DEFAULT_SELECTIVE_SAVE_NAMESPACES = frozenset(
    {
        "_c10d_functional",
        "flash_attn",
        "flash_attn_3",
        "prime_rl_attn",
        "prime_rl_collectives",
        "prime_rl_ring",
    }
)
DEFAULT_SELECTIVE_SAVE_OPERATIONS = frozenset(
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
        "aten::addmm",
        "aten::bmm",
        "aten::convolution",
        "aten::linear",
        "aten::mm",
        "prime_rl::dsv4_sparse_attn",
        "prime_rl::fp8_blockwise_mm",
        "prime_rl::grouped_fp8_gemm",
        "prime_rl::sparse_mla",
    }
)
# An operation target matches one qualified operator name, while a namespace
# target matches every operation registered in that namespace.
DEFAULT_SELECTIVE_TARGETS = DEFAULT_SELECTIVE_SAVE_NAMESPACES | DEFAULT_SELECTIVE_SAVE_OPERATIONS


# PyTorch calls checkpoint policies with the dispatched operation's operands.
# Keyword operands let us distinguish CUDA-to-CPU copies from other _to_copy calls.
def _mandatory_checkpoint_policy(
    _context: SelectiveCheckpointContext,
    operation: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
    *args,
    **kwargs,
) -> CheckpointPolicy:
    if operation.namespace in MANDATORY_SAVE_NAMESPACES or operation.name() in MANDATORY_SAVE_OPERATIONS:
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
    targets: frozenset[str] = DEFAULT_SELECTIVE_TARGETS,
    **kwargs,
) -> CheckpointPolicy:
    runtime_policy = _mandatory_checkpoint_policy(context, operation, *args, **kwargs)
    if runtime_policy is CheckpointPolicy.MUST_SAVE:
        return runtime_policy
    if operation.namespace in targets or operation.name() in targets:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


# With the default targets, every second matmul is recomputed to balance memory and compute.
ALTERNATING_SAVE_OPERATIONS = frozenset({"aten::linear", "aten::mm"})


def _alternating_matmul_policy(targets: frozenset[str]) -> Callable[..., CheckpointPolicy]:
    """Build a per-checkpoint policy that saves only every second matmul among ``targets``."""
    matmul_counts = {False: 0, True: 0}

    def policy(
        context: SelectiveCheckpointContext,
        operation: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
        *args,
        **kwargs,
    ) -> CheckpointPolicy:
        decision = _selective_checkpoint_policy(context, operation, *args, targets=targets, **kwargs)
        if decision is CheckpointPolicy.MUST_SAVE and operation.name() in ALTERNATING_SAVE_OPERATIONS:
            matmul_counts[context.is_recompute] += 1
            if (matmul_counts[context.is_recompute] & 1) == 0:
                return CheckpointPolicy.PREFER_RECOMPUTE
        return decision

    return policy


def get_activation_checkpoint_wrapper(config: ActivationCheckpointConfig) -> Callable[[nn.Module], nn.Module]:
    if config.mode == "full":
        context_fn = partial(create_selective_checkpoint_contexts, _mandatory_checkpoint_policy)
    elif config.targets is None:
        # The matmul counter must restart for every checkpointed call, so build the policy per call.
        def context_fn():
            return create_selective_checkpoint_contexts(_alternating_matmul_policy(DEFAULT_SELECTIVE_TARGETS))
    else:
        context_fn = partial(
            create_selective_checkpoint_contexts,
            partial(_selective_checkpoint_policy, targets=frozenset(config.targets)),
        )

    return partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=context_fn,
    )


__all__ = [
    "ALTERNATING_SAVE_OPERATIONS",
    "DEFAULT_SELECTIVE_SAVE_NAMESPACES",
    "DEFAULT_SELECTIVE_SAVE_OPERATIONS",
    "DEFAULT_SELECTIVE_TARGETS",
    "MANDATORY_SAVE_NAMESPACES",
    "MANDATORY_SAVE_OPERATIONS",
    "get_activation_checkpoint_wrapper",
]
