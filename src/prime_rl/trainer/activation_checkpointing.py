"""Whole-block activation checkpointing with an operator-based policy."""

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
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
        "prime_kernels::select_indexed_blocks",
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
        "aten::_grouped_mm",
        "aten::addmm",
        "aten::bmm",
        "aten::convolution",
        "aten::linear",
        "aten::mm",
        "prime_rl::dsv4_sparse_attn",
        "prime_rl::fp8_blockwise_mm",
        "prime_rl::grouped_fp8_gemm",
        "prime_rl::sparse_mla",
        "quack::gemm_act_out",
        "quack::gemm_gated_out",
        "quack::gemm_out",
        "prime_kernels::indexed_attention_forward",
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


def get_activation_checkpoint_wrapper(config: ActivationCheckpointConfig) -> Callable[[nn.Module], nn.Module]:
    if config.mode == "full":
        policy = _mandatory_checkpoint_policy
    else:
        targets = DEFAULT_SELECTIVE_TARGETS if config.targets is None else frozenset(config.targets)
        policy = partial(_selective_checkpoint_policy, targets=targets)

    return partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=partial(_checkpoint_contexts, policy),
    )


def _checkpoint_contexts(
    policy: Callable[..., CheckpointPolicy],
) -> tuple[AbstractContextManager, AbstractContextManager]:
    forward_context, recompute_context = create_selective_checkpoint_contexts(policy)
    if torch.compiler.is_compiling():
        return forward_context, recompute_context
    # Dynamo settings are thread-local, and backward recomputes on the autograd thread. A compiled block under
    # this eager checkpoint recompiles there, so it must use the forward's settings: with the defaults it
    # specializes on every new int and falls back to eager after 8 recompiles, which breaks the replay.
    dynamo_settings = {
        "recompile_limit": torch._dynamo.config.recompile_limit,
        "capture_scalar_outputs": torch._dynamo.config.capture_scalar_outputs,
    }
    return forward_context, _with_dynamo_settings(
        recompute_context, dynamo_settings, torch.compiler.config.dynamic_sources
    )


@contextmanager
def _with_dynamo_settings(context: AbstractContextManager, dynamo_settings: dict, dynamic_sources: str):
    with (
        torch._dynamo.config.patch(dynamo_settings),
        torch.compiler.config.patch(dynamic_sources=dynamic_sources),
        context,
    ):
        yield


__all__ = [
    "DEFAULT_SELECTIVE_SAVE_NAMESPACES",
    "DEFAULT_SELECTIVE_SAVE_OPERATIONS",
    "DEFAULT_SELECTIVE_TARGETS",
    "MANDATORY_SAVE_NAMESPACES",
    "MANDATORY_SAVE_OPERATIONS",
    "get_activation_checkpoint_wrapper",
]
