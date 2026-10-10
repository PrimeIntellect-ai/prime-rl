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
        "prime_rl::dsv41_index_topk",
        "prime_rl::deepep_moe",
        "prime_rl::mega_moe",
        "prime_rl::mega_moe_fp8",
        "prime_kernels::dsv41_index_topk",
        "prime_kernels::select_indexed_blocks",
        "prime_rl::record_moe_routing_statistics",
    }
)

# Mandatory saves that `mode = "full_moe"` recomputes instead: the FP8 Mega MoE re-runs its dispatch, expert
# GEMMs and combine in backward. Its collectives stay matched because every expert-parallel rank checkpoints
# the same layers in the same order.
REPLAYABLE_EXPERT_OPERATIONS = frozenset({"prime_rl::mega_moe_fp8"})

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
        "prime_rl::dsv41_sparse_attn",
        "prime_rl::dsv41_sparse_attn_rope",
        "prime_rl::dsv4_linear_rope",
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
# Fused operations that run in place of a target operation: naming the target saves them too.
FUSED_OPERATIONS = {
    "prime_rl::dsv41_sparse_attn": frozenset({"prime_rl::dsv41_sparse_attn_rope"}),
    # The shared expert's gate and up projections as one GEMM.
    "prime_rl::fp8_blockwise_mm": frozenset({"prime_rl::fp8_gate_up_mm"}),
}


# The levels between `full` and `selective`, each retaining more than the one before (see the
# `mode` docstring). `projections`: the projection outputs, except the attention query.
PROJECTION_TARGETS = frozenset(
    {
        "_c10d_functional",
        "prime_rl_collectives",
        "aten::_grouped_mm",
        "aten::_scaled_grouped_mm",
        "aten::_scaled_mm",
        "aten::_scaled_mm_v2",
        "aten::addmm",
        "aten::bmm",
        "aten::linear",
        "aten::mm",
        "prime_kernels::mhc_gates_forward",
        "prime_rl::fp8_blockwise_mm",
        "prime_rl::grouped_fp8_gemm",
        "prime_rl::grouped_linear",
        "quack::gemm_act_out",
        "quack::gemm_gated_out",
        "quack::gemm_out",
    }
)
# `attention`: also the attention outputs and the mHC collapses.
ATTENTION_TARGETS = PROJECTION_TARGETS | frozenset(
    {
        "flash_attn",
        "flash_attn_3",
        "prime_kernels::indexed_attention_forward",
        "prime_kernels::mhc_projection_forward",
        "prime_kernels::mhc_update_projection_forward",
        "prime_rl::dsv4_sparse_attn",
        "prime_rl::dsv41_sparse_attn",
        "prime_rl::sparse_mla",
        "prime_rl_attn",
        "prime_rl_ring",
    }
)
# `matmul`: also the query (its projection fused with RoPE), so recompute runs no matmul.
MATMUL_TARGETS = ATTENTION_TARGETS | frozenset({"prime_rl::dsv4_linear_rope", "prime_rl::dsv4_q_norm_rope"})
MODE_TARGETS = {"projections": PROJECTION_TARGETS, "attention": ATTENTION_TARGETS, "matmul": MATMUL_TARGETS}


# PyTorch calls checkpoint policies with the dispatched operation's operands.
# Keyword operands let us distinguish CUDA-to-CPU copies from other _to_copy calls.
def _mandatory_checkpoint_policy(
    _context: SelectiveCheckpointContext,
    operation: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
    *args,
    recompute_experts: bool = False,
    **kwargs,
) -> CheckpointPolicy:
    if recompute_experts and operation.name() in REPLAYABLE_EXPERT_OPERATIONS:
        return CheckpointPolicy.PREFER_RECOMPUTE
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


def get_layer_modes(
    config: ActivationCheckpointConfig,
    layer_names: list[str],
    num_layers: int,
    parts: list[str | None] | None = None,
) -> list[str]:
    """The checkpointing mode of each named decoder layer (`none` for an unchecked one).

    `layer_names` are the layers' indices in the full model, as a pipeline stage keeps them; `freq`
    counts the layers this rank holds. `parts` is the half each layer keeps when a pipeline stage boundary
    cuts it (`"attention"` or `"moe"`, else `None`), which picks the half's mode from a pair."""
    if config.layer_modes is None:
        return [config.mode if position % config.freq == 0 else "none" for position in range(len(layer_names))]
    if len(config.layer_modes) != num_layers:
        raise ValueError(f"model.ac.layer_modes has {len(config.layer_modes)} entries for {num_layers} decoder layers")
    parts = parts or [None] * len(layer_names)
    modes = []
    for name, part in zip(layer_names, parts):
        mode = config.layer_modes[int(name)]
        if isinstance(mode, str):
            modes.append(mode)
        elif part is None:
            raise ValueError(f"model.ac.layer_modes gives layer {name} a mode per half, but no stage boundary cuts it")
        else:
            modes.append(mode[0] if part == "attention" else mode[1])
    return modes


def get_checkpoint_context_fn(config: ActivationCheckpointConfig, mode: str | None = None) -> Callable:
    """`torch.utils.checkpoint`'s `context_fn` for `mode` (default: the configured mode)."""
    mode = config.mode if mode is None else mode
    if mode in ("full", "full_moe"):
        policy = partial(_mandatory_checkpoint_policy, recompute_experts=mode == "full_moe")
    else:
        if mode in MODE_TARGETS:
            targets = MODE_TARGETS[mode]
        else:
            targets = DEFAULT_SELECTIVE_TARGETS if config.targets is None else frozenset(config.targets)
        targets = targets.union(*(FUSED_OPERATIONS.get(target, frozenset()) for target in targets))
        policy = partial(_selective_checkpoint_policy, targets=targets)
    return partial(create_selective_checkpoint_contexts, policy)


def get_activation_checkpoint_wrapper(
    config: ActivationCheckpointConfig, mode: str | None = None
) -> Callable[[nn.Module], nn.Module]:
    return partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=get_checkpoint_context_fn(config, mode),
    )


__all__ = [
    "DEFAULT_SELECTIVE_SAVE_NAMESPACES",
    "DEFAULT_SELECTIVE_SAVE_OPERATIONS",
    "DEFAULT_SELECTIVE_TARGETS",
    "MANDATORY_SAVE_NAMESPACES",
    "MANDATORY_SAVE_OPERATIONS",
    "MODE_TARGETS",
    "REPLAYABLE_EXPERT_OPERATIONS",
    "get_activation_checkpoint_wrapper",
    "get_checkpoint_context_fn",
    "get_layer_modes",
]
