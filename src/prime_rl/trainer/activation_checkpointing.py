"""Whole-block activation checkpointing with an operator-based policy."""

import re
from collections.abc import Callable, Iterable
from fnmatch import fnmatchcase
from functools import partial

import torch
import torch.fx.traceback as fx_traceback
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
    CheckpointImpl,
    checkpoint_wrapper,
)
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
        "prime_kernels::indexed_attention_plan",
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
        "prime_rl::fp8_blockwise_bmm",
        "prime_rl::fp8_blockwise_mm",
        "prime_rl::mega_moe_forward",
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

# A module-qualified target, "<operation or namespace>@<module path pattern>", matches only inside
# the modules whose path relative to the checkpointed block matches the pattern.
MODULE_TARGET_SEPARATOR = "@"
# Dynamo names a module by its source, e.g. L['self'].__dict__['_modules']['_checkpoint_wrapped_module'].mlp.
_DYNAMO_MODULE_SUBSCRIPT = re.compile(r"(?:\.__dict__\['_modules'\])?\['([^']+)'\]")


def _split_targets(targets: Iterable[str]) -> tuple[frozenset[str], tuple[tuple[str, str], ...]]:
    """Split targets into plain operation targets and (operation target, module path pattern) pairs."""
    operation_targets = set()
    module_targets = []
    for target in targets:
        operation_target, separator, module_pattern = target.partition(MODULE_TARGET_SEPARATOR)
        if separator:
            module_targets.append((operation_target, module_pattern))
        else:
            operation_targets.add(target)
    return frozenset(operation_targets), tuple(module_targets)


def _matches_operation(operation: torch._ops.OpOverload | torch._ops.HigherOrderOperator, targets) -> bool:
    return operation.namespace in targets or operation.name() in targets


def _traced_block_module_paths() -> list[str]:
    """Return the block-relative paths of the modules enclosing the operation that torch.compile is tracing."""
    if not fx_traceback.has_preserved_node_meta():
        raise RuntimeError(
            "Module-qualified activation checkpointing targets need the checkpointed blocks to run under "
            "torch.compile without graph breaks, which records the module of each traced operation."
        )
    paths = []
    for source_name, _module_class in fx_traceback.get_current_meta().get("nn_module_stack", {}).values():
        _, found, path = _DYNAMO_MODULE_SUBSCRIPT.sub(r".\1", source_name).rpartition(_CHECKPOINT_PREFIX)
        if found:
            paths.append(path)
    return paths


def _matches_module_target(
    operation: torch._ops.OpOverload | torch._ops.HigherOrderOperator,
    module_targets: tuple[tuple[str, str], ...],
) -> bool:
    module_paths = _traced_block_module_paths()
    return any(
        _matches_operation(operation, (operation_target,)) and any(fnmatchcase(path, pattern) for path in module_paths)
        for operation_target, pattern in module_targets
    )


def unmatched_module_patterns(targets: Iterable[str], blocks: Iterable[nn.Module]) -> list[str]:
    """Return the module path patterns in targets that match no submodule of any block."""
    _, module_targets = _split_targets(targets)
    paths = {path for block in blocks for path, _ in block.named_modules() if path}
    return [pattern for _, pattern in module_targets if not any(fnmatchcase(path, pattern) for path in paths)]


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
    module_targets: tuple[tuple[str, str], ...] = (),
    **kwargs,
) -> CheckpointPolicy:
    runtime_policy = _mandatory_checkpoint_policy(context, operation, *args, **kwargs)
    if runtime_policy is CheckpointPolicy.MUST_SAVE:
        return runtime_policy
    if _matches_operation(operation, targets):
        return CheckpointPolicy.MUST_SAVE
    if module_targets and _matches_module_target(operation, module_targets):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def get_activation_checkpoint_wrapper(config: ActivationCheckpointConfig) -> Callable[[nn.Module], nn.Module]:
    if config.mode == "full":
        policy = _mandatory_checkpoint_policy
    else:
        targets, module_targets = _split_targets(
            DEFAULT_SELECTIVE_TARGETS if config.targets is None else config.targets
        )
        policy = partial(_selective_checkpoint_policy, targets=targets, module_targets=module_targets)

    return partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=partial(create_selective_checkpoint_contexts, policy),
    )


__all__ = [
    "DEFAULT_SELECTIVE_SAVE_NAMESPACES",
    "DEFAULT_SELECTIVE_SAVE_OPERATIONS",
    "DEFAULT_SELECTIVE_TARGETS",
    "MANDATORY_SAVE_NAMESPACES",
    "MANDATORY_SAVE_OPERATIONS",
    "get_activation_checkpoint_wrapper",
    "unmatched_module_patterns",
]
