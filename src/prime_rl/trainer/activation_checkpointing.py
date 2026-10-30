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
    CheckpointWrapper,
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
MANDATORY_SAVE_OPERATIONS = frozenset(
    {
        "aten::topk",
        "prime_rl::fp8_indexer",
        "prime_rl::fp8_indexer_prequantized",
        "prime_kernels::select_indexed_blocks",
        "prime_kernels::indexed_attention_plan",
        "prime_rl::record_moe_routing_statistics",
        # The EP dispatch splits are a saved CUDA-to-CPU copy of these counts. Replayed under torch.compile, the
        # counts exchange would no longer feed the dispatch all-to-all, so inductor could order the two differently
        # on ranks with different sequence lengths, leaving the ranks in mismatched collectives.
        "prime_rl_collectives::all_to_all_counts",
    }
)

DEFAULT_SELECTIVE_SAVE_NAMESPACES = frozenset(
    {
        "_c10d_functional",
        "deepep",
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
    if operation.name() in MANDATORY_SAVE_OPERATIONS:
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


class TokenChunkedCheckpointWrapper(CheckpointWrapper):
    """Experimental. Checkpoint a block as one checkpoint per chunk of at most ``token_chunk_size`` tokens.

    The block's recompute and backward then hold one chunk's activations at a time instead of the
    whole sequence's. Only valid for blocks in which every op is independent per position, so the
    output at each token depends on that token alone: the hidden states and every tensor argument
    laid out like them (``[batch, seq, ...]``) are split along the sequence; other arguments go to
    every chunk as-is.

    The chunk loop always runs eagerly. ``compile_chunk`` compiles the checkpointed chunk call inside
    it; chunk lengths vary within and across rows, so the CUDA-graph compile modes would re-record often.
    """

    def __init__(self, mod: nn.Module, token_chunk_size: int, **checkpoint_kwargs) -> None:
        super().__init__(mod, **checkpoint_kwargs)
        self.token_chunk_size = token_chunk_size
        self._checkpoint_chunk = partial(self.checkpoint_fn, self._checkpoint_wrapped_module)

    def compile_chunk(self, **compile_kwargs) -> None:
        """Compile one checkpointed chunk. Use instead of ``nn.Module.compile``, which would compile the chunk
        loop: its bounds follow the row length, so torch.compile would recompile for every new length."""
        self._checkpoint_chunk = torch.compile(self._checkpoint_chunk, **compile_kwargs)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        if seq_len <= self.token_chunk_size:
            return self._checkpoint_chunk(hidden_states, *args, **kwargs)

        def is_token_aligned(value) -> bool:
            return isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[:2] == (batch_size, seq_len)

        # Equal chunks rather than full chunks plus a remainder, so no chunk is a 1-token tail: torch.compile
        # specializes on size 1, which costs a separate graph and gave wrong parameter gradients on torch 2.13.
        num_chunks = -(-seq_len // self.token_chunk_size)
        base, extra = divmod(seq_len, num_chunks)
        assert base > 1, f"1-token chunk: seq_len={seq_len}, token_chunk_size={self.token_chunk_size}"
        outputs = []
        start = 0
        for chunk in range(num_chunks):
            end = start + base + (chunk < extra)
            chunk_args = [arg[:, start:end] if is_token_aligned(arg) else arg for arg in args]
            chunk_kwargs = {k: v[:, start:end] if is_token_aligned(v) else v for k, v in kwargs.items()}
            outputs.append(self._checkpoint_chunk(hidden_states[:, start:end], *chunk_args, **chunk_kwargs))
            start = end
        return torch.cat(outputs, dim=1)


def get_activation_checkpoint_wrapper(config: ActivationCheckpointConfig) -> Callable[[nn.Module], nn.Module]:
    if config.mode == "full":
        policy = _mandatory_checkpoint_policy
    else:
        targets, module_targets = _split_targets(
            DEFAULT_SELECTIVE_TARGETS if config.targets is None else config.targets
        )
        policy = partial(_selective_checkpoint_policy, targets=targets, module_targets=module_targets)
    checkpoint_kwargs = {
        "checkpoint_impl": CheckpointImpl.NO_REENTRANT,
        "context_fn": partial(create_selective_checkpoint_contexts, policy),
        # Keep SAC-saved tensors out of surrounding saved-tensor hooks (activation offloading), as before
        # torch 2.14 made this explicit; leaving it unset warns whenever a hook is in scope.
        "respect_saved_tensors_hooks": False,
    }

    def wrap(block: nn.Module) -> nn.Module:
        if config.token_chunk_size is not None and getattr(block, "is_token_local", False):
            return TokenChunkedCheckpointWrapper(block, config.token_chunk_size, **checkpoint_kwargs)
        return checkpoint_wrapper(block, **checkpoint_kwargs)

    return wrap


__all__ = [
    "DEFAULT_SELECTIVE_SAVE_NAMESPACES",
    "DEFAULT_SELECTIVE_SAVE_OPERATIONS",
    "DEFAULT_SELECTIVE_TARGETS",
    "MANDATORY_SAVE_OPERATIONS",
    "TokenChunkedCheckpointWrapper",
    "get_activation_checkpoint_wrapper",
    "unmatched_module_patterns",
]
