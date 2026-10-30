"""Whole-block activation checkpointing with an operator-based policy."""

from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
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


class TokenChunkedCheckpointWrapper(CheckpointWrapper):
    """Checkpoint a token-local block as one checkpoint per chunk of at most ``token_chunk_size`` tokens.

    The block's recompute and backward then hold one chunk's activations at a time instead of the
    whole sequence's. Only valid for blocks whose output at each token depends on that token alone
    (e.g. a MoE layer): the hidden states and every tensor argument laid out like them
    (``[batch, seq, ...]``) are split along the sequence; other arguments go to every chunk as-is.
    """

    def __init__(self, mod: nn.Module, token_chunk_size: int, **checkpoint_kwargs) -> None:
        super().__init__(mod, **checkpoint_kwargs)
        self.token_chunk_size = token_chunk_size
        self._checkpoint_chunk = super().forward

    def compile(self, *args, **kwargs) -> None:
        # Compile one checkpointed chunk, not the chunk loop: the loop's bounds follow the row length, which
        # torch.compile would specialize on and recompile for. The checkpoint stays inside the compiled call so
        # its policy still applies.
        self._checkpoint_chunk = torch.compile(super().forward, *args, **kwargs)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        if seq_len <= self.token_chunk_size:
            return self._checkpoint_chunk(hidden_states, *args, **kwargs)

        def is_token_aligned(value) -> bool:
            return isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[:2] == (batch_size, seq_len)

        # Equal chunks rather than full chunks plus a remainder: a 1-token tail would hit torch.compile's size-1
        # specialization, whose separate graph gave wrong parameter gradients.
        num_chunks = -(-seq_len // self.token_chunk_size)
        base, extra = divmod(seq_len, num_chunks)
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
        targets = DEFAULT_SELECTIVE_TARGETS if config.targets is None else frozenset(config.targets)
        policy = partial(_selective_checkpoint_policy, targets=targets)
    checkpoint_kwargs = {
        "checkpoint_impl": CheckpointImpl.NO_REENTRANT,
        "context_fn": partial(create_selective_checkpoint_contexts, policy),
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
    "MANDATORY_SAVE_NAMESPACES",
    "MANDATORY_SAVE_OPERATIONS",
    "TokenChunkedCheckpointWrapper",
    "get_activation_checkpoint_wrapper",
]
