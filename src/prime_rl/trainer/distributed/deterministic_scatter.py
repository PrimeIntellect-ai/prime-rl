# Adapted from torchtitan/ops/scatter_add.py (BSD-3-Clause).
"""Compile-compatible scatter addition with scoped PyTorch determinism."""

import torch


@torch.library.custom_op("prime_rl::deterministic_scatter_add", mutates_args=())
def deterministic_scatter_add(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    # Keep the cast inside the custom op so compilation preserves rounded inputs.
    dtype = torch.float32 if src.dtype in (torch.bfloat16, torch.float16) else src.dtype
    accumulation_out = out.to(dtype)
    accumulation_src = src.to(dtype)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        result = accumulation_out.scatter_add(0, index, accumulation_src)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    return result.to(out.dtype)


@deterministic_scatter_add.register_fake
def _fake(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


def _setup_context(ctx, inputs, output):
    _, index, _ = inputs
    ctx.save_for_backward(index)


def _backward(ctx, grad_output):
    (index,) = ctx.saved_tensors
    return grad_output, None, torch.gather(grad_output, 0, index)


deterministic_scatter_add.register_autograd(_backward, setup_context=_setup_context)
