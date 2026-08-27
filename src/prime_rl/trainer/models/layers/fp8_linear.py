from __future__ import annotations

import re

import torch
from torch import nn

try:
    import deep_gemm
except ImportError:
    deep_gemm = None  # CPU-only environments don't ship deep_gemm; FP8 paths
    # are GPU-only at runtime, so leaving the symbol None is safe — only the
    # autograd Function bodies below actually call into it.

from prime_rl.trainer.models.kernels.fp8_utils import (
    per_block_cast_to_fp8_tp_triton,
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_tp_triton,
    per_token_cast_to_fp8_triton,
    ue8m0_for_device,
)
from prime_rl.utils.logger import get_logger


@torch.library.custom_op("prime_rl::fp8_blockwise_mm", mutates_args=())
def _fp8_blockwise_mm(x: torch.Tensor, weight: torch.Tensor, block_size: int) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    use_ue8m0 = ue8m0_for_device(x.device)
    x_fp8 = per_token_cast_to_fp8_triton(x_2d, use_ue8m0, block_size)
    weight_fp8 = per_block_cast_to_fp8_triton(weight, use_ue8m0, block_size)

    out = torch.empty((x_2d.size(0), weight.size(0)), device=x.device, dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(x_fp8, weight_fp8, out)
    return out.reshape(*x.shape[:-1], out.size(-1))


@_fp8_blockwise_mm.register_fake
def _fp8_blockwise_mm_fake(x: torch.Tensor, weight: torch.Tensor, block_size: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("prime_rl::fp8_blockwise_mm_backward", mutates_args=())
def _fp8_blockwise_mm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: int,
    needs_grad_x: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    use_ue8m0 = ue8m0_for_device(grad_output.device)
    grad_x = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)

    if needs_grad_x:
        grad_output_fp8 = per_token_cast_to_fp8_triton(grad_output_2d, use_ue8m0, block_size)
        weight_dx_fp8 = per_block_cast_to_fp8_tp_triton(weight, use_ue8m0, block_size)
        grad_x_2d = torch.empty_like(x_2d)
        deep_gemm.fp8_gemm_nt(grad_output_fp8, weight_dx_fp8, grad_x_2d)
        grad_x = grad_x_2d.reshape(x.shape)

    if needs_grad_weight:
        # DeepGEMM's (1, 1, 128) recipe requires the token dimension to be aligned.
        num_tokens = grad_output_2d.size(0)
        padded_tokens = (num_tokens + block_size - 1) // block_size * block_size
        if padded_tokens != num_tokens:
            pad_rows = padded_tokens - num_tokens
            grad_output_2d = torch.nn.functional.pad(grad_output_2d, (0, 0, 0, pad_rows))
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, pad_rows))
        grad_output_t_fp8 = per_token_cast_to_fp8_tp_triton(grad_output_2d, use_ue8m0, block_size)
        x_t_fp8 = per_token_cast_to_fp8_tp_triton(x_2d, use_ue8m0, block_size)
        grad_weight_fp32 = torch.zeros_like(weight, dtype=torch.float32)
        deep_gemm.fp8_gemm_nt(
            grad_output_t_fp8,
            x_t_fp8,
            grad_weight_fp32,
            c=grad_weight_fp32,
            recipe=(1, 1, 128),
        )
        grad_weight = grad_weight_fp32.to(weight.dtype)

    return grad_x, grad_weight


@_fp8_blockwise_mm_backward.register_fake
def _fp8_blockwise_mm_backward_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: int,
    needs_grad_x: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


def _fp8_blockwise_mm_setup_context(ctx, inputs, output) -> None:
    x, weight, block_size = inputs
    ctx.save_for_backward(x, weight)
    ctx.block_size = block_size


def _fp8_blockwise_mm_autograd_backward(ctx, grad_output: torch.Tensor):
    x, weight = ctx.saved_tensors
    needs_grad_x, needs_grad_weight, _ = ctx.needs_input_grad
    grad_x, grad_weight = _fp8_blockwise_mm_backward(
        grad_output,
        x.detach(),
        weight.detach(),
        ctx.block_size,
        needs_grad_x,
        needs_grad_weight,
    )
    return grad_x if needs_grad_x else None, grad_weight if needs_grad_weight else None, None


_fp8_blockwise_mm.register_autograd(
    _fp8_blockwise_mm_autograd_backward,
    setup_context=_fp8_blockwise_mm_setup_context,
)


class Float8BlockwiseLinear(nn.Linear):
    """nn.Linear replacement that uses FP8 blockwise matmul via DeepGEMM.

    Requires:
    - SM90 (Hopper) or SM100 (Blackwell) GPU
    - bfloat16 inputs/weights
    - No bias
    - in_features and out_features divisible by 128
    """

    def __init__(self, *args, block_size: int = 128, dtype=torch.bfloat16, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fp8_blockwise_mm(x, self.weight, self.block_size)

    @classmethod
    def from_linear(cls, mod: nn.Linear) -> "Float8BlockwiseLinear":
        """Convert an existing nn.Linear to Float8BlockwiseLinear."""
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def replace_linear_with_fp8_blockwise_linear(model: nn.Module, ignore_modules: list[str]) -> None:
    """Replace nn.Linear in `model` with Float8BlockwiseLinear, skipping any
    module whose qualified name matches an ignore pattern (substring or regex).

    The default ignore list covers layers that should never be quantized:
    - lm_head
    - MoE routers and gates (router, mlp.gate., shared_expert.output_gate)
    - sparse-MLA scalar projection (weights_proj)
    - GLM-5.1 MTP head (eh_proj)
    - hybrid-Mamba projections (in_proj_a, in_proj_b)

    Independently of the name-based ignore list, we also skip any nn.Linear
    whose in_features or out_features is not a multiple of 128. Float8BlockwiseLinear
    documents that requirement and DeepGEMM's fp8_gemm_nt crashes at runtime
    on unaligned dims — better to keep them in BF16 with a clear log line than
    silently break in the kernel.

    Conv1d, layer norms, and embedding tables are not nn.Linear and are
    skipped automatically by the type check; we don't need to list them.
    """
    logger = get_logger()
    logger.info(f"Replacing linear layers with FP8 blockwise linear layers (ignore={ignore_modules})")
    replaced_modules = []
    skipped_modules = []
    skipped_unaligned: list[str] = []
    named_modules = dict(model.named_modules())
    for name, module in named_modules.items():
        if not isinstance(module, nn.Linear):
            continue
        if any(re.search(pattern, name) for pattern in ignore_modules):
            skipped_modules.append(name)
            continue
        if module.in_features % 128 != 0 or module.out_features % 128 != 0:
            skipped_unaligned.append(f"{name}({module.in_features}->{module.out_features})")
            continue
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, Float8BlockwiseLinear.from_linear(module))
        replaced_modules.append(name)

    logger.info(
        f"Replaced {len(replaced_modules)} linear layers with FP8 blockwise linear "
        f"(skipped {len(skipped_modules)} by name, "
        f"{len(skipped_unaligned)} by 128-divisibility); "
        f"first replaced={replaced_modules[:3]}, "
        f"first skipped(name)={skipped_modules[:3]}, "
        f"first skipped(unaligned)={skipped_unaligned[:3]}"
    )
