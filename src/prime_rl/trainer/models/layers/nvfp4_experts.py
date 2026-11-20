"""Routed experts with the numerics of vLLM's online ``nvfp4_per_token`` MoE (FlashInfer TRT-LLM kernel).

Forward: the GEMM operands are bit-identical to the engines'. Activations go through FlashInfer's per-token NVFP4
quantizer (the kernel the engines run on the GEMM1 input and, after SwiGLU, on the GEMM2 input), weights through
vLLM's load-time NVFP4 quantizer (one FP32 scale per expert, E4M3 scales per 16 values). The GEMMs run on the FP4
tensor cores (cuBLASLt block-scaled NVFP4) with FP32 output, and the per-token and per-expert scales apply to the
FP32 result, as in the engine's GEMM epilogue before SwiGLU and the GEMM2 quantizer.

Backward: MXFP8 dgrad and wgrad on the forward's NVFP4 operands, requantized to MXFP8 straight from their packed form
(straight through the quantizers).
"""

import importlib
import os
from types import ModuleType
from typing import TYPE_CHECKING

import torch

from prime_rl.trainer.models.kernels.nvfp4_to_mxfp8 import nvfp4_to_mxfp8_columns
from prime_rl.trainer.models.layers.expert_compute import to_local

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts

# FlashInfer's per-token recipe: per-token decode scale = row amax / (448 * 6).
_PER_TOKEN_GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)


def quantize_activations(x: torch.Tensor, four_over_six: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashInfer's per-token NVFP4 quantizer, as ``quantize_nvfp4_per_token_input`` and the TRT-LLM MoE's GEMM2
    requantization run it: packed E2M1 ``[M, K // 2]``, E4M3 block scales ``[M, K // 16]``, FP32 decode scales
    ``[M]``."""
    from flashinfer import SfLayout, nvfp4_quantize
    from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config

    packed, block_scales, token_scales = nvfp4_quantize(
        x.contiguous(),
        _PER_TOKEN_GLOBAL_SCALE_INV,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
        nvfp4_4over6=NVFP44Over6Config() if four_over_six else None,
    )
    return packed.view(torch.uint8), block_scales.view(torch.float8_e4m3fn).view(x.shape[0], -1), token_scales.view(-1)


def quantize_weights(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLLM's load-time quantizer for ``[E, N, K]`` expert weights: packed E2M1, E4M3 block scales, and one FP32
    decode scale per expert."""
    from vllm.model_executor.layers.quantization.online.nvfp4 import _quantize_moe_weight_to_nvfp4

    packed, block_scales, expert_scales = _quantize_moe_weight_to_nvfp4(weight.bfloat16())
    num_experts, n, k = weight.shape
    return (
        packed.view(torch.uint8).view(num_experts, n, k // 2),
        block_scales.view(torch.float8_e4m3fn).view(num_experts, n, k // 16),
        expert_scales,
    )


def _blocked(block_scales: torch.Tensor) -> torch.Tensor:
    """``[..., rows, K // 16]`` E4M3 block scales, ``rows`` a multiple of 128 and ``K // 16`` of 4, in cuBLAS's 128x4
    block layout: ``[..., rows // 128 * K // 64, 512]``, one 128x4 tile per row."""
    *batch, rows, cols = block_scales.shape
    tiles = block_scales.view(torch.uint8).view(*batch, rows // 128, 4, 32, cols // 4, 4)
    return tiles.permute(*range(len(batch)), -5, -2, -3, -4, -1).reshape(*batch, -1, 512).view(torch.float8_e4m3fn)


def _nvfp4_grouped_mm(x, weight, bounds: list[int]) -> torch.Tensor:
    """Per expert group, ``x @ weight[e].T`` on NVFP4 operands with an FP32 result, scaled per token and per expert;
    rows past the last group are 0."""
    x_packed, x_block_scales, token_scales = x
    w_packed, w_block_scales, expert_scales = weight
    device = x_packed.device
    counts = [end - start for start, end in zip(bounds[:-1], bounds[1:])]
    # Every group's activation scales padded to whole 128-row tiles, so each group's blocked scales are one slice
    tiles = [-(-count // 128) for count in counts]
    tile_starts = [0]
    for count_tiles in tiles:
        tile_starts.append(tile_starts[-1] + count_tiles)
    shift = torch.tensor([128 * tile - start for tile, start in zip(tile_starts, bounds)], device=device)
    rows = torch.arange(bounds[-1], device=device)
    rows += torch.repeat_interleave(shift[:-1], torch.tensor(counts, device=device), output_size=bounds[-1])
    padded = torch.zeros(128 * tile_starts[-1], x_block_scales.shape[1], dtype=torch.uint8, device=device)
    padded[rows] = x_block_scales[: bounds[-1]].view(torch.uint8)
    x_scales = _blocked(padded)
    w_scales = _blocked(w_block_scales)
    tile_rows = x_block_scales.shape[1] // 4

    out = torch.empty(x_packed.shape[0], w_packed.shape[1], dtype=torch.float32, device=device)
    out[bounds[-1] :].zero_()
    for expert, (start, count) in enumerate(zip(bounds[:-1], counts)):
        if count:
            first = tile_starts[expert] * tile_rows
            torch._scaled_mm(
                x_packed[start : start + count].view(torch.float4_e2m1fn_x2),
                w_packed[expert].view(torch.float4_e2m1fn_x2).T,
                x_scales[first : first + tiles[expert] * tile_rows].flatten(),
                w_scales[expert].flatten(),
                out_dtype=torch.float32,
                out=out[start : start + count],
            )
    row_expert_scales = torch.repeat_interleave(
        expert_scales, torch.tensor(counts, device=device), output_size=bounds[-1]
    )
    out[: bounds[-1]].mul_((token_scales[: bounds[-1]] * row_expert_scales).unsqueeze(-1))
    return out


class _NVFP4GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, offsets, bounds, four_over_six, mxfp8):
        x_quantized = quantize_activations(x, four_over_six)
        w_quantized = quantize_weights(weight)
        out = _nvfp4_grouped_mm(x_quantized, w_quantized, bounds)
        ctx.save_for_backward(*x_quantized, *w_quantized, offsets)
        ctx.mxfp8 = mxfp8
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_packed, x_block_scales, token_scales, w_packed, w_block_scales, expert_scales, offsets = ctx.saved_tensors
        mx = ctx.mxfp8
        grad_output = grad_output.bfloat16().contiguous()
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_data, grad_scales = mx.quantize_rows(grad_output)
            # The weights as [E, K, N], quantized along N (the dgrad contraction)
            num_experts, n, _ = w_packed.shape
            weight_data, weight_scales = nvfp4_to_mxfp8_columns(
                w_packed, w_block_scales, expert_scales.unsqueeze(-1).expand(num_experts, n)
            )
            grad_input = torch._scaled_grouped_mm(
                grad_data,
                weight_data.transpose(-2, -1),
                mx._rearrange_token_scales(grad_scales, offsets),
                mx.triton_mx_block_rearrange_per_group_3d(weight_scales),
                offs=offsets,
                out_dtype=torch.bfloat16,
            )
        if ctx.needs_input_grad[1]:
            grad_data, grad_scales = mx._quantize_dim1(grad_output)
            # The activations as [K, M], quantized along M (the wgrad contraction)
            input_data, input_scales = nvfp4_to_mxfp8_columns(
                x_packed.unsqueeze(0), x_block_scales.unsqueeze(0), token_scales.unsqueeze(0)
            )
            scale_offsets = mx._pow2_groups(offsets // mx.TOKEN_GROUP_ALIGNMENT)
            grad_weight = torch._scaled_grouped_mm(
                grad_data,
                input_data[0].transpose(-2, -1),
                mx.triton_mx_block_rearrange_2d_K_groups(grad_scales, scale_offsets),
                mx.triton_mx_block_rearrange_2d_K_groups(input_scales[0], scale_offsets),
                offs=offsets,
                out_dtype=torch.bfloat16,
            )
        return grad_input, grad_weight, None, None, None, None


class NVFP4ExpertCompute:
    """Routed experts as vLLM's online ``nvfp4_per_token`` MoE computes them, MXFP8 backward."""

    def __init__(self, mxfp8_kernel: ModuleType, four_over_six: bool, flashinfer_env: dict[str, str]) -> None:
        if torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("NVFP4 expert compute requires a Blackwell (SM100/SM103) GPU.")
        # FlashInfer's quantizer reads its division mode from the environment, as on the engines.
        os.environ.update(flashinfer_env)
        # The MXFP8 grouped GEMM's own quantizers and scale layouts.
        self.mxfp8 = importlib.import_module(f"{mxfp8_kernel.__name__}.grouped_gemm")
        self.four_over_six = four_over_six
        self.token_group_alignment = mxfp8_kernel.TOKEN_GROUP_ALIGNMENT

    def validate(self, experts: "GroupedExperts") -> None:
        if experts.gate_proj is None and experts.gate_up_proj is None:
            raise ValueError("NVFP4 expert compute supports gated experts only.")
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("NVFP4 expert compute does not support expert biases.")
        # Whole 128x4 scale tiles for the weights, whole 64-value tiles for the MXFP8 requantization
        dim = experts.down_proj.shape[-2]
        if dim % 128 or experts.hidden_dim % 64:
            raise ValueError(
                f"NVFP4 expert compute needs hidden size divisible by 128 and expert intermediate size by 64, got "
                f"{dim} and {experts.hidden_dim}."
            )

    def _linear(self, x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor, bounds: list[int]) -> torch.Tensor:
        return _NVFP4GroupedLinear.apply(x, weight, offsets, bounds, self.four_over_six, self.mxfp8)

    # Eager kernels with a host-side group loop: nothing for the compiler to fuse.
    @torch.compiler.disable
    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        bounds = [0, *offsets.tolist()]
        if experts.gate_up_proj is not None:
            gate_up_proj = to_local(experts.gate_up_proj)
        else:
            # vLLM quantizes gate and up as one w13 weight with one scale per expert.
            gate_up_proj = torch.cat((to_local(experts.gate_proj), to_local(experts.up_proj)), dim=1)
        # The engine's GEMM1 epilogue applies the scales and SwiGLU in FP32 and stores BF16 for the GEMM2 quantizer.
        gate, up = self._linear(x.bfloat16(), gate_up_proj.bfloat16(), offsets, bounds).chunk(2, dim=-1)
        hidden = experts.activation.apply(gate, up).bfloat16()
        return self._linear(hidden, to_local(experts.down_proj).bfloat16(), offsets, bounds).type_as(x)
