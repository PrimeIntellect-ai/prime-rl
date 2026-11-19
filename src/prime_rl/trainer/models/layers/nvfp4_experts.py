"""Routed experts with the numerics of vLLM's online ``nvfp4_per_token`` MoE (FlashInfer TRT-LLM kernel).

Forward: the GEMM operands are bit-identical to the engines'. Activations go through FlashInfer's per-token NVFP4
quantizer (the kernel the engines run on the GEMM1 input and, after SwiGLU, on the GEMM2 input), weights through
vLLM's load-time NVFP4 quantizer (one FP32 scale per expert, E4M3 scales per 16 values). The GEMM multiplies the
dequantized operands in BF16, which holds every E2M1 x E4M3 product exactly, accumulates in FP32 and applies the
per-token and per-expert scales after, as the FP4 tensor-core GEMM does; only the accumulation order differs.

Backward: MXFP8 dgrad and wgrad on the dequantized NVFP4 operands (straight-through the quantizers).
"""

import importlib
import os
from types import ModuleType
from typing import TYPE_CHECKING

import torch

from prime_rl.trainer.models.layers.expert_compute import to_local

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts

# FlashInfer's per-token recipe: per-token decode scale = row amax / (448 * 6).
_PER_TOKEN_GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequantize_nvfp4(packed: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """``[..., K // 2]`` packed E2M1 pairs (low nibble first) times their ``[..., K // 16]`` E4M3 block scales, in
    BF16 (exact: at most 5 significant bits)."""
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.bfloat16, device=packed.device)
    packed = packed.view(torch.uint8)
    values = torch.stack((lut[(packed & 0xF).long()], lut[(packed >> 4).long()]), dim=-1)
    values = values.view(*packed.shape[:-1], -1, 16)
    return (values * block_scales.view(torch.float8_e4m3fn).to(torch.bfloat16).unsqueeze(-1)).flatten(-2)


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
    return packed, block_scales.view(x.shape[0], -1), token_scales.view(-1)


def quantize_weights(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLLM's load-time quantizer for ``[E, N, K]`` expert weights: packed E2M1, E4M3 block scales, and one FP32
    decode scale per expert."""
    from vllm.model_executor.layers.quantization.online.nvfp4 import _quantize_moe_weight_to_nvfp4

    packed, block_scales, expert_scales = _quantize_moe_weight_to_nvfp4(weight.bfloat16())
    num_experts, n, k = weight.shape
    return packed.view(num_experts, n, k // 2), block_scales.view(num_experts, n, k // 16), expert_scales


def _row_expert_scales(expert_scales: torch.Tensor, offsets: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Each row's expert decode scale; rows past the last group (alignment padding) get 0."""
    counts = torch.diff(offsets, prepend=offsets.new_zeros(1)).long()
    padding = offsets.new_tensor(num_rows, dtype=torch.long) - offsets[-1].long()
    return torch.repeat_interleave(
        torch.cat((expert_scales, expert_scales.new_zeros(1))),
        torch.cat((counts, padding.unsqueeze(0))),
        output_size=num_rows,
    )


def _grouped_mm_fp32(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``x[rows of group e] @ weight[e].T`` per group, FP32 out (the FP4 GEMM applies its scales to the FP32
    accumulator); rows past the last group are 0."""
    out = x.new_zeros(x.shape[0], weight.shape[1], dtype=torch.float32)
    bounds = [0, *offsets.tolist()]
    for expert, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        if end > start:
            torch.mm(x[start:end], weight[expert].T, out_dtype=torch.float32, out=out[start:end])
    return out


class _NVFP4GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, offsets, four_over_six, mxfp8):
        x_packed, x_block_scales, token_scales = quantize_activations(x, four_over_six)
        w_packed, w_block_scales, expert_scales = quantize_weights(weight)
        out = _grouped_mm_fp32(
            dequantize_nvfp4(x_packed, x_block_scales), dequantize_nvfp4(w_packed, w_block_scales), offsets
        )
        out = out * (token_scales * _row_expert_scales(expert_scales, offsets, x.shape[0])).unsqueeze(-1)
        ctx.save_for_backward(x_packed, x_block_scales, token_scales, weight, offsets)
        ctx.mxfp8 = mxfp8
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_packed, x_block_scales, token_scales, weight, offsets = ctx.saved_tensors
        grad_output = grad_output.bfloat16()
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            w_packed, w_block_scales, expert_scales = quantize_weights(weight)
            w_hat = (dequantize_nvfp4(w_packed, w_block_scales) * expert_scales.view(-1, 1, 1)).bfloat16()
            grad_input = ctx.mxfp8._dgrad(grad_output, w_hat.transpose(-2, -1), offsets)
        if ctx.needs_input_grad[1]:
            x_hat = (dequantize_nvfp4(x_packed, x_block_scales) * token_scales.unsqueeze(-1)).bfloat16()
            grad_weight = ctx.mxfp8._wgrad(grad_output, x_hat, offsets, high_precision=False).transpose(-2, -1)
        return grad_input, grad_weight, None, None, None


class NVFP4ExpertCompute:
    """Routed experts as vLLM's online ``nvfp4_per_token`` MoE computes them, MXFP8 backward."""

    def __init__(self, mxfp8_kernel: ModuleType, four_over_six: bool, flashinfer_env: dict[str, str]) -> None:
        if torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("NVFP4 expert compute requires a Blackwell (SM100/SM103) GPU.")
        # FlashInfer's quantizer reads its division mode from the environment, as on the engines.
        os.environ.update(flashinfer_env)
        # The MXFP8 grouped GEMM's own dgrad and wgrad.
        self.mxfp8 = importlib.import_module(f"{mxfp8_kernel.__name__}.grouped_gemm")
        self.four_over_six = four_over_six
        self.token_group_alignment = mxfp8_kernel.TOKEN_GROUP_ALIGNMENT

    def validate(self, experts: "GroupedExperts") -> None:
        if experts.gate_proj is None and experts.gate_up_proj is None:
            raise ValueError("NVFP4 expert compute supports gated experts only.")
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("NVFP4 expert compute does not support expert biases.")

    def _linear(self, x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return _NVFP4GroupedLinear.apply(x, weight, offsets, self.four_over_six, self.mxfp8)

    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        if experts.gate_up_proj is not None:
            gate_up_proj = to_local(experts.gate_up_proj)
        else:
            # vLLM quantizes gate and up as one w13 weight with one scale per expert.
            gate_up_proj = torch.cat((to_local(experts.gate_proj), to_local(experts.up_proj)), dim=1)
        # The engine's GEMM1 epilogue applies the scales and SwiGLU in FP32 and stores BF16 for the GEMM2 quantizer.
        gate, up = self._linear(x.bfloat16(), gate_up_proj.bfloat16(), offsets).chunk(2, dim=-1)
        hidden = experts.activation.apply(gate, up).bfloat16()
        return self._linear(hidden, to_local(experts.down_proj).bfloat16(), offsets).type_as(x)
