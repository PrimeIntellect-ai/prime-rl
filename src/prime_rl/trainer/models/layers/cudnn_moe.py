"""Routed experts on cuDNN frontend's fused MXFP8 grouped GEMMs (SM100 / SM103).

The whole expert MLP is two custom ops, so compile sees it as opaque and selective activation
checkpointing saves its outputs instead of replaying it:

- forward: quantize the tokens and both weights (`mxfp8_quantize`), then the first GEMM with the
  GLU, its clamps and the MXFP8 quantization of its output fused into the epilogue, then the
  second GEMM.
- backward: the second layer's dgrad with the GLU backward fused (using the saved pre-activation),
  the first layer's dgrad, and both weight gradients.

cuDNN's GLU kernels compute `(clamp(up, min=-clamp, max=clamp) + linear_offset)
* g * sigmoid(alpha * g)` with `g = gate.clamp(max=clamp)` over a weight whose gate and up rows
interleave in 32-row blocks; `alpha=1, linear_offset=0` is SwiGLU. Every expert's token group
must be a multiple of 256 rows with zero padding rows, which the token dispatcher provides; rows
past the last group are left unset in the outputs and their gradients, and the dispatcher drops
them. Routing weights are applied by the dispatcher, so the kernels' per-row probabilities are ones.
"""

from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.kernels.mxfp8_quant import mxfp8_quantize, mxfp8_split_interleaved_columns
from prime_rl.trainer.models.layers.activations import ClampedSwiglu, Silu

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts

TOKEN_GROUP_ALIGNMENT = 256
SF_VEC_SIZE = 32


def _glu_params(activation) -> tuple[float, float, float]:
    """`(alpha, linear_offset, clamp)` of cuDNN's GLU for a supported expert activation."""
    from prime_rl.trainer.models.deepseek_v4.moe import ClampedSwiglu as DeepseekV4ClampedSwiglu

    if activation is Silu:
        return 1.0, 0.0, float("inf")
    if isinstance(activation, DeepseekV4ClampedSwiglu):
        return 1.0, 0.0, float(activation.limit)
    if activation is ClampedSwiglu:
        return 1.702, 1.0, 7.0
    raise ValueError(f"cuDNN MXFP8 expert compute does not support the {activation!r} expert activation.")


def _rows3(t: torch.Tensor) -> torch.Tensor:
    """`(m, k)` contiguous -> the `(m, k, 1)` operand view cuDNN's grouped kernels take."""
    m, k = t.shape
    return torch.as_strided(t, (m, k, 1), (k, 1, m * k))


def _mma_scales(scales: torch.Tensor, rows: int, cols: int, groups: int) -> torch.Tensor:
    """Flat swizzled scales of `groups` stacked `(rows, cols)` operands -> cuDNN's 6-D MMA view."""
    return scales.view(groups, rows // 128, cols // SF_VEC_SIZE // 4, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)


def _flat_scales(mma_scales: torch.Tensor) -> torch.Tensor:
    """cuDNN's 6-D MMA scale view -> the flat swizzled buffer under it."""
    return mma_scales.permute(5, 2, 4, 0, 1, 3).reshape(-1)


def _weight_offsets(groups: int, rows: int, device: torch.device) -> torch.Tensor:
    return torch.arange(1, groups + 1, dtype=torch.int32, device=device) * rows


@torch.library.custom_op("prime_rl::cudnn_mxfp8_moe", mutates_args=())
def cudnn_mxfp8_moe(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    offsets: torch.Tensor,
    alpha: float,
    linear_offset: float,
    clamp: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """`x` `(m, dim)`, `gate_proj`/`up_proj` `(experts, hidden, dim)`, `down_proj` `(experts, dim, hidden)`.

    Returns the `(m, dim)` bf16 output followed by what the backward reads: the interleaved
    pre-activation, the column-wise quantized activation and tokens, and the dgrad-oriented
    quantized weights (data, scales) each.
    """
    from cudnn.gemm.cutedsl.grouped import grouped_gemm_glu_wrapper_sm100, grouped_gemm_quant_wrapper_sm100

    m, dim = x.shape
    experts, hidden, _ = gate_proj.shape
    device = x.device
    ones = torch.ones(experts, dtype=torch.float32, device=device)
    norm_const = torch.ones(1, dtype=torch.float32, device=device)

    x_row, x_row_scales, x_col, x_col_scales = mxfp8_quantize(x, offsets)
    w1_row, w1_row_scales, w1_col, w1_col_scales = mxfp8_quantize(
        gate_proj, _weight_offsets(experts, 2 * hidden, device), interleave_with=up_proj
    )
    w2_row, w2_row_scales, w2_col, w2_col_scales = mxfp8_quantize(down_proj, _weight_offsets(experts, dim, device))

    fc1 = grouped_gemm_glu_wrapper_sm100(
        a_tensor=_rows3(x_row),
        sfa_tensor=_mma_scales(x_row_scales, m, dim, 1),
        padded_offsets=offsets,
        alpha_tensor=ones,
        b_tensor=w1_row.view(experts, 2 * hidden, dim).permute(1, 2, 0),
        sfb_tensor=_mma_scales(w1_row_scales, 2 * hidden, dim, experts),
        prob_tensor=torch.ones(m, 1, 1, dtype=torch.float32, device=device),
        norm_const_tensor=norm_const,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=SF_VEC_SIZE,
        generate_c=True,
        discrete_col_sfd=True,
        act_func="geglu",
        geglu_alpha=alpha,
        linear_offset=linear_offset,
        glu_clamp_max=clamp,
        glu_clamp_min=-clamp,
    )
    fc2 = grouped_gemm_quant_wrapper_sm100(
        a_tensor=fc1["d_tensor"],
        sfa_tensor=fc1["sfd_row_tensor"],
        padded_offsets=offsets,
        alpha_tensor=ones,
        b_tensor=w2_row.view(experts, dim, hidden).permute(1, 2, 0),
        sfb_tensor=_mma_scales(w2_row_scales, dim, hidden, experts),
        norm_const_tensor=norm_const,
        d_dtype=torch.bfloat16,
        sf_vec_size=SF_VEC_SIZE,
    )
    return (
        fc2["d_tensor"].squeeze(-1),
        fc1["c_tensor"].squeeze(-1),
        fc1["d_col_tensor"].squeeze(-1),
        _flat_scales(fc1["sfd_col_tensor"]),
        x_col,
        x_col_scales,
        w1_col,
        w1_col_scales,
        w2_col,
        w2_col_scales,
    )


@cudnn_mxfp8_moe.register_fake
def _cudnn_mxfp8_moe_fake(x, gate_proj, up_proj, down_proj, offsets, alpha, linear_offset, clamp):
    m, dim = x.shape
    experts, hidden, _ = gate_proj.shape

    def fp8(*shape):
        return x.new_empty(shape, dtype=torch.float8_e4m3fn)

    def scales(rows, cols):
        return x.new_empty((rows * cols // SF_VEC_SIZE,), dtype=torch.float8_e8m0fnu)

    return (
        x.new_empty((m, dim), dtype=torch.bfloat16),
        x.new_empty((m, 2 * hidden), dtype=torch.bfloat16),
        fp8(m, hidden),
        scales(m, hidden),
        fp8(m, dim),
        scales(m, dim),
        fp8(experts * 2 * hidden, dim),
        scales(experts * 2 * hidden, dim),
        fp8(experts * dim, hidden),
        scales(experts * dim, hidden),
    )


@torch.library.custom_op("prime_rl::cudnn_mxfp8_moe_backward", mutates_args=())
def cudnn_mxfp8_moe_backward(
    grad_output: torch.Tensor,
    preact: torch.Tensor,
    act_col: torch.Tensor,
    act_col_scales: torch.Tensor,
    x_col: torch.Tensor,
    x_col_scales: torch.Tensor,
    w1_col: torch.Tensor,
    w1_col_scales: torch.Tensor,
    w2_col: torch.Tensor,
    w2_col_scales: torch.Tensor,
    offsets: torch.Tensor,
    alpha: float,
    linear_offset: float,
    clamp: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(grad_x, grad_gate_proj, grad_up_proj, grad_down_proj)`, all bf16."""
    from cudnn.gemm.cutedsl.grouped import (
        grouped_gemm_dglu_wrapper_sm100,
        grouped_gemm_quant_wrapper_sm100,
        grouped_gemm_wgrad_wrapper_sm100,
    )

    m, dim = grad_output.shape
    hidden = act_col.shape[1]
    experts = offsets.numel()
    device = grad_output.device
    ones = torch.ones(experts, dtype=torch.float32, device=device)
    norm_const = torch.ones(1, dtype=torch.float32, device=device)

    dy_row, dy_row_scales, dy_col, dy_col_scales = mxfp8_quantize(grad_output, offsets)

    # The dgrad weights are the forward weights quantized along the other axis, read N-major.
    dglu = grouped_gemm_dglu_wrapper_sm100(
        a_tensor=_rows3(dy_row),
        c_tensor=_rows3(preact),
        sfa_tensor=_mma_scales(dy_row_scales, m, dim, 1),
        padded_offsets=offsets,
        alpha_tensor=ones,
        beta_tensor=ones,
        prob_tensor=torch.ones(m, 1, 1, dtype=torch.float32, device=device),
        dprob_tensor=torch.zeros(m, 1, 1, dtype=torch.float32, device=device),
        b_tensor=w2_col.view(experts, dim, hidden).permute(2, 1, 0),
        sfb_tensor=_mma_scales(w2_col_scales, hidden, dim, experts),
        norm_const_tensor=norm_const,
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=SF_VEC_SIZE,
        discrete_col_sfd=True,
        act_func="dgeglu",
        geglu_alpha=alpha,
        linear_offset=linear_offset,
        glu_clamp_max=clamp,
        glu_clamp_min=-clamp,
    )
    grad_x = grouped_gemm_quant_wrapper_sm100(
        a_tensor=dglu["d_row_tensor"],
        sfa_tensor=dglu["sfd_row_tensor"],
        padded_offsets=offsets,
        alpha_tensor=ones,
        b_tensor=w1_col.view(experts, 2 * hidden, dim).permute(2, 1, 0),
        sfb_tensor=_mma_scales(w1_col_scales, dim, 2 * hidden, experts),
        norm_const_tensor=norm_const,
        d_dtype=torch.bfloat16,
        sf_vec_size=SF_VEC_SIZE,
    )["d_tensor"].squeeze(-1)

    # Weight gradients reduce over tokens: both operands are the column-wise quantized copies,
    # the left one read M-major, and the scales are swizzled per expert.
    def wgrad(left, left_scales, right, right_scales):
        return grouped_gemm_wgrad_wrapper_sm100(
            a_tensor=left.t(),
            b_tensor=right,
            sfa_tensor=left_scales.view(left.shape[1], -1),
            sfb_tensor=right_scales.view(right.shape[1], -1),
            offsets_tensor=offsets,
            wgrad_dtype=torch.bfloat16,
            sf_vec_size=SF_VEC_SIZE,
        )["wgrad_tensor"]

    grad_down_proj = wgrad(dy_col, dy_col_scales, act_col, act_col_scales)
    gate_col, gate_col_scales, up_col, up_col_scales = mxfp8_split_interleaved_columns(
        dglu["d_col_tensor"].squeeze(-1), _flat_scales(dglu["sfd_col_tensor"]), offsets
    )
    grad_gate_proj = wgrad(gate_col, gate_col_scales, x_col, x_col_scales)
    grad_up_proj = wgrad(up_col, up_col_scales, x_col, x_col_scales)
    return grad_x, grad_gate_proj, grad_up_proj, grad_down_proj


@cudnn_mxfp8_moe_backward.register_fake
def _cudnn_mxfp8_moe_backward_fake(
    grad_output,
    preact,
    act_col,
    act_col_scales,
    x_col,
    x_col_scales,
    w1_col,
    w1_col_scales,
    w2_col,
    w2_col_scales,
    offsets,
    alpha,
    linear_offset,
    clamp,
):
    m, dim = grad_output.shape
    hidden = act_col.shape[1]
    experts = offsets.numel()
    return (
        torch.empty_like(grad_output),
        grad_output.new_empty((experts, hidden, dim)),
        grad_output.new_empty((experts, hidden, dim)),
        grad_output.new_empty((experts, dim, hidden)),
    )


def _setup_context(ctx, inputs, output) -> None:
    x, gate_proj, up_proj, down_proj, offsets, alpha, linear_offset, clamp = inputs
    ctx.save_for_backward(*output[1:], offsets)
    # Only the output is differentiable; without this autograd zero-fills a gradient for every
    # tensor saved for the backward.
    ctx.mark_non_differentiable(*output[1:])
    ctx.set_materialize_grads(False)
    ctx.glu_params = (alpha, linear_offset, clamp)
    ctx.dtypes = (x.dtype, gate_proj.dtype, up_proj.dtype, down_proj.dtype)


def _backward(ctx, grad_output: torch.Tensor, *_unused_grads):
    *saved, offsets = ctx.saved_tensors
    grads = cudnn_mxfp8_moe_backward(grad_output.contiguous(), *saved, offsets, *ctx.glu_params)
    grad_x, grad_gate_proj, grad_up_proj, grad_down_proj = (
        grad.to(dtype) for grad, dtype in zip(grads, ctx.dtypes, strict=True)
    )
    return grad_x, grad_gate_proj, grad_up_proj, grad_down_proj, None, None, None, None


cudnn_mxfp8_moe.register_autograd(_backward, setup_context=_setup_context)


class CudnnMXFP8ExpertCompute:
    token_group_alignment = TOKEN_GROUP_ALIGNMENT

    def validate(self, experts: "GroupedExperts") -> None:
        if experts.gate_proj is None and experts.gate_up_proj is None:
            raise ValueError("cuDNN MXFP8 expert compute requires gated experts.")
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("cuDNN MXFP8 expert compute does not support expert biases.")
        _glu_params(experts.activation)

    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        def to_local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        if experts.gate_up_proj is None:
            gate_proj, up_proj = to_local(experts.gate_proj), to_local(experts.up_proj)
        else:
            gate_proj, up_proj = to_local(experts.gate_up_proj).chunk(2, dim=1)
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        output = cudnn_mxfp8_moe(
            x.bfloat16(), gate_proj, up_proj, to_local(experts.down_proj), offsets, *_glu_params(experts.activation)
        )[0]
        return output.type_as(x)
