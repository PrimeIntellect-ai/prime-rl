"""The NVFP4 expert GEMMs' epilogue in one pass: scale an expert group's FP32 GEMM result per token and per expert,
then store it in BF16, or after GEMM1 store the BF16 gate/up and their SwiGLU."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK_ROWS = 32
_BLOCK_COLS = 128


@triton.jit
def _epilogue_kernel(
    result_ptr,
    token_scale_ptr,
    expert_scale_ptr,
    expert,
    out_ptr,
    gate_up_ptr,
    rows,
    cols,
    limit,
    GATED: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    # The per-token and per-expert decode scales multiply first, as in the reference path
    scale = tl.load(token_scale_ptr + r, mask=r < rows, other=0.0) * tl.load(expert_scale_ptr + expert)
    in_cols = 2 * cols if GATED else cols
    first = tl.load(result_ptr + r[:, None] * in_cols + c[None, :], mask=mask, other=0.0) * scale[:, None]
    if GATED:
        up = tl.load(result_ptr + r[:, None] * in_cols + cols + c[None, :], mask=mask, other=0.0) * scale[:, None]
        tl.store(gate_up_ptr + r[:, None] * in_cols + c[None, :], first.to(tl.bfloat16), mask=mask)
        tl.store(gate_up_ptr + r[:, None] * in_cols + cols + c[None, :], up.to(tl.bfloat16), mask=mask)
        gate = first
        if CLAMP:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
        # torch's F.silu: x / (1 + exp(-x)) in IEEE FP32
        first = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up
    tl.store(out_ptr + r[:, None] * cols + c[None, :], first.to(tl.bfloat16), mask=mask)


def nvfp4_epilogue(
    result: torch.Tensor,
    token_scales: torch.Tensor,
    expert_scales: torch.Tensor,
    expert: int,
    out: torch.Tensor,
    gate_up: torch.Tensor | None = None,
    limit: float | None = None,
) -> None:
    """``result`` ``[rows, N]`` FP32 times ``token_scales[rows] * expert_scales[expert]`` into BF16 ``out``. With
    ``gate_up``, ``result`` holds gate and up halves: they are stored there in BF16, and ``out`` gets
    ``silu(gate) * up``, with gate clamped to ``max=limit`` and up to ``[-limit, limit]`` when ``limit`` is set."""
    rows, cols = out.shape
    grid = (triton.cdiv(rows, _BLOCK_ROWS), triton.cdiv(cols, _BLOCK_COLS))
    _epilogue_kernel[grid](
        result,
        token_scales,
        expert_scales,
        expert,
        out,
        gate_up if gate_up is not None else out,
        rows,
        cols,
        limit if limit is not None else 0.0,
        GATED=gate_up is not None,
        CLAMP=limit is not None,
        BLOCK_ROWS=_BLOCK_ROWS,
        BLOCK_COLS=_BLOCK_COLS,
    )


@triton.jit
def _swiglu_backward_kernel(
    gate_up_ptr,
    grad_ptr,
    out_ptr,
    rows,
    cols,
    limit,
    CLAMP: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    gate_offsets = r[:, None] * (2 * cols) + c[None, :]
    gate = tl.load(gate_up_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + gate_offsets + cols, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + r[:, None] * cols + c[None, :], mask=mask, other=0.0).to(tl.float32)
    if CLAMP:
        # as torch's clamp backward: the gradient passes strictly inside the range
        gate_pass = gate < limit
        up_pass = (up > -limit) & (up < limit)
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    grad_gate = grad * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    grad_up = grad * gate * sigmoid
    if CLAMP:
        grad_gate = tl.where(gate_pass, grad_gate, 0.0)
        grad_up = tl.where(up_pass, grad_up, 0.0)
    tl.store(out_ptr + gate_offsets, grad_gate.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + gate_offsets + cols, grad_up.to(tl.bfloat16), mask=mask)


def swiglu_backward(gate_up: torch.Tensor, grad_hidden: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    """Gradient of ``silu(gate) * up`` (gate clamped to ``max=limit``, up to ``[-limit, limit]``) with respect to the
    BF16 ``[rows, 2 * cols]`` gate/up, given the BF16 ``[rows, cols]`` output gradient."""
    rows, cols = grad_hidden.shape
    out = torch.empty_like(gate_up)
    grid = (triton.cdiv(rows, _BLOCK_ROWS), triton.cdiv(cols, _BLOCK_COLS))
    _swiglu_backward_kernel[grid](
        gate_up,
        grad_hidden.contiguous(),
        out,
        rows,
        cols,
        limit if limit is not None else 0.0,
        CLAMP=limit is not None,
        BLOCK_ROWS=_BLOCK_ROWS,
        BLOCK_COLS=_BLOCK_COLS,
    )
    return out
