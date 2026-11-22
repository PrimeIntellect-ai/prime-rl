"""Fused Triton FWHT-128 and K-Pool compression kernels for GLM-5.3 Flash's frozen indexer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fwht128_stage_vec(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    """One radix-2 Walsh-Hadamard butterfly stage in registers on a 128-wide FP32 vector."""
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    a, b = tl.split(tl.permute(x3, (0, 2, 1)))
    s = a + b
    d = a - b
    return tl.reshape(tl.permute(tl.join(s, d), (0, 2, 1)), (128,))


@triton.jit
def _fwht128_vec(x):
    """Full 7-stage orthonormal FWHT-128 in FP32 registers (bit-exact match to butterfly order)."""
    x = _fwht128_stage_vec(x, 64, 1)
    x = _fwht128_stage_vec(x, 32, 2)
    x = _fwht128_stage_vec(x, 16, 4)
    x = _fwht128_stage_vec(x, 8, 8)
    x = _fwht128_stage_vec(x, 4, 16)
    x = _fwht128_stage_vec(x, 2, 32)
    x = _fwht128_stage_vec(x, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _triton_fwht128_kernel(X_ptr, Out_ptr, n_rows, stride_x, stride_o):
    row = tl.program_id(0).to(tl.int64)
    if row >= n_rows:
        return
    offs = tl.arange(0, 128)
    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)
    y = _fwht128_vec(x)
    tl.store(Out_ptr + row * stride_o + offs, y.to(Out_ptr.dtype.element_ty))


@triton.jit
def _triton_kpool_compress_fwht128_kernel(
    Key_ptr,
    Gate_ptr,
    Ape_ptr,
    Starts_ptr,
    Out_ptr,
    n_pools,
    stride_k,
    stride_g,
    stride_a,
    stride_o,
    POOL_SIZE: tl.constexpr,
):
    """Single-pass online-softmax weighted key pooling + fused FWHT-128 in registers."""
    pid = tl.program_id(0).to(tl.int64)
    if pid >= n_pools:
        return
    start = tl.load(Starts_ptr + pid).to(tl.int64)
    offs = tl.arange(0, 128)

    max_l = tl.full((128,), -float("inf"), dtype=tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        g = tl.load(Gate_ptr + (start + slot) * stride_g + offs).to(tl.float32)
        a = tl.load(Ape_ptr + slot * stride_a + offs).to(tl.float32)
        max_l = tl.maximum(max_l, g + a)

    pooled = tl.zeros((128,), dtype=tl.float32)
    norm = tl.zeros((128,), dtype=tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        g = tl.load(Gate_ptr + (start + slot) * stride_g + offs).to(tl.float32)
        a = tl.load(Ape_ptr + slot * stride_a + offs).to(tl.float32)
        w = tl.exp((g + a) - max_l)
        k = tl.load(Key_ptr + (start + slot) * stride_k + offs).to(tl.float32)
        pooled += k * w
        norm += w

    # Round to key dtype before FWHT to match reference precision.
    pooled = (pooled / norm).to(Out_ptr.dtype.element_ty).to(tl.float32)
    rotated = _fwht128_vec(pooled)
    tl.store(Out_ptr + pid * stride_o + offs, rotated.to(Out_ptr.dtype.element_ty))


@torch.library.custom_op("prime_rl::hopper_fwht128", mutates_args=())
def _fwht128_op(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x2 = x.contiguous().view(-1, 128)
    n_rows = x2.shape[0]
    out = torch.empty_like(x2)
    if n_rows > 0:
        _triton_fwht128_kernel[(n_rows,)](x2, out, n_rows, x2.stride(0), out.stride(0), num_warps=2)
    return out.view(orig_shape)


@_fwht128_op.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::hopper_kpool_compress_fwht128", mutates_args=())
def _kpool_compress_fwht128_op(
    key: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    pool_start_tokens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    key_c = key.contiguous()
    gate_c = gate.contiguous()
    ape_c = ape.contiguous()
    starts_c = pool_start_tokens.contiguous()
    n_pools = starts_c.shape[0]
    out = torch.empty((n_pools, 128), device=key_c.device, dtype=key_c.dtype)
    if n_pools > 0:
        _triton_kpool_compress_fwht128_kernel[(n_pools,)](
            key_c,
            gate_c,
            ape_c,
            starts_c,
            out,
            n_pools,
            key_c.stride(0),
            gate_c.stride(0),
            ape_c.stride(0),
            out.stride(0),
            POOL_SIZE=pool_size,
            num_warps=2,
        )
    return out


@_kpool_compress_fwht128_op.register_fake
def _(
    key: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    pool_start_tokens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    return key.new_empty((pool_start_tokens.shape[0], 128))


def triton_fwht128(x: torch.Tensor) -> torch.Tensor:
    return _fwht128_op(x)


def triton_kpool_compress_fwht128(
    key: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    pool_start_tokens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    return _kpool_compress_fwht128_op(key, gate, ape, pool_start_tokens, pool_size)


__all__ = ["triton_fwht128", "triton_kpool_compress_fwht128"]
