"""Fused mHC input side for DeepSeek-V4: RMS norm, gate projection and pre-collapse of the streams.

For streams `x` of shape `(T, hc, D)` (`K = hc * D` per token), projection `fn` of shape `(N, K)`
and per-column `scale` and `base` of shape `(N,)`:

    rstd      = rsqrt(mean(x ** 2) + norm_eps)                        per token, fp32
    mixes     = (x @ fn.T) * rstd                                     (T, N), fp32
    logits    = mixes * scale + base                                  (T, N), fp32
    pre       = sigmoid(logits[:, :hc]) + hc_eps
    collapsed = sum_i pre[:, i] * x[:, i, :]                          (T, D), x.dtype

The RMS norm is folded into the projection, so the fp32 normalized streams are never built.
Autograd keeps `x`, `mixes` and `rstd`. The forward is one kernel over the tokens that reads the
streams twice (projection, then collapse). On SM90 and SM100 the projection and sum of squares come
from DeepGEMM's TF32 `tf32_hc_prenorm_gemm`, the kernel vLLM runs for mHC, with `fn` zero-padded to
its multiple of 8 rows, and the forward kernel only collapses. `PRIME_RL_DSV4_MHC_PAD_PROJECTION=0`
restores the multiple-of-8 gate, keeping such projections on the IEEE Triton path. The backward also reads the streams
twice: a token-blocked kernel reduces the collapse gradient against them into the gate gradients, then
a kernel over hidden slices writes the input gradient and accumulates the weight gradient over fixed
token splits. Off DeepGEMM, and in the backward, the projection follows
`torch.get_float32_matmul_precision` (TF32 unless it is "highest"), like the fp32 `F.linear` it replaces.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DW_SPLITS = 16


def _configs(**extra):
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_D": bd, **extra}, num_warps=nw, num_stages=ns)
        for bm in (32, 64)
        for bd in (64, 128)
        for nw in (4, 8)
        for ns in (2, 3)
    ]


@triton.autotune(configs=_configs(), key=["HC", "D", "N", "PRECISION", "PRECOMPUTED"])
@triton.jit
def _fwd_kernel(
    x_ptr,
    w_ptr,
    projection_ptr,
    sqrsum_ptr,
    scale_ptr,
    base_ptr,
    mixes_ptr,
    rstd_ptr,
    logits_ptr,
    collapsed_ptr,
    T,
    norm_eps,
    hc_eps,
    HC: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    PRECISION: tl.constexpr,
    PRECOMPUTED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    K: tl.constexpr = HC * D
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < T
    rows64 = rows.to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_d = tl.arange(0, BLOCK_D)

    if PRECOMPUTED:
        acc = tl.load(
            projection_ptr + rows64[:, None] * N + offs_n[None, :], mask=row_mask[:, None] & n_mask[None, :], other=0.0
        )
        sumsq = tl.load(sqrsum_ptr + rows, mask=row_mask, other=0.0)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + offs_d
            d_mask = d < D
            for i in tl.static_range(HC):
                k = i * D + d
                x = tl.load(
                    x_ptr + rows64[:, None] * K + k[None, :], mask=row_mask[:, None] & d_mask[None, :], other=0.0
                )
                x = x.to(tl.float32)
                w = tl.load(w_ptr + offs_n[:, None] * K + k[None, :], mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                acc = tl.dot(x, tl.trans(w.to(tl.float32)), acc, input_precision=PRECISION)
                sumsq += tl.sum(x * x, axis=1)

    rstd = tl.rsqrt(sumsq / K + norm_eps)
    mixes = acc * rstd[:, None]
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    logits = mixes * scale[None, :] + base[None, :]
    out_mask = row_mask[:, None] & n_mask[None, :]
    tl.store(rstd_ptr + rows, rstd, mask=row_mask)
    tl.store(mixes_ptr + rows64[:, None] * N + offs_n[None, :], mixes, mask=out_mask)
    tl.store(logits_ptr + rows64[:, None] * N + offs_n[None, :], logits, mask=out_mask)

    pre = tl.sigmoid(logits) + hc_eps
    for d0 in range(0, D, BLOCK_D):
        d = d0 + offs_d
        mask = row_mask[:, None] & (d < D)[None, :]
        collapsed = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for i in tl.static_range(HC):
            pre_i = tl.sum(tl.where(offs_n[None, :] == i, pre, 0.0), axis=1)
            x = tl.load(x_ptr + rows64[:, None] * K + i * D + d[None, :], mask=mask, other=0.0)
            collapsed += pre_i[:, None] * x.to(tl.float32)
        tl.store(
            collapsed_ptr + rows64[:, None] * D + d[None, :], collapsed.to(collapsed_ptr.dtype.element_ty), mask=mask
        )


@triton.autotune(configs=_configs(), key=["HC", "D", "N", "HAS_GRAD_LOGITS"])
@triton.jit
def _bwd_gate_kernel(
    x_ptr,
    scale_ptr,
    base_ptr,
    mixes_ptr,
    rstd_ptr,
    grad_logits_ptr,
    grad_collapsed_ptr,
    grad_z_ptr,
    grad_proj_ptr,
    pre_ptr,
    coef_ptr,
    T,
    hc_eps,
    HC: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    HAS_GRAD_LOGITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Writes the logits' total gradient `grad_z` (T, N), `grad_proj = grad_z * scale * rstd` (T, BLOCK_N,
    zero-padded), the gates `pre` (T, HC) and the coefficient `coef` (T,) of `x` in the input gradient."""
    K: tl.constexpr = HC * D
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < T
    rows64 = rows.to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_d = tl.arange(0, BLOCK_D)
    out_mask = row_mask[:, None] & n_mask[None, :]

    grad_pre = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + offs_d
        mask = row_mask[:, None] & (d < D)[None, :]
        grad_c = tl.load(grad_collapsed_ptr + rows64[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
        for i in tl.static_range(HC):
            x = tl.load(x_ptr + rows64[:, None] * K + i * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
            grad_pre += tl.where(offs_n[None, :] == i, tl.sum(grad_c * x, axis=1)[:, None], 0.0)

    mixes = tl.load(mixes_ptr + rows64[:, None] * N + offs_n[None, :], mask=out_mask, other=0.0)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(mixes * scale[None, :] + base[None, :])
    grad_z = grad_pre * sig * (1.0 - sig)
    if HAS_GRAD_LOGITS:
        grad_z += tl.load(grad_logits_ptr + rows64[:, None] * N + offs_n[None, :], mask=out_mask, other=0.0)
    tl.store(grad_z_ptr + rows64[:, None] * N + offs_n[None, :], grad_z, mask=out_mask)
    grad_mixes = grad_z * scale[None, :]
    # rstd's dependence on x contributes `coef * x`.
    tl.store(coef_ptr + rows, -rstd * rstd * tl.sum(grad_mixes * mixes, axis=1) / K, mask=row_mask)
    tl.store(
        grad_proj_ptr + rows64[:, None] * BLOCK_N + offs_n[None, :], grad_mixes * rstd[:, None], mask=row_mask[:, None]
    )
    pre_mask = row_mask[:, None] & (offs_n < HC)[None, :]
    tl.store(pre_ptr + rows64[:, None] * HC + offs_n[None, :], sig + hc_eps, mask=pre_mask)


@triton.jit
def _round_to_tf32(x):
    """Round to nearest (ties away) at TF32's 10 mantissa bits, so the MMA's truncation does not bias the dot."""
    return (x.to(tl.int32, bitcast=True) + 0x1000).to(tl.float32, bitcast=True)


@triton.jit
def _bwd_stream(
    x_ptr,
    w_ptr,
    pre_ptr,
    grad_x_ptr,
    rows,
    row_mask,
    mask,
    w_mask,
    d,
    grad_proj,
    grad_c,
    coef,
    acc,
    I: tl.constexpr,
    HC: tl.constexpr,
    D: tl.constexpr,
    PRECISION: tl.constexpr,
    ROUND_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One stream's slice of `grad_x` (stored) and of the weight gradient (returned, accumulated onto `acc`).
    Streams `I >= HC` do not exist and leave `acc` unchanged."""
    if I < HC:
        K: tl.constexpr = HC * D
        k = I * D + d
        offs_n = tl.arange(0, BLOCK_N)
        pre = tl.load(pre_ptr + rows * HC + I, mask=row_mask, other=0.0)
        x = tl.load(x_ptr + rows[:, None] * K + k[None, :], mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * K + k[None, :], mask=w_mask, other=0.0).to(tl.float32)
        if ROUND_TF32:
            w = _round_to_tf32(w)
        grad_x = tl.dot(grad_proj, w, input_precision=PRECISION) + pre[:, None] * grad_c + coef[:, None] * x
        tl.store(grad_x_ptr + rows[:, None] * K + k[None, :], grad_x.to(grad_x_ptr.dtype.element_ty), mask=mask)
        acc = tl.dot(tl.trans(grad_proj), x, acc, input_precision=PRECISION)
    return acc


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_D": bd}, num_warps=nw, num_stages=ns)
        for bm in (32, 64)
        for bd in (64, 128)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["D", "BLOCK_N", "PRECISION", "ROUND_TF32", "SPLITS"],
)
@triton.jit
def _bwd_input_weight_kernel(
    x_ptr,
    w_ptr,
    grad_collapsed_ptr,
    grad_proj_ptr,
    pre_ptr,
    coef_ptr,
    grad_x_ptr,
    partial_ptr,
    T,
    N,
    HC: tl.constexpr,
    D: tl.constexpr,
    PRECISION: tl.constexpr,
    ROUND_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Grid: (cdiv(D, BLOCK_D), SPLITS). For one hidden slice of every stream (at most 4) over one split's tokens,
    writes `grad_x = grad_proj @ fn + pre * grad_collapsed + coef * x` and `partial[split] = grad_proj.T @ x`, so the
    streams are read once for both gradients."""
    tl.static_assert(HC <= 4)
    K: tl.constexpr = HC * D
    d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < D
    split = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    w_mask = (offs_n < N)[:, None] & d_mask[None, :]
    m_tiles = tl.cdiv(T, BLOCK_M)
    per_split = tl.cdiv(m_tiles, SPLITS)
    acc0 = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for m_tile in range(split * per_split, tl.minimum((split + 1) * per_split, m_tiles)):
        rows = (m_tile * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        row_mask = rows < T
        mask = row_mask[:, None] & d_mask[None, :]
        grad_proj = tl.load(
            grad_proj_ptr + rows[:, None] * BLOCK_N + offs_n[None, :], mask=row_mask[:, None], other=0.0
        )
        grad_c = tl.load(grad_collapsed_ptr + rows[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
        coef = tl.load(coef_ptr + rows, mask=row_mask, other=0.0)
        if ROUND_TF32:
            grad_proj = _round_to_tf32(grad_proj)
        acc0 = _bwd_stream(x_ptr, w_ptr, pre_ptr, grad_x_ptr, rows, row_mask, mask, w_mask, d, grad_proj, grad_c, coef, acc0,
                           0, HC, D, PRECISION, ROUND_TF32, BLOCK_N)  # fmt: skip
        acc1 = _bwd_stream(x_ptr, w_ptr, pre_ptr, grad_x_ptr, rows, row_mask, mask, w_mask, d, grad_proj, grad_c, coef, acc1,
                           1, HC, D, PRECISION, ROUND_TF32, BLOCK_N)  # fmt: skip
        acc2 = _bwd_stream(x_ptr, w_ptr, pre_ptr, grad_x_ptr, rows, row_mask, mask, w_mask, d, grad_proj, grad_c, coef, acc2,
                           2, HC, D, PRECISION, ROUND_TF32, BLOCK_N)  # fmt: skip
        acc3 = _bwd_stream(x_ptr, w_ptr, pre_ptr, grad_x_ptr, rows, row_mask, mask, w_mask, d, grad_proj, grad_c, coef, acc3,
                           3, HC, D, PRECISION, ROUND_TF32, BLOCK_N)  # fmt: skip
    out = partial_ptr + (split * BLOCK_N + offs_n[:, None]) * K + d[None, :]
    tl.store(out, acc0, mask=d_mask[None, :])
    if HC > 1:
        tl.store(out + D, acc1, mask=d_mask[None, :])
    if HC > 2:
        tl.store(out + 2 * D, acc2, mask=d_mask[None, :])
    if HC > 3:
        tl.store(out + 3 * D, acc3, mask=d_mask[None, :])


def _precision() -> str:
    return "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"


def _backward_precision() -> str:
    """`PRIME_RL_DSV4_MHC_BWD_PRECISION` (a `tl.dot` input precision) overrides the matmul precision for the
    backward's dots."""
    return os.environ.get("PRIME_RL_DSV4_MHC_BWD_PRECISION") or _precision()


def _rounds_to_tf32(precision: str) -> bool:
    return precision == "tf32"


def _block_n(n: int) -> int:
    return max(16, triton.next_power_of_2(n))


def _grid(num_tokens):
    return lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_M"]),)


def _pads_projection() -> bool:
    return os.environ.get("PRIME_RL_DSV4_MHC_PAD_PROJECTION", "1") != "0"


def _uses_deep_gemm_projection(x: torch.Tensor, fn: torch.Tensor) -> bool:
    return (
        torch.cuda.get_device_capability(x.device)[0] in (9, 10)
        and x.dtype == torch.bfloat16
        and fn.shape[0] <= 32
        and fn.shape[1] % 64 == 0
        and (fn.shape[0] % 8 == 0 or _pads_projection())
    )


def _deep_gemm_projection(x: torch.Tensor, fn: torch.Tensor, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`x @ fn.T` and the per-token sum of squares, split over K into 1, 4 or 16 parts like vLLM, so short rows
    still fill the GPU and DeepGEMM JIT-compiles at most three variants. `fn` is zero-padded to the kernel's
    multiple of 8 rows, as DeepSeek's TileKernels does for `hc_head`."""
    import deep_gemm

    n, k = fn.shape
    padded_n = triton.cdiv(n, 8) * 8
    weight = F.pad(fn.float(), (0, 0, 0, padded_n - n)) if padded_n != n else fn.float()
    token_blocks = triton.cdiv(num_tokens, 64)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    splits = max(1, min(sm_count // token_blocks, triton.cdiv(k, 64) // 4))
    num_splits = 1 if splits == 1 else 4 if splits <= 4 else 16
    projection = x.new_empty(num_splits, num_tokens, padded_n, dtype=torch.float32)
    sqrsum = x.new_empty(num_splits, num_tokens, dtype=torch.float32)
    deep_gemm.tf32_hc_prenorm_gemm(x.view(num_tokens, k), weight, projection, sqrsum, num_splits)
    return projection.sum(0)[:, :n].contiguous(), sqrsum.sum(0)


@torch.library.custom_op("prime_rl::dsv4_mhc_pre", mutates_args=())
def _mhc_pre(
    x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, norm_eps: float, hc_eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    *lead, hc, hidden = x.shape
    num_tokens, n = x.numel() // (hc * hidden), fn.shape[0]
    mixes = x.new_empty(num_tokens, n, dtype=torch.float32)
    rstd = x.new_empty(num_tokens, dtype=torch.float32)
    logits = x.new_empty(*lead, n, dtype=torch.float32)
    collapsed = x.new_empty(*lead, hidden)
    precomputed = _uses_deep_gemm_projection(x, fn)
    projection, sqrsum = _deep_gemm_projection(x, fn, num_tokens) if precomputed else (None, None)
    _fwd_kernel[_grid(num_tokens)](
        x, fn, projection, sqrsum, scale, base, mixes, rstd, logits, collapsed, num_tokens, norm_eps, hc_eps,
        HC=hc, D=hidden, N=n, PRECISION=_precision(), PRECOMPUTED=precomputed, BLOCK_N=_block_n(n),
    )  # fmt: skip
    return collapsed, logits, mixes, rstd


@_mhc_pre.register_fake
def _mhc_pre_fake(x, fn, scale, base, norm_eps, hc_eps):
    *lead, hc, hidden = x.shape
    num_tokens, n = x.numel() // (hc * hidden), fn.shape[0]
    return (
        x.new_empty(*lead, hidden),
        x.new_empty(*lead, n, dtype=torch.float32),
        x.new_empty(num_tokens, n, dtype=torch.float32),
        x.new_empty(num_tokens, dtype=torch.float32),
    )


@torch.library.custom_op("prime_rl::dsv4_mhc_pre_backward", mutates_args=())
def _mhc_pre_backward(
    grad_collapsed: torch.Tensor,
    grad_logits: torch.Tensor | None,
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    mixes: torch.Tensor,
    rstd: torch.Tensor,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc, hidden = x.shape[-2:]
    num_tokens, n = mixes.shape
    block_n, precision = _block_n(n), _backward_precision()
    grad_z = torch.empty_like(mixes)
    grad_proj = mixes.new_empty(num_tokens, block_n)
    pre = mixes.new_empty(num_tokens, hc)
    coef = mixes.new_empty(num_tokens)
    _bwd_gate_kernel[_grid(num_tokens)](
        x, scale, base, mixes, rstd, grad_logits, grad_collapsed, grad_z, grad_proj, pre, coef, num_tokens, hc_eps,
        HC=hc, D=hidden, N=n, HAS_GRAD_LOGITS=grad_logits is not None, BLOCK_N=block_n,
    )  # fmt: skip
    grad_x = torch.empty_like(x)
    partial = mixes.new_empty(DW_SPLITS, block_n, hc * hidden)
    _bwd_input_weight_kernel[lambda meta: (triton.cdiv(hidden, meta["BLOCK_D"]), DW_SPLITS)](
        x, fn, grad_collapsed, grad_proj, pre, coef, grad_x, partial, num_tokens, n,
        HC=hc, D=hidden, PRECISION=precision, ROUND_TF32=_rounds_to_tf32(precision), BLOCK_N=block_n, SPLITS=DW_SPLITS,
    )  # fmt: skip
    grad_fn = partial.sum(0)[:n].to(fn.dtype)
    grad_scale = (grad_z * mixes).sum(0).to(scale.dtype)
    grad_base = grad_z.sum(0).to(base.dtype)
    return grad_x, grad_fn, grad_scale, grad_base


@_mhc_pre_backward.register_fake
def _mhc_pre_backward_fake(grad_collapsed, grad_logits, x, fn, scale, base, mixes, rstd, hc_eps):
    return torch.empty_like(x), torch.empty_like(fn), torch.empty_like(scale), torch.empty_like(base)


def _mhc_pre_setup_context(ctx, inputs, output) -> None:
    x, fn, scale, base, _, hc_eps = inputs
    _, _, mixes, rstd = output
    ctx.mark_non_differentiable(mixes, rstd)
    ctx.save_for_backward(x, fn, scale, base, mixes, rstd)
    ctx.hc_eps = hc_eps


def _mhc_pre_autograd_backward(ctx, grad_collapsed, grad_logits, _grad_mixes, _grad_rstd):
    x, fn, scale, base, mixes, rstd = ctx.saved_tensors
    grads = _mhc_pre_backward(
        grad_collapsed.contiguous(),
        None if grad_logits is None else grad_logits.contiguous(),
        x, fn, scale, base, mixes, rstd, ctx.hc_eps,
    )  # fmt: skip
    return *grads, None, None


_mhc_pre.register_autograd(_mhc_pre_autograd_backward, setup_context=_mhc_pre_setup_context)


def fused_mhc_pre(
    x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, norm_eps: float, hc_eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(collapsed, logits)` for streams `x` of shape `(..., hc, D)`; see the module docstring."""
    collapsed, logits, _, _ = _mhc_pre(
        x.contiguous(), fn.contiguous(), scale.contiguous(), base.contiguous(), norm_eps, hc_eps
    )
    return collapsed, logits
