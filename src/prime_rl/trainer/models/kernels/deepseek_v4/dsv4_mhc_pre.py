"""Fused mHC input side for DeepSeek-V4: RMS norm, gate projection and pre-collapse of the streams.

For streams `x` of shape `(T, hc, D)` (`K = hc * D` per token), projection `fn` of shape `(N, K)`
and per-column `scale` and `base` of shape `(N,)`:

    rstd      = rsqrt(mean(x ** 2) + norm_eps)                        per token, fp32
    mixes     = (x @ fn.T) * rstd                                     (T, N), fp32
    logits    = mixes * scale + base                                  (T, N), fp32
    pre       = sigmoid(logits[:, :hc]) + hc_eps
    collapsed = sum_i pre[:, i] * x[:, i, :]                          (T, D), x.dtype

The RMS norm is folded into the projection, so the fp32 normalized streams are never built.
Autograd keeps `x`, `mixes` and `rstd`. Each direction is one kernel over the tokens that reads the
streams twice (projection, then collapse or input gradient); the weight gradient is a second kernel
that sums fixed split-token partials. The projection follows `torch.get_float32_matmul_precision`
(TF32 unless it is "highest"), like the fp32 `F.linear` it replaces.
"""

import torch
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


@triton.autotune(configs=_configs(), key=["HC", "D", "N", "PRECISION"])
@triton.jit
def _fwd_kernel(
    x_ptr,
    w_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + offs_d
        d_mask = d < D
        for i in tl.static_range(HC):
            k = i * D + d
            x = tl.load(x_ptr + rows64[:, None] * K + k[None, :], mask=row_mask[:, None] & d_mask[None, :], other=0.0)
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


@triton.autotune(configs=_configs(), key=["HC", "D", "N", "PRECISION", "HAS_GRAD_LOGITS"])
@triton.jit
def _bwd_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    base_ptr,
    mixes_ptr,
    rstd_ptr,
    grad_logits_ptr,
    grad_collapsed_ptr,
    grad_x_ptr,
    grad_z_ptr,
    grad_proj_ptr,
    T,
    hc_eps,
    HC: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    PRECISION: tl.constexpr,
    HAS_GRAD_LOGITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Writes `grad_x`, the logits' total gradient `grad_z` (T, N) and `grad_proj = grad_z * scale * rstd`
    (T, BLOCK_N, zero-padded), the operand of the weight gradient."""
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
    coef = -rstd * rstd * tl.sum(grad_mixes * mixes, axis=1) / K
    grad_proj = grad_mixes * rstd[:, None]
    tl.store(grad_proj_ptr + rows64[:, None] * BLOCK_N + offs_n[None, :], grad_proj, mask=row_mask[:, None])

    pre = sig + hc_eps
    for d0 in range(0, D, BLOCK_D):
        d = d0 + offs_d
        d_mask = d < D
        mask = row_mask[:, None] & d_mask[None, :]
        grad_c = tl.load(grad_collapsed_ptr + rows64[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
        for i in tl.static_range(HC):
            k = i * D + d
            pre_i = tl.sum(tl.where(offs_n[None, :] == i, pre, 0.0), axis=1)
            x = tl.load(x_ptr + rows64[:, None] * K + k[None, :], mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs_n[:, None] * K + k[None, :], mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            grad_x = tl.dot(grad_proj, w.to(tl.float32), input_precision=PRECISION)
            grad_x += pre_i[:, None] * grad_c + coef[:, None] * x
            tl.store(grad_x_ptr + rows64[:, None] * K + k[None, :], grad_x.to(grad_x_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for bm in (32, 64)
        for bk in (64, 128)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["K", "BLOCK_N", "PRECISION"],
)
@triton.jit
def _bwd_weight_kernel(
    x_ptr,
    grad_proj_ptr,
    partial_ptr,
    T,
    K: tl.constexpr,
    PRECISION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid: (cdiv(K, BLOCK_K), SPLITS). `partial[split] = grad_proj[tokens].T @ x[tokens]` over one split's tokens."""
    k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k < K
    split = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    m_tiles = tl.cdiv(T, BLOCK_M)
    per_split = tl.cdiv(m_tiles, SPLITS)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m_tile in range(split * per_split, tl.minimum((split + 1) * per_split, m_tiles)):
        rows = (m_tile * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        row_mask = rows < T
        g = tl.load(grad_proj_ptr + rows[:, None] * BLOCK_N + offs_n[None, :], mask=row_mask[:, None], other=0.0)
        x = tl.load(x_ptr + rows[:, None] * K + k[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        acc = tl.dot(tl.trans(g), x.to(tl.float32), acc, input_precision=PRECISION)
    tl.store(partial_ptr + (split * BLOCK_N + offs_n[:, None]) * K + k[None, :], acc, mask=k_mask[None, :])


def _precision() -> str:
    return "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"


def _block_n(n: int) -> int:
    return max(16, triton.next_power_of_2(n))


def _grid(num_tokens):
    return lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_M"]),)


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
    _fwd_kernel[_grid(num_tokens)](
        x, fn, scale, base, mixes, rstd, logits, collapsed, num_tokens, norm_eps, hc_eps,
        HC=hc, D=hidden, N=n, PRECISION=_precision(), BLOCK_N=_block_n(n),
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
    block_n, precision = _block_n(n), _precision()
    grad_x = torch.empty_like(x)
    grad_z = torch.empty_like(mixes)
    grad_proj = mixes.new_empty(num_tokens, block_n)
    _bwd_kernel[_grid(num_tokens)](
        x, fn, scale, base, mixes, rstd, grad_logits, grad_collapsed, grad_x, grad_z, grad_proj, num_tokens, hc_eps,
        HC=hc, D=hidden, N=n, PRECISION=precision, HAS_GRAD_LOGITS=grad_logits is not None, BLOCK_N=block_n,
    )  # fmt: skip
    k = hc * hidden
    partial = mixes.new_empty(DW_SPLITS, block_n, k)
    _bwd_weight_kernel[lambda meta: (triton.cdiv(k, meta["BLOCK_K"]), DW_SPLITS)](
        x, grad_proj, partial, num_tokens, K=k, PRECISION=precision, BLOCK_N=block_n, SPLITS=DW_SPLITS
    )
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
