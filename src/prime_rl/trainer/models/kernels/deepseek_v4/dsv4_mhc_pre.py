"""Fused mHC input side for DeepSeek-V4: RMS norm, gate projection, pre-collapse of the streams and, optionally,
the sublayer's weighted input RMSNorm.

For streams `x` of shape `(T, hc, D)` (`K = hc * D` per token), projection `fn` of shape `(N, K)`
and per-column `scale` and `base` of shape `(N,)`:

    rstd      = rsqrt(mean(x ** 2) + norm_eps)                        per token, fp32
    mixes     = (x @ fn.T) * rstd                                     (T, N), fp32
    logits    = mixes * scale + base                                  (T, N), fp32
    pre       = sigmoid(logits[:, :hc]) + hc_eps
    c         = sum_i pre[:, i] * x[:, i, :]                          (T, D), fp32
    out       = c                                                     (T, D), x.dtype, without a norm weight
    out       = bf16(c) * rsqrt(mean(c ** 2) + out_eps) * weight      with one, as vLLM's fused mHC pre-pass

With a norm weight the statistic comes from the fp32 collapse and normalizes its bf16 rounding, as vLLM serves it,
so the sublayer's RMSNorm runs inside this op instead of as a separate kernel on the rounded collapse.

The forward projection and sum of squares come from DeepGEMM's TF32 `tf32_hc_prenorm_gemm` on SM90 and SM100, the
kernel vLLM runs for mHC, with `fn` zero-padded to a multiple of 8 rows; `PRIME_RL_DSV4_MHC_PAD_PROJECTION=0`
restores the multiple-of-8 gate, keeping such projections on the IEEE Triton path. Both operands are bf16-valued
under FSDP, so its TF32 products are exact. A second kernel, one token per program, reduces DeepGEMM's split-K partials and
collapses (and normalizes) the token's streams. Autograd keeps `x`, `mixes` and `rstd`.

The backward reads the streams twice. A token-blocked kernel reduces the output gradient against the streams into
the gate gradients (recomputing the collapse for the norm's terms), then a kernel over hidden slices writes the
input gradient and accumulates the weight gradients over fixed token splits from one more read of the streams.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DW_SPLITS = 16


@triton.autotune(configs=[triton.Config({}, num_warps=nw) for nw in (4, 8, 16)], key=["HC", "D", "N", "NORM", "SPLITS"])
@triton.jit
def _fwd_kernel(
    projection_ptr,
    sqrsum_ptr,
    x_ptr,
    scale_ptr,
    base_ptr,
    norm_weight_ptr,
    mixes_ptr,
    rstd_ptr,
    logits_ptr,
    out_ptr,
    T,
    norm_eps,
    hc_eps,
    out_eps,
    HC: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    PROJECTION_N: tl.constexpr,
    SPLITS: tl.constexpr,
    NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One token per program: reduce the split-K partials in order, then collapse (and normalize) its streams."""
    K: tl.constexpr = HC * D
    t = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    projection = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sumsq = 0.0
    for s in tl.static_range(SPLITS):
        projection += tl.load(projection_ptr + (s * T + t) * PROJECTION_N + offs_n, mask=n_mask, other=0.0)
        sumsq += tl.load(sqrsum_ptr + s * T + t)
    rstd = tl.rsqrt(sumsq / K + norm_eps)
    mixes = projection * rstd
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    logits = mixes * scale + base
    tl.store(rstd_ptr + t, rstd)
    tl.store(mixes_ptr + t * N + offs_n, mixes, mask=n_mask)
    tl.store(logits_ptr + t * N + offs_n, logits, mask=n_mask)

    pre = tl.sigmoid(logits) + hc_eps
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    c = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in tl.static_range(HC):
        pre_i = tl.sum(tl.where(offs_n == i, pre, 0.0))
        x = tl.load(x_ptr + t * K + i * D + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        c += pre_i * x
    if NORM:
        norm_rstd = tl.rsqrt(tl.sum(c * c) / D + out_eps)
        weight = tl.load(norm_weight_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        c = c.to(out_ptr.dtype.element_ty).to(tl.float32) * norm_rstd * weight
    tl.store(out_ptr + t * D + offs_d, c.to(out_ptr.dtype.element_ty), mask=d_mask)


def _configs(**extra):
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_D": bd, **extra}, num_warps=nw, num_stages=ns)
        for bm in (32, 64)
        for bd in (64, 128)
        for nw in (4, 8)
        for ns in (2, 3)
    ]


@triton.autotune(configs=_configs(), key=["HC", "D", "N", "HAS_GRAD_LOGITS", "NORM"])
@triton.jit
def _bwd_gate_kernel(
    x_ptr,
    scale_ptr,
    base_ptr,
    norm_weight_ptr,
    mixes_ptr,
    rstd_ptr,
    grad_logits_ptr,
    grad_out_ptr,
    grad_z_ptr,
    grad_proj_ptr,
    pre_ptr,
    coef_ptr,
    norm_coef_ptr,
    T,
    hc_eps,
    out_eps,
    HC: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    HAS_GRAD_LOGITS: tl.constexpr,
    NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Writes the logits' total gradient `grad_z` (T, N), `grad_proj = grad_z * scale * rstd` (T, BLOCK_N,
    zero-padded), the gates `pre` (T, HC) and the coefficient `coef` (T,) of `x` in the input gradient. With a norm,
    `grad_out` is the gradient of the normalized output, the collapse is rebuilt from `x` and `pre`, and
    `norm_coef` (T, 2) holds `(r, r ** 3 * sum(grad_out * weight * bf16(c)) / D)`, so the collapse's gradient is
    `r * weight * grad_out - norm_coef[:, 1] * c`."""
    K: tl.constexpr = HC * D
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < T
    rows64 = rows.to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_d = tl.arange(0, BLOCK_D)
    out_mask = row_mask[:, None] & n_mask[None, :]

    mixes = tl.load(mixes_ptr + rows64[:, None] * N + offs_n[None, :], mask=out_mask, other=0.0)
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(mixes * scale[None, :] + base[None, :])
    pre = sig + hc_eps

    grad_pre = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if NORM:
        tl.static_assert(HC <= 4)
        gates = tl.where(offs_n[None, :] < HC, pre, 0.0)
        pre0 = tl.sum(tl.where(offs_n[None, :] == 0, gates, 0.0), axis=1)
        pre1 = tl.sum(tl.where(offs_n[None, :] == 1, gates, 0.0), axis=1)
        pre2 = tl.sum(tl.where(offs_n[None, :] == 2, gates, 0.0), axis=1)
        pre3 = tl.sum(tl.where(offs_n[None, :] == 3, gates, 0.0), axis=1)
        weighted_x = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        c_x = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        grad_dot_c = tl.zeros((BLOCK_M,), dtype=tl.float32)
        sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + offs_d
            d_mask = d < D
            mask = row_mask[:, None] & d_mask[None, :]
            weight = tl.load(norm_weight_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
            g = tl.load(grad_out_ptr + rows64[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
            g = g * weight[None, :]
            stream = x_ptr + rows64[:, None] * K + d[None, :]
            x0 = _load_stream(stream, mask, 0, HC, D, BLOCK_M, BLOCK_D)
            x1 = _load_stream(stream, mask, 1, HC, D, BLOCK_M, BLOCK_D)
            x2 = _load_stream(stream, mask, 2, HC, D, BLOCK_M, BLOCK_D)
            x3 = _load_stream(stream, mask, 3, HC, D, BLOCK_M, BLOCK_D)
            c = pre0[:, None] * x0 + pre1[:, None] * x1 + pre2[:, None] * x2 + pre3[:, None] * x3
            weighted_x += tl.where(offs_n[None, :] == 0, tl.sum(g * x0, axis=1)[:, None], 0.0)
            weighted_x += tl.where(offs_n[None, :] == 1, tl.sum(g * x1, axis=1)[:, None], 0.0)
            weighted_x += tl.where(offs_n[None, :] == 2, tl.sum(g * x2, axis=1)[:, None], 0.0)
            weighted_x += tl.where(offs_n[None, :] == 3, tl.sum(g * x3, axis=1)[:, None], 0.0)
            c_x += tl.where(offs_n[None, :] == 0, tl.sum(c * x0, axis=1)[:, None], 0.0)
            c_x += tl.where(offs_n[None, :] == 1, tl.sum(c * x1, axis=1)[:, None], 0.0)
            c_x += tl.where(offs_n[None, :] == 2, tl.sum(c * x2, axis=1)[:, None], 0.0)
            c_x += tl.where(offs_n[None, :] == 3, tl.sum(c * x3, axis=1)[:, None], 0.0)
            grad_dot_c += tl.sum(g * c.to(x_ptr.dtype.element_ty).to(tl.float32), axis=1)
            sumsq += tl.sum(c * c, axis=1)
        norm_rstd = tl.rsqrt(sumsq / D + out_eps)
        norm_b = norm_rstd * norm_rstd * norm_rstd * grad_dot_c / D
        grad_pre = norm_rstd[:, None] * weighted_x - norm_b[:, None] * c_x
        tl.store(norm_coef_ptr + rows64 * 2, norm_rstd, mask=row_mask)
        tl.store(norm_coef_ptr + rows64 * 2 + 1, norm_b, mask=row_mask)
    else:
        for d0 in range(0, D, BLOCK_D):
            d = d0 + offs_d
            mask = row_mask[:, None] & (d < D)[None, :]
            g = tl.load(grad_out_ptr + rows64[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
            for i in tl.static_range(HC):
                x = tl.load(x_ptr + rows64[:, None] * K + i * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
                grad_pre += tl.where(offs_n[None, :] == i, tl.sum(g * x, axis=1)[:, None], 0.0)

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
    tl.store(pre_ptr + rows64[:, None] * HC + offs_n[None, :], pre, mask=pre_mask)


@triton.jit
def _load_stream(
    stream, mask, I: tl.constexpr, HC: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    """Stream `I`'s tile in fp32, or zeros for a stream past `HC`."""
    if I < HC:
        return tl.load(stream + I * D, mask=mask, other=0.0).to(tl.float32)
    else:
        return tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)


@triton.jit
def _round_to_tf32(x):
    """Round to nearest (ties away) at TF32's 10 mantissa bits, so the MMA's truncation does not bias the dot."""
    return (x.to(tl.int32, bitcast=True) + 0x1000).to(tl.float32, bitcast=True)


@triton.jit
def _bwd_stream(
    w_ptr,
    pre_ptr,
    grad_x_ptr,
    x,
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
        w = tl.load(w_ptr + offs_n[:, None] * K + k[None, :], mask=w_mask, other=0.0).to(tl.float32)
        if ROUND_TF32:
            w = _round_to_tf32(w)
        grad_x = tl.dot(grad_proj, w, input_precision=PRECISION) + pre[:, None] * grad_c + coef[:, None] * x
        tl.store(grad_x_ptr + rows[:, None] * K + k[None, :], grad_x.to(grad_x_ptr.dtype.element_ty), mask=mask)
        acc = tl.dot(tl.trans(grad_proj), x, acc, input_precision=PRECISION)
    return acc


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_D": bd}, num_warps=nw, num_stages=2)
        for bm in (32, 64)
        for bd in (32, 64, 128)
        for nw in (4, 8)
    ],
    key=["D", "BLOCK_N", "PRECISION", "ROUND_TF32", "SPLITS", "NORM"],
)
@triton.jit
def _bwd_input_weight_kernel(
    x_ptr,
    w_ptr,
    norm_weight_ptr,
    grad_out_ptr,
    grad_proj_ptr,
    pre_ptr,
    coef_ptr,
    norm_coef_ptr,
    grad_x_ptr,
    partial_ptr,
    norm_partial_ptr,
    T,
    N,
    HC: tl.constexpr,
    D: tl.constexpr,
    PRECISION: tl.constexpr,
    ROUND_TF32: tl.constexpr,
    NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Grid: (cdiv(D, BLOCK_D), SPLITS). For one hidden slice of every stream over one split's tokens, writes
    `grad_x = grad_proj @ fn + pre * grad_c + coef * x` and `partial[split] = grad_proj.T @ x`, so the streams are
    read once for both gradients. With a norm, `grad_c` is rebuilt from the normalized output's gradient and the
    recomputed collapse, and `norm_partial[split]` accumulates the norm weight's gradient."""
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
    acc_norm = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if NORM:
        weight = tl.load(norm_weight_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    for m_tile in range(split * per_split, tl.minimum((split + 1) * per_split, m_tiles)):
        rows = (m_tile * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        row_mask = rows < T
        mask = row_mask[:, None] & d_mask[None, :]
        grad_proj = tl.load(
            grad_proj_ptr + rows[:, None] * BLOCK_N + offs_n[None, :], mask=row_mask[:, None], other=0.0
        )
        g = tl.load(grad_out_ptr + rows[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
        coef = tl.load(coef_ptr + rows, mask=row_mask, other=0.0)
        if ROUND_TF32:
            grad_proj = _round_to_tf32(grad_proj)
        stream = x_ptr + rows[:, None] * K + d[None, :]
        x0 = _load_stream(stream, mask, 0, HC, D, BLOCK_M, BLOCK_D)
        x1 = _load_stream(stream, mask, 1, HC, D, BLOCK_M, BLOCK_D)
        x2 = _load_stream(stream, mask, 2, HC, D, BLOCK_M, BLOCK_D)
        x3 = _load_stream(stream, mask, 3, HC, D, BLOCK_M, BLOCK_D)
        if NORM:
            gates = tl.load(
                pre_ptr + rows[:, None] * HC + tl.arange(0, 4)[None, :],
                mask=row_mask[:, None] & (tl.arange(0, 4) < HC)[None, :],
                other=0.0,
            )
            c = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 0, gates, 0.0), axis=1)[:, None] * x0
            c += tl.sum(tl.where(tl.arange(0, 4)[None, :] == 1, gates, 0.0), axis=1)[:, None] * x1
            c += tl.sum(tl.where(tl.arange(0, 4)[None, :] == 2, gates, 0.0), axis=1)[:, None] * x2
            c += tl.sum(tl.where(tl.arange(0, 4)[None, :] == 3, gates, 0.0), axis=1)[:, None] * x3
            norm_rstd = tl.load(norm_coef_ptr + rows * 2, mask=row_mask, other=0.0)
            norm_b = tl.load(norm_coef_ptr + rows * 2 + 1, mask=row_mask, other=0.0)
            acc_norm += tl.sum(g * c.to(x_ptr.dtype.element_ty).to(tl.float32) * norm_rstd[:, None], axis=0)
            grad_c = norm_rstd[:, None] * weight[None, :] * g - norm_b[:, None] * c
        else:
            grad_c = g
        acc0 = _bwd_stream(w_ptr, pre_ptr, grad_x_ptr, x0, rows, row_mask, mask, w_mask, d, grad_proj, grad_c, coef, acc0,
                           0, HC, D, PRECISION, ROUND_TF32, BLOCK_N)  # fmt: skip
        acc1 = _bwd_stream(w_ptr, pre_ptr, grad_x_ptr, x1, rows, row_mask, mask, w_mask, d, grad_proj, grad_c, coef, acc1,
                           1, HC, D, PRECISION, ROUND_TF32, BLOCK_N)  # fmt: skip
        acc2 = _bwd_stream(w_ptr, pre_ptr, grad_x_ptr, x2, rows, row_mask, mask, w_mask, d, grad_proj, grad_c, coef, acc2,
                           2, HC, D, PRECISION, ROUND_TF32, BLOCK_N)  # fmt: skip
        acc3 = _bwd_stream(w_ptr, pre_ptr, grad_x_ptr, x3, rows, row_mask, mask, w_mask, d, grad_proj, grad_c, coef, acc3,
                           3, HC, D, PRECISION, ROUND_TF32, BLOCK_N)  # fmt: skip
    out = partial_ptr + (split * BLOCK_N + offs_n[:, None]) * K + d[None, :]
    tl.store(out, acc0, mask=d_mask[None, :])
    if HC > 1:
        tl.store(out + D, acc1, mask=d_mask[None, :])
    if HC > 2:
        tl.store(out + 2 * D, acc2, mask=d_mask[None, :])
    if HC > 3:
        tl.store(out + 3 * D, acc3, mask=d_mask[None, :])
    if NORM:
        tl.store(norm_partial_ptr + split * D + d, acc_norm, mask=d_mask)


def _precision() -> str:
    return "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"


def _backward_precision() -> str:
    """`PRIME_RL_DSV4_MHC_BWD_PRECISION` (a `tl.dot` input precision) overrides the matmul precision for the
    backward's dots."""
    return os.environ.get("PRIME_RL_DSV4_MHC_BWD_PRECISION") or _precision()


def _pads_projection() -> bool:
    return os.environ.get("PRIME_RL_DSV4_MHC_PAD_PROJECTION", "1") != "0"


def _rounds_to_tf32(precision: str) -> bool:
    return precision == "tf32"


def _block_n(n: int) -> int:
    return max(16, triton.next_power_of_2(n))


def _grid(num_tokens):
    return lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_M"]),)


def _uses_deep_gemm_projection(x: torch.Tensor, fn: torch.Tensor) -> bool:
    return (
        torch.cuda.get_device_capability(x.device)[0] in (9, 10)
        and x.dtype == torch.bfloat16
        and fn.shape[0] <= 32
        and fn.shape[1] % 64 == 0
        and (fn.shape[0] % 8 == 0 or _pads_projection())
    )


def _projection(x: torch.Tensor, fn: torch.Tensor, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-K partials of `x @ fn.T` (splits, T, padded N) and of the per-token sum of squares (splits, T).

    DeepGEMM splits K into 1, 4 or 16 parts like vLLM, so short rows still fill the GPU and it JIT-compiles at most
    three variants. `fn` is zero-padded to the kernel's multiple of 8 rows, as DeepSeek's TileKernels does for
    `hc_head`. Elsewhere one fp32 matmul follows `torch.get_float32_matmul_precision`."""
    n, k = fn.shape
    flat = x.view(num_tokens, k)
    if not _uses_deep_gemm_projection(x, fn):
        flat = flat.float()
        return (flat @ fn.float().T).unsqueeze(0), flat.square().sum(-1).unsqueeze(0)
    import deep_gemm

    padded_n = triton.cdiv(n, 8) * 8
    weight = F.pad(fn.float(), (0, 0, 0, padded_n - n))
    token_blocks = triton.cdiv(num_tokens, 64)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    splits = max(1, min(sm_count // token_blocks, triton.cdiv(k, 64) // 4))
    num_splits = 1 if splits == 1 else 4 if splits <= 4 else 16
    projection = x.new_empty(num_splits, num_tokens, padded_n, dtype=torch.float32)
    sqrsum = x.new_empty(num_splits, num_tokens, dtype=torch.float32)
    deep_gemm.tf32_hc_prenorm_gemm(flat, weight, projection, sqrsum, num_splits)
    return projection, sqrsum


@torch.library.custom_op("prime_rl::dsv4_mhc_pre", mutates_args=())
def _mhc_pre(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    hc_eps: float,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    *lead, hc, hidden = x.shape
    num_tokens, n = x.numel() // (hc * hidden), fn.shape[0]
    mixes = x.new_empty(num_tokens, n, dtype=torch.float32)
    rstd = x.new_empty(num_tokens, dtype=torch.float32)
    logits = x.new_empty(*lead, n, dtype=torch.float32)
    out = x.new_empty(*lead, hidden)
    projection, sqrsum = _projection(x, fn, num_tokens)
    _fwd_kernel[(num_tokens,)](
        projection, sqrsum, x, scale, base, norm_weight, mixes, rstd, logits, out, num_tokens, norm_eps, hc_eps, out_eps,
        HC=hc, D=hidden, N=n, PROJECTION_N=projection.shape[-1], SPLITS=projection.shape[0], NORM=norm_weight is not None,
        BLOCK_N=_block_n(n), BLOCK_D=triton.next_power_of_2(hidden),
    )  # fmt: skip
    return out, logits, mixes, rstd


@_mhc_pre.register_fake
def _mhc_pre_fake(x, fn, scale, base, norm_weight, norm_eps, hc_eps, out_eps):
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
    grad_out: torch.Tensor,
    grad_logits: torch.Tensor | None,
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor | None,
    mixes: torch.Tensor,
    rstd: torch.Tensor,
    hc_eps: float,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc, hidden = x.shape[-2:]
    num_tokens, n = mixes.shape
    norm = norm_weight is not None
    block_n, precision = _block_n(n), _backward_precision()
    grad_z = torch.empty_like(mixes)
    grad_proj = mixes.new_empty(num_tokens, block_n)
    pre = mixes.new_empty(num_tokens, hc)
    coef = mixes.new_empty(num_tokens)
    norm_coef = mixes.new_empty(num_tokens, 2) if norm else None
    _bwd_gate_kernel[_grid(num_tokens)](
        x, scale, base, norm_weight, mixes, rstd, grad_logits, grad_out, grad_z, grad_proj, pre, coef, norm_coef,
        num_tokens, hc_eps, out_eps,
        HC=hc, D=hidden, N=n, HAS_GRAD_LOGITS=grad_logits is not None, NORM=norm, BLOCK_N=block_n,
    )  # fmt: skip
    grad_x = torch.empty_like(x)
    partial = mixes.new_empty(DW_SPLITS, block_n, hc * hidden)
    norm_partial = mixes.new_empty(DW_SPLITS, hidden) if norm else None
    _bwd_input_weight_kernel[lambda meta: (triton.cdiv(hidden, meta["BLOCK_D"]), DW_SPLITS)](
        x, fn, norm_weight, grad_out, grad_proj, pre, coef, norm_coef, grad_x, partial, norm_partial, num_tokens, n,
        HC=hc, D=hidden, PRECISION=precision, ROUND_TF32=_rounds_to_tf32(precision), NORM=norm, BLOCK_N=block_n,
        SPLITS=DW_SPLITS,
    )  # fmt: skip
    grad_fn = partial.sum(0)[:n].to(fn.dtype)
    grad_scale = (grad_z * mixes).sum(0).to(scale.dtype)
    grad_base = grad_z.sum(0).to(base.dtype)
    grad_norm_weight = norm_partial.sum(0).to(norm_weight.dtype) if norm else x.new_empty(0)
    return grad_x, grad_fn, grad_scale, grad_base, grad_norm_weight


@_mhc_pre_backward.register_fake
def _mhc_pre_backward_fake(grad_out, grad_logits, x, fn, scale, base, norm_weight, mixes, rstd, hc_eps, out_eps):
    grad_norm_weight = torch.empty_like(norm_weight) if norm_weight is not None else x.new_empty(0)
    return torch.empty_like(x), torch.empty_like(fn), torch.empty_like(scale), torch.empty_like(base), grad_norm_weight


def _mhc_pre_setup_context(ctx, inputs, output) -> None:
    x, fn, scale, base, norm_weight, _, hc_eps, out_eps = inputs
    _, _, mixes, rstd = output
    ctx.mark_non_differentiable(mixes, rstd)
    ctx.save_for_backward(x, fn, scale, base, norm_weight, mixes, rstd)
    ctx.hc_eps = hc_eps
    ctx.out_eps = out_eps


def _mhc_pre_autograd_backward(ctx, grad_out, grad_logits, _grad_mixes, _grad_rstd):
    x, fn, scale, base, norm_weight, mixes, rstd = ctx.saved_tensors
    *grads, grad_norm_weight = _mhc_pre_backward(
        grad_out.contiguous(),
        None if grad_logits is None else grad_logits.contiguous(),
        x, fn, scale, base, norm_weight, mixes, rstd, ctx.hc_eps, ctx.out_eps,
    )  # fmt: skip
    return *grads, grad_norm_weight if norm_weight is not None else None, None, None, None


_mhc_pre.register_autograd(_mhc_pre_autograd_backward, setup_context=_mhc_pre_setup_context)


def fused_mhc_pre(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    norm_weight: torch.Tensor | None = None,
    out_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(out, logits)` for streams `x` of shape `(..., hc, D)`; see the module docstring."""
    out, logits, _, _ = _mhc_pre(
        x.contiguous(), fn.contiguous(), scale.contiguous(), base.contiguous(),
        None if norm_weight is None else norm_weight.contiguous(), norm_eps, hc_eps, out_eps,
    )  # fmt: skip
    return out, logits
