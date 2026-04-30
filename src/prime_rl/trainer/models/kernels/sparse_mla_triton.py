import torch
import triton
import triton.language as tl

_D = 512
_D_TAIL = 64
_D_TOTAL = _D + _D_TAIL
_LOG2E = 1.4426950408889634


@triton.jit
def _sparse_mla_fwd_kernel(
    Q,
    KV,
    Indices,
    Output,
    Lse,
    S: tl.constexpr,
    S_KV: tl.constexpr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SM_SCALE_LOG2: tl.constexpr,
    D: tl.constexpr,
    D_TAIL: tl.constexpr,
    D_TOTAL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_tail = tl.arange(0, D_TAIL)

    q_base = ((pid_b * S + pid_s) * H + offs_h[:, None]) * D_TOTAL
    q = tl.load(Q + q_base + offs_d[None, :], mask=offs_h[:, None] < H, other=0.0)
    q_tail = tl.load(Q + q_base + D + offs_tail[None, :], mask=offs_h[:, None] < H, other=0.0)

    m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, D), tl.float32)

    for start_n in tl.range(0, TOPK, BLOCK_N, num_stages=2):
        sparse_offsets = start_n + offs_n
        idx = tl.load(Indices + (pid_b * S + pid_s) * TOPK + sparse_offsets)
        valid_n = idx <= S_KV - 2

        k_offsets = (pid_b * S_KV + idx[None, :]) * D_TOTAL
        k = tl.load(KV + k_offsets + offs_d[:, None], mask=valid_n[None, :], other=0.0)
        k_tail = tl.load(KV + k_offsets + D + offs_tail[:, None], mask=valid_n[None, :], other=0.0)

        scores = tl.dot(q, k)
        scores += tl.dot(q_tail, k_tail)
        scores *= SM_SCALE_LOG2
        scores = tl.where((offs_h[:, None] < H) & valid_n[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp2(scores - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), tl.trans(k))
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        Output + ((pid_b * S + pid_s) * H + offs_h[:, None]) * D + offs_d[None, :],
        acc,
        mask=offs_h[:, None] < H,
    )
    tl.store(Lse + (pid_b * S + pid_s) * H + offs_h, tl.log2(l_i) + m_i, mask=offs_h < H)


@triton.jit
def _sparse_mla_bwd_kernel(
    Q,
    KV,
    dO,
    Indices,
    Lse,
    Delta,
    dQ,
    dKV,
    S: tl.constexpr,
    S_KV: tl.constexpr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SM_SCALE: tl.constexpr,
    SM_SCALE_LOG2: tl.constexpr,
    D: tl.constexpr,
    D_TAIL: tl.constexpr,
    D_TOTAL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_tail = tl.arange(0, D_TAIL)

    q_base = ((pid_b * S + pid_s) * H + offs_h[:, None]) * D_TOTAL
    q = tl.load(Q + q_base + offs_d[None, :], mask=offs_h[:, None] < H, other=0.0)
    q_tail = tl.load(Q + q_base + D + offs_tail[None, :], mask=offs_h[:, None] < H, other=0.0)
    do = tl.load(
        dO + ((pid_b * S + pid_s) * H + offs_h[:, None]) * D + offs_d[None, :],
        mask=offs_h[:, None] < H,
        other=0.0,
    )
    lse = tl.load(Lse + (pid_b * S + pid_s) * H + offs_h, mask=offs_h < H, other=float("inf"))
    delta = tl.load(Delta + (pid_b * S + pid_s) * H + offs_h, mask=offs_h < H, other=0.0)

    dq = tl.zeros((BLOCK_H, D), tl.float32)
    dq_tail = tl.zeros((BLOCK_H, D_TAIL), tl.float32)

    for start_n in tl.range(0, TOPK, BLOCK_N, num_stages=2):
        sparse_offsets = start_n + offs_n
        idx = tl.load(Indices + (pid_b * S + pid_s) * TOPK + sparse_offsets)
        valid_n = idx <= S_KV - 2

        k_offsets = (pid_b * S_KV + idx[None, :]) * D_TOTAL
        k = tl.load(KV + k_offsets + offs_d[:, None], mask=valid_n[None, :], other=0.0)
        k_tail = tl.load(KV + k_offsets + D + offs_tail[:, None], mask=valid_n[None, :], other=0.0)

        scores = tl.dot(q, k)
        scores += tl.dot(q_tail, k_tail)
        scores *= SM_SCALE_LOG2
        scores = tl.where((offs_h[:, None] < H) & valid_n[None, :], scores, -float("inf"))
        p = tl.exp2(scores - lse[:, None])

        dp = tl.dot(do, k)
        ds = p * (dp - delta[:, None]) * SM_SCALE
        ds_bf16 = ds.to(tl.bfloat16)
        p_bf16 = p.to(tl.bfloat16)

        dq += tl.dot(ds_bf16, tl.trans(k))
        dq_tail += tl.dot(ds_bf16, tl.trans(k_tail))

        dkv = tl.dot(tl.trans(ds_bf16), q)
        dkv += tl.dot(tl.trans(p_bf16), do)
        dkv_tail = tl.dot(tl.trans(ds_bf16), q_tail)

        dkv_offsets = (pid_b * S_KV + idx[:, None]) * D_TOTAL
        tl.atomic_add(dKV + dkv_offsets + offs_d[None, :], dkv, sem="relaxed", mask=valid_n[:, None])
        tl.atomic_add(
            dKV + dkv_offsets + D + offs_tail[None, :],
            dkv_tail,
            sem="relaxed",
            mask=valid_n[:, None],
        )

    tl.store(
        dQ + ((pid_b * S + pid_s) * H + offs_h[:, None]) * D_TOTAL + offs_d[None, :],
        dq,
        mask=offs_h[:, None] < H,
    )
    tl.store(
        dQ + ((pid_b * S + pid_s) * H + offs_h[:, None]) * D_TOTAL + D + offs_tail[None, :],
        dq_tail,
        mask=offs_h[:, None] < H,
    )


def _check_inputs(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> tuple[int, int, int, int, int]:
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert indices.is_contiguous()
    batch, seq_len, heads, dim_total = q.shape
    kv_batch, seq_len_kv, kv_group, kv_dim_total = kv.shape
    assert batch == kv_batch
    assert kv_group == 1
    assert dim_total == _D_TOTAL
    assert kv_dim_total == _D_TOTAL
    assert heads % 16 == 0
    assert indices.shape[:3] == (batch, seq_len, kv_group)
    topk = indices.shape[-1]
    assert topk % 32 == 0
    assert indices.dtype == torch.int32
    assert q.dtype == torch.bfloat16
    assert kv.dtype == torch.bfloat16
    return batch, seq_len, heads, seq_len_kv, topk


def sparse_mla_fwd_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float | None = None,
    d_v: int = _D,
    block_I: int = 64,
    num_stages: int = 2,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    del num_stages, threads
    assert d_v == _D
    batch, seq_len, heads, seq_len_kv, topk = _check_inputs(q, kv, indices)
    if sm_scale is None:
        sm_scale = _D_TOTAL ** -0.5

    block_h = 16
    block_n = block_I
    assert block_n in (32, 64)
    assert topk % block_n == 0

    out = torch.empty((batch, seq_len, heads, _D), device=q.device, dtype=q.dtype)
    lse = torch.empty((batch, seq_len, heads), device=q.device, dtype=torch.float32)
    grid = (seq_len, triton.cdiv(heads, block_h), batch)
    _sparse_mla_fwd_kernel[grid](
        q,
        kv,
        indices,
        out,
        lse,
        seq_len,
        seq_len_kv,
        heads,
        topk,
        sm_scale * _LOG2E,
        _D,
        _D_TAIL,
        _D_TOTAL,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out, lse


def sparse_mla_bwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, seq_len_kv, topk = _check_inputs(q, kv, indices)
    assert o.shape == (batch, seq_len, heads, _D)
    assert do.shape == (batch, seq_len, heads, _D)
    assert lse.shape == (batch, seq_len, heads)
    assert o.dtype == torch.bfloat16
    assert do.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    if sm_scale is None:
        sm_scale = _D_TOTAL ** -0.5

    do = do.contiguous()
    delta = (o.float() * do.float()).sum(dim=-1)
    dq = torch.empty_like(q)
    dkv_accum = torch.zeros_like(kv, dtype=torch.float32)

    block_h = 16
    block_n = 32
    grid = (seq_len, triton.cdiv(heads, block_h), batch)
    _sparse_mla_bwd_kernel[grid](
        q,
        kv,
        do,
        indices,
        lse,
        delta,
        dq,
        dkv_accum,
        seq_len,
        seq_len_kv,
        heads,
        topk,
        sm_scale,
        sm_scale * _LOG2E,
        _D,
        _D_TAIL,
        _D_TOTAL,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return dq, dkv_accum.to(dtype=kv.dtype)
