"""The indexer's top-k on BF16 logits, picking exactly what DeepSelect's FP32 kernel picks on their FP32 copy.

On SM100 `fp8_fp4_mqa_logits` returns BF16 logits. Copied to FP32 (exactly), DeepSelect's FP32 kernel picks their
top-k, but the copy and that kernel's twice as wide reads cost more than DeepSelect's BF16 kernel on the logits as
they are. The BF16 kernel picks the same values but breaks ties among equal logits differently, and BF16 logits tie
at the top-k boundary in most rows. Both kernels' tie rules follow from DeepSelect's source:

* A kernel visits a row's 512-element segments in a fixed order, filters with a strict `>` against the running
  k-th value and compacts its candidates stably. It keeps the top-k under (value descending, visit rank ascending),
  values compared as floats (-0 == +0), and writes them out in visit order. A row of length E <= topk comes back
  as 0..E-1, then the fill value.
* The visit order depends on E alone. A row is visited in index order up to E = 8192 by the FP32 kernel and up to
  E = 16384 by the BF16 kernel. A longer row is visited from its tail [n_perm * 512, E), then segments
  P(0), ..., P(n_perm - 1), with n_perm = (ceil(E / 512) - 1) // 8 * 8 and
  P(l) = ((l % n_perm) * (0xB559EB75 % n_perm) + 0x22262226) % n_perm, by both kernels.

So the two kernels agree, picks and order, on rows up to 8192 and past 16384. `exact_topk_bf16` rewrites each row in
between into the FP32 kernel's visit order before the BF16 kernel runs, so that its index-order visit is the FP32
kernel's visit, and maps the picks back. `_check_deep_select_visit_orders` checks all of this against the installed
DeepSelect once per process and raises if a DeepSelect release visits rows differently.
"""

import functools

import torch
import triton
import triton.language as tl

TAIL_SEGS = 8  # A permuted row's last 8 segments of 512 are visited first
FP32_INDEX_ORDER_MAX = 8192  # The FP32 kernel visits rows up to this length in index order
BF16_INDEX_ORDER_MAX = 16384  # The BF16 kernel up to this one; both visit longer rows in the same order
PERM_ADD_BASE = 0x22262226
PERM_MUL_PRIME = 0xB559EB75
MAX_TOPK = 2048


@triton.jit
def _fp32_visit_layout(E, TAIL_SEGS: tl.constexpr, PERM_MUL_PRIME: tl.constexpr):
    """The FP32 kernel's visit order of a row of length E in (8192, 16384]: the tail [n_perm * 512, E), then segment
    P(l) = ((l % n_perm) * mul + PERM_ADD_BASE) % n_perm for l = 0, 1, ..., n_perm - 1."""
    n_perm = (tl.cdiv(E, 512) - 1) // TAIL_SEGS * TAIL_SEGS
    mul = PERM_MUL_PRIME % n_perm.to(tl.int64)
    return n_perm, E - n_perm * 512, mul


@triton.jit
def _to_fp32_visit_order(
    logits_ptr,
    logits_stride,
    len_ptr,
    TAIL_SEGS: tl.constexpr,
    FP32_INDEX_ORDER_MAX: tl.constexpr,
    BF16_INDEX_ORDER_MAX: tl.constexpr,
    PERM_ADD_BASE: tl.constexpr,
    PERM_MUL_PRIME: tl.constexpr,
):
    """Rewrites, in place, each row that the two kernels visit in different orders into the FP32 kernel's visit
    order, which the BF16 kernel then visits in index order."""
    row = tl.program_id(0).to(tl.int64)
    E = tl.load(len_ptr + row)
    if (E > FP32_INDEX_ORDER_MAX) & (E <= BF16_INDEX_ORDER_MAX):
        row_ptr = logits_ptr + row * logits_stride
        n_perm, tail_len, mul = _fp32_visit_layout(E, TAIL_SEGS, PERM_MUL_PRIME)
        lin = tl.arange(0, BF16_INDEX_ORDER_MAX // 512)
        src_seg = ((lin * mul + PERM_ADD_BASE) % n_perm).to(tl.int32)
        cols = tl.arange(0, 512)
        in_perm = (lin < n_perm)[:, None]
        body = tl.load(row_ptr + src_seg[:, None] * 512 + cols[None, :], mask=in_perm)
        t = tl.arange(0, TAIL_SEGS * 512)
        tail = tl.load(row_ptr + n_perm * 512 + t, mask=t < tail_len)
        tl.debug_barrier()  # every thread holds its source values before any is overwritten
        tl.store(row_ptr + tail_len + lin[:, None] * 512 + cols[None, :], body, mask=in_perm)
        tl.store(row_ptr + t, tail, mask=t < tail_len)


@triton.jit
def _picks_to_indices(
    picks_ptr,
    picks_stride,
    len_ptr,
    ks_ptr,
    out_ptr,
    out_stride,
    sentinel,
    TOPK: tl.constexpr,
    TAIL_SEGS: tl.constexpr,
    FP32_INDEX_ORDER_MAX: tl.constexpr,
    BF16_INDEX_ORDER_MAX: tl.constexpr,
    PERM_ADD_BASE: tl.constexpr,
    PERM_MUL_PRIME: tl.constexpr,
):
    """Key indices of the picks (`ks + column`, `sentinel` for unused slots), mapping a rewritten row back."""
    row = tl.program_id(0).to(tl.int64)
    E = tl.load(len_ptr + row)
    ks = tl.load(ks_ptr + row)
    picks = tl.load(picks_ptr + row * picks_stride + tl.arange(0, TOPK))
    if (E > FP32_INDEX_ORDER_MAX) & (E <= BF16_INDEX_ORDER_MAX):
        n_perm, tail_len, mul = _fp32_visit_layout(E, TAIL_SEGS, PERM_MUL_PRIME)
        body = tl.maximum(picks - tail_len, 0)
        seg = (((body // 512) * mul + PERM_ADD_BASE) % n_perm).to(tl.int32)
        picks = tl.where(picks < tail_len, n_perm * 512 + picks, seg * 512 + body % 512)
    tl.store(out_ptr + row * out_stride + tl.arange(0, TOPK), tl.where(picks >= 0, picks + ks, sentinel))


def exact_topk_bf16_supported(topk: int, n_cols: int) -> bool:
    """Whether `exact_topk_bf16` takes this top-k of rows `n_cols` wide (DeepSelect itself is the caller's check)."""
    return topk > 0 and (topk & (topk - 1)) == 0 and topk <= min(MAX_TOPK, n_cols)


def exact_topk_bf16(
    logits: torch.Tensor, lengths: torch.Tensor, ks: torch.Tensor, topk: int, sentinel: int, out: torch.Tensor
) -> None:
    """Writes to `out` the indexer's top-k picks of these BF16 logits, exactly as DeepSelect's FP32 kernel makes them.

    Row i holds the logits of keys [ks[i], ks[i] + lengths[i]) from column 0; the rest of the row is never read.
    The picks are key indices `ks[i] + column` in the FP32 kernel's output order; unused slots hold `sentinel`.
    Rows of length in (8192, 16384] are rewritten in place.
    """
    assert logits.dtype == torch.bfloat16 and logits.stride(1) == 1
    assert out.dtype == torch.int32 and out.stride(1) == 1 and out.shape == (logits.shape[0], topk)
    assert exact_topk_bf16_supported(topk, logits.shape[1])
    if logits.shape[0] == 0:
        return
    _check_deep_select_visit_orders(logits.device, topk)
    _exact_topk_bf16(logits, lengths, ks, topk, sentinel, out)


def _exact_topk_bf16(logits, lengths, ks, topk, sentinel, out):
    import deep_select

    n_rows = logits.shape[0]
    lengths = lengths.to(torch.int32)
    meta = dict(
        TAIL_SEGS=TAIL_SEGS,
        FP32_INDEX_ORDER_MAX=FP32_INDEX_ORDER_MAX,
        BF16_INDEX_ORDER_MAX=BF16_INDEX_ORDER_MAX,
        PERM_ADD_BASE=PERM_ADD_BASE,
        PERM_MUL_PRIME=PERM_MUL_PRIME,
    )
    _to_fp32_visit_order[(n_rows,)](logits, logits.stride(0), lengths, **meta, num_warps=8)
    _, picks = deep_select.topk(
        logits,
        topk,
        end=lengths,
        indices_type=torch.int32,
        idx_oob_fill_value=-1,
        return_value=False,
        abort_when_nan_found=False,
    )
    _picks_to_indices[(n_rows,)](
        picks,
        picks.stride(0),
        lengths,
        ks.to(torch.int32),
        out,
        out.stride(0),
        sentinel,
        TOPK=topk,
        **meta,
        num_warps=4,
    )


def fp32_topk_reference(logits: torch.Tensor, lengths: torch.Tensor, ks: torch.Tensor, topk: int, sentinel: int):
    """DeepSelect's FP32 kernel on the FP32 copy of BF16 logits, picks offset and masked as the indexer does."""
    import deep_select

    _, picks = deep_select.topk(
        logits.float(),
        topk,
        end=lengths.to(torch.int32),
        indices_type=torch.int32,
        idx_oob_fill_value=sentinel,
        return_value=False,
        abort_when_nan_found=False,
    )
    picks = picks + ks[:, None]
    return picks.masked_fill_((picks < ks[:, None]) | (picks >= (ks + lengths)[:, None]), sentinel)


@functools.cache
def _check_deep_select_visit_orders(device: torch.device, topk: int) -> None:
    """Raises if the installed DeepSelect visits rows in another order than this module assumes.

    Rows of each visit-order regime (short, index order in both kernels, rewritten, permuted in both) with logits
    that tie heavily at the k-th place, in batches small and large enough for DeepSelect's single- and multi-wave
    BF16 kernels: a change in either kernel's visit order changes some of their picks."""
    gen = torch.Generator(device=device).manual_seed(0)
    n_cols = 65536
    row_lengths = [0, 1, topk - 1, topk, topk + 1, 4097, 8192, 8193, 8705, 12289, 16383, 16384, 16385, 24577, 65536]
    n_sms = torch.cuda.get_device_properties(device).multi_processor_count
    for n_rows in (len(row_lengths), 2 * n_sms + len(row_lengths)):
        lengths = torch.tensor(row_lengths, device=device).repeat(-(-n_rows // len(row_lengths)))[:n_rows]
        logits = torch.randint(0, 3, (n_rows, n_cols), device=device, generator=gen, dtype=torch.int8).bfloat16()
        logits[1::2] = torch.randn(n_rows // 2, n_cols, device=device, generator=gen).bfloat16()
        ks = torch.zeros(n_rows, dtype=torch.int32, device=device)
        expected = fp32_topk_reference(logits, lengths, ks, topk, n_cols)
        out = torch.empty(n_rows, topk, dtype=torch.int32, device=device)
        _exact_topk_bf16(logits, lengths, ks, topk, n_cols, out)
        if not torch.equal(out, expected):
            differing = (out != expected).any(1).sum().item()
            raise RuntimeError(
                f"The exact BF16 top-k differs from DeepSelect's FP32 kernel on {differing}/{n_rows} rows (top-{topk}): "
                "the installed DeepSelect visits rows in another order than indexer_topk.py assumes"
            )
