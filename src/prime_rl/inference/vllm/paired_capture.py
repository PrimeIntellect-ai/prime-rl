"""Paired support evidence carried through vLLM's sampling-mask transport as CSR rows."""

import time

import numpy as np
import pybase64
import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import SamplingMaskLists

NEG_INF_BITS = -8388608  # IEEE float32 -inf; plain IDs decode to missing evidence.
CUTOFF_FLAG = 1 << 31  # Marks the cutoff entry of a partial support; vocab ids stay below it.
LOG_INTERVAL_S = 60.0

logger = init_logger("vllm.prime_rl.sampling_support")


@triton.jit
def _order_key(x):
    """Signed int32 key ordered like the float32 value."""
    bits = x.to(tl.int32, bitcast=True)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


@triton.jit
def _pack(logprob, token_id):
    bits = (logprob.to(tl.int32, bitcast=True) ^ -8388608).to(tl.int64)
    return (bits << 32) | token_id


@triton.jit
def _support_stats_kernel(
    logits_ptr,
    row_stride,
    cu_num_logits_ptr,
    committed_ptr,
    sizes_ptr,
    lse_ptr,
    cutoffs_ptr,
    vocab_size,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    if tl.load(committed_ptr + req) == 0:
        tl.store(sizes_ptr + req, 0)
        return
    row = logits_ptr + tl.load(cu_num_logits_ptr + req).to(tl.int64) * row_stride
    count = tl.zeros((), dtype=tl.int64)
    m = tl.full((), -float("inf"), dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)
    low = tl.full((), float("inf"), dtype=tl.float32)
    cutoff = tl.zeros((), dtype=tl.int64)
    for start in range(0, vocab_size, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        x = tl.load(row + offsets, mask=offsets < vocab_size, other=-float("inf")).to(tl.float32)
        keep = (x > -float("inf")) & (x < float("inf"))
        count += tl.sum(keep.to(tl.int64))
        m_new = tl.maximum(m, tl.max(tl.where(keep, x, -float("inf"))))
        shift = tl.where(m_new > -float("inf"), m_new, 0.0)
        s = s * tl.exp(m - shift) + tl.sum(tl.where(keep, tl.exp(x - shift), 0.0))
        m = m_new
        kept = tl.where(keep, x, float("inf"))
        block_low = tl.min(kept)
        cutoff = tl.where(block_low < low, start + tl.argmin(kept, axis=0).to(tl.int64), cutoff)
        low = tl.minimum(low, block_low)
    tl.store(sizes_ptr + req, count)
    tl.store(lse_ptr + req, m + tl.log(s))
    tl.store(cutoffs_ptr + req, cutoff)


@triton.jit
def _support_select_kernel(
    logits_ptr,
    row_stride,
    cu_num_logits_ptr,
    sampled_ptr,
    sampled_stride,
    sizes_ptr,
    cutoffs_ptr,
    counts_ptr,
    thresholds_ptr,
    width,
    vocab_size,
    BLOCK: tl.constexpr,
):
    """Row size for each support: all of it when it fits in ``width``, else the most ids at or above
    one logit threshold that fit in ``width - 2`` (bisected over order keys), plus the sampled token
    and the cutoff."""
    req = tl.program_id(0)
    size = tl.load(sizes_ptr + req)
    if size <= width:
        tl.store(counts_ptr + req, size)
        return
    row = logits_ptr + tl.load(cu_num_logits_ptr + req).to(tl.int64) * row_stride
    cutoff = tl.load(cutoffs_ptr + req)
    # count(key >= lo) > width - 2 >= count(key >= hi); +inf's key is above every finite logit.
    lo = _order_key(tl.load(row + cutoff).to(tl.float32)).to(tl.int64)
    hi = tl.full((), 0x7F800000, dtype=tl.int64)
    head = tl.zeros((), dtype=tl.int64)
    while hi - lo > 1:
        mid = lo + (hi - lo) // 2
        n = tl.zeros((), dtype=tl.int64)
        for start in range(0, vocab_size, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            x = tl.load(row + offsets, mask=offsets < vocab_size, other=-float("inf")).to(tl.float32)
            keep = (x > -float("inf")) & (x < float("inf")) & (_order_key(x).to(tl.int64) >= mid)
            n += tl.sum(keep.to(tl.int64))
        over = n > width - 2
        lo = tl.where(over, mid, lo)
        hi = tl.where(over, hi, mid)
        head = tl.where(over, head, n)
    sampled = tl.load(sampled_ptr + req * sampled_stride).to(tl.int64)
    below = _order_key(tl.load(row + sampled).to(tl.float32)).to(tl.int64) < hi
    tl.store(thresholds_ptr + req, hi.to(tl.int32))
    tl.store(counts_ptr + req, head + 1 + (below & (sampled != cutoff)).to(tl.int64))


@triton.jit
def _support_compact_kernel(
    logits_ptr,
    row_stride,
    cu_num_logits_ptr,
    sampled_ptr,
    sampled_stride,
    sizes_ptr,
    starts_ptr,
    lse_ptr,
    cutoffs_ptr,
    thresholds_ptr,
    out_ptr,
    width,
    vocab_size,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    size = tl.load(sizes_ptr + req)
    if size <= 0:
        return
    row = logits_ptr + tl.load(cu_num_logits_ptr + req).to(tl.int64) * row_stride
    pos = tl.load(starts_ptr + req)
    lse = tl.load(lse_ptr + req)
    partial = size > width
    threshold = tl.where(partial, tl.load(thresholds_ptr + req), -2147483648)
    for start in range(0, vocab_size, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        x = tl.load(row + offsets, mask=offsets < vocab_size, other=-float("inf")).to(tl.float32)
        keep = (x > -float("inf")) & (x < float("inf")) & (_order_key(x) >= threshold)
        keep_i64 = keep.to(tl.int64)
        slot = pos + tl.cumsum(keep_i64, axis=0) - keep_i64
        tl.store(out_ptr + slot, _pack(x - lse, offsets.to(tl.int64)), mask=keep)
        pos += tl.sum(keep_i64)
    if partial:
        cutoff = tl.load(cutoffs_ptr + req)
        sampled = tl.load(sampled_ptr + req * sampled_stride).to(tl.int64)
        x = tl.load(row + sampled).to(tl.float32)
        if (_order_key(x) < threshold) & (sampled != cutoff):
            tl.store(out_ptr + pos, _pack(x - lse, sampled))
            pos += 1
        x = tl.load(row + cutoff).to(tl.float32)
        tl.store(out_ptr + pos, _pack(x - lse, cutoff | 2147483648))  # CUTOFF_FLAG


class CsrSupports:
    """Per-request supports packed as ``(logprob bits << 32) | id`` in one flat device buffer.

    Every replay row holds at most ``width`` entries, so the buffer is sized up front and capture never
    syncs. The host copy is two-phase: counts ride on vLLM's async copy, then exactly the step's total
    entries are copied once vLLM consumes the step."""

    replayed = 0
    partial = 0
    largest = 0
    last_log = time.monotonic()

    def __init__(self, flat: torch.Tensor, counts: torch.Tensor, sizes: torch.Tensor, width: int):
        self.flat = flat
        self.counts = counts
        self.sizes = sizes
        self.width = width

    def to_cpu_nonblocking(self) -> "CsrSupports":
        self.counts_cpu = self.counts.to("cpu", non_blocking=True)
        self.sizes_cpu = self.sizes.to("cpu", non_blocking=True)
        self.ready = torch.cuda.Event()
        self.ready.record()
        return self

    def tolists(self, num_sampled_tokens: np.ndarray | None = None) -> SamplingMaskLists:
        counts = self.counts_cpu.numpy()
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        total = int(offsets[-1])
        host = torch.empty(total, dtype=torch.int64, pin_memory=True)
        if total:
            stream = _copy_stream(self.flat.device)
            with torch.cuda.stream(stream):
                stream.wait_event(self.ready)
                host.copy_(self.flat[:total], non_blocking=True)
            stream.synchronize()
        self._log(self.sizes_cpu.numpy())
        return SamplingMaskLists(host.numpy(), offsets)

    def _log(self, sizes: np.ndarray) -> None:
        cls = CsrSupports
        sizes = sizes[sizes > 0]
        cls.replayed += len(sizes)
        cls.partial += int((sizes > self.width).sum())
        cls.largest = max(cls.largest, int(sizes.max(initial=0)))
        now = time.monotonic()
        if now - cls.last_log >= LOG_INTERVAL_S:
            logger.info(
                "Sampling supports over the last %.0fs: %d of %d replayed tokens exceeded "
                "inference.sampling_support_width (%d) and returned a partial support; largest %d ids",
                now - cls.last_log,
                cls.partial,
                cls.replayed,
                self.width,
                cls.largest,
            )
            cls.replayed, cls.partial, cls.largest, cls.last_log = 0, 0, 0, now


_COPY_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    if device not in _COPY_STREAMS:
        _COPY_STREAMS[device] = torch.cuda.Stream(device)
    return _COPY_STREAMS[device]


def capture_supports(logits, cu_num_logits, committed, sampled, width, sampler_rows=None) -> CsrSupports:
    """CSR supports for one sampling step. A replay request's support is its finite processed logits (its
    kept set), with logprobs normalized over the whole set. A support of at most ``width`` ids is captured
    whole; a larger one keeps its top ids, the sampled token and its least likely id (the cutoff, flagged
    with ``CUTOFF_FLAG``), at most ``width`` entries. Sampler-rows requests add their packed rows."""
    num_reqs, vocab_size = committed.shape[0], logits.shape[1]
    device = logits.device
    sizes = torch.empty(num_reqs, dtype=torch.int64, device=device)
    lse = torch.empty(num_reqs, dtype=torch.float32, device=device)
    cutoffs = torch.empty(num_reqs, dtype=torch.int64, device=device)
    _support_stats_kernel[(num_reqs,)](
        logits, logits.stride(0), cu_num_logits, committed, sizes, lse, cutoffs, vocab_size, BLOCK=8192
    )
    counts = torch.empty_like(sizes)
    thresholds = torch.empty(num_reqs, dtype=torch.int32, device=device)
    _support_select_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        cu_num_logits,
        sampled,
        sampled.stride(0),
        sizes,
        cutoffs,
        counts,
        thresholds,
        width,
        vocab_size,
        BLOCK=8192,
    )
    capacity = num_reqs * width
    if sampler_rows is not None:
        index, packed, row_counts = sampler_rows
        counts.index_put_((index,), row_counts)
        capacity += packed.numel()
    starts = counts.cumsum(0) - counts
    flat = torch.empty(capacity + 1, dtype=torch.int64, device=device)
    _support_compact_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        cu_num_logits,
        sampled,
        sampled.stride(0),
        sizes,
        starts,
        lse,
        cutoffs,
        thresholds,
        flat,
        width,
        vocab_size,
        BLOCK=8192,
    )
    if sampler_rows is not None:
        columns = torch.arange(packed.shape[1], device=packed.device)
        slots = starts[index].unsqueeze(-1) + columns
        valid = columns < counts[index].unsqueeze(-1)
        flat[torch.where(valid, slots, capacity)] = packed
    return CsrSupports(flat, counts, sizes, width)


def pack_sampler_rows(logits, cu_num_logits, num_sampled, sampled, num_rows, draws):
    """Per selected request: the sampled token, then ``num_rows`` ids from the processed distribution q (i.i.d.
    draws with replacement, or the top ids), each packed with its full-vocabulary processed logprob. The rows
    copy vLLM's ``[sampled, top...]`` logprob layout. Draws use the global generator, so they are independent
    of the sampled token. Returns the selected request indices, packed rows and row sizes."""
    selected = np.flatnonzero(num_rows)
    width = int(num_rows.max())
    index = torch.from_numpy(selected).pin_memory().to(logits.device, non_blocking=True)
    x = logits[cu_num_logits[:-1].long()[index]].float()
    if draws[selected].all():
        alternatives = torch.multinomial(x.softmax(-1), width, replacement=True)
    elif not draws[selected].any():
        alternatives = x.topk(width).indices
    else:
        use_draws = torch.from_numpy(draws[selected]).pin_memory().to(x.device, non_blocking=True).unsqueeze(-1)
        alternatives = torch.where(
            use_draws, torch.multinomial(x.softmax(-1), width, replacement=True), x.topk(width).indices
        )
    ids = torch.cat([sampled[index].view(-1, 1).long(), alternatives], dim=1)
    values = (x.gather(1, ids) - x.logsumexp(-1, keepdim=True)).clamp_min(-9999.0)
    packed = ((values.contiguous().view(torch.int32) ^ NEG_INF_BITS).long() << 32) | ids
    row_counts = (
        torch.from_numpy((num_rows[selected] + 1).astype(np.int64)).pin_memory().to(x.device, non_blocking=True)
    )
    return index, packed, torch.where(num_sampled[index] > 0, row_counts, 0)


def serialize_paired_support(packed, counts):
    """Return aligned mask IDs and existing compact-logprob wire data. Rows that hold a flagged cutoff
    entry are partial supports and are listed in ``partial_rows``."""
    bits = np.asarray(packed, dtype=np.uint64)
    ids = (bits & np.uint64(0xFFFFFFFF)).astype("<u4")
    cutoff = ids >= CUTOFF_FLAG
    ids &= np.uint32(CUTOFF_FLAG - 1)
    values = ((bits >> np.uint64(32)).astype(np.uint32).view(np.int32) ^ NEG_INF_BITS).view("<f4")
    counts = np.asarray(counts, dtype=np.int32)
    if int(counts.sum()) != len(ids) or not np.isfinite(values).all():
        raise ValueError("Incomplete paired sampling-support evidence")
    compact = {
        "data": pybase64.b64encode(ids.tobytes() + values.tobytes()).decode("ascii"),
        "format": "u32-f32-le-v1",
        "offsets": [0, *np.cumsum(counts).tolist()],
        "num_top_logprobs": int(counts.max(initial=0)),
    }
    if cutoff.any():
        compact["partial_rows"] = np.repeat(np.arange(len(counts)), counts)[cutoff].tolist()
    return ids.astype(np.int32, copy=False), compact
