"""Paired support evidence carried through vLLM's sampling-mask transport as CSR rows."""

import time

import numpy as np
import pybase64
import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import SamplingMaskLists

NEG_INF_BITS = -8388608  # IEEE float32 -inf; plain IDs decode to missing evidence.
LOG_INTERVAL_S = 60.0

logger = init_logger(__name__)


@triton.jit
def _support_stats_kernel(
    logits_ptr, row_stride, cu_num_logits_ptr, committed_ptr, counts_ptr, lse_ptr, vocab_size, BLOCK: tl.constexpr
):
    req = tl.program_id(0)
    if tl.load(committed_ptr + req) == 0:
        tl.store(counts_ptr + req, 0)
        return
    row = logits_ptr + tl.load(cu_num_logits_ptr + req).to(tl.int64) * row_stride
    count = tl.zeros((), dtype=tl.int64)
    m = tl.full((), -float("inf"), dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)
    for start in range(0, vocab_size, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        x = tl.load(row + offsets, mask=offsets < vocab_size, other=-float("inf")).to(tl.float32)
        keep = (x > -float("inf")) & (x < float("inf"))
        count += tl.sum(keep.to(tl.int64))
        m_new = tl.maximum(m, tl.max(tl.where(keep, x, -float("inf"))))
        shift = tl.where(m_new > -float("inf"), m_new, 0.0)
        s = s * tl.exp(m - shift) + tl.sum(tl.where(keep, tl.exp(x - shift), 0.0))
        m = m_new
    tl.store(counts_ptr + req, count)
    tl.store(lse_ptr + req, m + tl.log(s))


@triton.jit
def _support_compact_kernel(
    logits_ptr,
    row_stride,
    cu_num_logits_ptr,
    committed_ptr,
    starts_ptr,
    counts_ptr,
    lse_ptr,
    out_ptr,
    vocab_size,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    if (tl.load(committed_ptr + req) == 0) | (tl.load(counts_ptr + req) <= 0):
        return
    row = logits_ptr + tl.load(cu_num_logits_ptr + req).to(tl.int64) * row_stride
    pos = tl.load(starts_ptr + req)
    lse = tl.load(lse_ptr + req)
    for start in range(0, vocab_size, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        x = tl.load(row + offsets, mask=offsets < vocab_size, other=-float("inf")).to(tl.float32)
        keep = (x > -float("inf")) & (x < float("inf"))
        keep_i64 = keep.to(tl.int64)
        slot = pos + tl.cumsum(keep_i64, axis=0) - keep_i64
        bits = ((x - lse).to(tl.int32, bitcast=True) ^ -8388608).to(tl.int64)
        tl.store(out_ptr + slot, (bits << 32) | offsets.to(tl.int64), mask=keep)
        pos += tl.sum(keep_i64)


class CsrSupports:
    """Per-request supports packed as ``(logprob bits << 32) | id`` in one flat device buffer.

    The device buffer has a fixed capacity, so capture never syncs; requests whose rows do not fit get
    count -1 and no entries. The host copy is two-phase: counts ride on vLLM's async copy, then exactly
    the step's total entries are copied once vLLM consumes the step."""

    max_total = 0
    overflows = 0
    last_log = time.monotonic()

    def __init__(self, flat: torch.Tensor, counts: torch.Tensor):
        self.flat = flat
        self.counts = counts

    def to_cpu_nonblocking(self) -> "CsrSupports":
        self.counts_cpu = self.counts.to("cpu", non_blocking=True)
        self.ready = torch.cuda.Event()
        self.ready.record()
        return self

    def tolists(self, num_sampled_tokens: np.ndarray | None = None) -> SamplingMaskLists:
        counts = self.counts_cpu.numpy()
        kept = np.maximum(counts, 0)
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(kept, out=offsets[1:])
        total = int(offsets[-1])
        host = torch.empty(total, dtype=torch.int64, pin_memory=True)
        if total:
            stream = _copy_stream(self.flat.device)
            with torch.cuda.stream(stream):
                stream.wait_event(self.ready)
                host.copy_(self.flat[:total], non_blocking=True)
            stream.synchronize()
        self._log(total, int((counts < 0).sum()))
        return SamplingMaskLists(host.numpy(), offsets)

    def _log(self, total: int, overflows: int) -> None:
        cls = CsrSupports
        cls.max_total = max(cls.max_total, total)
        cls.overflows += overflows
        if overflows:
            logger.warning(
                "Sampling-support capture overflow: %d requests did not fit in %d entries "
                "(inference.sampling_support_capacity) and fail",
                overflows,
                self.flat.numel() - 1,
            )
        now = time.monotonic()
        if now - cls.last_log >= LOG_INTERVAL_S:
            logger.info(
                "Sampling-support capture: max %d entries per step over the last %.0fs (capacity %d, %d overflows)",
                cls.max_total,
                now - cls.last_log,
                self.flat.numel() - 1,
                cls.overflows,
            )
            cls.max_total, cls.overflows, cls.last_log = 0, 0, now


_COPY_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    if device not in _COPY_STREAMS:
        _COPY_STREAMS[device] = torch.cuda.Stream(device)
    return _COPY_STREAMS[device]


def capture_supports(logits, cu_num_logits, committed, capacity, sampler_rows=None) -> CsrSupports:
    """CSR supports for one sampling step: each replay request's finite processed logits (its exact
    kept set) with logprobs normalized over that set, and each sampler-rows request's packed rows."""
    num_reqs, vocab_size = committed.shape[0], logits.shape[1]
    counts = torch.empty(num_reqs, dtype=torch.int64, device=logits.device)
    lse = torch.empty(num_reqs, dtype=torch.float32, device=logits.device)
    _support_stats_kernel[(num_reqs,)](
        logits, logits.stride(0), cu_num_logits, committed, counts, lse, vocab_size, BLOCK=8192
    )
    if sampler_rows is not None:
        index, packed, row_counts = sampler_rows
        counts.index_put_((index,), row_counts)
    ends = counts.cumsum(0)
    counts = torch.where(ends <= capacity, counts, -1)
    starts = ends - counts.clamp_min(0)
    # One spare slot absorbs the masked scatter of sampler rows that do not fit.
    flat = torch.empty(capacity + 1, dtype=torch.int64, device=logits.device)
    _support_compact_kernel[(num_reqs,)](
        logits, logits.stride(0), cu_num_logits, committed, starts, counts, lse, flat, vocab_size, BLOCK=8192
    )
    if sampler_rows is not None:
        columns = torch.arange(packed.shape[1], device=packed.device)
        slots = starts[index].unsqueeze(-1) + columns
        valid = columns < counts[index].unsqueeze(-1)
        flat[torch.where(valid, slots, capacity)] = packed
    return CsrSupports(flat, counts)


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
    """Return aligned mask IDs and existing compact-logprob wire data."""
    bits = np.asarray(packed, dtype=np.uint64)
    ids = (bits & np.uint64(0xFFFFFFFF)).astype("<u4")
    values = ((bits >> np.uint64(32)).astype(np.uint32).view(np.int32) ^ NEG_INF_BITS).view("<f4")
    counts = np.asarray(counts, dtype=np.int32)
    if (counts <= 0).any():
        raise ValueError(
            "Missing paired sampling-support evidence: the step's supports exceeded inference.sampling_support_capacity"
        )
    if int(counts.sum()) != len(ids) or not np.isfinite(values).all():
        raise ValueError("Incomplete paired sampling-support evidence")
    return ids.astype(np.int32, copy=False), {
        "data": pybase64.b64encode(ids.tobytes() + values.tobytes()).decode("ascii"),
        "format": "u32-f32-le-v1",
        "offsets": [0, *np.cumsum(counts).tolist()],
        "num_top_logprobs": int(counts.max(initial=0)),
    }
