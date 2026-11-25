"""Building micro batches on a trainer rank.

The orchestrator ships each micro batch as a ``MicroBatchPlan``: its samples by reference.
A trainer rank fetches their bodies (``prime_rl.transports.payload``), turns each into a
``TrainingSample`` and packs them here into the ``MicroBatch`` it trains on: per-token streams,
routing rows and sampler support concatenated in packing order, the trailing padding, and the
zero-loss copies that even out the micro batch count across data ranks.

Nothing here imports torch.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

from prime_rl.transports.batch.types import (
    MICRO_BATCH_DTYPES,
    MicroBatch,
    MMImageRef,
    MMRefs,
    RoutedExperts,
    SamplerSupport,
    TrainingSample,
)

# Backfill value per component weight stream when a packed sample doesn't
# carry it: absent rl means weight 1.0 on the loss mask, absent ce/ref_kl
# means no component (weight 0.0).
STREAM_FILL = {"rl_weights": 1.0, "ce_weights": 0.0, "ref_kl_weights": 0.0}


def mm_cut(images: Sequence[Sequence[int]], num_tokens: int, seq_len: int) -> int:
    """How many of a sample's ``num_tokens`` fit in ``seq_len``: a cut never splits an image
    placeholder run (``[offset, length]``), so it moves back to the start of the run it lands in."""
    if num_tokens <= seq_len:
        return num_tokens
    for offset, length in images:
        if offset < seq_len < offset + length:
            return offset
    return seq_len


def truncate_mm_refs(refs: MMRefs | None, num_tokens: int) -> MMRefs | None:
    """The images that lie within the first ``num_tokens``."""
    if refs is None:
        return None
    kept = [image for image in refs.images if image.offset + image.length <= num_tokens]
    return MMRefs(images=kept) if kept else None


def multimodal_sample_error(sample: TrainingSample) -> str | None:
    mm_token_type_ids = sample.mm_token_type_ids
    if mm_token_type_ids is not None and len(mm_token_type_ids) != len(sample.token_ids):
        return (
            "mm_token_type_ids length must match token_ids length "
            f"({len(mm_token_type_ids)} != {len(sample.token_ids)})"
        )
    if sample.mm_refs is not None and mm_token_type_ids is None:
        return "multimodal samples require mm_token_type_ids"
    if sample.mm_refs is not None:
        previous_end = 0
        for image in sample.mm_refs.images:
            if image.offset < previous_end or image.offset + image.length > len(sample.token_ids):
                return "multimodal image refs must be ordered, non-overlapping, and within token_ids"
            previous_end = image.offset + image.length
    return None


def _tile(data: bytes, size: int) -> bytes:
    """``data`` repeated end to end and cut to ``size`` bytes."""
    return (data * (size // len(data) + 1))[:size]


def routed_experts_row_size(routed_experts: RoutedExperts) -> int:
    return routed_experts.shape[1] * routed_experts.shape[2] * np.dtype(routed_experts.dtype).itemsize


def slice_routed_experts(routed_experts: RoutedExperts, num_tokens: int) -> RoutedExperts:
    row_size = routed_experts_row_size(routed_experts)
    return RoutedExperts(
        data=routed_experts.data[: num_tokens * row_size],
        shape=[num_tokens, routed_experts.shape[1], routed_experts.shape[2]],
        dtype=routed_experts.dtype,
    )


def slice_sampler_support(support: SamplerSupport, num_tokens: int) -> SamplerSupport:
    counts = np.frombuffer(support.counts, dtype=np.int32)[:num_tokens]
    size = int(counts.sum()) * 4
    return SamplerSupport(
        ids=support.ids[:size],
        counts=counts.tobytes(),
        replayed=support.replayed[:num_tokens],
        logprobs=support.logprobs[:size] if support.logprobs is not None else None,
    )


def fill_routing(num_tokens: int, shape: Sequence[int], dtype: str, rows: bytes) -> RoutedExperts:
    """Routing rows for ``num_tokens`` tokens that have none: ``rows`` repeated, or, without
    any, distinct expert ids per token (every top-k slot routes to a different expert)."""
    np_dtype = np.dtype(dtype)
    if not rows:
        rows = np.broadcast_to(np.arange(shape[-1], dtype=np_dtype), (1, *shape)).tobytes()
    row_size = int(np.prod(shape)) * np_dtype.itemsize
    return RoutedExperts(data=_tile(rows, num_tokens * row_size), shape=[num_tokens, *shape], dtype=np_dtype.name)


def _empty_sampler_support(num_tokens: int) -> SamplerSupport:
    return SamplerSupport(ids=b"", counts=bytes(num_tokens * 4), replayed=bytes(num_tokens))


def _pack_routed_experts(samples: Sequence[MicroBatch]) -> RoutedExperts | None:
    routed = [sample.routed_experts for sample in samples if sample.routed_experts is not None]
    if not routed:
        return None
    assert len(routed) == len(samples), "routing must be on every sample of a micro batch or none"
    assert all(r.dtype == routed[0].dtype and r.shape[1:] == routed[0].shape[1:] for r in routed)
    return RoutedExperts(
        data=b"".join(r.data for r in routed),
        shape=[sum(r.shape[0] for r in routed), *routed[0].shape[1:]],
        dtype=routed[0].dtype,
    )


def _pack_sampler_support(samples: Sequence[MicroBatch]) -> SamplerSupport | None:
    """The samples' supports concatenated; the support is per-token optional, so samples
    without one get empty rows."""
    if all(sample.sampler_support is None for sample in samples):
        return None
    supports = [
        sample.sampler_support if sample.sampler_support is not None else _empty_sampler_support(len(sample.input_ids))
        for sample in samples
    ]
    logprobs = None
    if any(support.logprobs is not None for support in supports):
        # NaN marks entries that carry no sampler logprobs.
        logprobs = b"".join(
            support.logprobs
            if support.logprobs is not None
            else np.full(len(support.ids) // 4, np.nan, np.float32).tobytes()
            for support in supports
        )
    return SamplerSupport(
        ids=b"".join(support.ids for support in supports),
        counts=b"".join(support.counts for support in supports),
        replayed=b"".join(support.replayed for support in supports),
        logprobs=logprobs,
    )


def prepare_sample(sample: TrainingSample) -> MicroBatch:
    """One sample as a one-sequence micro batch."""
    if error := multimodal_sample_error(sample):
        raise ValueError(error)
    assert sample.env_name != "all", "env_name='all' is reserved for aggregate metric keys"
    input_ids = sample.token_ids
    loss_mask = sample.mask
    if sample.advantages is not None:
        advantages = sample.advantages
    else:
        rl_w = sample.rl_weights
        has_rl_members = loss_mask.any() if rl_w is None else (loss_mask & (rl_w != 0)).any()
        if has_rl_members:
            raise ValueError(
                f"sample from env '{sample.env_name}' has rl member tokens but no advantages — "
                "the orchestrator must credit its trace"
            )
        advantages = np.zeros(len(input_ids), dtype=np.float32)
    num_tokens = len(input_ids)
    for name in (
        "mask",
        "logprobs",
        "temperatures",
        "ref_logprobs",
        "rl_weights",
        "ce_weights",
        "ref_kl_weights",
        "staleness",
    ):
        stream = getattr(sample, name)
        assert stream is None or len(stream) == num_tokens, f"{name}: {len(stream)} for {num_tokens} tokens"
    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=np.arange(num_tokens, dtype=np.int32),
        inference_logprobs=sample.logprobs,
        sequence_lengths=[num_tokens],
        ref_logprobs=sample.ref_logprobs,
        temperatures=sample.temperatures,
        mm_token_type_ids=sample.mm_token_type_ids,
        env_names=[sample.env_name],
        mm_refs=sample.mm_refs,
        rl_weights=sample.rl_weights,
        ce_weights=sample.ce_weights,
        ref_kl_weights=sample.ref_kl_weights,
        seq_lens=[num_tokens],
        trace_ids=[sample.trace_id or ""],
        branch_indices=[sample.branch_index if sample.branch_index is not None else -1],
        staleness=sample.staleness,
        routed_experts=sample.routed_experts,
        sampler_support=sample.sampler_support,
    )


def materialize_bin(samples: Sequence[MicroBatch]) -> MicroBatch:
    """Concatenate one-sequence micro batches in order. A stream materializes as soon as one
    sample carries it; the samples that lack it get the stream's identity fill."""
    has_ref_logprobs = any(sample.ref_logprobs is not None for sample in samples)
    has_mm_token_type_ids = any(sample.mm_token_type_ids is not None for sample in samples)
    has_stream = {name: any(getattr(s, name) is not None for s in samples) for name in STREAM_FILL}
    has_staleness = any(sample.staleness is not None for sample in samples)

    input_ids: list[np.ndarray] = []
    loss_mask: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    inference_logprobs: list[np.ndarray] = []
    position_ids: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    env_names: list[str] = []
    ref_logprobs: list[np.ndarray] | None = [] if has_ref_logprobs else None
    mm_token_type_ids: list[np.ndarray] | None = [] if has_mm_token_type_ids else None
    mm_refs: MMRefs | None = None
    streams: dict[str, list[np.ndarray] | None] = {name: ([] if has_stream[name] else None) for name in STREAM_FILL}
    seq_lens: list[int] = []
    trace_ids: list[str] = []
    branch_indices: list[int] = []
    staleness: list[np.ndarray] | None = [] if has_staleness else None

    offset = 0
    for sample in samples:
        sample_len = len(sample.input_ids)
        input_ids.append(sample.input_ids)
        loss_mask.append(sample.loss_mask)
        advantages.append(sample.advantages)
        inference_logprobs.append(sample.inference_logprobs)
        position_ids.append(sample.position_ids)
        temperatures.append(sample.temperatures)
        env_names.extend(sample.env_names)
        if ref_logprobs is not None:
            ref_logprobs.append(
                sample.ref_logprobs if sample.ref_logprobs is not None else np.zeros(sample_len, dtype=np.float32)
            )
        for name, fill in STREAM_FILL.items():
            stream = streams[name]
            if stream is not None:
                sample_stream = getattr(sample, name)
                stream.append(
                    sample_stream if sample_stream is not None else np.full(sample_len, fill, dtype=np.float32)
                )
        if mm_token_type_ids is not None:
            mm_token_type_ids.append(
                sample.mm_token_type_ids
                if sample.mm_token_type_ids is not None
                else np.zeros(sample_len, dtype=np.int32)
            )
        if sample.mm_refs is not None:
            if mm_refs is None:
                mm_refs = MMRefs(images=[])
            mm_refs.images.extend(
                MMImageRef(url=image.url, offset=offset + image.offset, length=image.length)
                for image in sample.mm_refs.images
            )
        seq_lens.extend(sample.seq_lens)
        trace_ids.extend(sample.trace_ids or [""] * len(sample.sequence_lengths))
        branch_indices.extend(sample.branch_indices or [-1] * len(sample.sequence_lengths))
        if staleness is not None:
            staleness.append(
                sample.staleness if sample.staleness is not None else np.full(sample_len, -1, dtype=np.int16)
            )
        offset += sample_len

    sequence_lengths = [len(sample.input_ids) for sample in samples]
    assert sum(sequence_lengths) == offset, (sequence_lengths, offset)
    assert sum(seq_lens) == offset, (seq_lens, offset)

    return MicroBatch(
        input_ids=np.concatenate(input_ids),
        advantages=np.concatenate(advantages),
        loss_mask=np.concatenate(loss_mask),
        position_ids=np.concatenate(position_ids),
        inference_logprobs=np.concatenate(inference_logprobs),
        sequence_lengths=sequence_lengths,
        ref_logprobs=np.concatenate(ref_logprobs) if ref_logprobs is not None else None,
        temperatures=np.concatenate(temperatures),
        mm_token_type_ids=np.concatenate(mm_token_type_ids) if mm_token_type_ids is not None else None,
        env_names=env_names,
        mm_refs=mm_refs,
        **{name: np.concatenate(stream) if stream is not None else None for name, stream in streams.items()},
        seq_lens=seq_lens,
        trace_ids=trace_ids,
        branch_indices=branch_indices,
        staleness=np.concatenate(staleness) if staleness is not None else None,
        routed_experts=_pack_routed_experts(samples),
        sampler_support=_pack_sampler_support(samples),
    )


def pad_micro_batch(micro_batch: MicroBatch, padding_size: int) -> MicroBatch:
    """Append ``padding_size`` loss-masked tokens to the micro batch's last sequence."""
    if len(micro_batch.env_names) != len(micro_batch.sequence_lengths):
        raise ValueError(
            f"MicroBatch.env_names must name every sequence: "
            f"env_names={len(micro_batch.env_names)}, sequences={len(micro_batch.sequence_lengths)}"
        )
    if padding_size <= 0:
        return micro_batch

    # Identical pad tokens route to the same MoE experts and can overload one EP rank, so text rows pad with their
    # own tokens. Multimodal rows keep a constant id: a repeated image placeholder would have no feature to receive.
    # `np.pad` appends zeros of the stream's own dtype unless given another value.
    tail = (0, padding_size)
    if micro_batch.mm_refs is None:
        # `np.resize` repeats the row end to end.
        pad_ids = np.resize(micro_batch.input_ids, padding_size)
        micro_batch.input_ids = np.concatenate([micro_batch.input_ids, pad_ids])
    else:
        micro_batch.input_ids = np.pad(micro_batch.input_ids, tail, constant_values=1)
    micro_batch.advantages = np.pad(micro_batch.advantages, tail)
    micro_batch.loss_mask = np.pad(micro_batch.loss_mask, tail)
    micro_batch.position_ids = np.concatenate([micro_batch.position_ids, np.arange(padding_size, dtype=np.int32)])
    micro_batch.sequence_lengths[-1] += padding_size
    micro_batch.seq_lens[-1] += padding_size
    micro_batch.padding += padding_size
    micro_batch.inference_logprobs = np.pad(micro_batch.inference_logprobs, tail)
    # Use temperature 1.0 for padding tokens (doesn't matter since loss_mask is False)
    micro_batch.temperatures = np.pad(micro_batch.temperatures, tail, constant_values=1.0)
    if micro_batch.ref_logprobs is not None:
        micro_batch.ref_logprobs = np.pad(micro_batch.ref_logprobs, tail)
    # Padding is loss-masked, so no component trains it; fill every stream
    # with 0.0 (not the pack-boundary defaults) so a padded pure-ce batch
    # still reads as rl-empty to consumers that key off nonzero weights
    # (e.g. the per-env mismatch metrics).
    for stream_name in STREAM_FILL:
        stream = getattr(micro_batch, stream_name)
        if stream is not None:
            setattr(micro_batch, stream_name, np.pad(stream, tail))
    if micro_batch.mm_token_type_ids is not None:
        micro_batch.mm_token_type_ids = np.pad(micro_batch.mm_token_type_ids, tail)
    if micro_batch.staleness is not None:
        micro_batch.staleness = np.pad(micro_batch.staleness, tail, constant_values=-1)
    # Padding tokens route like the micro batch's own and carry no support.
    routed = micro_batch.routed_experts
    if routed is not None:
        routed.data += _tile(routed.data, padding_size * routed_experts_row_size(routed))
        routed.shape[0] += padding_size
    if micro_batch.sampler_support is not None:
        micro_batch.sampler_support.counts += bytes(padding_size * 4)
        micro_batch.sampler_support.replayed += bytes(padding_size)
    return micro_batch


def make_dummy_batch(source: MicroBatch) -> MicroBatch:
    """A zero-loss copy of a micro batch, keeping its modality and routing rows."""
    dummy = copy.deepcopy(source)
    dummy.advantages = np.zeros(len(dummy.input_ids), dtype=np.float32)
    dummy.loss_mask = np.zeros(len(dummy.input_ids), dtype=np.bool_)
    # ce/ref_kl membership is weight != 0 (independent of loss_mask), so the
    # streams must go too or the dummy would still train those tokens.
    dummy.rl_weights = None
    dummy.ce_weights = None
    dummy.ref_kl_weights = None
    # Fully loss-masked: the forward pass replays the routing rows, but the sampler support
    # would be pure wasted work.
    dummy.sampler_support = None
    # The copied identity would double-annotate the source's traces.
    dummy.trace_ids = None
    dummy.branch_indices = None
    return dummy


def assert_token_arrays_aligned(micro_batch: MicroBatch) -> None:
    """Every per-token array must stay position-aligned with ``input_ids`` through packing
    and padding — a field extended without backfill, or a misaligned routing row, would
    corrupt training silently."""
    num_tokens = len(micro_batch.input_ids)
    per_token_fields = (
        "loss_mask",
        "advantages",
        "inference_logprobs",
        "position_ids",
        "temperatures",
        "ref_logprobs",
        "rl_weights",
        "ce_weights",
        "ref_kl_weights",
        "mm_token_type_ids",
        "staleness",
    )
    for name in ("input_ids", *per_token_fields):
        values = getattr(micro_batch, name)
        assert values is None or len(values) == num_tokens, (
            f"{name} misaligned after packing: {len(values)} != {num_tokens} tokens"
        )
        # The trainer reads each stream's raw bytes as this dtype.
        assert values is None or values.dtype == MICRO_BATCH_DTYPES[name], (
            f"{name} must be {np.dtype(MICRO_BATCH_DTYPES[name])}, got {values.dtype}"
        )
    assert sum(micro_batch.sequence_lengths) == num_tokens, (
        f"sequence_lengths sum {sum(micro_batch.sequence_lengths)} != {num_tokens} tokens"
    )
    num_sequences = len(micro_batch.sequence_lengths)
    for name in ("env_names", "trace_ids", "branch_indices"):
        values = getattr(micro_batch, name)
        assert values is None or len(values) == num_sequences, (
            f"{name} misaligned after packing: {len(values)} != {num_sequences} sequences"
        )
    assert sum(micro_batch.seq_lens) == num_tokens, f"seq_lens sum {sum(micro_batch.seq_lens)} != {num_tokens} tokens"
    routed = micro_batch.routed_experts
    if routed is not None:
        assert routed.shape[0] == num_tokens, f"routed_experts: {routed.shape} for {num_tokens}"
        assert len(routed.data) == num_tokens * routed_experts_row_size(routed), (
            f"routed_experts: {len(routed.data)} bytes for {routed.shape}"
        )
    support = micro_batch.sampler_support
    if support is not None:
        counts = np.frombuffer(support.counts, dtype=np.int32)
        assert len(counts) == num_tokens, f"sampler_support counts: {len(counts)} for {num_tokens}"
        assert len(support.replayed) == num_tokens, f"sampler_support replayed: {len(support.replayed)}"
        assert (counts >= 0).all() and len(support.ids) == int(counts.sum()) * 4
        assert support.logprobs is None or len(support.logprobs) == len(support.ids)
