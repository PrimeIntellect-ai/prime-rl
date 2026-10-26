import copy
import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import msgspec
import numpy as np
import torch

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import get_model
from prime_rl.trainer.perf import FlopCounter
from prime_rl.transports.batch.types import (
    MICRO_BATCH_DTYPES,
    MicroBatch,
    MMImageRef,
    MMRefs,
    PayloadSlice,
    TrainingSample,
)

# Backfill value per component weight stream when a packed sample doesn't
# carry it: absent rl means weight 1.0 on the loss mask, absent ce/ref_kl
# means no component (weight 0.0).
STREAM_FILL = {"rl_weights": 1.0, "ce_weights": 0.0, "ref_kl_weights": 0.0}


@dataclass
class _WeightedSet:
    total: int = 0
    items: list[int] = field(default_factory=list)

    def add(self, idx: int, weight: int) -> None:
        self.items.append(idx)
        self.total += weight

    def merge(self, other: "_WeightedSet") -> None:
        self.items.extend(other.items)
        self.total += other.total

    def __lt__(self, other: "_WeightedSet") -> bool:
        if self.total != other.total:
            return self.total < other.total
        return self.items < other.items


class _KKState:
    def __init__(self, items: list[tuple[int, int]], k: int):
        self.sets = [_WeightedSet() for _ in range(k)]
        for set_idx, (idx, weight) in enumerate(items):
            self.sets[set_idx].add(idx, weight)
        self.sets.sort(reverse=True)

    @property
    def spread(self) -> int:
        return self.sets[0].total - self.sets[-1].total

    def merge(self, other: "_KKState") -> None:
        k = len(self.sets)
        for i in range(k):
            self.sets[i].merge(other.sets[k - 1 - i])
        self.sets.sort(reverse=True)

    def partitions(self) -> list[list[int]]:
        return [sorted(weighted_set.items) for weighted_set in self.sets]

    def __lt__(self, other: "_KKState") -> bool:
        if self.spread != other.spread:
            return self.spread > other.spread
        return self.sets[0] > other.sets[0]


def _karmarkar_karp(weights: Sequence[int], num_partitions: int) -> list[list[int]]:
    assert len(weights) >= num_partitions
    assert len(weights) % num_partitions == 0
    weighted_indices = sorted((weight, idx) for idx, weight in enumerate(weights))
    states: list[_KKState] = []
    for offset in range(0, len(weighted_indices), num_partitions):
        items = [(idx, weight) for weight, idx in weighted_indices[offset : offset + num_partitions]]
        heapq.heappush(states, _KKState(items, num_partitions))

    while len(states) > 1:
        state = heapq.heappop(states)
        state.merge(heapq.heappop(states))
        heapq.heappush(states, state)

    return states[0].partitions()


def _partition_loads(weights: Sequence[int], partitions: list[list[int]]) -> list[int]:
    return [sum(weights[i] for i in partition) for partition in partitions]


def _refine_by_swapping(weights: Sequence[int], partitions: list[list[int]]) -> list[list[int]]:
    """Apply the best pairwise swap until none lowers ``(max load, max - min load)``.

    Ties go to the first swap in (rank pair, left position, right position) order.
    Loads are taken relative to the current max load, so the int64 math stays within
    the spread plus the largest weight even when total FLOPs exceed int64.
    """
    partitions = [list(partition) for partition in partitions]
    loads = _partition_loads(weights, partitions)
    item_weights = np.asarray(weights, dtype=np.int64)
    num_partitions = len(partitions)
    no_other_max, no_other_min = np.iinfo(np.int64).min, np.iinfo(np.int64).max

    while True:
        top = max(loads)
        relative = np.array([load - top for load in loads], dtype=np.int64)
        best_score = (0, top - min(loads))
        best_swap = None
        for left_rank in range(num_partitions):
            for right_rank in range(left_rank + 1, num_partitions):
                others = np.delete(relative, [left_rank, right_rank])
                other_max = others.max() if len(others) else no_other_max
                other_min = others.min() if len(others) else no_other_min
                left_weights = item_weights[partitions[left_rank]][:, None]
                right_weights = item_weights[partitions[right_rank]][None, :]
                new_left = relative[left_rank] - left_weights + right_weights
                new_right = relative[right_rank] - right_weights + left_weights
                new_max = np.maximum(np.maximum(new_left, new_right), other_max)
                lowest_max = int(new_max.min())
                if lowest_max > best_score[0]:
                    continue
                new_min = np.minimum(np.minimum(new_left, new_right), other_min)
                spread = np.where(new_max == lowest_max, new_max - new_min, no_other_min)
                flat = int(spread.argmin())
                score = (lowest_max, int(spread.flat[flat]))
                if score < best_score:
                    best_score = score
                    best_swap = (left_rank, right_rank, *divmod(flat, right_weights.shape[1]))
        if best_swap is None:
            return partitions

        left_rank, right_rank, left_pos, right_pos = best_swap
        left_idx, right_idx = partitions[left_rank][left_pos], partitions[right_rank][right_pos]
        partitions[left_rank][left_pos], partitions[right_rank][right_pos] = right_idx, left_idx
        loads[left_rank] += weights[right_idx] - weights[left_idx]
        loads[right_rank] += weights[left_idx] - weights[right_idx]


def balanced_partition(weights: Sequence[int], num_partitions: int) -> list[list[int]]:
    """Partition item indices into ``num_partitions`` groups of near-equal total weight.

    Requires ``len(weights)`` to be a positive multiple of ``num_partitions``.
    """
    partitions = _karmarkar_karp(weights, num_partitions)
    return _refine_by_swapping(weights, partitions)


def _truncate_mm_refs(refs: MMRefs, seq_len: int) -> tuple[int, MMRefs | None]:
    cut = seq_len
    kept: list[MMImageRef] = []
    for image in refs.images:
        end = image.offset + image.length
        if image.offset < seq_len < end:
            cut = image.offset
            break
        if end <= seq_len:
            kept.append(image)
    return cut, MMRefs(images=kept) if kept else None


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


def prepare_sample(training_example: TrainingSample, seq_len: int) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    if error := multimodal_sample_error(training_example):
        raise ValueError(error)
    input_ids = training_example.token_ids
    loss_mask = training_example.mask
    inference_logprobs = training_example.logprobs
    if training_example.advantages is not None:
        advantages = training_example.advantages
    else:
        rl_w = training_example.rl_weights
        has_rl_members = loss_mask.any() if rl_w is None else (loss_mask & (rl_w != 0)).any()
        if has_rl_members:
            raise ValueError(
                f"sample from env '{training_example.env_name}' has rl member tokens but no advantages — "
                "the producer must stamp the advantage stream (the orchestrator broadcasts the rollout scalar)"
            )
        advantages = np.zeros(len(input_ids), dtype=np.float32)
    # Component weight streams: keep absent streams None (rl weight 1.0 on the
    # loss mask, no ce/ref_kl component) so the packed batch stays as small as before.
    rl_weights = training_example.rl_weights
    ce_weights = training_example.ce_weights
    ref_kl_weights = training_example.ref_kl_weights
    position_ids = np.arange(len(input_ids), dtype=np.int32)
    mm_token_type_ids = training_example.mm_token_type_ids
    mm_refs = training_example.mm_refs
    assert training_example.env_name != "all", "env_name='all' is reserved for aggregate metric keys"

    # Per-token sampling temperatures (context tokens are masked out, so theirs are don't-care).
    temperatures = training_example.temperatures

    # Ref logprobs already cover the full sequence (prompt + completion),
    # computed via prefill in the orchestrator when the algorithm scores against a reference
    ref_logprobs = training_example.ref_logprobs
    payload = training_example.payload

    if len(input_ids) > seq_len:
        cut = seq_len
        if mm_refs is not None:
            cut, mm_refs = _truncate_mm_refs(mm_refs, seq_len)
        input_ids = input_ids[:cut]
        loss_mask = loss_mask[:cut]
        inference_logprobs = inference_logprobs[:cut]
        position_ids = position_ids[:cut]
        advantages = advantages[:cut]
        temperatures = temperatures[:cut]
        if ref_logprobs is not None:
            ref_logprobs = ref_logprobs[:cut]
        if rl_weights is not None:
            rl_weights = rl_weights[:cut]
        if ce_weights is not None:
            ce_weights = ce_weights[:cut]
        if ref_kl_weights is not None:
            ref_kl_weights = ref_kl_weights[:cut]
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[:cut]
        if payload is not None:
            payload = msgspec.structs.replace(payload, num_tokens=cut)

    assert (
        len(input_ids)
        == len(advantages)
        == len(loss_mask)
        == len(position_ids)
        == len(inference_logprobs)
        == len(temperatures)
    ), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}, temperatures: {len(temperatures)}"
    )
    if ref_logprobs is not None:
        assert len(ref_logprobs) == len(input_ids), f"ref_logprobs: {len(ref_logprobs)}"
    for stream_name, stream in (
        ("rl_weights", rl_weights),
        ("ce_weights", ce_weights),
        ("ref_kl_weights", ref_kl_weights),
    ):
        if stream is not None:
            assert len(stream) == len(input_ids), f"{stream_name}: {len(stream)}"

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        sequence_lengths=[len(input_ids)],
        ref_logprobs=ref_logprobs,
        temperatures=temperatures,
        mm_token_type_ids=mm_token_type_ids,
        env_names=[training_example.env_name],
        mm_refs=mm_refs,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=ref_kl_weights,
        seq_lens=[len(input_ids)],
        trace_ids=[training_example.trace_id or ""],
        branch_indices=[training_example.branch_index if training_example.branch_index is not None else -1],
        payloads=[payload] if payload is not None else None,
    )


def _is_multimodal_sample(sample: MicroBatch) -> bool:
    """Check if a sample contains multimodal data (images)."""
    return sample.mm_refs is not None


def _has_routing(sample: MicroBatch) -> bool:
    """Whether the sample's arrays, kept by an env server, include routing rows."""
    return any(payload is not None and "routed_experts" in payload.fields for payload in sample.payloads or [])


@dataclass
class _MicroBatchBin:
    samples: list[MicroBatch]
    length: int

    @classmethod
    def from_sample(cls, sample: MicroBatch) -> "_MicroBatchBin":
        return cls(samples=[sample], length=len(sample.input_ids))

    @property
    def first_sample(self) -> MicroBatch:
        return self.samples[0]

    def can_add(self, sample: MicroBatch, max_seq_len: int) -> bool:
        # Loss routing is per token (component weight streams), so samples of
        # different loss types and raw-image refs pack together freely.
        first_sample = self.first_sample
        if self.length + len(sample.input_ids) > max_seq_len:
            return False
        # Routing replays per micro batch: every sample in it has routing rows or none does.
        if _has_routing(first_sample) != _has_routing(sample):
            return False

        return True

    def add(self, sample: MicroBatch) -> None:
        self.samples.append(sample)
        self.length += len(sample.input_ids)

    def workload(self, bin_cost: Callable[[Sequence[int]], int]) -> int:
        return bin_cost([len(sample.input_ids) for sample in self.samples])

    def split_by_workload(self, bin_cost: Callable[[Sequence[int]], int]) -> tuple["_MicroBatchBin", "_MicroBatchBin"]:
        # Greedily place the heaviest sample on the currently lighter side (longest-processing-time).
        ranked = sorted(self.samples, key=lambda sample: -bin_cost([len(sample.input_ids)]))
        left: list[MicroBatch] = []
        right: list[MicroBatch] = []
        left_workload = right_workload = 0
        for sample in ranked:
            sample_workload = bin_cost([len(sample.input_ids)])
            if left_workload <= right_workload:
                left.append(sample)
                left_workload += sample_workload
            else:
                right.append(sample)
                right_workload += sample_workload
        return (
            _MicroBatchBin(left, sum(len(sample.input_ids) for sample in left)),
            _MicroBatchBin(right, sum(len(sample.input_ids) for sample in right)),
        )


def _materialize_bin(bin_content: _MicroBatchBin) -> MicroBatch:
    has_ref_logprobs = any(sample.ref_logprobs is not None for sample in bin_content.samples)
    has_mm_token_type_ids = any(sample.mm_token_type_ids is not None for sample in bin_content.samples)
    # A weight stream materializes as soon as one packed sample carries it; the
    # samples that lack it get the stream's identity fill (STREAM_FILL).
    has_stream = {name: any(getattr(s, name) is not None for s in bin_content.samples) for name in STREAM_FILL}

    # Each sample's per-token arrays, concatenated once the bin is walked.
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
    payloads: list[PayloadSlice | None] = []

    offset = 0
    for sample in bin_content.samples:
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
                MMImageRef(
                    url=image.url,
                    offset=offset + image.offset,
                    length=image.length,
                )
                for image in sample.mm_refs.images
            )
        seq_lens.extend(sample.seq_lens)
        trace_ids.extend(sample.trace_ids or [""] * len(sample.sequence_lengths))
        branch_indices.extend(sample.branch_indices or [-1] * len(sample.sequence_lengths))
        payloads.extend(sample.payloads or [None] * len(sample.sequence_lengths))
        offset += sample_len

    sequence_lengths = [len(sample.input_ids) for sample in bin_content.samples]
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
        payloads=payloads if any(payload is not None for payload in payloads) else None,
    )


def _expand_bins_by_splitting(
    bins: list[_MicroBatchBin], target_count: int, bin_cost: Callable[[Sequence[int]], int]
) -> None:
    while len(bins) < target_count:
        candidates = [
            (bin_content.workload(bin_cost), idx)
            for idx, bin_content in enumerate(bins)
            if len(bin_content.samples) > 1
        ]
        if not candidates:
            break
        _, idx = max(candidates)
        left, right = bins[idx].split_by_workload(bin_cost)
        bins[idx] = left
        bins.append(right)


def packed_samples_into_micro_bs(
    samples: list[MicroBatch],
    max_seq_len: int,
    num_train_workers: int,
    bin_cost: Callable[[Sequence[int]], int],
) -> list[MicroBatch]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    With per-token temperatures, samples can be packed together regardless of their temperature values.

    Multimodal samples pack with text spans and other raw-image samples. Packed
    batches preserve sample boundaries in ``seq_lens``.
    """
    # Sort by decreasing length for packing efficiency
    samples.sort(key=lambda sample: -len(sample.input_ids))

    bins: list[_MicroBatchBin] = []

    for sample in samples:
        # Try to find a bin that can fit this sequence. Multimodal samples only
        # pack when their sidecar tensors are compatible.
        for bin_content in bins:
            if bin_content.can_add(sample, max_seq_len):
                bin_content.add(sample)
                break
        else:
            bins.append(_MicroBatchBin.from_sample(sample))

    if num_train_workers > 1:
        target_count = max(
            ((len(bins) + num_train_workers - 1) // num_train_workers) * num_train_workers,
            num_train_workers,
        )
        _expand_bins_by_splitting(bins, target_count, bin_cost)

    return [_materialize_bin(bin_content) for bin_content in bins]


def _distribute_group(
    group: list[MicroBatch],
    num_train_workers: int,
    bin_cost: Callable[[Sequence[int]], int],
) -> list[list[MicroBatch]]:
    # Callers pad each group to a positive multiple of num_train_workers first.
    assert len(group) % num_train_workers == 0, "Number of micro batches is not divisible by number of data ranks"
    if not group:
        return [[] for _ in range(num_train_workers)]

    weights = [bin_cost(micro_batch.sequence_lengths) for micro_batch in group]
    partitions = balanced_partition(weights, num_train_workers)
    return [[group[i] for i in partition] for partition in partitions]


def pad_micro_batch(micro_batch: MicroBatch, pad_to_multiple_of: int) -> MicroBatch:
    """
    Pad a micro batch with loss-masked tokens up to the next multiple of `pad_to_multiple_of`.
    Args:
        micro_batch: The micro batch to pad.
        pad_to_multiple_of: The multiple to pad the sequence length up to.
    Returns:
        The padded micro batch.
    """

    padding_size = (pad_to_multiple_of - (len(micro_batch.input_ids) % pad_to_multiple_of)) % pad_to_multiple_of

    if len(micro_batch.env_names) != len(micro_batch.sequence_lengths):
        raise ValueError(
            f"MicroBatch.env_names must name every sequence: "
            f"env_names={len(micro_batch.env_names)}, sequences={len(micro_batch.sequence_lengths)}"
        )

    if not (pad_to_multiple_of > 1 and padding_size > 0):
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

    return micro_batch


def _assert_token_arrays_aligned(micro_batch: MicroBatch) -> None:
    """Every per-token array must stay position-aligned with ``input_ids``
    through packing and padding — a field extended without backfill would
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
    assert micro_batch.payloads is None or len(micro_batch.payloads) == num_sequences, (
        f"payloads misaligned after packing: {len(micro_batch.payloads)} != {num_sequences} sequences"
    )


def _make_dummy_batch(source: MicroBatch) -> MicroBatch:
    """Create a zero-loss dummy batch from an existing batch, preserving its modality."""
    dummy = copy.deepcopy(source)
    dummy.advantages = np.zeros(len(dummy.input_ids), dtype=np.float32)
    dummy.loss_mask = np.zeros(len(dummy.input_ids), dtype=np.bool_)
    # ce/ref_kl membership is weight != 0 (independent of loss_mask), so the
    # streams must go too or the dummy would still train those tokens.
    dummy.rl_weights = None
    dummy.ce_weights = None
    dummy.ref_kl_weights = None
    # Fully loss-masked: fetch only the routing rows the forward pass replays, since
    # replaying sampling masks would be pure wasted work.
    dummy.payloads = (
        [msgspec.structs.replace(payload, fields=["routed_experts"]) for payload in dummy.payloads]
        if _has_routing(dummy)
        else None
    )
    # The copied identity would double-annotate the source's traces.
    dummy.trace_ids = None
    dummy.branch_indices = None
    return dummy


def _pad_group_for_distribution(group: list[MicroBatch], num_train_workers: int) -> list[MicroBatch]:
    """Pad a group of micro batches so its length is divisible by num_train_workers."""
    num_padding = -len(group) % num_train_workers
    if num_padding > 0 and len(group) > 0:
        dummy = _make_dummy_batch(group[0])
        group.extend([dummy] * num_padding)
    return group


def prepare_batch(
    rollouts: list[TrainingSample],
    seq_len: int,
    num_train_workers: int,
    bin_cost: Callable[[Sequence[int]], int],
    pad_to_multiple_of: int = 1,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, seq_len], the number of samples is not fixed per micro batch.

    FSDP requires all ranks to execute the same operations at each step. If one rank
    processes a multimodal batch (triggering the vision encoder) while another processes
    a text-only batch, the all-gather will hang. We separate micro batches by modality
    and distribute them so that at each step index, all ranks see the same modality.
    """
    all_samples = [prepare_sample(rollout, seq_len) for rollout in rollouts]

    micro_batches = packed_samples_into_micro_bs(all_samples, seq_len, num_train_workers, bin_cost)
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]

    # Separate by modality so each step index has uniform modality across all ranks
    mm_batches = [b for b in micro_batches if _is_multimodal_sample(b)]
    text_batches = [b for b in micro_batches if not _is_multimodal_sample(b)]

    # Pad each group independently so its count is divisible by num_train_workers
    mm_batches = _pad_group_for_distribution(mm_batches, num_train_workers)
    text_batches = _pad_group_for_distribution(text_batches, num_train_workers)

    # Alignment check after distribution padding so the dummy batches are covered too
    for micro_batch in (*mm_batches, *text_batches):
        _assert_token_arrays_aligned(micro_batch)

    batches_per_gpu: list[list[MicroBatch]] = [[] for _ in range(num_train_workers)]
    for group in (mm_batches, text_batches):
        group_batches_per_gpu = _distribute_group(group, num_train_workers, bin_cost)
        for worker_idx, worker_batches in enumerate(group_batches_per_gpu):
            batches_per_gpu[worker_idx].extend(worker_batches)

    return batches_per_gpu


class BatchPacker:
    """Bin-packs a step's training samples into one micro-batch list per trainer DP rank."""

    def __init__(self, config: OrchestratorConfig):
        self.seq_len = config.seq_len
        self.num_train_workers = config.num_train_workers
        # Rows hold at most `seq_len` tokens, so padding to a multiple of it pads every row to exactly `seq_len`
        self.pad_to_multiple_of = config.seq_len if config.pad_to_seq_len else config.pad_to_multiple_of
        # Built as the trainer builds it. The flash-attention version selects kernels, not modules.
        model_config = ModelConfig(
            name=config.model.name, attn="flash_attention_3", trust_remote_code=config.tokenizer.trust_remote_code
        )
        self.bin_cost = FlopCounter(get_model(model_config, device=torch.device("meta")))

    def pack(self, samples: list[TrainingSample]) -> list[list[MicroBatch]]:
        return prepare_batch(
            rollouts=samples,
            seq_len=self.seq_len,
            num_train_workers=self.num_train_workers,
            bin_cost=self.bin_cost,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
