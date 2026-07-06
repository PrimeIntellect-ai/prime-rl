import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from prime_rl.trainer.utils import balanced_partition
from prime_rl.transport.types import EncodedTensor, MicroBatch, RoutedExperts, TrainingSample

ROUTED_EXPERTS_DTYPE_ITEMSIZE = {
    "uint8": 1,
    "int16": 2,
    "int32": 4,
}

# Backfill value per component weight stream when a packed sample doesn't
# carry it: absent rl means weight 1.0 on the loss mask, absent ce/ref_kl
# means no component (weight 0.0).
STREAM_FILL = {"rl_weights": 1.0, "ce_weights": 0.0, "ref_kl_weights": 0.0}


def _copy_routed_experts(routed_experts: RoutedExperts) -> RoutedExperts:
    return RoutedExperts(
        data=routed_experts.data,
        shape=list(routed_experts.shape),
        dtype=routed_experts.dtype,
    )


def _routed_experts_row_size(routed_experts: RoutedExperts) -> int:
    return routed_experts.shape[1] * routed_experts.shape[2] * ROUTED_EXPERTS_DTYPE_ITEMSIZE[routed_experts.dtype]


def _slice_routed_experts(routed_experts: RoutedExperts, seq_len: int) -> RoutedExperts:
    row_size = _routed_experts_row_size(routed_experts)
    return RoutedExperts(
        data=routed_experts.data[: seq_len * row_size],
        shape=[seq_len, routed_experts.shape[1], routed_experts.shape[2]],
        dtype=routed_experts.dtype,
    )


def _pad_routed_experts(micro_batch: MicroBatch, padding_size: int) -> None:
    routed_experts = micro_batch.routed_experts
    assert routed_experts is not None
    row_size = _routed_experts_row_size(routed_experts)
    routed_experts.data += b"\0" * (padding_size * row_size)
    routed_experts.shape[0] += padding_size


def _slice_encoded(tensor: EncodedTensor, n_rows: int) -> EncodedTensor:
    """First `n_rows` rows of a dim-0-stacked encoded tensor (e.g. pixel_values, image_grid_thw)."""
    row = int(np.prod(tensor.shape[1:])) if len(tensor.shape) > 1 else 1
    itemsize = np.dtype(tensor.dtype).itemsize
    return EncodedTensor(
        dtype=tensor.dtype,
        shape=[n_rows, *tensor.shape[1:]],
        data=tensor.data[: n_rows * row * itemsize],
    )


def _truncate_mm(
    mm_token_type_ids: list[int], mm_kwargs: dict[str, EncodedTensor], seq_len: int
) -> tuple[int, dict[str, EncodedTensor] | None]:
    """Truncating a sample must not split an image's placeholder block, else the surviving image
    token count no longer matches the image embeddings in `mm_kwargs`. Returns the cut point
    (<= seq_len, never inside an image block) and `mm_kwargs` sliced to the images whose
    placeholders fully survive (None if no image survives)."""
    grid = np.frombuffer(bytearray(mm_kwargs["image_grid_thw"].data), dtype=mm_kwargs["image_grid_thw"].dtype).reshape(
        mm_kwargs["image_grid_thw"].shape
    )
    patches_per_image = [int(g.prod()) for g in grid]
    total_patches = mm_kwargs["pixel_values"].shape[0]
    total_tokens = sum(1 for t in mm_token_type_ids if t)
    ppt = total_patches // total_tokens if total_tokens else 1  # patches per token (merge^2)
    tokens_per_image = [p // ppt for p in patches_per_image]

    surviving = sum(1 for t in mm_token_type_ids[:seq_len] if t)
    kept = acc = 0
    for n in tokens_per_image:
        if acc + n > surviving:
            break
        acc += n
        kept += 1
    if acc == surviving:
        cut = seq_len  # surviving image tokens are exactly `kept` whole images
    else:
        # `surviving` lands inside image `kept`; cut to its first placeholder, dropping it.
        seen, cut = 0, seq_len
        for i, t in enumerate(mm_token_type_ids):
            if t:
                seen += 1
                if seen == acc + 1:
                    cut = i
                    break
    if not kept:
        return cut, None
    kept_patches = sum(patches_per_image[:kept])
    sliced = {k: _slice_encoded(v, kept if k == "image_grid_thw" else kept_patches) for k, v in mm_kwargs.items()}
    return cut, sliced


def _append_mm_kwargs(dst: dict[str, EncodedTensor], src: dict[str, EncodedTensor]) -> None:
    for key in dst:
        dst[key].data += src[key].data
        dst[key].shape[0] += src[key].shape[0]


def _can_pack_mm_kwargs(dst: dict[str, EncodedTensor] | None, src: dict[str, EncodedTensor] | None) -> bool:
    if dst is None or src is None or set(dst) != set(src):
        return False
    return all(
        dst[key].dtype == src[key].dtype
        and len(dst[key].shape) > 0
        and len(dst[key].shape) == len(src[key].shape)
        and dst[key].shape[1:] == src[key].shape[1:]
        for key in dst
    )


def prepare_sample(training_example: TrainingSample, seq_len: int) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    input_ids = training_example.token_ids
    loss_mask = training_example.mask
    inference_logprobs = training_example.logprobs
    if training_example.advantages is not None:
        advantages = list(training_example.advantages)
    else:
        rl_w = training_example.rl_weights
        has_rl_members = any(loss_mask) if rl_w is None else any(m and w != 0 for m, w in zip(loss_mask, rl_w))
        if has_rl_members:
            raise ValueError(
                f"sample from env '{training_example.env_name}' has rl member tokens but no advantages — "
                "the producer must stamp the advantage stream (the orchestrator broadcasts the rollout scalar)"
            )
        advantages = [0.0] * len(input_ids)
    # Component weight streams: keep absent streams None (rl weight 1.0 on the
    # loss mask, no ce/ref_kl component) so the packed batch stays as small as before.
    rl_weights = list(training_example.rl_weights) if training_example.rl_weights is not None else None
    ce_weights = list(training_example.ce_weights) if training_example.ce_weights is not None else None
    ref_kl_weights = list(training_example.ref_kl_weights) if training_example.ref_kl_weights is not None else None
    position_ids = list(range(len(input_ids)))
    mm_token_type_ids = training_example.mm_token_type_ids
    mm_kwargs = training_example.mm_kwargs
    assert training_example.env_name != "all", "env_name='all' is reserved for aggregate metric keys"
    env_names = [training_example.env_name] * len(input_ids)

    # Per-token sampling temperatures (context tokens are masked out, so theirs are don't-care).
    temperatures = training_example.temperatures

    # Ref logprobs already cover the full sequence (prompt + completion),
    # computed via prefill in the orchestrator when the algorithm scores against a reference
    ref_logprobs = training_example.ref_logprobs
    routed_experts = (
        _copy_routed_experts(training_example.routed_experts) if training_example.routed_experts is not None else None
    )

    if len(input_ids) > seq_len:
        # Multimodal: never split an image's placeholder block — cut to a whole-image boundary
        # and slice mm_kwargs to match, so image-token count == image-embedding count.
        cut = seq_len
        if mm_token_type_ids is not None and mm_kwargs is not None:
            cut, mm_kwargs = _truncate_mm(mm_token_type_ids, mm_kwargs, seq_len)
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
        if routed_experts is not None:
            routed_experts = _slice_routed_experts(routed_experts, cut)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[:cut]
        env_names = env_names[:cut]

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

    if routed_experts is not None:
        assert routed_experts.shape[0] == len(input_ids), (
            f"routed_experts: {routed_experts.shape}, input_ids: {len(input_ids)}"
        )
        assert len(routed_experts.data) == len(input_ids) * _routed_experts_row_size(routed_experts)

    if mm_token_type_ids is not None:
        assert len(mm_token_type_ids) == len(input_ids), (
            f"mm_token_type_ids: {len(mm_token_type_ids)}, input_ids: {len(input_ids)}"
        )
    if mm_kwargs is not None and "image_grid_thw" in mm_kwargs and mm_token_type_ids is None:
        raise ValueError("image_grid_thw multimodal samples require mm_token_type_ids")
    assert len(env_names) == len(input_ids), f"env_names: {len(env_names)}, input_ids: {len(input_ids)}"

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        sequence_lengths=[len(input_ids)],
        ref_logprobs=ref_logprobs,
        temperatures=temperatures,
        routed_experts=routed_experts,
        mm_token_type_ids=mm_token_type_ids,
        env_names=env_names,
        mm_kwargs=mm_kwargs,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=ref_kl_weights,
        seq_lens=[len(input_ids)],
    )


def _is_multimodal_sample(sample: MicroBatch) -> bool:
    """Check if a sample contains multimodal data (images)."""
    return sample.mm_kwargs is not None


@dataclass
class _MicroBatchBin:
    samples: list[tuple[int, MicroBatch]]
    length: int

    @classmethod
    def from_sample(cls, lora_idx: int, sample: MicroBatch) -> "_MicroBatchBin":
        return cls(samples=[(lora_idx, sample)], length=len(sample.input_ids))

    @property
    def first_sample(self) -> MicroBatch:
        return self.samples[0][1]

    @property
    def first_lora_idx(self) -> int:
        return self.samples[0][0]

    @property
    def first_multimodal_sample(self) -> MicroBatch | None:
        for _, sample in self.samples:
            if _is_multimodal_sample(sample):
                return sample
        return None

    def can_add(self, sample: MicroBatch, max_seq_len: int, lora_idx: int) -> bool:
        # Loss routing is per token (component weight streams), so samples of
        # different loss types pack together freely. Multimodal packing is still
        # constrained by modality sidecars, length, routed experts, and LoRA/run.
        first_sample = self.first_sample
        if self.length + len(sample.input_ids) > max_seq_len:
            return False
        if (first_sample.routed_experts is None) != (sample.routed_experts is None):
            return False

        sample_is_mm = _is_multimodal_sample(sample)
        existing_mm_sample = self.first_multimodal_sample
        if existing_mm_sample is None and not sample_is_mm:
            return True
        if self.first_lora_idx != lora_idx:
            return False
        if existing_mm_sample is not None and sample_is_mm:
            return _can_pack_mm_kwargs(existing_mm_sample.mm_kwargs, sample.mm_kwargs)
        return True

    def add(self, lora_idx: int, sample: MicroBatch) -> None:
        self.samples.append((lora_idx, sample))
        self.length += len(sample.input_ids)

    def workload(self, bin_cost: Callable[[Sequence[int]], int]) -> int:
        return bin_cost([len(sample.input_ids) for _, sample in self.samples])

    def split_by_workload(self, bin_cost: Callable[[Sequence[int]], int]) -> tuple["_MicroBatchBin", "_MicroBatchBin"]:
        # Greedily place the heaviest sample on the currently lighter side (longest-processing-time).
        ranked = sorted(self.samples, key=lambda pair: -bin_cost([len(pair[1].input_ids)]))
        left: list[tuple[int, MicroBatch]] = []
        right: list[tuple[int, MicroBatch]] = []
        left_workload = right_workload = 0
        for lora_idx, sample in ranked:
            sample_workload = bin_cost([len(sample.input_ids)])
            if left_workload <= right_workload:
                left.append((lora_idx, sample))
                left_workload += sample_workload
            else:
                right.append((lora_idx, sample))
                right_workload += sample_workload
        return (
            _MicroBatchBin(left, sum(len(sample.input_ids) for _, sample in left)),
            _MicroBatchBin(right, sum(len(sample.input_ids) for _, sample in right)),
        )


def _materialize_bin(bin_content: _MicroBatchBin, num_loras: int) -> MicroBatch:
    has_ref_logprobs = any(sample.ref_logprobs is not None for _, sample in bin_content.samples)
    has_mm_token_type_ids = any(sample.mm_token_type_ids is not None for _, sample in bin_content.samples)
    # A weight stream materializes as soon as one packed sample carries it; the
    # samples that lack it get the stream's identity fill (STREAM_FILL).
    has_stream = {name: any(getattr(s, name) is not None for _, s in bin_content.samples) for name in STREAM_FILL}

    input_ids: list[int] = []
    loss_mask: list[bool] = []
    advantages: list[float] = []
    inference_logprobs: list[float] = []
    position_ids: list[int] = []
    temperatures: list[float] = []
    env_names: list[str] = []
    ref_logprobs: list[float] | None = [] if has_ref_logprobs else None
    mm_token_type_ids: list[int] | None = [] if has_mm_token_type_ids else None
    mm_kwargs: dict[str, EncodedTensor] | None = None
    streams: dict[str, list[float] | None] = {name: ([] if has_stream[name] else None) for name in STREAM_FILL}
    seq_lens: list[int] = []
    routed_experts: RoutedExperts | None = None
    lora_num_tokens = [0] * num_loras

    for lora_idx, sample in bin_content.samples:
        sample_len = len(sample.input_ids)
        input_ids.extend(sample.input_ids)
        loss_mask.extend(sample.loss_mask)
        advantages.extend(sample.advantages)
        inference_logprobs.extend(sample.inference_logprobs)
        position_ids.extend(sample.position_ids)
        temperatures.extend(sample.temperatures)
        env_names.extend(sample.env_names)
        if ref_logprobs is not None:
            ref_logprobs.extend(sample.ref_logprobs if sample.ref_logprobs is not None else [0.0] * sample_len)
        for name, fill in STREAM_FILL.items():
            stream = streams[name]
            if stream is not None:
                sample_stream = getattr(sample, name)
                stream.extend(sample_stream if sample_stream is not None else [fill] * sample_len)
        if mm_token_type_ids is not None:
            mm_token_type_ids.extend(
                sample.mm_token_type_ids if sample.mm_token_type_ids is not None else [0] * sample_len
            )
        if sample.routed_experts is not None:
            if routed_experts is None:
                routed_experts = _copy_routed_experts(sample.routed_experts)
            else:
                assert routed_experts.dtype == sample.routed_experts.dtype
                assert routed_experts.shape[1:] == sample.routed_experts.shape[1:]
                routed_experts.data += sample.routed_experts.data
                routed_experts.shape[0] += sample.routed_experts.shape[0]
        if sample.mm_kwargs is not None:
            if mm_kwargs is None:
                mm_kwargs = copy.deepcopy(sample.mm_kwargs)
            else:
                _append_mm_kwargs(mm_kwargs, sample.mm_kwargs)
        seq_lens.extend(sample.seq_lens if sample.seq_lens is not None else [sample_len])
        lora_num_tokens[lora_idx] += sample_len

    sequence_lengths = [len(sample.input_ids) for _, sample in bin_content.samples]
    assert sum(sequence_lengths) == len(input_ids), (sequence_lengths, len(input_ids))
    assert sum(seq_lens) == len(input_ids), (seq_lens, len(input_ids))

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        sequence_lengths=sequence_lengths,
        ref_logprobs=ref_logprobs,
        temperatures=temperatures,
        lora_num_tokens=lora_num_tokens,
        routed_experts=routed_experts,
        mm_token_type_ids=mm_token_type_ids,
        env_names=env_names,
        mm_kwargs=mm_kwargs,
        rl_weights=streams["rl_weights"],
        ce_weights=streams["ce_weights"],
        ref_kl_weights=streams["ref_kl_weights"],
        seq_lens=seq_lens,
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
    samples: list[tuple[int, MicroBatch]],
    max_seq_len: int,
    num_loras: int,
    num_train_workers: int,
    bin_cost: Callable[[Sequence[int]], int],
) -> list[MicroBatch]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    With per-token temperatures, samples can be packed together regardless of their temperature values.

    Multimodal samples pack with text spans from the same run/LoRA and with
    compatible eager ``mm_kwargs`` samples. Packed batches preserve sample
    boundaries in ``seq_lens``.
    """
    # Sort by (lora_idx, -length) for packing efficiency
    samples.sort(key=lambda x: (x[0], -len(x[1].input_ids)))

    bins: list[_MicroBatchBin] = []

    for idx, sample in samples:
        # Try to find a bin that can fit this sequence. Multimodal samples only
        # pack when their sidecar tensors are compatible.
        for bin_content in bins:
            if bin_content.can_add(sample, max_seq_len, idx):
                bin_content.add(idx, sample)
                break
        else:
            bins.append(_MicroBatchBin.from_sample(idx, sample))

    if num_train_workers > 1:
        target_count = max(
            ((len(bins) + num_train_workers - 1) // num_train_workers) * num_train_workers,
            num_train_workers,
        )
        _expand_bins_by_splitting(bins, target_count, bin_cost)

    return [_materialize_bin(bin_content, num_loras) for bin_content in bins]


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
    Pad a micro batch with the given padding size sample
    Return the padded micro batch.
    Args:
        micro_batch: The micro batch to pad.
        padding_size: The number of padding tokens to add.
    Returns:
        The padded micro batch.
    """

    padding_size = (pad_to_multiple_of - (len(micro_batch.input_ids) % pad_to_multiple_of)) % pad_to_multiple_of

    if len(micro_batch.env_names) != len(micro_batch.input_ids):
        raise ValueError(
            f"MicroBatch.env_names must match input_ids length before padding: "
            f"env_names={len(micro_batch.env_names)}, input_ids={len(micro_batch.input_ids)}"
        )

    if not (pad_to_multiple_of > 1 and padding_size > 0):
        return micro_batch

    micro_batch.input_ids.extend([1] * padding_size)
    micro_batch.advantages.extend([0.0] * padding_size)
    micro_batch.loss_mask.extend([False] * padding_size)
    micro_batch.position_ids.extend(list(range(padding_size)))
    micro_batch.sequence_lengths.append(padding_size)
    micro_batch.inference_logprobs.extend([0.0] * padding_size)
    # Use temperature 1.0 for padding tokens (doesn't matter since loss_mask is False)
    micro_batch.temperatures.extend([1.0] * padding_size)
    if micro_batch.ref_logprobs is not None:
        micro_batch.ref_logprobs.extend([0.0] * padding_size)
    # Padding is loss-masked, so no component trains it; fill every stream
    # with 0.0 (not the pack-boundary defaults) so a padded pure-ce batch
    # still reads as rl-empty in token export, which keys off nonzero weights.
    for stream_name in STREAM_FILL:
        stream = getattr(micro_batch, stream_name)
        if stream is not None:
            stream.extend([0.0] * padding_size)
    if micro_batch.lora_num_tokens is not None:
        micro_batch.lora_num_tokens[-1] += (
            padding_size  # We send padding to the last lora so that tokens have ascending lora idx
        )
    if micro_batch.mm_token_type_ids is not None:
        micro_batch.mm_token_type_ids.extend([0] * padding_size)
    if micro_batch.seq_lens is not None:
        micro_batch.seq_lens.append(padding_size)
    if micro_batch.routed_experts is not None:
        _pad_routed_experts(micro_batch, padding_size)
    micro_batch.env_names.extend([""] * padding_size)

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
        "env_names",
        "ref_logprobs",
        "rl_weights",
        "ce_weights",
        "ref_kl_weights",
        "mm_token_type_ids",
    )
    for name in per_token_fields:
        values = getattr(micro_batch, name)
        assert values is None or len(values) == num_tokens, (
            f"{name} misaligned after packing: {len(values)} != {num_tokens} tokens"
        )
    assert sum(micro_batch.sequence_lengths) == num_tokens, (
        f"sequence_lengths sum {sum(micro_batch.sequence_lengths)} != {num_tokens} tokens"
    )
    if micro_batch.routed_experts is not None:
        assert micro_batch.routed_experts.shape[0] == num_tokens, (
            f"routed_experts misaligned after packing: {micro_batch.routed_experts.shape[0]} != {num_tokens} tokens"
        )


def _make_dummy_batch(source: MicroBatch) -> MicroBatch:
    """Create a zero-loss dummy batch from an existing batch, preserving its modality."""
    dummy = copy.deepcopy(source)
    dummy.advantages = [0.0] * len(dummy.input_ids)
    dummy.loss_mask = [False] * len(dummy.input_ids)
    # ce/ref_kl membership is weight != 0 (independent of loss_mask), so the
    # streams must go too or the dummy would still train those tokens.
    dummy.rl_weights = None
    dummy.ce_weights = None
    dummy.ref_kl_weights = None
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
    idxs: list[int],
    num_loras: int,
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
    all_samples = [(idx, prepare_sample(rollout, seq_len)) for idx, rollout in zip(idxs, rollouts)]

    micro_batches = packed_samples_into_micro_bs(
        all_samples,
        seq_len,
        num_loras,
        num_train_workers,
        bin_cost,
    )
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
