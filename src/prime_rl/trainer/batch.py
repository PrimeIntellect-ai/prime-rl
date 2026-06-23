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


def prepare_sample(training_example: TrainingSample, seq_len: int) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    input_ids = training_example.token_ids
    loss_mask = training_example.mask
    inference_logprobs = training_example.logprobs
    advantages = [training_example.advantage] * len(input_ids)
    reward = training_example.reward if training_example.reward is not None else float("nan")
    rewards = [reward] * len(input_ids)
    position_ids = list(range(len(input_ids)))
    mm_token_type_ids = training_example.mm_token_type_ids
    mm_kwargs = training_example.mm_kwargs
    assert training_example.env_name != "all", "env_name='all' is reserved for aggregate metric keys"
    env_names = [training_example.env_name] * len(input_ids)

    # Per-token sampling temperatures (context tokens are masked out, so theirs are don't-care).
    temperatures = training_example.temperatures

    # Teacher logprobs already cover the full sequence (prompt + completion),
    # computed via prefill in the orchestrator when a teacher model is configured
    teacher_logprobs = training_example.teacher_logprobs
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
        rewards = rewards[:cut]
        temperatures = temperatures[:cut]
        if teacher_logprobs is not None:
            teacher_logprobs = teacher_logprobs[:cut]
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
        == len(rewards)
        == len(temperatures)
    ), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}, rewards: {len(rewards)}, temperatures: {len(temperatures)}"
    )
    if teacher_logprobs is not None:
        assert len(teacher_logprobs) == len(input_ids), f"teacher_logprobs: {len(teacher_logprobs)}"

    if routed_experts is not None:
        assert routed_experts.shape[0] == len(input_ids), (
            f"routed_experts: {routed_experts.shape}, input_ids: {len(input_ids)}"
        )
        assert len(routed_experts.data) == len(input_ids) * _routed_experts_row_size(routed_experts)

    if mm_token_type_ids is not None:
        assert len(mm_token_type_ids) == len(input_ids), (
            f"mm_token_type_ids: {len(mm_token_type_ids)}, input_ids: {len(input_ids)}"
        )
    assert len(env_names) == len(input_ids), f"env_names: {len(env_names)}, input_ids: {len(input_ids)}"

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        sequence_lengths=[len(input_ids)],
        teacher_logprobs=teacher_logprobs,
        temperatures=temperatures,
        rewards=rewards,
        routed_experts=routed_experts,
        mm_token_type_ids=mm_token_type_ids,
        env_names=env_names,
        mm_kwargs=mm_kwargs,
        training_mode=training_example.training_mode,
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

    def can_add(self, sample: MicroBatch, max_seq_len: int) -> bool:
        first_sample = self.first_sample
        return (
            not _is_multimodal_sample(first_sample)
            and not _is_multimodal_sample(sample)
            and self.length + len(sample.input_ids) <= max_seq_len
            and first_sample.training_mode == sample.training_mode
            and (first_sample.routed_experts is None) == (sample.routed_experts is None)
        )

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
    has_rewards = any(sample.rewards is not None for _, sample in bin_content.samples)
    has_teacher_logprobs = any(sample.teacher_logprobs is not None for _, sample in bin_content.samples)
    has_mm_token_type_ids = any(sample.mm_token_type_ids is not None for _, sample in bin_content.samples)

    input_ids: list[int] = []
    loss_mask: list[bool] = []
    advantages: list[float] = []
    inference_logprobs: list[float] = []
    position_ids: list[int] = []
    temperatures: list[float] = []
    env_names: list[str] = []
    rewards: list[float] | None = [] if has_rewards else None
    teacher_logprobs: list[float] | None = [] if has_teacher_logprobs else None
    mm_token_type_ids: list[int] | None = [] if has_mm_token_type_ids else None
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
        if rewards is not None:
            rewards.extend(sample.rewards if sample.rewards is not None else [float("nan")] * sample_len)
        if teacher_logprobs is not None:
            teacher_logprobs.extend(
                sample.teacher_logprobs if sample.teacher_logprobs is not None else [0.0] * sample_len
            )
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
        lora_num_tokens[lora_idx] += sample_len

    sequence_lengths = [len(sample.input_ids) for _, sample in bin_content.samples]
    assert sum(sequence_lengths) == len(input_ids), (sequence_lengths, len(input_ids))
    first_sample = bin_content.first_sample

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        sequence_lengths=sequence_lengths,
        teacher_logprobs=teacher_logprobs,
        temperatures=temperatures,
        rewards=rewards,
        lora_num_tokens=lora_num_tokens,
        routed_experts=routed_experts,
        mm_token_type_ids=mm_token_type_ids,
        env_names=env_names,
        mm_kwargs=first_sample.mm_kwargs if _is_multimodal_sample(first_sample) else None,
        training_mode=first_sample.training_mode,
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

    NOTE: Multimodal samples (with mm_kwargs) are NOT packed together as they have variable-sized
    vision data that doesn't pack well. Each multimodal sample becomes its own micro batch.
    """
    # Sort by (lora_idx, -length) for packing efficiency
    samples.sort(key=lambda x: (x[0], -len(x[1].input_ids)))

    bins: list[_MicroBatchBin] = []

    for idx, sample in samples:
        # Try to find a bin that can fit this sequence (only pack text-only samples)
        for bin_content in bins:
            if bin_content.can_add(sample, max_seq_len):
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
    if micro_batch.rewards is not None:
        micro_batch.rewards.extend([float("nan")] * padding_size)
    micro_batch.loss_mask.extend([False] * padding_size)
    micro_batch.position_ids.extend(list(range(padding_size)))
    micro_batch.sequence_lengths.append(padding_size)
    micro_batch.inference_logprobs.extend([0.0] * padding_size)
    # Use temperature 1.0 for padding tokens (doesn't matter since loss_mask is False)
    micro_batch.temperatures.extend([1.0] * padding_size)
    if micro_batch.teacher_logprobs is not None:
        micro_batch.teacher_logprobs.extend([0.0] * padding_size)
    if micro_batch.lora_num_tokens is not None:
        micro_batch.lora_num_tokens[-1] += (
            padding_size  # We send padding to the last lora so that tokens have ascending lora idx
        )
    if micro_batch.mm_token_type_ids is not None:
        micro_batch.mm_token_type_ids.extend([0] * padding_size)
    if micro_batch.routed_experts is not None:
        _pad_routed_experts(micro_batch, padding_size)
    micro_batch.env_names.extend([""] * padding_size)

    return micro_batch


def _make_dummy_batch(source: MicroBatch) -> MicroBatch:
    """Create a zero-loss dummy batch from an existing batch, preserving its modality."""
    dummy = copy.deepcopy(source)
    dummy.advantages = [0.0] * len(dummy.input_ids)
    dummy.loss_mask = [False] * len(dummy.input_ids)
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

    micro_batches = packed_samples_into_micro_bs(all_samples, seq_len, num_loras, num_train_workers, bin_cost)
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]

    # Separate by modality so each step index has uniform modality across all ranks
    mm_batches = [b for b in micro_batches if _is_multimodal_sample(b)]
    text_batches = [b for b in micro_batches if not _is_multimodal_sample(b)]

    # Pad each group independently so its count is divisible by num_train_workers
    mm_batches = _pad_group_for_distribution(mm_batches, num_train_workers)
    text_batches = _pad_group_for_distribution(text_batches, num_train_workers)

    batches_per_gpu: list[list[MicroBatch]] = [[] for _ in range(num_train_workers)]
    for group in (mm_batches, text_batches):
        group_batches_per_gpu = _distribute_group(group, num_train_workers, bin_cost)
        for worker_idx, worker_batches in enumerate(group_batches_per_gpu):
            batches_per_gpu[worker_idx].extend(worker_batches)

    return batches_per_gpu
