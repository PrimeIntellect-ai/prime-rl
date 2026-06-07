import copy
import heapq
from dataclasses import dataclass, field
from typing import Any

from prime_rl.transport.types import MicroBatch, RoutedExperts, TrainingSample

ROUTED_EXPERTS_DTYPE_ITEMSIZE = {
    "uint8": 1,
    "int16": 2,
    "int32": 4,
}


def get_packing_flops_config(model_config: Any) -> Any:
    """Return the text config used for model-aware packing FLOP estimates."""
    return getattr(model_config, "text_config", model_config)


def _is_mla(config: Any) -> bool:
    return bool(getattr(config, "multi_latent_attention", False) or hasattr(config, "q_lora_rank"))


def _calculate_qkv_projection_flops(config: Any, seqlen: int) -> float:
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    kv_channels = getattr(config, "kv_channels", getattr(config, "head_dim", hidden_size // num_attention_heads))
    is_mla = _is_mla(config)
    if is_mla and getattr(config, "q_lora_rank", None) is not None:
        q_flops = (
            2
            * seqlen
            * config.q_lora_rank
            * (
                hidden_size
                + num_attention_heads * (getattr(config, "qk_head_dim", 0) + getattr(config, "qk_pos_emb_head_dim", 0))
            )
        )
    else:
        q_head_dim = (
            getattr(config, "qk_head_dim", 0) + getattr(config, "qk_pos_emb_head_dim", 0) if is_mla else kv_channels
        )
        q_flops = 2 * seqlen * hidden_size * num_attention_heads * q_head_dim

    if is_mla and getattr(config, "kv_lora_rank", None) is not None:
        kv_flops = (
            2
            * seqlen
            * (
                config.kv_lora_rank
                * (
                    hidden_size
                    + num_attention_heads * (getattr(config, "qk_head_dim", 0) + getattr(config, "v_head_dim", 0))
                )
                + hidden_size * getattr(config, "qk_pos_emb_head_dim", 0)
            )
        )
    else:
        num_query_groups = getattr(
            config, "num_query_groups", getattr(config, "num_key_value_heads", num_attention_heads)
        )
        kv_flops = 4 * seqlen * hidden_size * num_query_groups * kv_channels
    return q_flops + kv_flops


def _calculate_attention_flops(config: Any, seqlen: int) -> float:
    num_attention_heads = config.num_attention_heads
    kv_channels = getattr(config, "kv_channels", getattr(config, "head_dim", config.hidden_size // num_attention_heads))
    if _is_mla(config):
        flops = (
            num_attention_heads
            * seqlen
            * seqlen
            * (getattr(config, "qk_head_dim", 0) + getattr(config, "qk_pos_emb_head_dim", 0))
        )
        flops += num_attention_heads * seqlen * seqlen * getattr(config, "v_head_dim", kv_channels)
    else:
        flops = 2 * num_attention_heads * seqlen * seqlen * kv_channels
    return flops


def _calculate_layer_flops(config: Any, seqlen: int, ffn_hidden_size: int) -> float:
    hidden_size = config.hidden_size
    return (
        _calculate_qkv_projection_flops(config, seqlen)
        + _calculate_attention_flops(config, seqlen)
        + 2 * seqlen * hidden_size * hidden_size
        + 6 * seqlen * hidden_size * ffn_hidden_size
    )


def calculate_packing_fwd_flops(seqlens: list[int], config: Any) -> float:
    """Model-aware forward FLOP estimate copied in spirit from slime."""
    num_experts = getattr(config, "num_experts", getattr(config, "n_routed_experts", None))
    dense_ffn = getattr(
        config, "ffn_hidden_size", getattr(config, "intermediate_size", getattr(config, "moe_intermediate_size", 0))
    )
    if num_experts is None:
        num_dense_layers = config.num_hidden_layers
        num_moe_layers = 0
        moe_ffn = dense_ffn
    else:
        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        if isinstance(moe_layer_freq, list):
            num_dense_layers = sum(1 for freq in moe_layer_freq if freq == 0)
            num_moe_layers = sum(1 for freq in moe_layer_freq if freq > 0)
        elif isinstance(moe_layer_freq, int):
            num_dense_layers = sum(1 for i in range(config.num_hidden_layers) if i % moe_layer_freq != 0)
            num_moe_layers = config.num_hidden_layers - num_dense_layers
        elif getattr(config, "first_k_dense_replace", None) is not None:
            num_dense_layers = config.first_k_dense_replace
            num_moe_layers = config.num_hidden_layers - num_dense_layers
        else:
            num_dense_layers = 0
            num_moe_layers = config.num_hidden_layers

        routed_topk = getattr(config, "moe_router_topk", getattr(config, "num_experts_per_tok", 1))
        moe_ffn = getattr(config, "moe_ffn_hidden_size", getattr(config, "moe_intermediate_size", dense_ffn))
        moe_ffn = moe_ffn * routed_topk + (getattr(config, "moe_shared_expert_intermediate_size", None) or 0)

    total_flops = 0.0
    for seqlen in seqlens:
        if num_dense_layers > 0:
            total_flops += _calculate_layer_flops(config, seqlen, dense_ffn) * num_dense_layers
        if num_moe_layers > 0:
            total_flops += _calculate_layer_flops(config, seqlen, moe_ffn) * num_moe_layers
        total_flops += 2 * seqlen * config.hidden_size * config.vocab_size
    return total_flops


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


def prepare_sample(training_example: TrainingSample, seq_len: int) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    input_ids = training_example.prompt_ids + training_example.completion_ids
    loss_mask = training_example.prompt_mask + training_example.completion_mask
    inference_logprobs = [0.0] * len(training_example.prompt_ids) + training_example.completion_logprobs
    advantages = [training_example.advantage] * len(input_ids)
    reward = training_example.reward if training_example.reward is not None else float("nan")
    rewards = [reward] * len(input_ids)
    position_ids = list(range(len(input_ids)))
    mm_token_type_ids = training_example.mm_token_type_ids
    assert training_example.env_name != "all", "env_name='all' is reserved for aggregate metric keys"
    env_names = [training_example.env_name] * len(input_ids)

    # Per-token temperatures: prompt tokens use first completion temp (masked out anyway)
    # Default to 1.0 if completion is empty (e.g., model generated only tool calls with no text)
    prompt_temp = training_example.completion_temperatures[0] if training_example.completion_temperatures else 1.0
    temperatures = [prompt_temp] * len(training_example.prompt_ids) + training_example.completion_temperatures

    # Teacher logprobs already cover the full sequence (prompt + completion),
    # computed via prefill in the orchestrator when a teacher model is configured
    teacher_logprobs = training_example.teacher_logprobs
    routed_experts = (
        _copy_routed_experts(training_example.routed_experts) if training_example.routed_experts is not None else None
    )

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
        inference_logprobs = inference_logprobs[:seq_len]
        position_ids = position_ids[:seq_len]
        advantages = advantages[:seq_len]
        rewards = rewards[:seq_len]
        temperatures = temperatures[:seq_len]
        if teacher_logprobs is not None:
            teacher_logprobs = teacher_logprobs[:seq_len]
        if routed_experts is not None:
            routed_experts = _slice_routed_experts(routed_experts, seq_len)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[:seq_len]
        env_names = env_names[:seq_len]

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
        mm_kwargs=training_example.mm_kwargs,
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

    def workload(self, flops_config: Any | None) -> float:
        if flops_config is None:
            return self.length
        return calculate_packing_fwd_flops([len(sample.input_ids) for _, sample in self.samples], flops_config)

    def _sample_workload(self, sample: MicroBatch, flops_config: Any | None) -> float:
        if flops_config is None:
            return len(sample.input_ids)
        return calculate_packing_fwd_flops([len(sample.input_ids)], flops_config)

    def split_by_workload(self, flops_config: Any | None) -> tuple["_MicroBatchBin", "_MicroBatchBin"]:
        left: list[tuple[int, MicroBatch]] = []
        right: list[tuple[int, MicroBatch]] = []
        left_workload = 0.0
        right_workload = 0.0
        for lora_idx, sample in sorted(self.samples, key=lambda x: -self._sample_workload(x[1], flops_config)):
            sample_workload = self._sample_workload(sample, flops_config)
            if left_workload <= right_workload:
                left.append((lora_idx, sample))
                left_workload += sample_workload
            else:
                right.append((lora_idx, sample))
                right_workload += sample_workload
        return _MicroBatchBin(left, sum(len(sample.input_ids) for _, sample in left)), _MicroBatchBin(
            right, sum(len(sample.input_ids) for _, sample in right)
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


def _balanced_partitions(weights: list[int], num_partitions: int) -> list[list[int]]:
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


def _partition_loads(weights: list[float], partitions: list[list[int]]) -> list[float]:
    return [sum(weights[i] for i in partition) for partition in partitions]


def _improve_partitions_by_swapping(weights: list[float], partitions: list[list[int]]) -> list[list[int]]:
    partitions = [list(partition) for partition in partitions]
    loads = _partition_loads(weights, partitions)

    while True:
        current_score = (max(loads), max(loads) - min(loads))
        best_swap = None
        best_score = current_score
        for left_rank in range(len(partitions)):
            for right_rank in range(left_rank + 1, len(partitions)):
                for left_pos, left_idx in enumerate(partitions[left_rank]):
                    for right_pos, right_idx in enumerate(partitions[right_rank]):
                        new_left = loads[left_rank] - weights[left_idx] + weights[right_idx]
                        new_right = loads[right_rank] - weights[right_idx] + weights[left_idx]
                        new_loads = list(loads)
                        new_loads[left_rank] = new_left
                        new_loads[right_rank] = new_right
                        score = (max(new_loads), max(new_loads) - min(new_loads))
                        if score < best_score:
                            best_score = score
                            best_swap = (left_rank, right_rank, left_pos, right_pos, new_loads)
        if best_swap is None:
            return partitions

        left_rank, right_rank, left_pos, right_pos, loads = best_swap
        partitions[left_rank][left_pos], partitions[right_rank][right_pos] = (
            partitions[right_rank][right_pos],
            partitions[left_rank][left_pos],
        )


def _expand_bins_by_splitting(bins: list[_MicroBatchBin], target_count: int, flops_config: Any | None) -> None:
    while len(bins) < target_count:
        candidates = [
            (bin_content.workload(flops_config), idx)
            for idx, bin_content in enumerate(bins)
            if len(bin_content.samples) > 1
        ]
        if not candidates:
            break
        _, idx = max(candidates)
        left, right = bins[idx].split_by_workload(flops_config)
        bins[idx] = left
        bins.append(right)


def packed_samples_into_micro_bs(
    samples: list[tuple[int, MicroBatch]],
    max_seq_len: int,
    num_loras: int,
    num_train_workers: int,
    flops_config: Any | None = None,
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
        _expand_bins_by_splitting(bins, target_count, flops_config)

    return [_materialize_bin(bin_content, num_loras) for bin_content in bins]


def _distribute_group(
    group: list[MicroBatch],
    num_train_workers: int,
    flops_config: Any | None,
) -> list[list[MicroBatch]]:
    assert len(group) % num_train_workers == 0, "Number of micro batches is not divisible by number of data ranks"
    if not group:
        return [[] for _ in range(num_train_workers)]

    if len(group) >= num_train_workers:
        weights = [
            calculate_packing_fwd_flops(micro_batch.sequence_lengths, flops_config)
            if flops_config is not None
            else len(micro_batch.input_ids)
            for micro_batch in group
        ]
        partitions = _balanced_partitions(weights, num_train_workers)
        partitions = _improve_partitions_by_swapping(weights, partitions)
        return [[group[i] for i in partition] for partition in partitions]

    return [group] + [[] for _ in range(num_train_workers - 1)]


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
    pad_to_multiple_of: int = 1,
    flops_config: Any | None = None,
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
        flops_config=flops_config,
    )
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]

    # Separate by modality so each step index has uniform modality across all ranks
    mm_batches = [b for b in micro_batches if _is_multimodal_sample(b)]
    text_batches = [b for b in micro_batches if not _is_multimodal_sample(b)]

    # Pad each group independently so its count is divisible by num_train_workers
    mm_batches = _pad_group_for_distribution(mm_batches, num_train_workers)
    text_batches = _pad_group_for_distribution(text_batches, num_train_workers)

    batches_per_gpu: list[list[MicroBatch]] = [[] for _ in range(num_train_workers)]
    for group in (mm_batches, text_batches):
        group_batches_per_gpu = _distribute_group(
            group,
            num_train_workers,
            flops_config,
        )
        for worker_idx, worker_batches in enumerate(group_batches_per_gpu):
            batches_per_gpu[worker_idx].extend(worker_batches)

    return batches_per_gpu
