import copy
import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prime_rl.transports.batch.types import EncodedTensor, MicroBatch, RoutedExperts, SamplingMask, TrainingSample

# Backfill value per component weight stream when a packed sample doesn't
# carry it: absent rl means weight 1.0 on the loss mask, absent ce/ref_kl
# means no component (weight 0.0).
STREAM_FILL = {"rl_weights": 1.0, "ce_weights": 0.0, "ref_kl_weights": 0.0}


def _text_config(model_config: Any) -> Any:
    """Unwrap a multimodal model's text config, else return the config unchanged."""
    return getattr(model_config, "text_config", model_config)


def _is_mla(config: Any) -> bool:
    return bool(getattr(config, "multi_latent_attention", False) or hasattr(config, "q_lora_rank"))


def _kv_channels(config: Any) -> int:
    return getattr(config, "kv_channels", getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))


def _qkv_projection_flops_per_token(config: Any) -> int:
    """Linear-in-seqlen FLOPs of the Q/K/V projections (per token)."""
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    kv_channels = _kv_channels(config)
    is_mla = _is_mla(config)
    qk_head_dim = getattr(config, "qk_head_dim", 0)
    qk_pos_emb_head_dim = getattr(config, "qk_pos_emb_head_dim", 0)

    if is_mla and getattr(config, "q_lora_rank", None) is not None:
        q_flops = 2 * config.q_lora_rank * (hidden_size + num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim))
    else:
        q_head_dim = (qk_head_dim + qk_pos_emb_head_dim) if is_mla else kv_channels
        q_flops = 2 * hidden_size * num_attention_heads * q_head_dim

    if is_mla and getattr(config, "kv_lora_rank", None) is not None:
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_flops = 2 * (
            config.kv_lora_rank * (hidden_size + num_attention_heads * (qk_head_dim + v_head_dim))
            + hidden_size * qk_pos_emb_head_dim
        )
    else:
        num_query_groups = getattr(
            config, "num_query_groups", getattr(config, "num_key_value_heads", num_attention_heads)
        )
        kv_flops = 4 * hidden_size * num_query_groups * kv_channels
    return q_flops + kv_flops


def _attention_flops_per_token_squared(config: Any) -> int:
    """Quadratic-in-seqlen FLOPs of the attention scores and values (per token^2)."""
    num_attention_heads = config.num_attention_heads
    kv_channels = _kv_channels(config)
    if _is_mla(config):
        qk_head_dim = getattr(config, "qk_head_dim", 0)
        qk_pos_emb_head_dim = getattr(config, "qk_pos_emb_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", kv_channels)
        return num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim) + num_attention_heads * v_head_dim
    return 2 * num_attention_heads * kv_channels


def _ffn_flops_per_token(hidden_size: int, ffn_hidden_size: int) -> int:
    return 6 * hidden_size * ffn_hidden_size


def _dense_ffn_hidden_size(config: Any) -> int:
    return getattr(
        config, "ffn_hidden_size", getattr(config, "intermediate_size", getattr(config, "moe_intermediate_size", 0))
    )


def _moe_ffn_hidden_size(config: Any) -> int:
    """Effective FFN width of an MoE layer: routed experts (topk) plus shared experts."""
    dense_ffn = _dense_ffn_hidden_size(config)
    routed_topk = getattr(config, "moe_router_topk", getattr(config, "num_experts_per_tok", 1))
    moe_ffn = getattr(config, "moe_ffn_hidden_size", getattr(config, "moe_intermediate_size", dense_ffn))
    return moe_ffn * routed_topk + (getattr(config, "moe_shared_expert_intermediate_size", None) or 0)


def _count_dense_and_moe_layers(config: Any) -> tuple[int, int]:
    """Split the model's layers into (dense, moe) counts."""
    num_experts = getattr(config, "num_experts", getattr(config, "n_routed_experts", None))
    if num_experts is None:
        return config.num_hidden_layers, 0

    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if isinstance(moe_layer_freq, list):
        num_dense = sum(1 for freq in moe_layer_freq if freq == 0)
        num_moe = sum(1 for freq in moe_layer_freq if freq > 0)
    elif isinstance(moe_layer_freq, int):
        num_dense = sum(1 for i in range(config.num_hidden_layers) if i % moe_layer_freq != 0)
        num_moe = config.num_hidden_layers - num_dense
    elif getattr(config, "first_k_dense_replace", None) is not None:
        num_dense = config.first_k_dense_replace
        num_moe = config.num_hidden_layers - num_dense
    else:
        num_dense = 0
        num_moe = config.num_hidden_layers
    return num_dense, num_moe


def _packing_cost_coeffs(config: Any) -> tuple[int, int]:
    """Return the (linear, quadratic) coefficients of the per-sequence forward FLOPs."""
    hidden_size = config.hidden_size
    num_dense_layers, num_moe_layers = _count_dense_and_moe_layers(config)
    qkv_per_token = _qkv_projection_flops_per_token(config)
    attn_per_token_squared = _attention_flops_per_token_squared(config)

    def layer_linear(ffn_hidden_size: int) -> int:
        return qkv_per_token + 2 * hidden_size * hidden_size + _ffn_flops_per_token(hidden_size, ffn_hidden_size)

    linear = (
        num_dense_layers * layer_linear(_dense_ffn_hidden_size(config))
        + num_moe_layers * layer_linear(_moe_ffn_hidden_size(config))
        + 2 * hidden_size * config.vocab_size
    )
    quadratic = (num_dense_layers + num_moe_layers) * attn_per_token_squared
    return linear, quadratic


def build_bin_cost(model_config: Any | None) -> Callable[[Sequence[int]], int]:
    """Build a closure scoring a packed bin by estimated forward compute.

    With ``model_config=None`` the linear/quadratic coefficients are ``(1, 0)``, so
    the cost reduces to the token count and balancing falls back to sequence length.
    """
    if model_config is None:
        linear, quadratic = 1, 0
    else:
        linear, quadratic = _packing_cost_coeffs(_text_config(model_config))

    def bin_cost(seqlens: Sequence[int]) -> int:
        return linear * sum(seqlens) + quadratic * sum(n * n for n in seqlens)

    return bin_cost


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
    partitions = [list(partition) for partition in partitions]
    loads = _partition_loads(weights, partitions)

    while True:
        best_swap = None
        best_score = (max(loads), max(loads) - min(loads))
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


def balanced_partition(weights: Sequence[int], num_partitions: int) -> list[list[int]]:
    """Partition item indices into ``num_partitions`` groups of near-equal total weight.

    Requires ``len(weights)`` to be a positive multiple of ``num_partitions``.
    """
    partitions = _karmarkar_karp(weights, num_partitions)
    return _refine_by_swapping(weights, partitions)


def _copy_routed_experts(routed_experts: RoutedExperts) -> RoutedExperts:
    return RoutedExperts(
        data=routed_experts.data,
        shape=list(routed_experts.shape),
        dtype=routed_experts.dtype,
    )


def _routed_experts_row_size(routed_experts: RoutedExperts) -> int:
    return routed_experts.shape[1] * routed_experts.shape[2] * np.dtype(routed_experts.dtype).itemsize


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


_SAMPLING_MASK_ITEMSIZE = np.dtype(np.int32).itemsize


def _empty_sampling_mask(num_tokens: int) -> SamplingMask:
    return SamplingMask(ids=b"", counts=b"\0" * (num_tokens * _SAMPLING_MASK_ITEMSIZE))


def _slice_sampling_mask(sampling_mask: SamplingMask, seq_len: int) -> SamplingMask:
    counts = np.frombuffer(sampling_mask.counts, dtype=np.int32)[:seq_len]
    return SamplingMask(
        ids=sampling_mask.ids[: int(counts.sum()) * _SAMPLING_MASK_ITEMSIZE],
        counts=counts.tobytes(),
    )


def _pad_sampling_mask(micro_batch: MicroBatch, padding_size: int) -> None:
    """Add zero-count mask rows for sequence-padding tokens.

    Padding adds token positions but no eligible token ids, so only `counts` grows.
    """
    sampling_mask = micro_batch.sampling_mask
    assert sampling_mask is not None
    sampling_mask.counts += b"\0" * (padding_size * _SAMPLING_MASK_ITEMSIZE)


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


def multimodal_sample_error(sample: TrainingSample) -> str | None:
    mm_token_type_ids = sample.mm_token_type_ids
    if mm_token_type_ids is not None and len(mm_token_type_ids) != len(sample.token_ids):
        return (
            "mm_token_type_ids length must match token_ids length "
            f"({len(mm_token_type_ids)} != {len(sample.token_ids)})"
        )
    if sample.mm_kwargs is not None and "image_grid_thw" in sample.mm_kwargs and mm_token_type_ids is None:
        return "image_grid_thw multimodal samples require mm_token_type_ids"
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
    # No copy needed: SamplingMask holds immutable bytes, and _pad_sampling_mask only
    # ever mutates _materialize_bin's own accumulator.
    sampling_mask = training_example.sampling_mask

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
        if sampling_mask is not None:
            sampling_mask = _slice_sampling_mask(sampling_mask, cut)
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

    if sampling_mask is not None:
        mask_counts = np.frombuffer(sampling_mask.counts, dtype=np.int32)
        assert len(mask_counts) == len(input_ids), (
            f"sampling_mask counts: {len(mask_counts)}, input_ids: {len(input_ids)}"
        )
        assert len(sampling_mask.ids) == int(mask_counts.sum()) * _SAMPLING_MASK_ITEMSIZE

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
        sampling_mask=sampling_mask,
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
    samples: list[MicroBatch]
    length: int

    @classmethod
    def from_sample(cls, sample: MicroBatch) -> "_MicroBatchBin":
        return cls(samples=[sample], length=len(sample.input_ids))

    @property
    def first_sample(self) -> MicroBatch:
        return self.samples[0]

    @property
    def first_multimodal_sample(self) -> MicroBatch | None:
        for sample in self.samples:
            if _is_multimodal_sample(sample):
                return sample
        return None

    def can_add(self, sample: MicroBatch, max_seq_len: int) -> bool:
        # Loss routing is per token (component weight streams), so samples of
        # different loss types pack together freely. Multimodal packing is still
        # constrained by modality sidecars, length, and routed experts.
        first_sample = self.first_sample
        if self.length + len(sample.input_ids) > max_seq_len:
            return False
        if (first_sample.routed_experts is None) != (sample.routed_experts is None):
            return False

        sample_is_mm = _is_multimodal_sample(sample)
        existing_mm_sample = self.first_multimodal_sample
        if existing_mm_sample is not None and sample_is_mm:
            dst = existing_mm_sample.mm_kwargs
            src = sample.mm_kwargs
            assert dst is not None and src is not None, "multimodal samples must carry mm_kwargs"
            return set(dst) == set(src) and all(
                dst[key].dtype == src[key].dtype
                and len(dst[key].shape) > 0
                and len(dst[key].shape) == len(src[key].shape)
                and dst[key].shape[1:] == src[key].shape[1:]
                for key in dst
            )
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
    # Sampling masks are per-token optional (unlike routed_experts): samples
    # without them get zero-count backfill instead of constraining packing.
    has_sampling_mask = any(sample.sampling_mask is not None for sample in bin_content.samples)

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
    sampling_mask: SamplingMask | None = SamplingMask(ids=b"", counts=b"") if has_sampling_mask else None

    for sample in bin_content.samples:
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
                for key in mm_kwargs:
                    mm_kwargs[key].data += sample.mm_kwargs[key].data
                    mm_kwargs[key].shape[0] += sample.mm_kwargs[key].shape[0]
        seq_lens.extend(sample.seq_lens)
        if sampling_mask is not None:
            sample_mask = sample.sampling_mask if sample.sampling_mask is not None else _empty_sampling_mask(sample_len)
            sampling_mask.ids += sample_mask.ids
            sampling_mask.counts += sample_mask.counts

    sequence_lengths = [len(sample.input_ids) for sample in bin_content.samples]
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
        routed_experts=routed_experts,
        sampling_mask=sampling_mask,
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
            if len(bin_content.samples) > 1 and bin_content.first_multimodal_sample is None
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

    Multimodal samples pack with text spans and with compatible eager
    ``mm_kwargs`` samples. Packed batches preserve sample boundaries in
    ``seq_lens``.
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
    micro_batch.sequence_lengths[-1] += padding_size
    micro_batch.seq_lens[-1] += padding_size
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
    if micro_batch.mm_token_type_ids is not None:
        micro_batch.mm_token_type_ids.extend([0] * padding_size)
    if micro_batch.routed_experts is not None:
        _pad_routed_experts(micro_batch, padding_size)
    if micro_batch.sampling_mask is not None:
        _pad_sampling_mask(micro_batch, padding_size)
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
    assert sum(micro_batch.seq_lens) == num_tokens, f"seq_lens sum {sum(micro_batch.seq_lens)} != {num_tokens} tokens"
    if micro_batch.routed_experts is not None:
        assert micro_batch.routed_experts.shape[0] == num_tokens, (
            f"routed_experts misaligned after packing: {micro_batch.routed_experts.shape[0]} != {num_tokens} tokens"
        )
    if micro_batch.sampling_mask is not None:
        mask_counts = np.frombuffer(micro_batch.sampling_mask.counts, dtype=np.int32)
        assert len(mask_counts) == num_tokens, (
            f"sampling_mask misaligned after packing: {len(mask_counts)} != {num_tokens} tokens"
        )
        assert len(micro_batch.sampling_mask.ids) == int(mask_counts.sum()) * _SAMPLING_MASK_ITEMSIZE, (
            f"sampling_mask ids/counts inconsistent after packing: "
            f"{len(micro_batch.sampling_mask.ids)} bytes != {int(mask_counts.sum())} ids"
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
    # Fully loss-masked, so replaying sampling masks would be pure wasted work.
    dummy.sampling_mask = None
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
