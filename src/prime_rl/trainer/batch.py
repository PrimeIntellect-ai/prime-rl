import copy
import math

from prime_rl.transport.types import EncodedTensor, MicroBatch, MMRefs, RoutedExperts, TrainingSample

ENCODED_TENSOR_DTYPE_ITEMSIZE = {
    "bool": 1,
    "bool_": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
}

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


def _append_routed_experts(dst: MicroBatch, src: MicroBatch) -> None:
    dst_routed = dst.routed_experts
    src_routed = src.routed_experts
    assert dst_routed is not None
    assert src_routed is not None
    assert dst_routed.dtype == src_routed.dtype
    assert dst_routed.shape[1:] == src_routed.shape[1:]
    dst_routed.data += src_routed.data
    dst_routed.shape[0] += src_routed.shape[0]


def _pad_routed_experts(micro_batch: MicroBatch, padding_size: int) -> None:
    routed_experts = micro_batch.routed_experts
    assert routed_experts is not None
    row_size = _routed_experts_row_size(routed_experts)
    routed_experts.data += b"\0" * (padding_size * row_size)
    routed_experts.shape[0] += padding_size


def _encoded_tensor_dtype_itemsize(encoded: EncodedTensor) -> int:
    dtype = encoded.dtype.replace("numpy.", "").replace("torch.", "")
    if dtype not in ENCODED_TENSOR_DTYPE_ITEMSIZE:
        raise ValueError(f"Unsupported EncodedTensor dtype for multimodal packing: {encoded.dtype}")
    return ENCODED_TENSOR_DTYPE_ITEMSIZE[dtype]


def _validate_encoded_tensor_payload(encoded: EncodedTensor) -> None:
    expected_nbytes = math.prod(encoded.shape) * _encoded_tensor_dtype_itemsize(encoded)
    if len(encoded.data) != expected_nbytes:
        raise ValueError(
            "EncodedTensor byte length does not match dtype and shape: "
            f"dtype={encoded.dtype}, shape={encoded.shape}, "
            f"data_nbytes={len(encoded.data)}, expected_nbytes={expected_nbytes}"
        )


def _append_encoded_tensor(dst: EncodedTensor, src: EncodedTensor, key: str) -> None:
    _validate_encoded_tensor_payload(dst)
    _validate_encoded_tensor_payload(src)
    if dst.dtype != src.dtype:
        raise ValueError(f"Cannot pack mm_kwargs[{key!r}] with different dtypes: {dst.dtype} vs {src.dtype}")
    if len(dst.shape) == 0 or len(dst.shape) != len(src.shape) or dst.shape[1:] != src.shape[1:]:
        raise ValueError(f"Cannot pack mm_kwargs[{key!r}] with incompatible shapes: {dst.shape} vs {src.shape}")
    dst.data += src.data
    dst.shape[0] += src.shape[0]


def _append_mm_kwargs(dst: dict[str, EncodedTensor], src: dict[str, EncodedTensor]) -> None:
    if set(dst) != set(src):
        raise ValueError(f"Cannot pack mm_kwargs with different keys: {sorted(dst)} vs {sorted(src)}")
    for key in dst:
        _append_encoded_tensor(dst[key], src[key], key)


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


def _append_mm_ref_descriptor_list(dst_map: dict, src_map: dict, field: str) -> None:
    if set(dst_map) != set(src_map):
        raise ValueError(f"Cannot pack mm_refs descriptor {field} with different modalities")
    for modality, src_items in src_map.items():
        dst_items = dst_map[modality]
        if not isinstance(dst_items, list) or not isinstance(src_items, list):
            raise ValueError(f"mm_refs descriptor {field}[{modality!r}] must be a list to pack")
        dst_items.extend(copy.deepcopy(src_items))


def _append_mm_refs(dst: MMRefs, src: MMRefs) -> None:
    dst_items = dst.descriptor.get("mm_items") or {}
    src_items = src.descriptor.get("mm_items") or {}
    dst_hashes = dst.descriptor.get("mm_hashes") or {}
    src_hashes = src.descriptor.get("mm_hashes") or {}

    _append_mm_ref_descriptor_list(dst_items, src_items, "mm_items")
    _append_mm_ref_descriptor_list(dst_hashes, src_hashes, "mm_hashes")
    dst.descriptor["mm_items"] = dst_items
    dst.descriptor["mm_hashes"] = dst_hashes
    dst.uris.extend(src.uris)


def _mm_sidecar_kind(sample: MicroBatch) -> str | None:
    if sample.mm_kwargs is not None and sample.mm_refs is not None:
        raise ValueError("A multimodal sample cannot carry both mm_kwargs and mm_refs")
    if sample.mm_refs is not None:
        return "refs"
    if sample.mm_kwargs is not None:
        return "kwargs"
    return None


def _single_lora_idx(sample: MicroBatch) -> int | None:
    if sample.lora_num_tokens is None:
        return None
    active = [idx for idx, tokens in enumerate(sample.lora_num_tokens) if tokens > 0]
    return active[0] if len(active) == 1 else None


def _has_video_tokens(sample: MicroBatch) -> bool:
    return sample.mm_token_type_ids is not None and 2 in sample.mm_token_type_ids


def _can_pack_sample(
    bin_content: MicroBatch,
    sample: MicroBatch,
    *,
    idx: int,
    max_seq_len: int,
    pack_multimodal: bool,
) -> bool:
    if len(bin_content.input_ids) + len(sample.input_ids) > max_seq_len:
        return False
    if bin_content.training_mode != sample.training_mode:
        return False

    bin_mm_kind = _mm_sidecar_kind(bin_content)
    sample_mm_kind = _mm_sidecar_kind(sample)
    if bin_mm_kind is None and sample_mm_kind is None:
        return True
    if not pack_multimodal or bin_mm_kind != sample_mm_kind:
        return False
    if _has_video_tokens(bin_content) or _has_video_tokens(sample):
        return False
    if bin_mm_kind == "kwargs" and not _can_pack_mm_kwargs(bin_content.mm_kwargs, sample.mm_kwargs):
        return False
    # Multimodal samples only pack with the same run: a multi-run microbatch would
    # break the MoE LoRA path (one adapter per microbatch). prepare_batch may be
    # called with multi-run input, so this guard is load-bearing, not redundant.
    return _single_lora_idx(bin_content) == idx


def _append_micro_batch(bin_content: MicroBatch, sample: MicroBatch, idx: int) -> None:
    existing_len = len(bin_content.input_ids)
    sample_len = len(sample.input_ids)

    bin_content.input_ids.extend(sample.input_ids)
    bin_content.loss_mask.extend(sample.loss_mask)
    bin_content.advantages.extend(sample.advantages)
    if sample.rewards is not None:
        if bin_content.rewards is None:
            bin_content.rewards = [float("nan")] * existing_len
        bin_content.rewards.extend(sample.rewards)
    elif bin_content.rewards is not None:
        bin_content.rewards.extend([float("nan")] * sample_len)
    bin_content.inference_logprobs.extend(sample.inference_logprobs)
    bin_content.temperatures.extend(sample.temperatures)
    if bin_content.teacher_logprobs is not None or sample.teacher_logprobs is not None:
        if bin_content.teacher_logprobs is None:
            bin_content.teacher_logprobs = [0.0] * existing_len
        bin_content.teacher_logprobs.extend(sample.teacher_logprobs or [0.0] * sample_len)

    assert (bin_content.routed_experts is None) == (sample.routed_experts is None)
    if sample.routed_experts is not None:
        if bin_content.routed_experts is None:
            bin_content.routed_experts = _copy_routed_experts(sample.routed_experts)
        else:
            _append_routed_experts(bin_content, sample)

    if bin_content.mm_token_type_ids is not None or sample.mm_token_type_ids is not None:
        if bin_content.mm_token_type_ids is None:
            bin_content.mm_token_type_ids = [0] * existing_len
        bin_content.mm_token_type_ids.extend(sample.mm_token_type_ids or [0] * sample_len)

    bin_content.env_names.extend(sample.env_names)
    bin_content.position_ids.extend(sample.position_ids)
    _extend_seq_lens(bin_content, sample, existing_len)
    assert bin_content.lora_num_tokens is not None
    bin_content.lora_num_tokens[idx] += sample_len

    # Concatenate the multimodal sidecar. _can_pack_sample already guaranteed the
    # bin and sample share the same kind, so dispatch on whichever is present.
    if bin_content.mm_refs is not None:
        _append_mm_refs(bin_content.mm_refs, sample.mm_refs)
    elif bin_content.mm_kwargs is not None:
        _append_mm_kwargs(bin_content.mm_kwargs, sample.mm_kwargs)


def _extend_seq_lens(bin_content: MicroBatch, sample: MicroBatch, existing_len: int) -> None:
    if bin_content.seq_lens is None and sample.seq_lens is None:
        return
    if bin_content.seq_lens is None:
        bin_content.seq_lens = [existing_len]
    bin_content.seq_lens.extend(sample.seq_lens if sample.seq_lens is not None else [len(sample.input_ids)])


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

    if (training_example.mm_kwargs is not None or training_example.mm_refs is not None) and len(input_ids) > seq_len:
        raise ValueError(
            "Cannot truncate multimodal training sample without also truncating its multimodal sidecars: "
            f"sample_len={len(input_ids)}, seq_len={seq_len}"
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
    if training_example.mm_kwargs is not None and "image_grid_thw" in training_example.mm_kwargs:
        if mm_token_type_ids is None:
            raise ValueError("image_grid_thw multimodal samples require mm_token_type_ids")
    assert len(env_names) == len(input_ids), f"env_names: {len(env_names)}, input_ids: {len(input_ids)}"

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        temperatures=temperatures,
        rewards=rewards,
        routed_experts=routed_experts,
        mm_token_type_ids=mm_token_type_ids,
        env_names=env_names,
        mm_kwargs=copy.deepcopy(training_example.mm_kwargs),
        mm_refs=copy.deepcopy(training_example.mm_refs),
        training_mode=training_example.training_mode,
        seq_lens=[len(input_ids)]
        if training_example.mm_kwargs is not None or training_example.mm_refs is not None
        else None,
    )


def _is_multimodal_sample(sample: MicroBatch) -> bool:
    """Check if a sample contains multimodal data (images). A deferred sample
    carries ``mm_refs`` and no ``mm_kwargs``; both count as multimodal so it is
    not mis-packed as text (which would break the FSDP per-step modality
    invariant)."""
    return sample.mm_kwargs is not None or sample.mm_refs is not None


def packed_samples_into_micro_bs(
    samples: list[tuple[int, MicroBatch]], max_seq_len: int, num_loras: int, pack_multimodal: bool = False
) -> list[MicroBatch]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    With per-token temperatures, samples can be packed together regardless of their temperature values.

    Multimodal samples are only packed when ``pack_multimodal`` is true. They
    pack with other multimodal samples of the same sidecar representation
    (deferred ``mm_refs`` or eager ``mm_kwargs``), never with text-only samples.
    Packed multimodal batches preserve sample boundaries in ``seq_lens``.
    """
    # Sort by (lora_idx, -length) for packing efficiency
    samples.sort(key=lambda x: (x[0], -len(x[1].input_ids)))

    ## we create bins
    micro_batches: list[MicroBatch] = []

    for idx, sample in samples:
        # Unsupported multimodal samples remain standalone. Supported multimodal
        # samples use the same token-side first-fit packing as text, with strict
        # sidecar concatenation in the same sample order.
        if _is_multimodal_sample(sample) and not pack_multimodal:
            sample.lora_num_tokens = [0] * num_loras
            sample.lora_num_tokens[idx] = len(sample.input_ids)
            micro_batches.append(sample)
            continue

        # Try to find a bin that can fit this sequence.
        for bin_content in micro_batches:
            if _can_pack_sample(
                bin_content,
                sample,
                idx=idx,
                max_seq_len=max_seq_len,
                pack_multimodal=pack_multimodal,
            ):
                _append_micro_batch(bin_content, sample, idx)
                break
        else:
            sample.lora_num_tokens = [0] * num_loras
            sample.lora_num_tokens[idx] = len(sample.input_ids)
            micro_batches.append(sample)

    return micro_batches


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
    micro_batch.inference_logprobs.extend([0.0] * padding_size)
    # Use temperature 1.0 for padding tokens (doesn't matter since loss_mask is False)
    micro_batch.temperatures.extend([1.0] * padding_size)
    if micro_batch.teacher_logprobs is not None:
        micro_batch.teacher_logprobs.extend([0.0] * padding_size)
    micro_batch.lora_num_tokens[-1] += (
        padding_size  # We send padding to the last lora so that tokens have ascending lora idx
    )
    if micro_batch.mm_token_type_ids is not None:
        micro_batch.mm_token_type_ids.extend([0] * padding_size)
    if micro_batch.routed_experts is not None:
        _pad_routed_experts(micro_batch, padding_size)
    micro_batch.env_names.extend([""] * padding_size)
    if micro_batch.seq_lens is not None:
        micro_batch.seq_lens.append(padding_size)

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
    pack_multimodal: bool = False,
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

    micro_batches = packed_samples_into_micro_bs(all_samples, seq_len, num_loras, pack_multimodal=pack_multimodal)
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]

    # Separate by modality so each step index has uniform modality across all ranks
    mm_batches = [b for b in micro_batches if _is_multimodal_sample(b)]
    text_batches = [b for b in micro_batches if not _is_multimodal_sample(b)]

    # Pad each group independently so its count is divisible by num_train_workers
    mm_batches = _pad_group_for_distribution(mm_batches, num_train_workers)
    text_batches = _pad_group_for_distribution(text_batches, num_train_workers)

    # Combine: all multimodal first, then all text-only. Since each group's length is
    # divisible by num_train_workers, the modality boundary aligns with distribution rows.
    ordered = mm_batches + text_batches

    assert len(ordered) % num_train_workers == 0, "Number of micro batches is not divisible by number of data ranks"

    # Distribute in strided order so each step index has the same modality across ranks
    batches_per_gpu: list[list[MicroBatch]] = [[] for _ in range(num_train_workers)]
    for i, batch in enumerate(ordered):
        batches_per_gpu[i % num_train_workers].append(batch)

    return batches_per_gpu
