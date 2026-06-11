import copy

from prime_rl.transport.types import MicroBatch, RoutedExperts, TrainingSample

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


def prepare_sample(training_example: TrainingSample, seq_len: int) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    input_ids = training_example.prompt_ids + training_example.completion_ids
    loss_mask = training_example.prompt_mask + training_example.completion_mask
    inference_logprobs = [0.0] * len(training_example.prompt_ids) + training_example.completion_logprobs
    if training_example.token_advantages is not None:
        advantages = list(training_example.token_advantages)
    else:
        advantage = training_example.advantage if training_example.advantage is not None else 0.0
        advantages = [advantage] * len(input_ids)
    # Component weight streams: keep absent streams None (rl weight 1.0 on the
    # loss mask, no ce/ref_kl component) so the packed batch stays as small as before.
    rl_weights = list(training_example.rl_weights) if training_example.rl_weights is not None else None
    ce_weights = list(training_example.ce_weights) if training_example.ce_weights is not None else None
    ref_kl_weights = list(training_example.ref_kl_weights) if training_example.ref_kl_weights is not None else None
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

    # Ref logprobs already cover the full sequence (prompt + completion),
    # computed via prefill in the orchestrator when the algorithm scores against a reference
    ref_logprobs = training_example.ref_logprobs
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
        if ref_logprobs is not None:
            ref_logprobs = ref_logprobs[:seq_len]
        if rl_weights is not None:
            rl_weights = rl_weights[:seq_len]
        if ce_weights is not None:
            ce_weights = ce_weights[:seq_len]
        if ref_kl_weights is not None:
            ref_kl_weights = ref_kl_weights[:seq_len]
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
    assert len(env_names) == len(input_ids), f"env_names: {len(env_names)}, input_ids: {len(input_ids)}"

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        ref_logprobs=ref_logprobs,
        temperatures=temperatures,
        rewards=rewards,
        routed_experts=routed_experts,
        mm_token_type_ids=mm_token_type_ids,
        env_names=env_names,
        mm_kwargs=training_example.mm_kwargs,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=ref_kl_weights,
    )


def _is_multimodal_sample(sample: MicroBatch) -> bool:
    """Check if a sample contains multimodal data (images)."""
    return sample.mm_kwargs is not None


# Backfill value per component weight stream when a packed sample doesn't
# carry it: absent rl means weight 1.0 on the loss mask, absent ce/ref_kl
# means no component (weight 0.0).
STREAM_FILL = {"rl_weights": 1.0, "ce_weights": 0.0, "ref_kl_weights": 0.0}


def _extend_stream(
    current: list[float] | None, values: list[float] | None, existing_len: int, new_len: int, fill: float
) -> list[float] | None:
    """Extend a per-token weight stream across a packing boundary, backfilling
    whichever side doesn't carry it with the stream's default."""
    if values is not None:
        if current is None:
            current = [fill] * existing_len
        current.extend(values)
    elif current is not None:
        current.extend([fill] * new_len)
    return current


def packed_samples_into_micro_bs(
    samples: list[tuple[int, MicroBatch]], max_seq_len: int, num_loras: int
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

    ## we create bins
    micro_batches: list[MicroBatch] = []

    for idx, sample in samples:
        # Multimodal samples cannot be packed - each becomes its own micro batch
        if _is_multimodal_sample(sample):
            sample.lora_num_tokens = [0] * num_loras
            sample.lora_num_tokens[idx] = len(sample.input_ids)
            micro_batches.append(sample)
            continue

        # Try to find a bin that can fit this sequence (only pack text-only samples)
        for bin_content in micro_batches:
            # Don't pack into multimodal micro batches
            if _is_multimodal_sample(bin_content):
                continue
            # Check if sequence fits in this bin. Loss routing is per token,
            # so samples of different loss types pack together freely.
            if len(bin_content.input_ids) + len(sample.input_ids) <= max_seq_len:
                existing_len = len(bin_content.input_ids)
                bin_content.input_ids.extend(sample.input_ids)
                bin_content.loss_mask.extend(sample.loss_mask)
                bin_content.advantages.extend(sample.advantages)
                sample_len = len(sample.input_ids)
                for stream_name, fill in STREAM_FILL.items():
                    extended = _extend_stream(
                        getattr(bin_content, stream_name), getattr(sample, stream_name), existing_len, sample_len, fill
                    )
                    setattr(bin_content, stream_name, extended)
                if sample.rewards is not None:
                    if bin_content.rewards is None:
                        bin_content.rewards = [float("nan")] * existing_len
                    bin_content.rewards.extend(sample.rewards)
                elif bin_content.rewards is not None:
                    bin_content.rewards.extend([float("nan")] * len(sample.input_ids))
                bin_content.inference_logprobs.extend(sample.inference_logprobs)
                bin_content.temperatures.extend(sample.temperatures)
                if sample.ref_logprobs is not None:
                    if bin_content.ref_logprobs is None:
                        bin_content.ref_logprobs = [0.0] * existing_len
                    bin_content.ref_logprobs.extend(sample.ref_logprobs)
                elif bin_content.ref_logprobs is not None:
                    bin_content.ref_logprobs.extend([0.0] * len(sample.input_ids))
                assert (bin_content.routed_experts is None) == (sample.routed_experts is None)
                if sample.routed_experts is not None:
                    if bin_content.routed_experts is None:
                        bin_content.routed_experts = _copy_routed_experts(sample.routed_experts)
                    else:
                        _append_routed_experts(bin_content, sample)
                if sample.mm_token_type_ids is not None:
                    if bin_content.mm_token_type_ids is None:
                        bin_content.mm_token_type_ids = []
                    bin_content.mm_token_type_ids.extend(sample.mm_token_type_ids)
                bin_content.env_names.extend(sample.env_names)
                bin_content.position_ids.extend(sample.position_ids)
                bin_content.lora_num_tokens[idx] += len(sample.input_ids)
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
    if micro_batch.ref_logprobs is not None:
        micro_batch.ref_logprobs.extend([0.0] * padding_size)
    # Padding tokens are loss-masked, so the rl fill value is irrelevant;
    # ce/ref_kl membership is weight != 0, so their fill must be 0.0.
    for stream_name, fill in STREAM_FILL.items():
        stream = getattr(micro_batch, stream_name)
        if stream is not None:
            stream.extend([fill] * padding_size)
    micro_batch.lora_num_tokens[-1] += (
        padding_size  # We send padding to the last lora so that tokens have ascending lora idx
    )
    if micro_batch.mm_token_type_ids is not None:
        micro_batch.mm_token_type_ids.extend([0] * padding_size)
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
        "rewards",
        "mm_token_type_ids",
    )
    for name in per_token_fields:
        values = getattr(micro_batch, name)
        assert values is None or len(values) == num_tokens, (
            f"{name} misaligned after packing: {len(values)} != {num_tokens} tokens"
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

    micro_batches = packed_samples_into_micro_bs(all_samples, seq_len, num_loras)
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]
    for micro_batch in micro_batches:
        _assert_token_arrays_aligned(micro_batch)

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
