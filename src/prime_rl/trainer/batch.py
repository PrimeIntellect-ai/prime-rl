import copy

from prime_rl.transport.types import MicroBatch, TrainingSample

# Qwen3-VL image placeholder token ID
_IMAGE_PAD_TOKEN_ID = 151655


def _compute_keep_mask_for_multimodal(
    input_ids: list[int],
    image_grid_thw: list[list[int]],
    merge_size: int = 2,
) -> tuple[list[bool], int, list[list[int]]]:
    """Compute a per-position keep mask that drops orphaned image_pad tokens.

    After seq_len truncation, the last image may have only partial placeholder tokens.
    This computes which image_pad tokens belong to complete images (keep=True) and which
    are orphaned partials (keep=False). Non-image tokens are always kept.

    Returns (keep_mask, kept_image_count, kept_grids).
    """
    num_image_tokens = sum(1 for t in input_ids if t == _IMAGE_PAD_TOKEN_ID)
    expected_tokens = sum(g[0] * g[1] * g[2] // (merge_size**2) for g in image_grid_thw)

    if num_image_tokens == expected_tokens:
        return [True] * len(input_ids), len(image_grid_thw), image_grid_thw

    # Figure out how many complete images fit
    kept_grids = []
    valid_image_tokens = 0
    for grid in image_grid_thw:
        img_tokens = grid[0] * grid[1] * grid[2] // (merge_size**2)
        if valid_image_tokens + img_tokens <= num_image_tokens:
            kept_grids.append(grid)
            valid_image_tokens += img_tokens
        else:
            break

    # Build keep mask: keep the first `valid_image_tokens` image_pad tokens, drop the rest
    keep_mask = []
    seen_image_tokens = 0
    for t in input_ids:
        if t == _IMAGE_PAD_TOKEN_ID:
            keep_mask.append(seen_image_tokens < valid_image_tokens)
            seen_image_tokens += 1
        else:
            keep_mask.append(True)

    return keep_mask, len(kept_grids), kept_grids


def _trim_multimodal_to_match(
    input_ids: list[int],
    pixel_values: bytes | None,
    pixel_values_shape: list[int] | None,
    image_grid_thw: list[list[int]] | None,
    merge_size: int = 2,
) -> tuple[list[int], list[bool] | None, bytes | None, list[int] | None, list[list[int]] | None]:
    """Ensure image tokens and pixel_values are consistent after seq_len truncation.

    Returns (input_ids, keep_mask, pixel_values, pixel_values_shape, image_grid_thw).
    keep_mask is None when no filtering was needed, otherwise a bool list aligned with
    the original input_ids indicating which positions to keep. Callers must apply
    keep_mask to all parallel arrays (loss_mask, logprobs, etc.).
    """
    if pixel_values is None or image_grid_thw is None:
        return input_ids, None, pixel_values, pixel_values_shape, image_grid_thw

    num_image_tokens = sum(1 for t in input_ids if t == _IMAGE_PAD_TOKEN_ID)
    expected_tokens = sum(g[0] * g[1] * g[2] // (merge_size**2) for g in image_grid_thw)

    if num_image_tokens == expected_tokens:
        return input_ids, None, pixel_values, pixel_values_shape, image_grid_thw

    keep_mask, _, kept_grids = _compute_keep_mask_for_multimodal(input_ids, image_grid_thw, merge_size)
    input_ids = [t for t, k in zip(input_ids, keep_mask) if k]

    if not kept_grids:
        return input_ids, keep_mask, None, None, None

    patch_dim = pixel_values_shape[1] if pixel_values_shape else 0
    kept_patches = sum(g[0] * g[1] * g[2] for g in kept_grids)
    bytes_per_patch = 4 * patch_dim  # float32
    kept_bytes = kept_patches * bytes_per_patch
    return input_ids, keep_mask, pixel_values[:kept_bytes], [kept_patches, patch_dim], kept_grids


def prepare_sample(training_example: TrainingSample, seq_len: int) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    input_ids = training_example.prompt_ids + training_example.completion_ids
    loss_mask = training_example.prompt_mask + training_example.completion_mask
    inference_logprobs = [0.0] * len(training_example.prompt_ids) + training_example.completion_logprobs
    advantages = [training_example.advantage] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    # Per-token temperatures: prompt tokens use first completion temp (masked out anyway)
    # Default to 1.0 if completion is empty (e.g., model generated only tool calls with no text)
    prompt_temp = training_example.completion_temperatures[0] if training_example.completion_temperatures else 1.0
    temperatures = [prompt_temp] * len(training_example.prompt_ids) + training_example.completion_temperatures

    # Teacher logprobs already cover the full sequence (prompt + completion),
    # computed via prefill in the orchestrator when a teacher model is configured
    teacher_logprobs = training_example.teacher_logprobs
    routed_experts = training_example.routed_experts

    pixel_values = training_example.pixel_values
    pixel_values_shape = training_example.pixel_values_shape
    image_grid_thw = training_example.image_grid_thw

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
        inference_logprobs = inference_logprobs[:seq_len]
        position_ids = position_ids[:seq_len]
        advantages = advantages[:seq_len]
        temperatures = temperatures[:seq_len]
        if teacher_logprobs is not None:
            teacher_logprobs = teacher_logprobs[:seq_len]
        if routed_experts is not None:
            routed_experts = routed_experts[:seq_len]

    # VLM: pixel_values/image_grid_thw are computed from the full conversation images,
    # but input_ids may have fewer image_pad tokens due to vLLM's prompt truncation
    # (left-truncation to max_model_len) or seq_len truncation above. Drop trailing
    # images whose placeholder tokens were removed so features match tokens.
    input_ids, keep_mask, pixel_values, pixel_values_shape, image_grid_thw = _trim_multimodal_to_match(
        input_ids, pixel_values, pixel_values_shape, image_grid_thw
    )
    if keep_mask is not None:
        loss_mask = [v for v, k in zip(loss_mask, keep_mask) if k]
        inference_logprobs = [v for v, k in zip(inference_logprobs, keep_mask) if k]
        position_ids = [v for v, k in zip(position_ids, keep_mask) if k]
        advantages = [v for v, k in zip(advantages, keep_mask) if k]
        temperatures = [v for v, k in zip(temperatures, keep_mask) if k]
        if teacher_logprobs is not None:
            teacher_logprobs = [v for v, k in zip(teacher_logprobs, keep_mask) if k]
        if routed_experts is not None:
            routed_experts = [v for v, k in zip(routed_experts, keep_mask) if k]

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
    if teacher_logprobs is not None:
        assert len(teacher_logprobs) == len(input_ids), f"teacher_logprobs: {len(teacher_logprobs)}"

    if routed_experts is not None:
        assert len(routed_experts) == len(input_ids), (
            f"routed_experts: {len(routed_experts)}, input_ids: {len(input_ids)}"
        )

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        temperatures=temperatures,
        routed_experts=routed_experts,
        pixel_values=pixel_values,
        pixel_values_shape=pixel_values_shape,
        image_grid_thw=image_grid_thw,
    )


def _is_multimodal_sample(sample: MicroBatch) -> bool:
    """Check if a sample contains multimodal data (images)."""
    return sample.pixel_values is not None


def packed_samples_into_micro_bs(
    samples: list[tuple[int, MicroBatch]], max_seq_len: int, num_loras: int
) -> list[MicroBatch]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    With per-token temperatures, samples can be packed together regardless of their temperature values.

    NOTE: Multimodal samples (with pixel_values) are NOT packed together as they have variable-sized
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
            # Check if sequence fits in this bin
            if len(bin_content.input_ids) + len(sample.input_ids) <= max_seq_len:
                bin_content.input_ids.extend(sample.input_ids)
                bin_content.loss_mask.extend(sample.loss_mask)
                bin_content.advantages.extend(sample.advantages)
                bin_content.inference_logprobs.extend(sample.inference_logprobs)
                bin_content.temperatures.extend(sample.temperatures)
                if sample.teacher_logprobs is not None:
                    if bin_content.teacher_logprobs is None:
                        bin_content.teacher_logprobs = []
                    bin_content.teacher_logprobs.extend(sample.teacher_logprobs)
                if sample.routed_experts is not None:
                    if bin_content.routed_experts is None:
                        bin_content.routed_experts = []
                    bin_content.routed_experts.extend(sample.routed_experts)
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

    if not (pad_to_multiple_of > 1 and padding_size > 0):
        return micro_batch

    micro_batch.input_ids.extend([1] * padding_size)
    micro_batch.advantages.extend([0.0] * padding_size)
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
