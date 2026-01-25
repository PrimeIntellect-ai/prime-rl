import copy

from prime_rl.transport.types import MicroBatch, TrainingSample


def _is_nested_expert_list(data: list) -> bool:
    return bool(data) and isinstance(data[0], list) and bool(data[0]) and isinstance(data[0][0], list)


def _normalize_expert_data(
    data: list | None, num_tokens: int
) -> tuple[list[list[int]] | list[list[list[int]]] | None, str | None]:
    if data is None:
        return None, None
    if _is_nested_expert_list(data):
        if len(data) == num_tokens:
            num_layers = len(data[0])
            layer_major: list[list[list[int]]] = [[] for _ in range(num_layers)]
            for token_entry in data:
                if len(token_entry) != num_layers:
                    raise ValueError("Inconsistent number of layers in routed expert metadata.")
                for layer_idx, layer_entry in enumerate(token_entry):
                    layer_major[layer_idx].append(layer_entry)
            return layer_major, "layer"
        return data, "layer"
    return data, "token"


def _merge_expert_metadata(
    prompt_data: list | None,
    completion_data: list | None,
    prompt_len: int,
    completion_len: int,
) -> list | None:
    if prompt_data is None and completion_data is None:
        return None
    if prompt_data is None or completion_data is None:
        return None

    prompt_norm, prompt_layout = _normalize_expert_data(prompt_data, prompt_len)
    completion_norm, completion_layout = _normalize_expert_data(completion_data, completion_len)

    if prompt_layout != completion_layout:
        return None

    if prompt_layout == "token":
        assert isinstance(prompt_norm, list) and isinstance(completion_norm, list)
        return prompt_norm + completion_norm

    assert isinstance(prompt_norm, list) and isinstance(completion_norm, list)
    if len(prompt_norm) != len(completion_norm):
        return None
    merged: list[list[list[int]]] = []
    for prompt_layer, completion_layer in zip(prompt_norm, completion_norm):
        merged.append(prompt_layer + completion_layer)
    return merged


def _extend_expert_metadata(
    target: list | None,
    source: list | None,
) -> list | None:
    if source is None:
        return target
    if target is None:
        return copy.deepcopy(source)
    if _is_nested_expert_list(target):
        if not _is_nested_expert_list(source):
            raise ValueError("Routed expert metadata layout mismatch while packing.")
        if len(target) != len(source):
            raise ValueError("Routed expert metadata layer count mismatch while packing.")
        for layer_idx, layer_entries in enumerate(source):
            target[layer_idx].extend(layer_entries)
        return target
    if _is_nested_expert_list(source):
        raise ValueError("Routed expert metadata layout mismatch while packing.")
    target.extend(source)
    return target


def _pad_expert_metadata(
    data: list | None,
    padding_size: int,
    pad_value: int | float,
) -> None:
    if data is None or padding_size <= 0:
        return
    if _is_nested_expert_list(data):
        for layer_entries in data:
            top_k = len(layer_entries[0]) if layer_entries else 0
            layer_entries.extend([[pad_value] * top_k for _ in range(padding_size)])
    else:
        top_k = len(data[0]) if data else 0
        data.extend([[pad_value] * top_k for _ in range(padding_size)])


def prepare_sample(
    training_example: TrainingSample,
    seq_len: int,
    temperature: float,
) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    # Prepare input_ids, loss_mask, position_ids, inference_logprobs, and advantages
    input_ids = training_example.prompt_ids + training_example.completion_ids
    loss_mask = training_example.prompt_mask + training_example.completion_mask
    # Inference logprobs only cover completion tokens, so prepend zeros for prompt tokens
    inference_logprobs = [0.0] * len(training_example.prompt_ids) + training_example.completion_logprobs
    advantages = [training_example.advantage] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    # Teacher logprobs already cover the full sequence (prompt + completion),
    # computed via prefill in the orchestrator when a teacher model is configured
    teacher_logprobs = training_example.teacher_logprobs
    routed_expert_indices = _merge_expert_metadata(
        training_example.prompt_expert_indices,
        training_example.completion_expert_indices,
        len(training_example.prompt_ids),
        len(training_example.completion_ids),
    )
    routed_expert_probs = _merge_expert_metadata(
        training_example.prompt_expert_probs,
        training_example.completion_expert_probs,
        len(training_example.prompt_ids),
        len(training_example.completion_ids),
    )

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
        inference_logprobs = inference_logprobs[:seq_len]
        position_ids = position_ids[:seq_len]
        advantages = advantages[:seq_len]
        if teacher_logprobs is not None:
            teacher_logprobs = teacher_logprobs[:seq_len]
        if routed_expert_indices is not None:
            if _is_nested_expert_list(routed_expert_indices):
                routed_expert_indices = [layer[:seq_len] for layer in routed_expert_indices]
            else:
                routed_expert_indices = routed_expert_indices[:seq_len]
        if routed_expert_probs is not None:
            if _is_nested_expert_list(routed_expert_probs):
                routed_expert_probs = [layer[:seq_len] for layer in routed_expert_probs]
            else:
                routed_expert_probs = routed_expert_probs[:seq_len]

    assert len(input_ids) == len(advantages) == len(loss_mask) == len(position_ids) == len(inference_logprobs), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}"
    )
    if teacher_logprobs is not None:
        assert len(teacher_logprobs) == len(input_ids), f"teacher_logprobs: {len(teacher_logprobs)}"
    if routed_expert_indices is not None:
        if _is_nested_expert_list(routed_expert_indices):
            for layer_entries in routed_expert_indices:
                assert len(layer_entries) == len(input_ids), "Routed expert indices length mismatch"
        else:
            assert len(routed_expert_indices) == len(input_ids), "Routed expert indices length mismatch"
    if routed_expert_probs is not None:
        if _is_nested_expert_list(routed_expert_probs):
            for layer_entries in routed_expert_probs:
                assert len(layer_entries) == len(input_ids), "Routed expert probs length mismatch"
        else:
            assert len(routed_expert_probs) == len(input_ids), "Routed expert probs length mismatch"
    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        temperature=temperature,
        routed_expert_indices=routed_expert_indices,
        routed_expert_probs=routed_expert_probs,
    )


def packed_samples_into_micro_bs(
    samples: list[tuple[int, MicroBatch]], max_seq_len: int, num_loras: int
) -> list[MicroBatch]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    """
    samples.sort(key=lambda x: (x[0], -len(x[1].input_ids)))

    ## we create bins
    micro_batches: list[MicroBatch] = []

    for idx, sample in samples:
        # Try to find a bin that can fit this sequence
        for bin_content in micro_batches:
            # Check if sequence fits in this bin
            if len(bin_content.input_ids) + len(sample.input_ids) <= max_seq_len:
                bin_content.input_ids.extend(sample.input_ids)
                bin_content.loss_mask.extend(sample.loss_mask)
                bin_content.advantages.extend(sample.advantages)
                bin_content.inference_logprobs.extend(sample.inference_logprobs)
                bin_content.routed_expert_indices = _extend_expert_metadata(
                    bin_content.routed_expert_indices,
                    sample.routed_expert_indices,
                )
                bin_content.routed_expert_probs = _extend_expert_metadata(
                    bin_content.routed_expert_probs,
                    sample.routed_expert_probs,
                )
                if sample.teacher_logprobs is not None:
                    if bin_content.teacher_logprobs is None:
                        bin_content.teacher_logprobs = []
                    bin_content.teacher_logprobs.extend(sample.teacher_logprobs)
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
    _pad_expert_metadata(micro_batch.routed_expert_indices, padding_size, pad_value=0)
    _pad_expert_metadata(micro_batch.routed_expert_probs, padding_size, pad_value=0.0)
    if micro_batch.teacher_logprobs is not None:
        micro_batch.teacher_logprobs.extend([0.0] * padding_size)
    micro_batch.lora_num_tokens[-1] += (
        padding_size  # We send padding to the last lora so that tokens have ascending lora idx
    )

    return micro_batch


def prepare_batch(
    rollouts: list[TrainingSample],
    temperature: float,
    seq_len: int,
    num_train_workers: int,
    idxs: list[int],
    num_loras: int,
    pad_to_multiple_of: int = 1,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, seq_len], the number of samples is not fixed per micro batch.
    """
    max_seq_len = seq_len

    all_samples = [(idx, prepare_sample(rollout, max_seq_len, temperature)) for idx, rollout in zip(idxs, rollouts)]

    micro_batches = packed_samples_into_micro_bs(all_samples, max_seq_len, num_loras)
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]

    num_padding_batch = -len(micro_batches) % num_train_workers

    # because of fsdp we need to make sure that each data ran has the same number of micro batches otherwise training will hang.
    # We create fake micro batches to fill the gap with real data but zero advantages, they would not contribute to the loss.
    if num_train_workers > 1 and num_padding_batch > 0:
        padded_batch = copy.deepcopy(micro_batches[0])
        padded_batch.advantages = [0.0] * len(padded_batch.input_ids)
        padded_batch.loss_mask = [False] * len(padded_batch.input_ids)
        micro_batches.extend([padded_batch for _ in range(num_padding_batch)])

    assert len(micro_batches) % num_train_workers == 0, (
        "Number of micro batches is not divisible by number of data ranks"
    )

    per_gpu_micro_batches = len(micro_batches) // num_train_workers
    batches_per_gpu = []
    for _ in range(num_train_workers):
        batches = []
        for _ in range(per_gpu_micro_batches):
            batches.append(micro_batches.pop(0))
        batches_per_gpu.append(batches)

    return batches_per_gpu
