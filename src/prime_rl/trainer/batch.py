import copy

from prime_rl.transport.types import MicroBatch, TrainingSample


def prepare_sample(
    training_example: TrainingSample,
    seq_len: int,
) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    # Prepare input_ids, loss_mask, position_ids, inference_logprobs, and advantages
    input_ids = training_example.prompt_ids + training_example.completion_ids
    loss_mask = training_example.prompt_mask + training_example.completion_mask
    inference_logprobs = [0.0] * len(training_example.prompt_ids) + training_example.completion_logprobs
    advantages = [training_example.advantage] * len(input_ids)
    position_ids = list(range(len(input_ids)))
    
    teacher_logprobs = training_example.teacher_logprobs

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
        inference_logprobs = inference_logprobs[:seq_len]
        position_ids = position_ids[:seq_len]
        advantages = advantages[:seq_len]
        if teacher_logprobs is not None:
            teacher_logprobs = teacher_logprobs[:seq_len]

    assert len(input_ids) == len(advantages) == len(loss_mask) == len(position_ids) == len(inference_logprobs), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}"
    )
    if teacher_logprobs is not None:
        assert len(teacher_logprobs) == len(input_ids), (
            f"teacher_logprobs: {len(teacher_logprobs)}, input_ids: {len(input_ids)}"
        )
    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
    )


def packed_samples_into_micro_bs(samples: list[MicroBatch], max_seq_len: int) -> list[MicroBatch]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    """
    sorted_samples = sorted(samples, key=lambda x: len(x.input_ids), reverse=True)

    ## we create bins
    micro_batches: list[MicroBatch] = []

    for sample in sorted_samples:
        # Try to find a bin that can fit this sequence
        for bin_content in micro_batches:
            # Check if sequence fits in this bin
            if len(bin_content.input_ids) + len(sample.input_ids) <= max_seq_len:
                bin_content.input_ids.extend(sample.input_ids)
                bin_content.loss_mask.extend(sample.loss_mask)
                bin_content.advantages.extend(sample.advantages)
                bin_content.inference_logprobs.extend(sample.inference_logprobs)
                bin_content.position_ids.extend(sample.position_ids)
                # Handle teacher_logprobs if present
                if sample.teacher_logprobs is not None:
                    if bin_content.teacher_logprobs is None:
                        bin_content.teacher_logprobs = []
                    bin_content.teacher_logprobs.extend(sample.teacher_logprobs)
                break
        else:
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

    micro_batch.input_ids.extend([1 for _ in range(padding_size)])
    micro_batch.advantages.extend([0.0 for _ in range(padding_size)])
    micro_batch.loss_mask.extend([False for _ in range(padding_size)])
    micro_batch.position_ids.extend(list(range(padding_size)))
    micro_batch.inference_logprobs.extend([0.0 for _ in range(padding_size)])
    if micro_batch.teacher_logprobs is not None:
        micro_batch.teacher_logprobs.extend([0.0 for _ in range(padding_size)])

    return micro_batch


def prepare_batch(
    rollouts: list[TrainingSample],
    temperature: float,
    seq_len: int,
    num_train_workers: int,
    pad_to_multiple_of: int = 1,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, seq_len], the number of samples is not fixed per micro batch.
    """
    max_seq_len = seq_len

    all_samples = [prepare_sample(rollout, max_seq_len) for rollout in rollouts]

    micro_batches = packed_samples_into_micro_bs(all_samples, max_seq_len)
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]

    for micro_batch in micro_batches:
        micro_batch.temperature = temperature

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
