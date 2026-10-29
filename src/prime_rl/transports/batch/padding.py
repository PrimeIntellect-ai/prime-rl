import numpy as np

from prime_rl.transports.batch.types import STREAM_FILL, MicroBatch


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
    if micro_batch.staleness is not None:
        micro_batch.staleness = np.pad(micro_batch.staleness, tail, constant_values=-1)

    return micro_batch
