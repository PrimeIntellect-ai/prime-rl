"""Numeric logprob transport for non-streaming token generation."""

from typing import Any

import numpy as np
import pybase64
from vllm.logprobs import FlatLogprobs, SampleLogprobs


def serialize_compact_logprobs(
    logprobs: SampleLogprobs, num_top_logprobs: int, drop_filtered: bool = False
) -> dict[str, Any]:
    """Pack ragged rows as little-endian uint32 IDs followed by float32 logprobs.

    ``drop_filtered`` omits candidates the sampler filtered out (-inf), which under
    truncated sampling leaves exactly the recorded sampling support in each row."""
    if isinstance(logprobs, FlatLogprobs):
        starts, ends = logprobs.start_indices, logprobs.end_indices
        ids = np.asarray(logprobs.token_ids, dtype="<u4")
        values = np.asarray(logprobs.logprobs, dtype="<f4")
        lengths = np.asarray(ends, dtype=np.int64) - np.asarray(starts, dtype=np.int64)
        # Slices can reference disjoint ranges within the flat buffers.
        if starts != [0, *ends[:-1]] or (ends and ends[-1] != len(ids)):
            ids = np.concatenate([ids[start:end] for start, end in zip(starts, ends)]) if starts else ids[:0]
            values = np.concatenate([values[start:end] for start, end in zip(starts, ends)]) if starts else values[:0]
    else:
        lengths = np.asarray([len(row) for row in logprobs], dtype=np.int64)
        ids = np.asarray([token_id for row in logprobs for token_id in row], dtype="<u4")
        values = np.asarray([item.logprob for row in logprobs for item in row.values()], dtype="<f4")

    if drop_filtered:
        kept = values > -np.inf
        rows = np.repeat(np.arange(len(lengths)), lengths)
        ids, values = ids[kept], values[kept]
        lengths = np.bincount(rows[kept], minlength=len(lengths))
    # Match the native HTTP clamp, including filtered-out (-inf) candidates.
    values = np.maximum(values, -9999.0)
    return {
        "data": pybase64.b64encode(ids.tobytes() + values.tobytes()).decode("ascii"),
        "format": "u32-f32-le-v1",
        "offsets": [0, *np.cumsum(lengths).tolist()],
        "num_top_logprobs": num_top_logprobs,
    }
