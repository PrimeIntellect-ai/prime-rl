"""Compact sampled and candidate logprobs for the token generate endpoint."""

from typing import Any

import numpy as np
import pybase64
from vllm.logprobs import FlatLogprobs, SampleLogprobs


def serialize_compact_logprobs(logprobs: SampleLogprobs, num_top_logprobs: int) -> dict[str, Any]:
    if isinstance(logprobs, FlatLogprobs):
        starts, ends = logprobs.start_indices, logprobs.end_indices
        ids = np.asarray(logprobs.token_ids, dtype="<u4")
        values = np.asarray(logprobs.logprobs, dtype="<f4")
        lengths = np.asarray(ends, dtype=np.int64) - np.asarray(starts, dtype=np.int64)
        # Sliced FlatLogprobs can reference nonzero or disjoint buffer ranges.
        if starts != [0, *ends[:-1]] or (ends and ends[-1] != len(ids)):
            ids = np.concatenate([ids[start:end] for start, end in zip(starts, ends)]) if starts else ids[:0]
            values = np.concatenate([values[start:end] for start, end in zip(starts, ends)]) if starts else values[:0]
    else:
        lengths = np.asarray([len(row) for row in logprobs], dtype=np.int64)
        ids = np.asarray([token_id for row in logprobs for token_id in row], dtype="<u4")
        values = np.asarray([item.logprob for row in logprobs for item in row.values()], dtype="<f4")

    # Match vLLM's HTTP logprob clamp without constructing its per-token models.
    values = np.maximum(values, -9999.0)
    data = ids.tobytes() + values.tobytes()
    return {
        "data": pybase64.b64encode(data).decode("ascii"),
        "format": "u32-f32-le-v1",
        "offsets": [0, *np.cumsum(lengths).tolist()],
        "num_top_logprobs": num_top_logprobs,
    }
