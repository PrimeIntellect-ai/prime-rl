"""Turn a v1 `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything here is
attribute access — no dicts. The trace is a message graph (`trace.nodes`); each `trace.branches`
entry (a root→leaf path) is first-class and carries its own flat token sequence
(`branch.token_ids` / `branch.sampled_mask` / `branch.logprobs`), so a branch yields one
training sample directly. Token-length readers (`completion_len`, `total_tokens`, `num_turns`)
live on `vf.Trace` itself.

Training is renderer-only across every mode (RL/OPD student, SFT teacher), so every node
always carries its tokens — no backfill needed. For multimodal rollouts the branch also carries
raw image refs and renderer descriptors (`branch.multi_modal_data`), preserved here as
`mm_refs` for trainer-side materialization.
"""

from __future__ import annotations

import numpy as np
import verifiers.v1 as vf

from prime_rl.transport import TrainingSample
from prime_rl.transport.types import MMRefs, RoutedExperts
from prime_rl.utils.logger import get_logger
from prime_rl.utils.mm import build_mm_refs


def _encode_routed_experts(arr: np.ndarray | None, num_tokens: int) -> RoutedExperts | None:
    """The branch's router-replay array (`[tokens, layers, top_k]`) -> the transport
    `RoutedExperts` the trainer replays. Defensively realigns the token axis to `num_tokens`
    (the trainer asserts `routed_experts.shape[0] == len(token_ids)`): truncate if longer,
    zero-pad the tail if shorter. `Branch.routed_experts` already guarantees alignment, so this
    is a backstop."""
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    if arr.shape[0] > num_tokens:
        arr = arr[:num_tokens]
    elif arr.shape[0] < num_tokens:
        pad = np.zeros((num_tokens - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return RoutedExperts(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def _validate_image_spans(mm_refs: MMRefs, mm_token_type_ids: list[int]) -> None:
    """Every image ref's placeholder span must land on image-typed tokens.

    Placeholder offsets flow through the renderer, bridge extension, and node
    attribution before arriving here — and the trainer truncates on them — so
    any drift anywhere upstream must fail loudly before the sample ships.
    """
    for image in mm_refs.images:
        span = mm_token_type_ids[image.offset : image.offset + image.length]
        if len(span) != image.length or any(t != 1 for t in span):
            raise ValueError(
                f"Raw image placeholder [{image.offset}, {image.offset + image.length}) does not "
                f"cover image-typed tokens (branch length {len(mm_token_type_ids)}) — placeholder "
                "offsets have drifted from the branch token stream"
            )


def trace_to_samples(
    trace: vf.Trace,
    *,
    env_name: str = "",
    mm_token_type_ids_mapping: dict[int, int] | None = None,
) -> list[TrainingSample]:
    """Convert a v1 `Trace` into `TrainingSample`s — one per branch.

    Each `trace.branches` entry is already a flat token sequence (`branch.token_ids` /
    `branch.sampled_mask` / `branch.logprobs`), so a sample carries it directly: `mask` marks
    the trainable (model-sampled) tokens, the context tokens between completions stay masked
    out. On a rollout error the whole completion is masked out. A branch carrying images also
    gets `mm_refs` (raw image URIs + JSON-safe renderer metadata) and
    `mm_token_type_ids` (the renderer's `mm_token_type_id_map` applied to the branch
    tokens). Branches with no sampled tokens (e.g. an openai client carrying none) yield
    nothing.
    """
    has_error = trace.has_error
    samples: list[TrainingSample] = []
    for branch in trace.branches:
        mask = branch.sampled_mask
        if not any(mask):
            continue
        token_ids = branch.token_ids
        mm_refs: MMRefs | None = None
        mm_token_type_ids: list[int] | None = None
        mmd = branch.multi_modal_data
        if mmd is not None:
            mm_refs = build_mm_refs(mmd)
            mapping = mm_token_type_ids_mapping or {}
            mm_token_type_ids = [mapping.get(t, 0) for t in token_ids]
            if mm_refs is not None and mapping:
                _validate_image_spans(mm_refs, mm_token_type_ids)
        samples.append(
            TrainingSample(
                token_ids=token_ids,
                mask=[m and not has_error for m in mask],
                logprobs=branch.logprobs,
                temperatures=[],  # filled by TrainSink.process_group
                env_name=env_name,
                mm_refs=mm_refs,
                mm_token_type_ids=mm_token_type_ids,
                routed_experts=_encode_routed_experts(branch.routed_experts, len(token_ids)),
            )
        )
    if not samples:
        get_logger().warning(
            f"No trainable samples (error={has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples
