"""Turn a v1 `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything here is
attribute access — no dicts. The trace is a message graph (`trace.nodes`); each `trace.branches`
entry (a root→leaf path) is first-class and carries its own flat token sequence
(`branch.token_ids` / `branch.sampled_mask` / `branch.logprobs`), so a branch yields one
training sample directly. Token-length readers (`completion_len`, `total_tokens`, `num_turns`)
live on `vf.Trace` itself.

Training is renderer-only across every mode (RL/OPD student, SFT teacher), so every node
always carries its tokens — no backfill needed. For multimodal rollouts the branch also carries
the images it introduced (`branch.multi_modal_data`), rebuilt here into the flat `mm_kwargs` /
`mm_token_type_ids` the trainer forwards.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import verifiers.v1 as vf

from prime_rl.transport import TrainingSample
from prime_rl.transport.types import EncodedTensor, RoutedExperts
from prime_rl.utils.logger import get_logger


def _to_numpy(val) -> np.ndarray:
    """A renderer mm item value (torch tensor or numpy array) -> a contiguous numpy array."""
    if hasattr(val, "detach"):  # torch tensor
        val = val.detach().cpu().numpy()
    return np.ascontiguousarray(val)


def _encode_mm_kwargs(mm_items: dict[str, list[dict]]) -> dict[str, EncodedTensor] | None:
    """Concatenate the branch's per-image renderer items into the flat `mm_kwargs` the trainer
    forwards — one `EncodedTensor` per kwarg key (e.g. `pixel_values`, `image_grid_thw`), images
    cat'd along dim 0 in branch token order. Model-agnostic: the keys are whatever the processor
    emits. Returns None when there are no items."""
    bins: dict[str, list[np.ndarray]] = {}
    for items in mm_items.values():  # per modality
        for item in items:  # per image
            for key, val in item.items():
                bins.setdefault(key, []).append(_to_numpy(val))
    encoded: dict[str, EncodedTensor] = {}
    for key, arrs in bins.items():
        arr = np.concatenate(arrs, axis=0)
        encoded[key] = EncodedTensor(dtype=str(arr.dtype), shape=list(arr.shape), data=arr.tobytes())
    return encoded or None


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


def iter_trainable_branches(trace: vf.Trace) -> Iterator[tuple[vf.Branch, list[bool]]]:
    """Yield each branch that yields a training sample, with its trainable-token mask.

    The mask is `branch.sampled_mask` except that a sampled node shared by several branches
    (a mid-trajectory fork) is trainable only in the first branch containing it; later
    branches carry its tokens as context (mask False). Branches left with no trainable
    tokens are skipped, so consumers pairing branches with `trace_to_samples` output
    (e.g. echo's observation weighting) must filter through here to stay aligned.
    """
    trained_nodes: set[int] = set()
    for branch in trace.branches:
        mask: list[bool] = []
        for node in branch.nodes:
            if node.sampled and any(node.mask) and id(node) in trained_nodes:
                mask.extend([False] * len(node.mask))
            else:
                if node.sampled and any(node.mask):
                    trained_nodes.add(id(node))
                mask.extend(node.mask)
        if any(mask):
            yield branch, mask


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
    out. Errored rollouts are dropped upstream (`TrainSink.process_rollout`), so no error
    handling happens here. A branch carrying images also gets `mm_kwargs` (the concatenated
    pixel tensors) and `mm_token_type_ids` (the renderer's `mm_token_type_id_map` applied to
    the branch tokens). Branches with no sampled tokens (e.g. an openai client carrying none)
    yield nothing.
    """
    samples: list[TrainingSample] = []
    for branch, mask in iter_trainable_branches(trace):
        token_ids = branch.token_ids
        mm_kwargs: dict[str, EncodedTensor] | None = None
        mm_token_type_ids: list[int] | None = None
        mmd = branch.multi_modal_data
        if mmd is not None:
            mm_kwargs = _encode_mm_kwargs(mmd.mm_items)
            mapping = mm_token_type_ids_mapping or {}
            mm_token_type_ids = [mapping.get(t, 0) for t in token_ids]
        samples.append(
            TrainingSample(
                token_ids=token_ids,
                mask=mask,
                logprobs=branch.logprobs,
                temperatures=[],  # filled by TrainSink.process_group
                env_name=env_name,
                mm_kwargs=mm_kwargs,
                mm_token_type_ids=mm_token_type_ids,
                routed_experts=_encode_routed_experts(branch.routed_experts, len(token_ids)),
            )
        )
    if not samples:
        get_logger().warning(
            f"No trainable samples (error={trace.has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples
