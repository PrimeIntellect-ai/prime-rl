"""Turn a v1 `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything here is
attribute access — no dicts. The trace is a message graph (`trace.nodes`); each `trace.branches`
entry (a root→leaf path) is first-class and carries its own flat token sequence
(`branch.token_ids` / `branch.sampled_mask` / `branch.logprobs`), so a branch yields one
training sample directly. Token-length readers (`completion_len`, `total_tokens`, `num_turns`)
live on `vf.Trace` itself.

Training is renderer-only across every mode (RL/OPD student, SFT teacher), so every node
always carries its tokens — no backfill needed. Multimodal RL keeps the inline image URLs on
the messages and pairs them with vLLM's expanded image-token runs here.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import verifiers.v1 as vf

from prime_rl.transports.batch import MMImageRef, MMRefs, TrainingSample
from prime_rl.transports.batch.types import RoutedExperts
from prime_rl.utils.logger import get_logger


def _image_urls(branch: vf.Branch) -> list[str]:
    urls: list[str] = []
    for node in branch.nodes:
        content = node.message.content
        if not isinstance(content, list):
            continue
        for part in content:
            if getattr(part, "type", None) == "image_url":
                urls.append(part.image_url.url)
    return urls


def _image_runs(token_types: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, token_type in enumerate([*token_types, 0]):
        if token_type == 1 and start is None:
            start = index
        elif token_type != 1 and start is not None:
            runs.append((start, index - start))
            start = None
    return runs


def _build_mm_refs(urls: list[str], token_types: list[int]) -> MMRefs | None:
    runs = _image_runs(token_types)
    if len(urls) != len(runs):
        raise ValueError(
            f"Inline image count does not match expanded placeholder runs: images={len(urls)}, runs={len(runs)}"
        )
    if not urls:
        return None
    return MMRefs(
        images=[
            MMImageRef(url=url, offset=offset, length=length) for url, (offset, length) in zip(urls, runs, strict=True)
        ]
    )


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


def _loss_weights(branch: vf.Branch, name: str, trained_nodes: set[int]) -> list[float] | None:
    """Flatten one graph-native loss stream, training each shared node once."""
    weights: list[float] = []
    for node in branch.nodes:
        node_weights = (node.loss_weights or {}).get(name)
        if node_weights is None or id(node) in trained_nodes:
            weights.extend([0.0] * len(node.token_ids))
            continue
        if len(node_weights) != len(node.token_ids):
            raise ValueError(
                f"loss weight stream {name!r} must align with node token_ids: "
                f"got {len(node_weights)}, expected {len(node.token_ids)}"
            )
        weights.extend(node_weights)
        if any(node_weights):
            trained_nodes.add(id(node))
    return weights if any(weights) else None


def trace_to_samples(trace: vf.Trace, *, env_name: str = "") -> list[TrainingSample]:
    """Convert a v1 `Trace` into `TrainingSample`s — one per branch.

    Each `trace.branches` entry is already a flat token sequence (`branch.token_ids` /
    `branch.sampled_mask` / `branch.logprobs`), so a sample carries it directly: `mask` marks
    the trainable (model-sampled) tokens, the context tokens between completions stay masked
    out. Errored traces are dropped upstream (`TrainSink.process_episode`), so no error
    handling happens here. A branch carrying images gets raw image refs paired with expanded
    placeholder ranges and `mm_token_type_ids`. Branches with no sampled tokens yield nothing.
    """
    samples: list[TrainingSample] = []
    trained_loss_nodes: dict[str, set[int]] = {"rl": set(), "ce": set(), "ref_kl": set()}
    for branch, mask in iter_trainable_branches(trace):
        token_ids = branch.token_ids
        mm_token_type_ids: list[int] | None = None
        mm_refs: MMRefs | None = None
        image_urls = _image_urls(branch)
        if image_urls:
            mm_token_type_ids = branch.mm_token_type_ids
            if mm_token_type_ids is None:
                raise ValueError("Inline images have no expanded multimodal prompt tokens")
            mm_refs = _build_mm_refs(image_urls, mm_token_type_ids)
        samples.append(
            TrainingSample(
                token_ids=token_ids,
                mask=mask,
                logprobs=branch.logprobs,
                temperatures=[],  # filled by TrainSink.process_group
                env_name=env_name,
                ref_logprobs=branch.reference_logprobs,
                mm_refs=mm_refs,
                mm_token_type_ids=mm_token_type_ids,
                routed_experts=_encode_routed_experts(branch.routed_experts, len(token_ids)),
                rl_weights=_loss_weights(branch, "rl", trained_loss_nodes["rl"]),
                ce_weights=_loss_weights(branch, "ce", trained_loss_nodes["ce"]),
                ref_kl_weights=_loss_weights(branch, "ref_kl", trained_loss_nodes["ref_kl"]),
                advantages=branch.advantages,
            )
        )
    if not samples:
        get_logger().warning(
            f"No trainable samples (error={trace.has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples
