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
from prime_rl.transports.batch.types import (
    PAYLOAD_INFO_KEY,
    BranchArrays,
    PayloadSlice,
    RoutedExperts,
    SamplingMask,
    TopLogprobs,
)
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


def _encode_sampler_evidence(
    mask: vf.SamplingMask | None, head: vf.TopLogprobs | None, num_tokens: int
) -> tuple[SamplingMask | None, TopLogprobs | None]:
    """Encode a branch's sampling mask and sampler top logprobs for a fixed token count.

    The mask stores row sizes in `counts` and concatenates every row in `ids`.
    For example, `counts=[2, 1]` and `ids=[4, 7, 9]` become `counts=[2]` and
    `ids=[4, 7]` for one token. For three tokens, they become
    `counts=[2, 1, 0]` with unchanged ids. Zero-count rows disable replay.

    A masked token's top logprobs cover exactly its mask, so they move onto the
    mask; the remaining top-logprob rows are heads of untruncated distributions.
    """
    if head is not None and (
        len(head.counts) != num_tokens or int(head.counts.sum()) != len(head.ids) or len(head.ids) != len(head.logprobs)
    ):
        raise ValueError("Top logprobs are not aligned to branch tokens")
    if mask is None:
        if head is None:
            return None, None
        return None, TopLogprobs(
            ids=np.ascontiguousarray(head.ids, dtype=np.int32).tobytes(),
            logprobs=np.ascontiguousarray(head.logprobs, dtype=np.float32).tobytes(),
            counts=np.ascontiguousarray(head.counts, dtype=np.int32).tobytes(),
        )

    mask_ids, mask_counts = np.asarray(mask.ids), np.asarray(mask.counts)
    if len(mask_counts) > num_tokens:
        mask_counts = mask_counts[:num_tokens]
        mask_ids = mask_ids[: int(mask_counts.sum())]
    elif len(mask_counts) < num_tokens:
        mask_counts = np.concatenate([mask_counts, np.zeros(num_tokens - len(mask_counts), dtype=np.int32)])
    if head is None:
        return SamplingMask(
            ids=np.ascontiguousarray(mask_ids, dtype=np.int32).tobytes(),
            counts=np.ascontiguousarray(mask_counts, dtype=np.int32).tobytes(),
        ), None

    head_ids, head_logprobs, head_counts = np.asarray(head.ids), np.asarray(head.logprobs), np.asarray(head.counts)
    if np.array_equal(mask_counts, head_counts) and np.array_equal(mask_ids, head_ids):
        return SamplingMask(
            ids=np.ascontiguousarray(head_ids, dtype=np.int32).tobytes(),
            counts=np.ascontiguousarray(mask_counts, dtype=np.int32).tobytes(),
            logprobs=np.ascontiguousarray(head_logprobs, dtype=np.float32).tobytes(),
        ), None
    rows = np.arange(num_tokens, dtype=np.int64)
    head_rows = np.repeat(rows, head_counts)
    masked = mask_counts > 0
    on_mask = masked[head_rows]
    mask_keys = np.repeat(rows, mask_counts) << 32 | mask_ids.astype(np.int64)
    head_keys = head_rows[on_mask] << 32 | head_ids[on_mask].astype(np.int64)
    if not np.array_equal(np.sort(mask_keys), np.sort(head_keys)):
        raise ValueError("Top logprobs must cover exactly the recorded sampling support")
    sampling_mask = SamplingMask(
        ids=np.ascontiguousarray(head_ids[on_mask], dtype=np.int32).tobytes(),
        counts=np.ascontiguousarray(mask_counts, dtype=np.int32).tobytes(),
        logprobs=np.ascontiguousarray(head_logprobs[on_mask], dtype=np.float32).tobytes(),
    )
    if on_mask.all():
        return sampling_mask, None
    return sampling_mask, TopLogprobs(
        ids=np.ascontiguousarray(head_ids[~on_mask], dtype=np.int32).tobytes(),
        logprobs=np.ascontiguousarray(head_logprobs[~on_mask], dtype=np.float32).tobytes(),
        counts=np.ascontiguousarray(np.where(masked, 0, head_counts), dtype=np.int32).tobytes(),
    )


def iter_trainable_branches(trace: vf.Trace) -> Iterator[tuple[vf.Branch, np.ndarray]]:
    """Yield each branch that yields a training sample, with its trainable-token mask.

    Branches excluded by trace semantics are skipped before shared-node accounting.
    The mask is `branch.sampled_mask` except that a sampled node shared by several branches
    (a mid-trajectory fork) is trainable only in the first branch containing it; later
    branches carry its tokens as context (mask False). Branches left with no trainable
    tokens are skipped, so consumers pairing branches with `trace_to_samples` output
    (e.g. echo's observation weighting) must filter through here to stay aligned.
    """
    trained_nodes: set[int] = set()
    for branch in trace.branches:
        if not branch.trainable:
            continue
        parts: list[np.ndarray] = []
        for node in branch.nodes:
            node_mask = np.asarray(node.mask, dtype=bool)
            if node.sampled and node_mask.any() and id(node) in trained_nodes:
                parts.append(np.zeros(len(node_mask), dtype=bool))
            else:
                if node.sampled and node_mask.any():
                    trained_nodes.add(id(node))
                parts.append(node_mask)
        mask = np.concatenate(parts) if parts else np.zeros(0, dtype=bool)
        if mask.any():
            yield branch, mask


def _loss_weights(branch: vf.Branch, name: str, trained_nodes: set[int]) -> np.ndarray | None:
    """Flatten one graph-native loss stream (float32), training each shared node once."""
    parts: list[np.ndarray] = []
    for node in branch.nodes:
        node_weights = (node.loss_weights or {}).get(name)
        if node_weights is None or id(node) in trained_nodes:
            parts.append(np.zeros(len(node.token_ids), dtype=np.float32))
            continue
        if len(node_weights) != len(node.token_ids):
            raise ValueError(
                f"loss weight stream {name!r} must align with node token_ids: "
                f"got {len(node_weights)}, expected {len(node.token_ids)}"
            )
        node_weights = np.asarray(node_weights, dtype=np.float32)
        parts.append(node_weights)
        if node_weights.any():
            trained_nodes.add(id(node))
    weights = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return weights if weights.any() else None


def encode_branch_arrays(branch: vf.Branch, num_tokens: int) -> BranchArrays:
    """The branch's routing rows, sampling mask and top-k head as the trainer replays them."""
    sampling_mask, top_logprobs = _encode_sampler_evidence(branch.sampling_mask, branch.top_logprobs, num_tokens)
    return BranchArrays(
        routed_experts=_encode_routed_experts(branch.routed_experts, num_tokens),
        sampling_mask=sampling_mask,
        top_logprobs=top_logprobs,
    )


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
    # Set by the env server when it keeps the arrays (`prime_rl.transports.payload`).
    references = (trace.info.get(PAYLOAD_INFO_KEY) or {}).get("branches", {})
    trained_loss_nodes: dict[str, set[int]] = {"rl": set(), "ce": set(), "ref_kl": set()}
    for branch, mask in iter_trainable_branches(trace):
        token_ids = branch.token_ids
        mm_token_type_ids: np.ndarray | None = None
        mm_refs: MMRefs | None = None
        image_urls = _image_urls(branch)
        if image_urls:
            mm_token_type_ids = branch.mm_token_type_ids
            if mm_token_type_ids is None:
                raise ValueError("Inline images have no expanded multimodal prompt tokens")
            mm_refs = _build_mm_refs(image_urls, mm_token_type_ids)
        arrays = encode_branch_arrays(branch, len(token_ids))
        samples.append(
            TrainingSample(
                token_ids=token_ids,
                mask=mask,
                logprobs=branch.logprobs,
                temperatures=np.zeros(0, dtype=np.float32),  # filled by TrainSink.process_group
                env_name=env_name,
                ref_logprobs=branch.reference_logprobs,
                mm_refs=mm_refs,
                mm_token_type_ids=mm_token_type_ids,
                routed_experts=arrays.routed_experts,
                rl_weights=_loss_weights(branch, "rl", trained_loss_nodes["rl"]),
                ce_weights=_loss_weights(branch, "ce", trained_loss_nodes["ce"]),
                ref_kl_weights=_loss_weights(branch, "ref_kl", trained_loss_nodes["ref_kl"]),
                advantages=branch.advantages,
                sampling_mask=arrays.sampling_mask,
                top_logprobs=arrays.top_logprobs,
                trace_id=trace.id,
                branch_index=branch.index,
                payload=PayloadSlice(
                    url=trace.info[PAYLOAD_INFO_KEY]["url"],
                    num_tokens=len(token_ids),
                    **references[str(branch.index)],
                )
                if str(branch.index) in references
                else None,
            )
        )
    if not samples:
        get_logger().warning(
            f"No trainable samples (error={trace.has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples
