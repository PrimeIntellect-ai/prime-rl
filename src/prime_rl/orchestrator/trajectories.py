"""Turn a v1 `Trace` (the env worker's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything here is
attribute access — no dicts. The trace is a message graph (`trace.nodes`); each `trace.branches`
entry (a root→leaf path) is first-class and carries its own flat token sequence
(`branch.token_ids` / `branch.sampled_mask` / `branch.logprobs`), so a branch yields one
training sample directly. Token-length readers (`completion_len`, `total_tokens`, `num_turns`)
live on `vf.Trace` itself.

Training is renderer-only across every mode (RL/OPD student, SFT teacher), so every node
always carries its tokens — no backfill needed. Multimodal RL keeps the inline image and video
URLs on the messages and pairs them with vLLM's expanded placeholder runs here.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import verifiers.v1 as vf

from prime_rl.transports.batch import MMImageRef, MMRefs, MMVideoRef, TrainingSample
from prime_rl.transports.batch.types import (
    PAYLOAD_INFO_KEY,
    PayloadSlice,
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


def _video_urls(branch: vf.Branch) -> list[str]:
    urls: list[str] = []
    for node in branch.nodes:
        content = node.message.content
        if not isinstance(content, list):
            continue
        for part in content:
            if getattr(part, "type", None) == "video_url":
                urls.append(part.video_url.url)
    return urls


def _placeholder_runs(token_types: np.ndarray, token_type: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, current in enumerate([*token_types, 0]):
        if current == token_type and start is None:
            start = index
        elif current != token_type and start is not None:
            runs.append((start, index - start))
            start = None
    return runs


def _video_ranges(token_ids: np.ndarray, token_types: np.ndarray) -> list[tuple[int, int]]:
    """vLLM's placeholder range of each video. vLLM expands a prompt's ``<|video_pad|>`` into frame groups
    ``<t seconds><|vision_start|><|video_pad|>...<|vision_end|>``, between the prompt's own
    ``<|vision_start|>`` and ``<|vision_end|>``; the range spans the frame groups."""
    frames = _placeholder_runs(token_types, 2)
    if not frames:
        return []
    vision_start = token_ids[frames[0][0] - 1]
    vision_end = token_ids[frames[0][0] + frames[0][1]]
    videos = [[frames[0]]]
    for offset, length in frames[1:]:
        previous_offset, previous_length = videos[-1][-1]
        gap = token_ids[previous_offset + previous_length : offset]
        # Consecutive frame groups of one video are separated by one <|vision_end|> and one <|vision_start|>.
        if np.count_nonzero(gap == vision_end) == 1 and np.count_nonzero(gap == vision_start) == 1:
            videos[-1].append((offset, length))
        else:
            videos.append([(offset, length)])
    ranges: list[tuple[int, int]] = []
    for video in videos:
        first_offset = video[0][0]
        last_offset, last_length = video[-1]
        # The first frame group's timestamp text follows the prompt's <|vision_start|>.
        prompt_vision_starts = np.flatnonzero(token_ids[: first_offset - 1] == vision_start)
        if len(prompt_vision_starts) == 0:
            raise ValueError("Video placeholder range has no enclosing <|vision_start|>")
        start = int(prompt_vision_starts[-1]) + 1
        ranges.append((start, last_offset + last_length + 1 - start))
    return ranges


def _build_mm_refs(
    image_urls: list[str], video_urls: list[str], token_ids: np.ndarray, token_types: np.ndarray
) -> MMRefs | None:
    runs = _placeholder_runs(token_types, 1)
    if len(image_urls) != len(runs):
        raise ValueError(
            f"Inline image count does not match expanded placeholder runs: images={len(image_urls)}, runs={len(runs)}"
        )
    video_ranges = _video_ranges(token_ids, token_types)
    if len(video_urls) != len(video_ranges):
        raise ValueError(
            "Inline video count does not match expanded placeholder ranges: "
            f"videos={len(video_urls)}, ranges={len(video_ranges)}"
        )
    if not image_urls and not video_urls:
        return None
    return MMRefs(
        images=[
            MMImageRef(url=url, offset=offset, length=length)
            for url, (offset, length) in zip(image_urls, runs, strict=True)
        ],
        videos=[
            MMVideoRef(url=url, offset=offset, length=length)
            for url, (offset, length) in zip(video_urls, video_ranges, strict=True)
        ],
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


def trace_to_samples(trace: vf.Trace, *, env_name: str = "") -> list[TrainingSample]:
    """Convert a v1 `Trace` into `TrainingSample`s — one per branch.

    Each `trace.branches` entry is already a flat token sequence (`branch.token_ids` /
    `branch.sampled_mask` / `branch.logprobs`), so a sample carries it directly: `mask` marks
    the trainable (model-sampled) tokens, the context tokens between completions stay masked
    out. Errored traces are dropped upstream (`TrainSink.process_episode`), so no error
    handling happens here. A branch carrying images or videos gets raw media refs paired with
    expanded placeholder ranges and `mm_token_type_ids`. Branches with no sampled tokens yield nothing.
    """
    samples: list[TrainingSample] = []
    # Set by the env worker when it keeps the arrays (`prime_rl.transports.payload`).
    references = (trace.info.get(PAYLOAD_INFO_KEY) or {}).get("branches", {})
    trained_loss_nodes: dict[str, set[int]] = {"rl": set(), "ce": set(), "ref_kl": set()}
    for branch, mask in iter_trainable_branches(trace):
        token_ids = branch.token_ids
        mm_token_type_ids: np.ndarray | None = None
        mm_refs: MMRefs | None = None
        image_urls = _image_urls(branch)
        video_urls = _video_urls(branch)
        if image_urls or video_urls:
            mm_token_type_ids = branch.mm_token_type_ids
            if mm_token_type_ids is None:
                raise ValueError("Inline images and videos have no expanded multimodal prompt tokens")
            mm_refs = _build_mm_refs(image_urls, video_urls, token_ids, mm_token_type_ids)
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
                rl_weights=_loss_weights(branch, "rl", trained_loss_nodes["rl"]),
                ce_weights=_loss_weights(branch, "ce", trained_loss_nodes["ce"]),
                ref_kl_weights=_loss_weights(branch, "ref_kl", trained_loss_nodes["ref_kl"]),
                advantages=branch.advantages,
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
