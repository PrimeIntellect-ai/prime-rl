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


def _build_mm_refs(urls: list[str], token_types: list[int]) -> MMRefs:
    runs = _image_runs(token_types)
    return MMRefs(
        images=[
            MMImageRef(url=url, offset=offset, length=length) for url, (offset, length) in zip(urls, runs, strict=True)
        ]
    )


def iter_trainable_branches(trace: vf.Trace) -> Iterator[tuple[vf.Branch, np.ndarray]]:
    """Yield each branch that yields a training sample, with its trainable-token mask.

    Branches excluded by trace semantics, or whose image placeholder runs do not match their
    inline images, are skipped before shared-node accounting.
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
        num_images = len(_image_urls(branch))
        token_types = branch.mm_token_type_ids
        num_runs = 0 if token_types is None else len(_image_runs(token_types))
        if num_images != num_runs:
            # e.g. the policy sampled an image placeholder token, which the trainer would read as an image
            get_logger().warning(
                f"Skipping branch {branch.index} of trace {trace.id}: {num_images} inline images "
                f"but {num_runs} image placeholder runs."
            )
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
            mm_refs = _build_mm_refs(image_urls, mm_token_type_ids)
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
