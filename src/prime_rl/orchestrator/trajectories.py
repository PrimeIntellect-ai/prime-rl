"""Turn a v1 `Trace` into training samples, in the env worker that ran it.

The trace is a message graph (`trace.nodes`); each `trace.branches` entry (a root→leaf path)
carries its own flat token sequence (`branch.token_ids` / `branch.sampled_mask` /
`branch.logprobs`), so a trainable branch yields one training sample directly. The worker keeps
each sample's columns (`prime_rl.transports.payload`); the orchestrator only ever sees references
to them, and the trainer fetches them.

Multimodal RL keeps the inline image URLs on the messages and pairs them with vLLM's expanded
image-token runs here.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import verifiers.v1 as vf

from prime_rl.transports.batch import MMImageRef, MMRefs
from prime_rl.transports.batch.types import UNLABELED, UNVERSIONED
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


def token_versions(branch: vf.Branch) -> np.ndarray:
    """The int32 policy version that sampled each of the branch's tokens, from the engine's
    ``weight_versions`` labels: ``UNLABELED`` on a sampled token without a step label (engines
    whose updates bypass ``/update_weights``, e.g. Dynamo), ``UNVERSIONED`` on context."""
    parts = []
    for node in branch.nodes:
        versions = np.full(len(node.token_ids), UNVERSIONED, dtype=np.int32)
        sampled = int(np.count_nonzero(node.mask)) if node.sampled else 0
        offset = len(node.mask) - sampled
        if node.weight_versions:
            for span in node.weight_versions:
                version = int(span.version) if span.version.isdigit() else UNLABELED
                versions[offset + span.start : offset + span.end] = version
        elif sampled:
            versions[offset : offset + sampled] = UNLABELED
        parts.append(versions)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


def trainable_token_layout(trace: vf.Trace) -> dict[int, list[int]]:
    """The trainable tokens of each of the trace's training samples, by branch index: one
    count per sampled node of the branch that trains in it, in order. A sampled node shared
    by several branches trains only in the first."""
    layout: dict[int, list[int]] = {}
    for branch, mask in iter_trainable_branches(trace):
        counts, offset = [], 0
        for node in branch.nodes:
            count = int(np.count_nonzero(mask[offset : offset + len(node.mask)]))
            offset += len(node.mask)
            if count:
                counts.append(count)
        layout[branch.index] = counts
    return layout


@dataclass
class BranchSample:
    """One trainable branch's per-token columns, before the orchestrator's credit and
    staleness: ``mask`` marks the tokens that train, ``sampled`` every model-sampled token."""

    branch: vf.Branch
    token_ids: np.ndarray
    mask: np.ndarray
    sampled: np.ndarray
    logprobs: np.ndarray
    versions: np.ndarray
    rl_weights: np.ndarray | None
    ce_weights: np.ndarray | None
    ref_kl_weights: np.ndarray | None
    ref_logprobs: np.ndarray | None
    mm_token_type_ids: np.ndarray | None
    mm_refs: MMRefs | None


def trace_to_samples(trace: vf.Trace) -> list[BranchSample]:
    """One sample per trainable branch: the branch's flat token sequence, ``mask`` marking the
    trainable (model-sampled) tokens; the context tokens between completions stay masked out.
    A branch carrying images gets raw image refs paired with expanded placeholder ranges and
    ``mm_token_type_ids``. Branches with no sampled tokens yield nothing."""
    samples: list[BranchSample] = []
    trained_loss_nodes: dict[str, set[int]] = {"rl": set(), "ce": set(), "ref_kl": set()}
    for branch, mask in iter_trainable_branches(trace):
        mm_token_type_ids: np.ndarray | None = None
        mm_refs: MMRefs | None = None
        image_urls = _image_urls(branch)
        if image_urls:
            mm_token_type_ids = branch.mm_token_type_ids
            if mm_token_type_ids is None:
                raise ValueError("Inline images have no expanded multimodal prompt tokens")
            mm_refs = _build_mm_refs(image_urls, mm_token_type_ids)
        samples.append(
            BranchSample(
                branch=branch,
                token_ids=branch.token_ids,
                mask=mask,
                sampled=branch.sampled_mask,
                logprobs=branch.logprobs,
                versions=token_versions(branch),
                rl_weights=_loss_weights(branch, "rl", trained_loss_nodes["rl"]),
                ce_weights=_loss_weights(branch, "ce", trained_loss_nodes["ce"]),
                ref_kl_weights=_loss_weights(branch, "ref_kl", trained_loss_nodes["ref_kl"]),
                ref_logprobs=branch.reference_logprobs,
                mm_token_type_ids=mm_token_type_ids,
                mm_refs=mm_refs,
            )
        )
    if not samples:
        get_logger().warning(
            f"No trainable samples (error={trace.has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples
