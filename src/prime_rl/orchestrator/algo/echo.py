from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import numpy as np

from prime_rl.configs.algorithm import EchoAlgoConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    import verifiers.v1 as vf

    from prime_rl.orchestrator.clients import InferenceClient


class EchoAlgorithm(GRPOAlgorithm):
    """GRPO on action tokens, plus weighted CE on env-provided tokens of
    later turns (tool output, user feedback), selected by message role —
    tool-response bodies at the vetted default. Selected tokens feed the
    ``ce`` loss component at their role's ``alpha`` and stay outside the rl
    mask and its denominator. An optional user filter narrows the selection
    per rollout (e.g. dropping tool-output warnings)."""

    def __init__(self, config: EchoAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.role_weights = {
            role: role_config.alpha
            for role in ("system", "user", "assistant", "tool")
            if (role_config := getattr(config.roles, role)) is not None
        }
        self.filter_fn: Callable[..., list[list[bool]]] | None = None
        if config.filter is not None:
            self.filter_fn = partial(import_object(config.filter.import_path), **config.filter.kwargs)

    async def score_episode(self, episode: vf.Episode) -> None:
        for trace in episode.traces:
            if not trace.has_error and trace.agent.trainable:
                self._weight_observations(trace)

    def _weight_observations(self, trace: vf.Trace) -> None:
        """Write graph-native ``ce`` weights over the env-provided
        observation tokens of later turns. Provenance is structural under v1:
        within a branch, the non-sampled nodes that follow the first model
        response (tool output, user feedback) are the env-provided
        observations — each such node's tokens get its message role's weight,
        narrowed by the optional user filter. The initial prompt (before the
        first response) is excluded. Selected tokens have ``mask`` False, so ce
        is the only component that trains them; samples where nothing is
        selected ship no ce stream.

        Content granularity: when a node carries the renderer's per-token
        ``is_content`` (``MessageNode.is_content``, parallel to ``token_ids``),
        only the message-body tokens are weighted — the chat-template scaffold
        (role tags, separators, tool-response wraps) is excluded. Nodes without
        attribution (the default renderer, or relay turns with no token ids)
        fall back to weighting the whole non-sampled span."""
        trainable_branches = [branch for branch, _ in iter_trainable_branches(trace)]
        filter_masks = self._filter_masks(trace, trainable_branches) if self.filter_fn is not None else None
        for branch_idx, branch in enumerate(trainable_branches):
            weights = np.zeros(len(branch.token_ids), dtype=np.float32)
            keep_mask = np.asarray(filter_masks[branch_idx], dtype=bool) if filter_masks is not None else None
            offset = 0
            seen_response = False
            for node in branch.nodes:
                span = len(node.token_ids)
                role = node.message.role
                if seen_response and not node.sampled and role in self.role_weights:
                    selected = np.ones(span, dtype=bool)
                    # Per-token content granularity when the renderer attributed it; otherwise
                    # the whole node span (is_content empty -> fall back to current behavior).
                    if len(node.is_content) == span:
                        selected &= np.asarray(node.is_content, dtype=bool)
                    if keep_mask is not None:
                        selected &= keep_mask[offset : offset + span]
                    weights[offset : offset + span][selected] = self.role_weights[role]
                if node.sampled:
                    seen_response = True
                offset += span
            offset = 0
            for node in branch.nodes:
                end = offset + len(node.token_ids)
                node_weights = weights[offset:end]
                if node_weights.any():
                    streams = dict(node.loss_weights or {})
                    current = streams.get("ce")
                    current = np.zeros(end - offset, dtype=np.float32) if current is None else current
                    streams["ce"] = np.maximum(np.asarray(current, dtype=np.float32), node_weights)
                    node.loss_weights = streams
                offset = end

    def _filter_masks(self, trace: vf.Trace, trainable_branches: list) -> list[list[bool]]:
        """Invoke the user echo filter and validate its shape: one keep-mask
        per trainable branch, each spanning that branch's ``token_ids``."""
        assert self.filter_fn is not None
        masks = self.filter_fn(trace)
        if not isinstance(masks, list) or len(masks) != len(trainable_branches):
            got = len(masks) if isinstance(masks, list) else type(masks).__name__
            raise ValueError(
                f"echo filter must return one keep-mask per trainable branch: got {got}, expected {len(trainable_branches)}"
            )
        for branch_idx, (branch, mask) in enumerate(zip(trainable_branches, masks)):
            expected = len(branch.token_ids)
            if not isinstance(mask, list) or len(mask) != expected:
                got = len(mask) if isinstance(mask, list) else type(mask).__name__
                raise ValueError(
                    f"echo filter mask for branch {branch_idx} must span the branch's tokens: "
                    f"got {got}, expected {expected}"
                )
        return masks
