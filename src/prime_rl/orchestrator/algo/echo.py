from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

from prime_rl.configs.algorithm import EchoAlgoConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    import verifiers.v1 as vf

    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class EchoAlgorithm(GRPOAlgorithm):
    """GRPO on action tokens, plus weighted CE on env-provided tokens of
    later turns (tool output, user feedback), selected by message role —
    tool-response bodies at the vetted default. Selected tokens feed the
    ``ce`` loss component at their role's ``alpha`` and stay outside the rl
    mask and its denominator. An optional user filter narrows the selection
    per rollout (e.g. dropping tool-output warnings)."""

    def __init__(self, config: EchoAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.role_weights = {
            role: role_config.alpha
            for role in ("system", "user", "assistant", "tool")
            if (role_config := getattr(config.roles, role)) is not None
        }
        self.filter_fn: Callable[..., list[list[bool]]] | None = None
        if config.filter is not None:
            self.filter_fn = partial(import_object(config.filter.import_path), **config.filter.kwargs)

    async def score_rollout(self, rollout: Rollout) -> None:
        # Observation weighting is rollout-local; the group-relative GRPO
        # baseline is inherited unchanged as ``score_group``.
        self._weight_observations(rollout)

    def _weight_observations(self, rollout: Rollout) -> None:
        """Write each sample's ``ce_weights`` stream over the env-provided
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
        # Same branch selection as ``trace_to_samples``, so the zip with ``rollout.samples``
        # stays aligned when fork dedup drops a branch.
        trainable_branches = [branch for branch, _ in iter_trainable_branches(rollout)]
        filter_masks = self._filter_masks(rollout, trainable_branches) if self.filter_fn is not None else None
        for sample_idx, (sample, branch) in enumerate(zip(rollout.samples, trainable_branches)):
            weights = [0.0] * len(sample.token_ids)
            offset = 0
            seen_response = False
            for node in branch.nodes:
                span = len(node.token_ids)
                role = node.message.role
                if seen_response and not node.sampled and role in self.role_weights:
                    weight = self.role_weights[role]
                    keep_mask = filter_masks[sample_idx] if filter_masks is not None else None
                    # Per-token content granularity when the renderer attributed it; otherwise
                    # the whole node span (is_content empty -> fall back to current behavior).
                    has_content = len(node.is_content) == span
                    for i in range(offset, offset + span):
                        if has_content and not node.is_content[i - offset]:
                            continue
                        if keep_mask is None or keep_mask[i]:
                            weights[i] = weight
                if node.sampled:
                    seen_response = True
                offset += span
            if any(weights):
                sample.ce_weights = weights

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
