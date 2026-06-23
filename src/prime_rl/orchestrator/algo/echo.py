from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, EchoAlgorithmConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import RolloutView
    from prime_rl.utils.client import InferencePool


class EchoAlgorithm(GRPOAlgorithm):
    """GRPO on action tokens, plus weighted CE on env-provided tokens of
    later turns (tool output, user feedback), selected by message role —
    tool-response bodies at the vetted default. Selected tokens feed the
    ``ce`` loss component at their role's ``alpha`` and stay outside the rl
    mask and its denominator. An optional user filter narrows the selection
    per rollout (e.g. dropping tool-output warnings)."""

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(config, policy_pool, renderer)
        assert isinstance(config, EchoAlgorithmConfig)
        self.role_weights = {
            role: role_config.alpha
            for role in ("system", "user", "assistant", "tool")
            if (role_config := getattr(config.roles, role)) is not None
        }
        if config.filter is not None:
            raise ValueError("echo filters use the old trajectory-step mask shape and are not supported for v1 traces")

    async def score_rollout(self, rollout: RolloutView) -> None:
        # Observation weighting is rollout-local; the group-relative GRPO
        # baseline is inherited unchanged as ``score_group``.
        self._weight_observations(rollout)

    def _weight_observations(self, rollout: RolloutView) -> None:
        """Write CE weights for non-sampled trace nodes selected by message role."""
        branches = [branch for branch in rollout.trace.branches if branch.token_ids and any(branch.sampled_mask)]
        for sample, branch in zip(rollout.samples, branches, strict=True):
            weights = [0.0] * len(sample.completion_ids)
            offset = 0
            saw_sampled_node = False
            for node in branch.nodes:
                if node.sampled:
                    saw_sampled_node = True
                elif saw_sampled_node:
                    role = getattr(node.message, "role", None)
                    role_weight = self.role_weights.get(role, 0.0)
                    for i in range(len(node.token_ids)):
                        weights[offset + i] = role_weight
                offset += len(node.token_ids)
            if any(weights):
                sample.ce_weights = weights
