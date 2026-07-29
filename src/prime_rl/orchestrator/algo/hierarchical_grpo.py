from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import HierarchicalGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class HierarchicalGRPOAlgorithm(Algorithm):
    """Hierarchical GRPO for task-generating envs (proposer-solver-v1 and
    friends): GRPO baselines computed at two levels of the episode tree.

    A group is ``group_size`` episodes of the same source task — the proposer
    asked ``group_size`` times to invent a problem — and each episode holds the
    proposer's trace plus the n solver traces its minted task fanned out to.
    Neither level is safe under plain GRPO: pooling solver rewards across
    episodes baselines them against attempts at *different* problems, and
    pooling proposer and solver traces mixes reward scales across agents.

    Every trainable trace is instead mean-centered against its peer set:

    - agents in ``episode_agents`` (the solvers) against the same-agent traces
      of their own episode — sibling attempts at the same minted task;
    - every other agent (the proposer) against its same-agent traces across
      the group — parallel attempts at the same source task, rewarded by what
      their minted tasks did to the solvers (e.g. learnability).

    A peer set of one centers to zero advantage (GRPO's singleton convention;
    the zero-advantage filter drops it). Nothing in a rollout reveals the tree,
    so the env itself is gated at config validation
    (:meth:`HierarchicalGRPOAlgoConfig.validate_env`) rather than here: on a flat
    env every peer set would be a singleton and every group train on zeros."""

    def __init__(self, config: HierarchicalGRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.episode_agents = set(config.episode_agents)

    async def score_group(self, group: list[Rollout]) -> None:
        peers: dict[tuple[str | None, str | None], list[Rollout]] = defaultdict(list)
        for rollout in group:
            episode_scoped = rollout.agent_name in self.episode_agents
            key = (rollout.agent_name, rollout.episode_id if episode_scoped else None)
            peers[key].append(rollout)
        for members in peers.values():
            baseline = sum(rollout.reward for rollout in members) / len(members)
            for rollout in members:
                rollout.assign_advantages(rollout.reward - baseline)
