from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import HierarchicalGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Episode, TrainRollout
    from prime_rl.utils.client import InferencePool


class HierarchicalGRPOAlgorithm(Algorithm):
    """GRPO for proposer-solver envs.

    Solver rewards are compared only with other attempts on the same proposed
    problem. Proposer rewards are compared with the other proposals generated
    from the same source task. Keeping those comparisons separate avoids
    treating different problem difficulties or different agent roles as
    interchangeable.

    ``episode_agents`` lists the roles, normally ``solver``, that are compared
    within one episode. Other roles are compared across the whole group. A
    comparison group with one trace produces zero advantage."""

    def __init__(self, config: HierarchicalGRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.episode_agents = set(config.episode_agents)

    async def score_group(self, group: list[Episode]) -> None:
        peers: dict[tuple[str, str | None], list[TrainRollout]] = defaultdict(list)
        for episode in group:
            for rollout in episode.rollouts:
                episode_scoped = rollout.agent.name in self.episode_agents
                peers[(rollout.agent.name, episode.id if episode_scoped else None)].append(rollout)
        for members in peers.values():
            baseline = sum(rollout.reward for rollout in members) / len(members)
            for rollout in members:
                rollout.assign_advantages(rollout.reward - baseline)
