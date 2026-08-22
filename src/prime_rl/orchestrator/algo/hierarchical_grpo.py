from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import HierarchicalGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages

if TYPE_CHECKING:
    from prime_rl.utils.client import InferencePool


class HierarchicalGRPOAlgorithm(Algorithm):
    """GRPO for proposer-solver envs.

    Solver rewards are compared only with other attempts on the same proposed
    problem. Proposer rewards are compared with the other proposals generated
    from the same source task. Keeping those comparisons separate avoids
    treating different problem difficulties or different agent roles as
    interchangeable.

    ``episode_agents`` lists the roles, normally ``solver``, that are compared
    within one episode. Other roles are compared across the full rollout group.
    A comparison group with one trace produces zero advantage."""

    def __init__(self, config: HierarchicalGRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.episode_agents = set(config.episode_agents)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        peers: dict[tuple[str, str | None], list[vf.Trace]] = defaultdict(list)
        for episode, trace in iter_trainable_traces(episodes):
            episode_scoped = trace.agent.name in self.episode_agents
            key = (trace.agent.name, episode.id if episode_scoped else None)
            peers[key].append(trace)
        for members in peers.values():
            baseline = sum(trace.reward for trace in members) / len(members)
            for trace in members:
                assign_advantages(trace, trace.reward - baseline)
