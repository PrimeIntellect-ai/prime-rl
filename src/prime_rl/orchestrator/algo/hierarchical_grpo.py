from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

import torch

from prime_rl.configs.algorithm import HierarchicalGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class HierarchicalGRPOAlgorithm(Algorithm):
    """GRPO over a multi-seat env's episodes: each seat is baselined against
    whoever attempted the same prompt.

    The group is ``group_size`` episodes of one task. A role that fans out
    within an episode (proposer-solver's solvers: n attempts at the *minted*
    problem) is baselined within its own episode; a role that appears once per
    episode (the proposer: every episode starts from the same seed task) is
    baselined across the group's episodes. Singleton partitions get zero
    advantage — the standard GRPO degenerate — so the across-episode signal
    needs ``group_size > 1``, exactly like plain GRPO. On a single-agent env
    every partition is the whole group and this *is* plain GRPO."""

    multi_seat: ClassVar[bool] = True

    def __init__(self, config: HierarchicalGRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)

    async def score_group(self, group: list[Rollout]) -> None:
        by_role_episode: dict[tuple[str | None, str], list[Rollout]] = defaultdict(list)
        for rollout in group:
            by_role_episode[(rollout.role, rollout.episode_id)].append(rollout)
        # A role that ever fans within an episode is per-episode everywhere in the
        # group — its members share a minted prompt, not the seed task.
        fanned = {role for (role, _), members in by_role_episode.items() if len(members) > 1}
        partitions: list[list[Rollout]] = []
        for role in {rollout.role for rollout in group}:
            if role in fanned:
                partitions.extend(members for (r, _), members in by_role_episode.items() if r == role)
            else:
                partitions.append([rollout for rollout in group if rollout.role == role])
        for partition in partitions:
            rewards = torch.tensor([rollout.reward for rollout in partition], dtype=torch.float32)
            advantages = rewards - rewards.mean()
            for rollout, advantage in zip(partition, advantages.tolist(), strict=True):
                rollout.assign_advantages(advantage)
