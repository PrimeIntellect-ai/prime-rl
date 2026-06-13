from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.advantage import assign_group_norm
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import RolloutView


class SFTDistillAlgorithm(Algorithm):
    """Hard distillation. Needs a teacher: the frozen model that generates the
    rollouts (``sampling.source``); the policy trains with CE on its tokens.

    The ``ce`` loss ignores credit, but group-relative advantages are still
    assigned so reward-based filtering keeps working."""

    action_loss_type = "ce"

    def score_group(self, group: list[RolloutView]) -> None:
        assign_group_norm(group, None)
