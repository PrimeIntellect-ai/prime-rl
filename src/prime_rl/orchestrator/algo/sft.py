from __future__ import annotations

from prime_rl.orchestrator.algo.base import Algorithm


class SFTDistillAlgorithm(Algorithm):
    """Hard distillation. Needs a teacher: the frozen model that generates the
    rollouts (``sampling.source``); the policy trains with CE on its tokens.

    Assigns no advantage — the ``ce`` loss ignores credit, and SFT trains on
    every sampled token. A curriculum can reject results using reward or any
    other finalized rollout data."""

    action_loss_type = "ce"
