from __future__ import annotations

from prime_rl.orchestrator.algo.base import Algorithm


class SFTAlgorithm(Algorithm):
    """Supervised fine-tuning on tokens from a frozen model or static dataset.

    Assigns no advantage — the ``ce`` loss ignores credit, and SFT trains on
    every sampled token. Reward-based filtering, if wanted, is an explicit
    filter, not smuggled through an unused advantage stream."""

    action_loss_type = "ce"
