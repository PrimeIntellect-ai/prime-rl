from __future__ import annotations

from prime_rl.orchestrator.algo.base import Algorithm


class SFTAlgorithm(Algorithm):
    """Supervised fine-tuning: cross-entropy on the source's target tokens.

    Assigns no advantage — the ``ce`` loss ignores credit, and SFT trains on
    every target token. Where the targets come from (a frozen teacher model's
    fresh rollouts, or a static dataset's stored traces) is the Sampler's
    concern."""

    action_loss_type = "ce"
