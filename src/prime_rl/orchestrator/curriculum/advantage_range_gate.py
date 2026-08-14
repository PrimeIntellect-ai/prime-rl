"""Advantage-based training-sample admission."""

from __future__ import annotations

from prime_rl.orchestrator.curriculum.base import AdmissionGate, CurriculumResult


class AdvantageRangeGate(AdmissionGate):
    """Reject groups whose trainable-token advantages all fall inside a range.

    The default ``[0, 0]`` interval implements zero-advantage rejection.
    Groups without an advantage stream are admitted.
    """

    def __init__(
        self,
        *,
        reject_min: float = 0.0,
        reject_max: float = 0.0,
    ) -> None:
        if reject_min > reject_max:
            raise ValueError("reject_min must be less than or equal to reject_max")
        self.reject_min = reject_min
        self.reject_max = reject_max

    def admit(self, result: CurriculumResult) -> bool:
        advantages: list[float] = []
        for rollout in result.rollouts:
            if rollout.advantages is None:
                continue
            trainable = [value for sample in rollout.samples for value in sample.mask]
            advantages.extend(advantage for advantage, keep in zip(rollout.advantages, trainable, strict=True) if keep)
        if not advantages:
            return True
        return not all(self.reject_min <= advantage <= self.reject_max for advantage in advantages)
