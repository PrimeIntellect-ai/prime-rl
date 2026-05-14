"""Trial-level early stopping for sweep studies.

The tracker observes each completed trial's objective value and decides
whether the controller should halt remaining work. All decisions are made
between trials (not in-flight): a worse-than-threshold value or a long enough
run of non-improving trials triggers a halt that prevents new submissions
while in-flight trials finish naturally.
"""

import threading
from dataclasses import dataclass
from typing import Literal

from prime_rl.configs.sweep import (
    EarlyStoppingConfig,
    ObjectiveConfig,
    PatienceStoppingConfig,
    ThresholdStoppingConfig,
)
from prime_rl.sweep.metrics import coerce_finite_float


@dataclass(frozen=True)
class TrialOutcome:
    trial_id: str
    label: str
    objective: float | None


@dataclass
class TrialOutcomeSummary:
    completed: int
    best_trial_id: str | None
    best_value: float | None
    halted_by_early_stopping: bool
    halt_reason: Literal["threshold", "patience"] | None


class TrialOutcomeTracker:
    """Thread-safe tracker over completed trials' objectives."""

    def __init__(
        self,
        objective: ObjectiveConfig | None,
        early_stopping: EarlyStoppingConfig | None,
    ):
        self._objective = objective
        self._early_stopping = early_stopping
        self._lock = threading.Lock()
        self._completed = 0
        self._best_value: float | None = None
        self._best_trial_id: str | None = None
        self._best_label: str | None = None
        self._steps_without_improvement = 0
        self._halted = False
        self._halt_reason: Literal["threshold", "patience"] | None = None
        self._outcomes: list[TrialOutcome] = []

    def observe(self, outcome: TrialOutcome) -> bool:
        """Record an outcome and return whether the study should halt."""
        with self._lock:
            self._outcomes.append(outcome)
            value = coerce_finite_float(outcome.objective)
            if value is None:
                # Missing metrics do not advance early-stopping decisions.
                return self._halted

            self._completed += 1
            if self._is_improvement(value):
                self._best_value = value
                self._best_trial_id = outcome.trial_id
                self._best_label = outcome.label
                self._steps_without_improvement = 0
            else:
                self._steps_without_improvement += 1

            if not self._halted and self._should_halt(value):
                self._halted = True

            return self._halted

    def _is_improvement(self, value: float) -> bool:
        if self._objective is None or self._best_value is None:
            return self._best_value is None
        if self._objective.direction == "maximize":
            return value > self._best_value
        return value < self._best_value

    def _should_halt(self, value: float) -> bool:
        config = self._early_stopping
        if config is None or self._objective is None:
            return False
        if self._completed < config.min_trials:
            return False
        if isinstance(config, ThresholdStoppingConfig):
            worse = (
                value < config.threshold
                if self._objective.direction == "maximize"
                else value > config.threshold
            )
            if worse:
                self._halt_reason = "threshold"
                return True
            return False
        if isinstance(config, PatienceStoppingConfig):
            if self._steps_without_improvement >= config.patience:
                self._halt_reason = "patience"
                return True
            return False
        return False

    @property
    def halted(self) -> bool:
        with self._lock:
            return self._halted

    def summary(self) -> TrialOutcomeSummary:
        with self._lock:
            return TrialOutcomeSummary(
                completed=self._completed,
                best_trial_id=self._best_trial_id,
                best_value=self._best_value,
                halted_by_early_stopping=self._halted,
                halt_reason=self._halt_reason,
            )

    @property
    def best_label(self) -> str | None:
        with self._lock:
            return self._best_label
