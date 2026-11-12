"""The trainer's capacity: the train rollouts it would consume per second if a batch were always ready.

Each trainer rank stamps its batch request with the time it asked (``BatchRequest.requested_at``). From one shipped
batch to the next, the trainer is busy until it asks (training, broadcasting weights, checkpointing) and waits after:
rollouts over busy seconds is what it sustains, whether or not inference keeps up.
"""

import time
from collections import deque
from dataclasses import dataclass

WINDOW_STEPS = 8
"""Trainer steps the capacity averages over."""

WARMUP_SHIPS = 2
"""Ships carrying the trainer's warmup, left out."""


@dataclass(frozen=True)
class Step:
    rollouts: int
    period: float
    busy: float


class TrainerCapacity:
    def __init__(self) -> None:
        self.steps: deque[Step] = deque(maxlen=WINDOW_STEPS)
        self.ships = 0
        self.shipped_at: float | None = None

    def shipped(self, rollouts: int, requested_at: float | None, now: float | None = None) -> None:
        """A batch of ``rollouts`` finished train rollouts shipped at ``now``, which the trainer requested at
        ``requested_at`` (wall clock; None when the transport did not report it)."""
        now = time.time() if now is None else now
        previous, self.shipped_at = self.shipped_at, now
        self.ships += 1
        if previous is None or self.ships <= WARMUP_SHIPS or requested_at is None or now <= previous:
            return
        period = now - previous
        self.steps.append(Step(rollouts, period, min(max(requested_at - previous, 0.0), period)))

    @property
    def rate(self) -> float | None:
        """Train rollouts per busy second over the window; None until a step was measured."""
        busy = sum(step.busy for step in self.steps)
        return sum(step.rollouts for step in self.steps) / busy if busy > 0 else None

    @property
    def busy(self) -> float | None:
        """The trainer's busy share over the window."""
        period = sum(step.period for step in self.steps)
        return sum(step.busy for step in self.steps) / period if period > 0 else None

    @property
    def waited(self) -> float:
        """The share of the last step the trainer waited for its batch."""
        return 1 - self.steps[-1].busy / self.steps[-1].period if self.steps else 0.0

    def gauges(self) -> dict[str, float]:
        if self.rate is None:
            return {}
        return {"trainer_capacity/rollouts_per_s": self.rate, "trainer_capacity/busy": self.busy}
