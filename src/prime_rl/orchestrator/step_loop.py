"""The orchestrator's step loop as an explicit state machine.

The orchestrator holds rollout heads and sample refs and coordinates: it takes results in,
cuts batches and hands them to the trainer. Its step loop is always in one ``Phase``, waiting
for one thing; ``progress.step`` is the step it works on.

    COLLECTING ─batch cut─▶ SHIPPING ─trainer took it─▶ SHIPPED ─▶ COLLECTING
                              │                            ├─next batch queued─▶ SHIPPING
                              │                            └─final step─▶ FINISHING ─final policy applied─▶ DRAINING
                              └─past max_steps─────────────────────────────────────────────────────────▶ DRAINING
    DRAINING ─nothing in flight─▶ STOPPED

A cut batch with nothing to train is skipped: ``SHIPPING`` moves on to the next queued batch,
or back to ``COLLECTING``.

Only the step loop moves the phase. It holds the pipeline lock while it moves and releases it
while it waits, so a rollout checkpoint, which reads the pipeline with the lock held, finds the
loop in a phase and never between two: a cut batch is in the sink or in ``StepLoop.batch``, and
a batch the trainer took is in neither.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from prime_rl.utils.logger import format_time, get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainBatch


class Phase(StrEnum):
    COLLECTING = "collecting"
    """Waits for a dispatcher result. Each one goes to its sink; a train sink that cuts a batch
    moves the loop to ``SHIPPING``."""
    SHIPPING = "shipping"
    """Waits for every trainer rank to take the cut batch (``StepLoop.batch``)."""
    SHIPPED = "shipped"
    """The trainer took the batch and ``progress.step`` advanced. The step's bookkeeping runs
    before the loop moves on."""
    FINISHING = "finishing"
    """The final step shipped. Waits for inference to apply the policy the trainer trains on it,
    which triggers the final evals: they measure the final weights."""
    DRAINING = "draining"
    """Schedules no new train work and waits for the work in flight to finish, so triggered evals
    complete. Train results are discarded."""
    STOPPED = "stopped"
    """The pipeline is empty."""


TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.COLLECTING: frozenset({Phase.SHIPPING}),
    Phase.SHIPPING: frozenset({Phase.SHIPPED, Phase.SHIPPING, Phase.COLLECTING, Phase.DRAINING}),
    Phase.SHIPPED: frozenset({Phase.COLLECTING, Phase.SHIPPING, Phase.FINISHING}),
    Phase.FINISHING: frozenset({Phase.DRAINING}),
    Phase.DRAINING: frozenset({Phase.STOPPED}),
    Phase.STOPPED: frozenset(),
}


@dataclass
class StepLoop:
    phase: Phase = Phase.COLLECTING
    batch: TrainBatch | None = None
    """The batch cut for ``progress.step``, while ``SHIPPING``."""
    since: float = field(default_factory=time.perf_counter)
    """When the loop entered its phase."""

    def move(self, phase: Phase, batch: TrainBatch | None = None) -> None:
        if phase not in TRANSITIONS[self.phase]:
            raise RuntimeError(f"The step loop cannot move from {self.phase} to {phase}")
        if (phase is Phase.SHIPPING) != (batch is not None):
            raise ValueError(f"The step loop holds a batch exactly while shipping, not in {phase}")
        now = time.perf_counter()
        get_logger().debug(f"Step loop {self.phase} -> {phase} after {format_time(now - self.since)}")
        self.phase, self.batch, self.since = phase, batch, now

    @property
    def draining(self) -> bool:
        """Whether the loop schedules no new train work: it drains or has drained."""
        return self.phase in (Phase.DRAINING, Phase.STOPPED)

    @property
    def unshipped_rollouts(self) -> int:
        """Rollouts of the cut batch, which the trainer has not taken yet."""
        return self.batch.cohort.num_traces if self.batch is not None else 0
