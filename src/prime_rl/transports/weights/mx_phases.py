"""Phase timing for the mx_refit refit cycle.

A refit costs time in five places and, until this existed, prime-RL could see
only the total. Measured runs attributed roughly 18% of the framework-visible
broadcast to phases anyone could name, which left 1.7-2.8s per step unaccounted
for and made it impossible to say whether an expensive refit was slow on the
wire or slow waiting for a peer.

The gap is not an oversight in the transport so much as a property of the
client: ModelExpress carries a normalized span recorder
(``modelexpress.refit.timing``), but only its first-generation namespace
contributes spans. The ``modelexpress_rl`` client this transport is built on
records none, so activating that recorder here would emit empty cycles.

What prime-RL can time without any client change is the boundary it already
drives, and that turns out to be the whole split, because each phase sits
behind a separate call:

  publish     trainer: create the version and publish every rank's shard
  rendezvous  trainer: blocked until a generator retires the version
  discovery   receiver: waiting for every rank to reach READY
  wire        worker: stage_weight, i.e. the actual RDMA pull
  install     worker: apply_weight, i.e. PWAL and the copy into kernel storage

Emitted as one JSON line per role per step so the existing benchmark tooling
can consume it without a parser change. Splitting ``install`` further into
post-load processing versus copy needs spans inside the client and is
deliberately left to ModelExpress.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Iterator

RECORD = "mx-refit-phases-v1"


class PhaseTimer:
    """Accumulates named phase durations for one refit and emits them once."""

    def __init__(self, role: str, step: int, version_uid: str) -> None:
        self.role = role
        self.step = step
        self.version_uid = version_uid
        self.phases: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            # Recorded even when the body raises: a refit that failed after 30s
            # on the wire is exactly the case the split needs to explain, and
            # dropping the span would leave the failure looking instantaneous.
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - started)

    def payload(self) -> dict:
        """The record for this cycle, separated from emitting it so the split can
        be asserted on directly instead of scraped back out of a log sink."""
        return {
            "record": RECORD,
            "role": self.role,
            "step": self.step,
            "version_uid": self.version_uid,
            "phases_s": {name: round(value, 6) for name, value in self.phases.items()},
            "accounted_s": round(sum(self.phases.values()), 6),
        }

    def emit(self) -> None:
        line = json.dumps(self.payload())
        # The generator role runs inside a vLLM worker process, which prime-RL
        # does not own and where its logger may never have been configured --
        # and, on some worker entrypoints, where importing it at module scope
        # is itself unsafe. Hence the local import and the fallback: measurement
        # must not be the thing that takes a refit down, and a record on stdout
        # is still a record the harness can collect.
        try:
            from prime_rl.utils.logger import get_logger

            get_logger().info(line)
        except Exception:  # noqa: BLE001
            print(line, flush=True)


@contextmanager
def timed_refit(role: str, step: int, version_uid: str) -> Iterator[PhaseTimer]:
    """Time one refit cycle for ``role`` and emit the split on the way out."""
    timer = PhaseTimer(role, step, version_uid)
    try:
        yield timer
    finally:
        timer.emit()
