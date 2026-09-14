"""Structured phase timing for ModelExpress refits."""

from __future__ import annotations

import json
import os
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
        self.marks: dict[str, float] = {}
        self.started = time.perf_counter()
        self.elapsed: float | None = None
        self.status = "running"

    def identify(self, version_uid: str) -> None:
        """Set the version ID after discovery."""
        self.version_uid = version_uid

    def mark(self, name: str, value: float) -> None:
        """Record a value that is not part of accounted phase time."""
        self.marks[name] = value

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - started)

    def payload(self) -> dict:
        """Build the structured timing record."""
        phases = {name: round(value, 6) for name, value in self.phases.items()}
        payload = {
            "record": RECORD,
            "role": self.role,
            "step": self.step,
            "version_uid": self.version_uid,
            "status": self.status,
            "elapsed_s": round(self.elapsed if self.elapsed is not None else time.perf_counter() - self.started, 6),
            "phases_s": phases,
            "accounted_s": round(sum(phases.values()), 6),
        }
        if self.marks:
            payload["marks"] = {name: round(value, 6) for name, value in self.marks.items()}
        return payload

    def emit(self) -> None:
        line = json.dumps(self.payload())
        if os.environ.get("MX_REFIT_TIMING_STDOUT") == "1":
            print(line, flush=True)
            return
        try:
            from prime_rl.utils.logger import get_logger
        except ImportError:
            print(line, flush=True)
        else:
            get_logger().info(line)


@contextmanager
def timed_refit(role: str, step: int, version_uid: str) -> Iterator[PhaseTimer]:
    """Time one refit cycle for ``role`` and emit the split on the way out."""
    timer = PhaseTimer(role, step, version_uid)
    try:
        yield timer
    except BaseException:
        timer.status = "failed"
        raise
    else:
        timer.status = "complete"
    finally:
        timer.elapsed = time.perf_counter() - timer.started
        timer.emit()
