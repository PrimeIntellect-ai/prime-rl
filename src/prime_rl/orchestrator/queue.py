"""Queue: finished train groups waiting for the trainer.

Groups arrive one at a time from the sink. Those with samples wait in arrival
order until ``batch_size`` traces are ready; those without are held for the next
cut's accounting, so every batch describes the whole window since the last one.
Queued groups past ``max_off_policy_steps`` are voided when the step advances —
the hard bound on trained staleness; the dispatcher's in-flight cancel only saves
compute. The gauges say which side limits progress: a queue pinned at zero means
rollouts cannot keep up, one pinned at the target means the trainer is the
constraint and staleness climbs.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable

from prime_rl.orchestrator.types import Batch, Group, cancel, is_cancelled
from prime_rl.utils.logger import get_logger


class Queue:
    def __init__(self, *, batch_size: int, max_off_policy_steps: int) -> None:
        self.batch_size = batch_size
        self.max_off_policy_steps = max_off_policy_steps
        self._step: Callable[[], int] = lambda: 1
        self._on_batch: Callable[[Batch], Awaitable[None]] | None = None

        self.ready: deque[Group] = deque()
        """Groups with samples, in arrival order."""
        self.window: list[Group] = []
        """Groups without samples since the last cut, reported with the next batch."""
        self.dropped_stale = 0
        self._swept_step = 0
        self._dry_traces = 0
        self._dry_reported = 0

    def bind(self, *, step: Callable[[], int], on_batch: Callable[[Batch], Awaitable[None]]) -> None:
        self._step = step
        self._on_batch = on_batch

    async def put(self, group: Group) -> None:
        self.sweep()
        if self.is_stale(group):
            self.void(group)
        if group.samples:
            self.ready.append(group)
            self._dry_traces = self._dry_reported = 0
        else:
            self.window.append(group)
            self.warn_if_dry(group)
        if self.size >= self.batch_size:
            assert self._on_batch is not None, "Queue.on_batch is not bound"
            await self._on_batch(self.cut())

    def cut(self) -> Batch:
        groups, taken = list(self.window), 0
        while taken < self.batch_size:
            group = self.ready.popleft()
            if taken + group.num_samples > self.batch_size:
                group, rest = group.split(self.batch_size - taken)
                self.ready.appendleft(rest)
            groups.append(group)
            taken += group.num_samples
        self.window = []
        return Batch(step=self._step(), groups=groups)

    # ── staleness ──────────────────────────────────────────────────────────

    def is_stale(self, group: Group) -> bool:
        """The batch being collected trains v{step-1}; a group dispatched more than
        ``max_off_policy_steps`` versions before it can never train."""
        return group.version is not None and (self._step() - 1) - group.version > self.max_off_policy_steps

    def sweep(self) -> None:
        """Queued groups only age when the step advances, so one sweep per step."""
        step = self._step()
        if step == self._swept_step:
            return
        self._swept_step = step
        stale = [group for group in self.ready if self.is_stale(group)]
        if not stale:
            return
        self.ready = deque(group for group in self.ready if group not in stale)
        dropped = sum(group.num_samples for group in stale)
        for group in stale:
            self.void(group)
            self.window.append(group)
        get_logger().warning(
            f"Dropped {dropped} queued traces past max_off_policy_steps={self.max_off_policy_steps}. "
            "Consider increasing it to avoid this."
        )

    def void(self, group: Group) -> None:
        for episode in group.episodes:
            if not is_cancelled(episode):
                cancel(episode, "stale")
        self.dropped_stale += group.num_samples
        group.samples = {}

    def warn_if_dry(self, group: Group) -> None:
        """A run whose groups keep producing nothing to train on says so once per
        batch-equivalent of traces, rather than filling a batch never."""
        self._dry_traces += len(group.traces) or len(group.episodes)
        windows = self._dry_traces // self.batch_size
        if windows > self._dry_reported:
            self._dry_reported = windows
            get_logger().warning(
                f"No train payload after {self._dry_traces} finalized traces ({windows} zero-output batch equivalents)"
            )

    # ── observability ──────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return sum(group.num_samples for group in self.ready)

    def staleness(self) -> list[int]:
        step = self._step()
        return [
            max(0, (step - 1) - group.version) if group.version is not None else 0
            for group in self.ready
            for _ in range(group.num_samples)
        ]

    def status(self) -> str:
        size = self.size
        part = f"Train batch {size}/{self.batch_size} ({size / self.batch_size:.1%})"
        by_env: dict[str, int] = {}
        for group in self.ready:
            by_env[group.env] = by_env.get(group.env, 0) + group.num_samples
        if len(by_env) > 1:
            part += " (" + ", ".join(f"{name}={count}" for name, count in sorted(by_env.items())) + ")"
        return part

    def gauges(self) -> dict[str, float]:
        staleness = self.staleness()
        return {
            "queue/size": float(self.size),
            "queue/fill": self.size / self.batch_size,
            "queue/staleness/max": float(max(staleness, default=0)),
            "queue/staleness/mean": sum(staleness) / len(staleness) if staleness else 0.0,
            "queue/dropped_stale": float(self.dropped_stale),
        }
