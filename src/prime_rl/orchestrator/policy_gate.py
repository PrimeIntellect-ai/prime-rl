"""Admission and in-flight accounting for mutable-policy inference calls."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from prime_rl.orchestrator.types import Policy


class PolicyRequestRejected(RuntimeError):
    """A request cannot safely run against the mutable policy version."""


@dataclass(frozen=True)
class SchedulingEpoch:
    """Snapshot revalidated at the dispatcher's non-yielding commit point."""

    value: int


@dataclass(frozen=True)
class PolicyUpdateToken:
    """Unique ownership of one closed-gate transition."""

    step: int
    epoch: int


class MutablePolicyGate:
    """Serialize policy mutation with every request that depends on its weights.

    Dispatcher scheduling prepares outside the lock, then revalidates a
    :class:`SchedulingEpoch` at its short commit point. Other policy I/O, such
    as OPSD prefill scoring, uses :meth:`request` for its full lifetime.
    Closing the gate prevents new work; callers can then cancel
    dispatcher-owned tasks and await :meth:`wait_idle` before pausing.
    """

    def __init__(self, policy: Policy, *, enabled: bool) -> None:
        self.policy = policy
        self.enabled = enabled
        self._admission_lock = asyncio.Lock()
        self._epoch = 0
        self._pending_token: PolicyUpdateToken | None = None
        self._active_requests = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def pending(self) -> bool:
        return self.enabled and self._pending_token is not None

    async def scheduling_epoch(self) -> SchedulingEpoch | None:
        """Take a short-lived admission snapshot before slow preparation."""
        if not self.enabled:
            return SchedulingEpoch(self._epoch)
        async with self._admission_lock:
            if self._pending_token is not None:
                return None
            return SchedulingEpoch(self._epoch)

    @asynccontextmanager
    async def scheduling_commit(self, epoch: SchedulingEpoch) -> AsyncIterator[bool]:
        """Revalidate and serialize only the scheduling state commit.

        The caller must not await inside the admitted branch. Client discovery
        and rate limiting belong before this context so an update can close the
        gate promptly.
        """
        if not self.enabled:
            yield True
            return
        async with self._admission_lock:
            yield self._pending_token is None and epoch.value == self._epoch

    @asynccontextmanager
    async def request(self, *, expected_version: int) -> AsyncIterator[None]:
        """Register one non-dispatcher policy call for its complete lifetime."""
        if not self.enabled:
            yield
            return

        async with self._admission_lock:
            if self._pending_token is not None:
                raise PolicyRequestRejected(
                    f"Mutable-policy request rejected because a policy update is pending (expected version "
                    f"{expected_version})"
                )
            if expected_version != self.policy.version:
                raise PolicyRequestRejected(
                    f"Mutable-policy request expected policy version {expected_version}, but current version is "
                    f"{self.policy.version}"
                )
            self._active_requests += 1
            self._idle.clear()

        try:
            yield
        finally:
            # No await here: even repeated cancellation must not strand the
            # active count and deadlock the mutation barrier.
            self._active_requests -= 1
            if self._active_requests == 0:
                self._idle.set()

    async def begin_update(self, *, step: int) -> PolicyUpdateToken:
        """Close admission after every in-progress scheduling commit."""
        if not self.enabled:
            return PolicyUpdateToken(step=step, epoch=self._epoch)
        async with self._admission_lock:
            if self._pending_token is not None:
                raise RuntimeError(f"A policy update is already pending while preparing step {step}")
            self._epoch += 1
            token = PolicyUpdateToken(step=step, epoch=self._epoch)
            self._pending_token = token
            return token

    async def finish_update(self, token: PolicyUpdateToken) -> None:
        """Reopen admission after a successful or proven pre-mutation failure."""
        if not self.enabled:
            return
        async with self._admission_lock:
            if self._pending_token is not token:
                raise RuntimeError(f"Policy update token for step {token.step} does not own the pending transition")
            self._pending_token = None

    async def wait_idle(self) -> None:
        """Wait until every already-admitted non-dispatcher request settles."""
        if self.enabled:
            await self._idle.wait()
