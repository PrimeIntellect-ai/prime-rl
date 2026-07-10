"""Admission and in-flight accounting for mutable-policy inference calls."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from prime_rl.orchestrator.types import Policy


class PolicyRequestRejected(RuntimeError):
    """A request cannot safely run against the mutable policy version."""


class MutablePolicyGate:
    """Serialize policy mutation with every request that depends on its weights.

    Dispatcher scheduling holds :meth:`scheduling_admission` until a rollout
    task is registered. Other policy I/O, such as OPSD prefill scoring, uses
    :meth:`request` for its full lifetime. Closing the gate prevents new work;
    callers can then cancel dispatcher-owned tasks and await :meth:`wait_idle`
    before pausing the engine.
    """

    def __init__(self, policy: Policy, *, enabled: bool) -> None:
        self.policy = policy
        self.enabled = enabled
        self._admission_lock = asyncio.Lock()
        self._pending = False
        self._active_requests = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def pending(self) -> bool:
        return self.enabled and self._pending

    @asynccontextmanager
    async def scheduling_admission(self) -> AsyncIterator[bool]:
        """Hold the admission serialization point through scheduling commit."""
        if not self.enabled:
            yield True
            return
        async with self._admission_lock:
            yield not self._pending

    @asynccontextmanager
    async def request(self, *, expected_version: int) -> AsyncIterator[None]:
        """Register one non-dispatcher policy call for its complete lifetime."""
        if not self.enabled:
            yield
            return

        async with self._admission_lock:
            if self._pending:
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

    async def begin_update(self, *, step: int) -> None:
        """Close admission after every in-progress scheduling commit."""
        if not self.enabled:
            return
        async with self._admission_lock:
            if self._pending:
                raise RuntimeError(f"A policy update is already pending while preparing step {step}")
            self._pending = True

    async def finish_update(self) -> None:
        """Reopen admission after a successful or proven pre-mutation failure."""
        if not self.enabled:
            return
        async with self._admission_lock:
            self._pending = False

    async def wait_idle(self) -> None:
        """Wait until every already-admitted non-dispatcher request settles."""
        if self.enabled:
            await self._idle.wait()
