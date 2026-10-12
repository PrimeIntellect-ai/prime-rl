"""Per-env circuit breaker for train dispatch.

An env whose episodes keep failing on infrastructure (sandbox API down, model
interception broken, env server unresponsive) is paused so the other envs fill
the batch. After a cooldown the breaker lets one probe episode through: a clean
probe closes it, a failed one reopens it with a doubled cooldown."""

from __future__ import annotations

import uuid
from typing import Literal

from prime_rl.configs.orchestrator import CircuitBreakerConfig

BreakerState = Literal["closed", "open"]


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self.state: BreakerState = "closed"
        self.failed_groups: set[uuid.UUID] = set()
        """Groups with a failed episode since the last success. A broken task fails all its group members at once, so
        counting groups keeps one bad task from opening the breaker."""
        self.reason: str | None = None
        """Last failure seen: what broke, e.g. ``SandboxError: ...``."""
        self.cooldown = config.cooldown
        self.opened_at = 0.0
        self.probing = False
        """Whether the probe episode of an open breaker is in flight."""
        self.server_down = False
        """Set while the env server misses health probes; no probe goes out until it answers."""

    def allows(self, now: float) -> bool:
        """Whether one more episode of this env may start now."""
        if self.state == "closed":
            return True
        if self.server_down or self.probing:
            return False
        return now - self.opened_at >= self.cooldown

    def on_dispatch(self) -> None:
        if self.state == "open":
            self.probing = True

    def record(self, ok: bool, reason: str | None, now: float, group_id: uuid.UUID) -> BreakerState | None:
        """Count one finished episode; return the new state on a transition."""
        if ok:
            self.failed_groups.clear()
            if self.state == "closed":
                return None
            self.state, self.probing, self.cooldown = "closed", False, self.config.cooldown
            return "closed"
        self.failed_groups.add(group_id)
        self.reason = reason
        if self.probing:
            self.cooldown = min(2 * self.cooldown, self.config.max_cooldown)
            return self.trip(reason, now)
        if self.state == "closed" and len(self.failed_groups) >= self.config.failures:
            return self.trip(reason, now)
        return None

    def trip(self, reason: str | None, now: float) -> BreakerState:
        self.state, self.probing, self.opened_at, self.reason = "open", False, now, reason
        return "open"
