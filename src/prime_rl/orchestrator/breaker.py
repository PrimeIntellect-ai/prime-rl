"""Per-env circuit breaker for train dispatch.

An env whose episodes keep failing (sandbox API down, grader unreachable, env
server unresponsive) is paused so the other envs fill the batch. After a cooldown
the breaker lets one probe episode through: a clean probe closes it, a failed one
reopens it with a doubled cooldown."""

from __future__ import annotations

from typing import Literal

from prime_rl.configs.orchestrator import CircuitBreakerConfig

BreakerState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self.state: BreakerState = "closed"
        self.failures = 0
        """Consecutive failed episodes."""
        self.reason: str | None = None
        """Last failure seen: what broke, e.g. ``SandboxError: ...``."""
        self.cooldown = config.cooldown
        self.opened_at = 0.0
        self.probing = False
        self.server_down = False
        """Set while the env server misses health probes; no probe goes out until it answers."""

    def allows(self, now: float) -> bool:
        """Whether one more episode of this env may start now."""
        if self.state == "closed":
            return True
        if self.server_down or self.probing:
            return False
        if self.state == "open" and now - self.opened_at < self.cooldown:
            return False
        self.state = "half_open"
        return True

    def on_dispatch(self) -> None:
        if self.state == "half_open":
            self.probing = True

    def record(self, ok: bool, reason: str | None, now: float) -> BreakerState | None:
        """Count one finished episode; return the new state on a transition."""
        if ok:
            self.failures = 0
            if self.state == "closed":
                return None
            self.state, self.probing, self.cooldown = "closed", False, self.config.cooldown
            return "closed"
        self.failures += 1
        self.reason = reason
        if self.state == "half_open":
            self.cooldown = min(2 * self.cooldown, self.config.max_cooldown)
            return self.trip(reason, now)
        if self.state == "closed" and self.failures >= self.config.failures:
            return self.trip(reason, now)
        return None

    def trip(self, reason: str | None, now: float) -> BreakerState:
        self.state, self.probing, self.opened_at, self.reason = "open", False, now, reason
        return "open"
