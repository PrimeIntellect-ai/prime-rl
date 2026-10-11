"""Run-health advisories: warnings for a run that degrades without crashing.

Each rule watches the metrics whose key matches its pattern, against that metric's
own recent history. Advisories only warn and report ``health/*`` metrics; nothing
acts on them."""

from __future__ import annotations

import re
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

from prime_rl.utils.logger import get_logger

WINDOW = 20
"""Steps of history a metric's baseline (median) is taken over."""
WARMUP = 10
"""Steps of history a metric needs before a rule checks it."""


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: str
    kind: Literal["drop", "rise", "held"]
    factor: float = 0.0
    floor: float = 0.0
    """drop: fires below ``factor`` x baseline when the baseline is above ``floor``.
    rise: fires above ``factor`` x baseline + ``floor``."""
    seconds: float = 0.0
    """held: fires once the metric has stayed above zero this long."""


TRAINER_RULES = (
    Rule("entropy_collapse", r"entropy/all/mean", "drop", factor=0.2, floor=1e-3),
    Rule("entropy_blowup", r"entropy/all/mean", "rise", factor=3.0),
    Rule("mismatch_kl_rising", r"mismatch_kl/all/mean", "rise", factor=3.0, floor=1e-3),
)
ORCHESTRATOR_RULES = (
    Rule("reward_collapse", r"train/agg/all/[^/]+/reward/mean", "drop", factor=0.5, floor=0.05),
    Rule("truncation_spike", r"train/agg/all/[^/]+/is_truncated/mean", "rise", factor=2.0, floor=0.1),
    Rule("error_spike", r"train/agg/all/has_error/mean", "rise", factor=2.0, floor=0.1),
    Rule("off_policy_rising", r"off_policy/mean", "rise", factor=2.0, floor=1.0),
    Rule("breaker_open", r"dispatcher/breaker_open/.+", "held", seconds=900),
)


class RunHealth:
    def __init__(self, rules: tuple[Rule, ...]) -> None:
        self.rules = rules
        self.history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.held_since: dict[str, float] = {}
        self.firing: set[tuple[str, str]] = set()

    def _fires(self, rule: Rule, key: str, value: float, now: float) -> bool:
        if rule.kind == "held":
            if value <= 0:
                self.held_since.pop(key, None)
                return False
            return now - self.held_since.setdefault(key, now) >= rule.seconds
        history = self.history[key]
        if len(history) < WARMUP:
            return False
        baseline = statistics.median(history)
        if rule.kind == "drop":
            return baseline > rule.floor and value < rule.factor * baseline
        return value > rule.factor * baseline + rule.floor

    def check(self, metrics: dict[str, float]) -> dict[str, float]:
        """Check one step's metrics; returns ``health/<rule>`` flags and the
        ``health/warnings`` count of firing (rule, metric) pairs."""
        now = time.monotonic()
        firing: set[tuple[str, str]] = set()
        watched: dict[str, float] = {}
        for rule in self.rules:
            for key, value in metrics.items():
                if not re.fullmatch(rule.pattern, key):
                    continue
                value = float(value)
                watched[key] = value
                if not self._fires(rule, key, value, now):
                    continue
                firing.add((rule.name, key))
                if (rule.name, key) not in self.firing:
                    history = self.history[key]
                    context = (
                        f"recent median {statistics.median(history):.4g}" if history else f"for {rule.seconds:.0f}s"
                    )
                    get_logger().warning(f"Run health: {rule.name} ({key}={value:.4g}, {context})")
        for name, key in self.firing - firing:
            get_logger().info(f"Run health: {name} cleared ({key})")
        self.firing = firing
        for key, value in watched.items():
            self.history[key].append(value)
        out = {f"health/{rule.name}": float(any(name == rule.name for name, _ in firing)) for rule in self.rules}
        out["health/warnings"] = float(len(firing))
        return out
