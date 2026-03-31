import random
from abc import ABC, abstractmethod
from typing import Any

import verifiers as vf


def sample_items_for_logging(items: list[Any], sample_ratio: float | None) -> list[Any]:
    """Apply monitor sample_ratio semantics to a batch of items.

    - ``None`` keeps the full batch.
    - ``<= 0`` logs nothing.
    - ``0 < ratio < 1`` logs a random subset with a minimum of 1 item.
    - ``>= 1`` keeps the full batch.
    """
    if sample_ratio is None:
        return items
    if sample_ratio <= 0.0:
        return []
    if sample_ratio >= 1.0 or len(items) <= 1:
        return items

    max_samples = max(1, int(len(items) * sample_ratio))
    if len(items) <= max_samples:
        return items

    return random.sample(items, max_samples)


class Monitor(ABC):
    """Base class for all monitoring implementations.

    Subclasses should initialize a `history` attribute as a list of dictionaries
    to store logged metrics.
    """

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    @abstractmethod
    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        pass

    @abstractmethod
    def log_eval_samples(self, rollouts: list[vf.RolloutOutput], env_name: str, step: int) -> None:
        pass

    @abstractmethod
    def log_final_samples(self) -> None:
        pass

    @abstractmethod
    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        pass

    @abstractmethod
    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        pass

    def close(self) -> None:
        """Close any resources held by the monitor. Override in subclasses that need cleanup."""
        pass


class NoOpMonitor(Monitor):
    """Monitor that does nothing. Used when no monitors are configured."""

    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.history.append(metrics)

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        pass

    def log_eval_samples(self, rollouts: list[vf.RolloutOutput], env_name: str, step: int) -> None:
        pass

    def log_final_samples(self) -> None:
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        pass

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        pass
