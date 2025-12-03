from abc import ABC, abstractmethod
from typing import Any

import verifiers as vf


class Monitor(ABC):
    """Base class for all monitoring implementations.
    
    Subclasses should initialize a `history` attribute as a list of dictionaries
    to store logged metrics.
    """

    @abstractmethod
    def log(self, metrics: dict[str, Any]) -> None:
        """Log metrics to the monitoring platform."""
        pass

    @abstractmethod
    def log_samples(self, rollouts: list[vf.State], step: int) -> None:
        """Log prompt/response samples."""
        pass

    @abstractmethod
    def log_final_samples(self) -> None:
        """Log final samples at the end of training."""
        pass

    @abstractmethod
    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary."""
        pass


class NoOpMonitor(Monitor):
    """Monitor that does nothing. Used when no monitors are configured."""

    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any]) -> None:
        """No-op: does nothing."""
        self.history.append(metrics)

    def log_samples(self, rollouts: list[vf.State], step: int) -> None:
        """No-op: does nothing."""
        pass

    def log_final_samples(self) -> None:
        """No-op: does nothing."""
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """No-op: does nothing."""
        pass

