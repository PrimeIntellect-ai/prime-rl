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
        pass

    @abstractmethod
    def log_samples(self, rollouts: list[vf.State], step: int) -> None:
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


class NoOpMonitor(Monitor):
    """Monitor that does nothing. Used when no monitors are configured."""

    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any]) -> None:
        self.history.append(metrics)

    def log_samples(self, rollouts: list[vf.State], step: int) -> None:
        pass

    def log_final_samples(self) -> None:
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        pass

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        pass

