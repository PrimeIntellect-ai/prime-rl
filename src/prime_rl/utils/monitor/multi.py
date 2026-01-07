from typing import Any

import verifiers as vf

from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor


class MultiMonitor(Monitor):
    """Monitor that wraps multiple monitors and delegates calls to all of them."""

    def __init__(self, monitors: list[Monitor]):
        self.monitors = monitors
        self.logger = get_logger()

    @property
    def history(self) -> list[dict[str, Any]]:
        if not self.monitors:
            return []
        return self.monitors[0].history

    def log(self, metrics: dict[str, Any], step: int | None = None, commit: bool | None = None) -> None:
        for monitor in self.monitors:
            try:
                monitor.log(metrics, step=step, commit=commit)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to {monitor.__class__.__name__}: {e}")

    def log_samples(self, rollouts: list[vf.State], step: int, commit: bool | None = None) -> None:
        for monitor in self.monitors:
            try:
                monitor.log_samples(rollouts=rollouts, step=step, commit=commit)
            except Exception as e:
                self.logger.warning(f"Failed to log samples to {monitor.__class__.__name__}: {e}")

    def log_final_samples(self) -> None:
        for monitor in self.monitors:
            try:
                monitor.log_final_samples()
            except Exception as e:
                self.logger.warning(f"Failed to log final samples to {monitor.__class__.__name__}: {e}")

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        for monitor in self.monitors:
            try:
                monitor.save_final_summary(filename=filename)
            except Exception as e:
                self.logger.warning(f"Failed to save final summary to {monitor.__class__.__name__}: {e}")

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        for monitor in self.monitors:
            try:
                monitor.log_distributions(distributions=distributions, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log distributions to {monitor.__class__.__name__}: {e}")

    def close(self) -> None:
        for monitor in self.monitors:
            try:
                monitor.close()
            except Exception as e:
                self.logger.warning(f"Failed to close {monitor.__class__.__name__}: {e}")
