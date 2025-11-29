from typing import Any

from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor


class MultiMonitor(Monitor):
    """Monitor that wraps multiple monitors and delegates calls to all of them."""

    def __init__(self, monitors: list[Monitor]):
        self.monitors = monitors
        self.logger = get_logger()

    @property
    def history(self) -> list[dict[str, Any]]:
        """Returns aggregated history from all monitors."""
        # Concatenate histories from all monitors (they should be identical)
        if not self.monitors:
            return []
        # Use first monitor's history as primary, others should match
        return self.monitors[0].history

    def log(self, metrics: dict[str, Any]) -> None:
        """Log metrics to all monitors."""
        for monitor in self.monitors:
            try:
                monitor.log(metrics)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to {monitor.__class__.__name__}: {e}")

    def log_samples(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        rollouts_per_problem: int,
        step: int,
    ) -> None:
        """Log samples to all monitors."""
        for monitor in self.monitors:
            try:
                monitor.log_samples(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    rewards=rewards,
                    advantages=advantages,
                    rollouts_per_problem=rollouts_per_problem,
                    step=step,
                )
            except Exception as e:
                self.logger.warning(f"Failed to log samples to {monitor.__class__.__name__}: {e}")

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        """Log distributions to all monitors."""
        for monitor in self.monitors:
            try:
                monitor.log_distributions(distributions=distributions, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log distributions to {monitor.__class__.__name__}: {e}")

    def log_final_samples(self) -> None:
        """Log final samples to all monitors."""
        for monitor in self.monitors:
            try:
                monitor.log_final_samples()
            except Exception as e:
                self.logger.warning(f"Failed to log final samples to {monitor.__class__.__name__}: {e}")

    def log_final_distributions(self) -> None:
        """Log final distributions to all monitors."""
        for monitor in self.monitors:
            try:
                monitor.log_final_distributions()
            except Exception as e:
                self.logger.warning(f"Failed to log final distributions to {monitor.__class__.__name__}: {e}")

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to all monitors."""
        for monitor in self.monitors:
            try:
                monitor.save_final_summary(filename=filename)
            except Exception as e:
                self.logger.warning(f"Failed to save final summary to {monitor.__class__.__name__}: {e}")

