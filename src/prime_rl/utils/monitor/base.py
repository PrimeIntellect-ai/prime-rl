import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any


class Monitor(ABC):
    """Base class for all monitoring implementations."""

    @property
    @abstractmethod
    def history(self) -> list[dict[str, Any]]:
        """Returns the history of logged metrics."""
        pass

    @abstractmethod
    def log(self, metrics: dict[str, Any]) -> None:
        """Log metrics to the monitoring platform."""
        pass

    @abstractmethod
    def log_samples(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        rollouts_per_problem: int,
        step: int,
    ) -> None:
        """Log prompt/response samples."""
        pass

    @abstractmethod
    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        """Log distributions (e.g., rewards, advantages)."""
        pass

    @abstractmethod
    def log_final_samples(self) -> None:
        """Log final samples at the end of training."""
        pass

    @abstractmethod
    def log_final_distributions(self) -> None:
        """Log final distributions at the end of training."""
        pass

    @abstractmethod
    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary."""
        pass

    @staticmethod
    def select_sample_problems(
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rollouts_per_problem: int,
    ) -> dict[str, int]:
        """Select 3 problems to log: min_len, max_len, and random.

        Args:
            input_tokens: List of input token sequences
            output_tokens: List of output token sequences
            rollouts_per_problem: Number of rollouts per problem

        Returns:
            Dictionary with keys "min_len", "max_len", "random" mapping to problem IDs
        """
        batch_size = len(input_tokens)
        num_problems = batch_size // rollouts_per_problem

        # Compute per-problem statistics
        per_problem_tokens: dict[int, list[list[int]]] = defaultdict(list)
        tokens = [input_tokens[i] + output_tokens[i] for i in range(batch_size)]
        for i, t in enumerate(tokens):
            problem_id = i // rollouts_per_problem
            per_problem_tokens[problem_id].append(t)
        assert len(per_problem_tokens) == num_problems
        assert list(per_problem_tokens.keys()) == list(range(num_problems))

        per_problem_seq_len = {
            problem_id: sum(len(t) for t in tokens) / len(tokens) for problem_id, tokens in per_problem_tokens.items()
        }
        min_len_problem_id = min(per_problem_seq_len.items(), key=lambda kv: kv[1])[0]
        max_len_problem_id = max(per_problem_seq_len.items(), key=lambda kv: kv[1])[0]
        random_problem_id = random.choice(list(range(num_problems)))

        return {
            "min_len": min_len_problem_id,
            "max_len": max_len_problem_id,
            "random": random_problem_id,
        }

    @staticmethod
    def should_log_samples(
        config: Any | None,
        step: int,
        last_log_step: int = -1,
    ) -> bool:
        """Check if samples should be logged at this step.

        Args:
            config: Monitor config with log_extras attribute
            step: Current step
            last_log_step: Last step samples were logged

        Returns:
            True if samples should be logged, False otherwise
        """
        if (
            not config
            or not config.log_extras
            or not config.log_extras.samples
            or step % config.log_extras.interval != 0
        ):
            return False
        if last_log_step >= step:
            return False
        return True

    @staticmethod
    def should_log_distributions(
        config: Any | None,
        step: int,
        last_log_step: int = -1,
    ) -> bool:
        """Check if distributions should be logged at this step.

        Args:
            config: Monitor config with log_extras attribute
            step: Current step
            last_log_step: Last step distributions were logged

        Returns:
            True if distributions should be logged, False otherwise
        """
        if (
            not config
            or not config.log_extras
            or not config.log_extras.distributions
            or step % config.log_extras.interval != 0
        ):
            return False
        if last_log_step >= step:
            return False
        return True


class NoOpMonitor(Monitor):
    """Monitor that does nothing. Used when no monitors are configured."""

    def __init__(self):
        self._history: list[dict[str, Any]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Returns empty history."""
        return self._history

    def log(self, metrics: dict[str, Any]) -> None:
        """No-op: does nothing."""
        self._history.append(metrics)

    def log_samples(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        rollouts_per_problem: int,
        step: int,
    ) -> None:
        """No-op: does nothing."""
        pass

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        """No-op: does nothing."""
        pass

    def log_final_samples(self) -> None:
        """No-op: does nothing."""
        pass

    def log_final_distributions(self) -> None:
        """No-op: does nothing."""
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """No-op: does nothing."""
        pass

