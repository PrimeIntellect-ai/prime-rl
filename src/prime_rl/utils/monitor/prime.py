import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import PrimeMonitorConfig, PrimeMonitorWithExtrasConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor
from prime_rl.utils.pydantic_config import BaseSettings


class PrimeMonitor(Monitor):
    """Logs to Prime Intellect API."""

    def __init__(
        self,
        config: PrimeMonitorConfig | PrimeMonitorWithExtrasConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseSettings | None = None,
    ):
        self.config = config
        self.logger = get_logger()
        self._history: list[dict[str, Any]] = []
        self.output_dir = output_dir
        self.tokenizer = tokenizer

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0
        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return
        assert config is not None
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")

        # Get API key from environment
        api_key = os.getenv(config.api_key_var)
        if not api_key:
            self.logger.warning(f"API key not found in environment variable {config.api_key_var}. PrimeMonitor will not be able to upload data.")
            self.enabled = False
            return

        self.api_key = api_key
        self.api_endpoint = config.api_endpoint
        self.run_name = config.run_name or "prime-rl-run"

        # Get run_id from environment variable
        run_id = os.getenv("RUN_ID")
        if not run_id:
            self.logger.warning("RUN_ID environment variable not set. PrimeMonitor will not be able to upload data.")
            self.enabled = False
            return
        self.run_id = run_id

        # Optionally, initialize sample logging attributes
        if config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.samples = []

            if config.log_extras.distributions:
                self.last_log_distributions_step = -1
                self.distributions = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Returns the history of logged metrics."""
        return self._history

    def _make_request(self, endpoint: str, data: dict[str, Any]) -> None:
        """Make a POST request to the Prime Intellect API."""
        if not self.enabled:
            return
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        try:
            # Construct endpoint path (no run_id in path)
            full_endpoint = f"{self.api_endpoint}/{endpoint}"
            response = requests.post(
                full_endpoint,
                headers=headers,
                json=data,
                timeout=30,
            )
            response.raise_for_status()
        except Exception as e:
            self.logger.warning(f"Failed to upload to Prime Intellect API: {e}")

    def log(self, metrics: dict[str, Any]) -> None:
        self._history.append(metrics)
        if not self.is_master:
            return
        if not self.enabled:
            return
        self._make_request(
            "metrics",
            {
                "run_id": self.run_id,
                "metrics": metrics,
            },
        )

    def log_samples(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        rollouts_per_problem: int,
        step: int,
    ) -> None:
        """Log prompt/response samples to Prime Intellect API.

        Args:
            input_tokens: List of input token sequences
            output_tokens: List of output token sequences
            rewards: List of rewards for each sample
            advantages: List of advantages for each sample
            rollouts_per_problem: Number of rollouts per problem
            step: Current training step
        """
        if not self.is_master:
            return
        if not Monitor.should_log_samples(self.config, step, self.last_log_samples_step):
            return
        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.logger is not None, "Logger is required for sample logging"
        self.logger.info(f"Logging samples to Prime Intellect API at step {step}")
        start_time = time.perf_counter()

        # Use shared sample selection logic
        problem_ids = Monitor.select_sample_problems(input_tokens, output_tokens, rollouts_per_problem)
        self.logger.debug(f"Logging samples for problems: {problem_ids}")

        # Prepare samples for selected problems
        samples = []
        for tag, problem_id in problem_ids.items():
            start_idx = problem_id * rollouts_per_problem
            for sample_id in range(start_idx, start_idx + rollouts_per_problem):
                sample = {
                    "step": step,
                    "tag": tag,
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "num_input_tokens": len(input_tokens[sample_id]),
                    "num_output_tokens": len(output_tokens[sample_id]),
                    "input_tokens": str(input_tokens[sample_id]),
                    "output_tokens": str(output_tokens[sample_id]),
                    "prompt": self.tokenizer.decode(input_tokens[sample_id]),
                    "completion": self.tokenizer.decode(output_tokens[sample_id]),
                    "reward": float(rewards[sample_id]),
                    "advantage": float(advantages[sample_id]),
                }
                samples.append(sample)
                self.samples.append(sample)

        # Upload samples
        self._make_request(
            "samples",
            {
                "run_id": self.run_id,
                "step": step,
                "samples": samples,
            },
        )
        self.last_log_samples_step = step
        self.logger.debug(f"Logged samples at step {step} to Prime Intellect API in {time.perf_counter() - start_time:.2f}s")

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if not self.is_master:
            return
        if not Monitor.should_log_distributions(self.config, step, self.last_log_distributions_step):
            return
        assert self.logger is not None
        self.logger.info(f"Logging distributions for keys {list(distributions.keys())} to Prime Intellect API at step {step}")

        row = {"step": step, **distributions}
        self.distributions.append(row)

        # Upload distributions
        start_time = time.perf_counter()
        self._make_request(
            "distributions",
            {
                "run_id": self.run_id,
                "step": step,
                "distributions": distributions,
            },
        )
        self.last_log_distributions_step = step
        self.logger.debug(
            f"Logged distributions at step {step} to Prime Intellect API in {time.perf_counter() - start_time:.2f}s"
        )

    def log_final_samples(self) -> None:
        """Log final samples to Prime Intellect API."""
        if not self.is_master:
            return
        if not self.config or not self.config.log_extras or not self.config.log_extras.samples:
            return
        self.logger.info("Logging final samples to Prime Intellect API")
        # Get step from last sample if available, otherwise use 0
        step = self.samples[-1].get("step", 0) if self.samples else 0
        self._make_request(
            "final-samples",
            {
                "run_id": self.run_id,
                "step": step,
                "samples": self.samples,
            },
        )

    def log_final_distributions(self) -> None:
        """Log final distributions to Prime Intellect API."""
        if not self.is_master:
            return
        if not self.config or not self.config.log_extras or not self.config.log_extras.distributions:
            return
        self.logger.info("Logging final distributions to Prime Intellect API")
        # Get step from last distribution if available, otherwise use 0
        step = self.distributions[-1].get("step", 0) if self.distributions else 0
        # Extract distributions dict (remove step key)
        distributions_dict = {k: v for k, v in self.distributions[-1].items() if k != "step"} if self.distributions else {}
        self._make_request(
            "final-distributions",
            {
                "run_id": self.run_id,
                "step": step,
                "distributions": distributions_dict,
            },
        )

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to Prime Intellect API."""
        if not self.is_master or not self.enabled:
            return
        self.logger.info("Saving final summary to Prime Intellect API")
        # For Prime Intellect, we can upload the summary as part of finalizing the run
        self._make_request(
            "finalize",
            {
                "run_id": self.run_id,
                "summary": self._history[-1] if self._history else {},
            },
        )
