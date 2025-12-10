import os
import time
from pathlib import Path
from typing import Any

import requests
import verifiers as vf
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
        self.history: list[dict[str, Any]] = []
        self.output_dir = output_dir

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
        if config is not None and isinstance(config, PrimeMonitorWithExtrasConfig) and config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.tokenizer = tokenizer
            if config.log_extras.distributions:
                self.last_log_distributions_step = -1

    def log(self, metrics: dict[str, Any]) -> None:
        self.history.append(metrics)
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

    def log_samples(self, rollouts: list[vf.State], step: int) -> None:
        """Logs rollouts to Prime Intellect API."""
        if not self.is_master:
            return
        if not self.enabled:
            return
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            # Do not log samples if not enabled or not log interval step
            return

        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for sample logging"

        self.logger.info(f"Logging samples to Prime Intellect API at step {step}")
        start_time = time.perf_counter()

        # Prepare samples for API
        samples = []
        for rollout in rollouts:
            # Extract prompt and completion separately from the last trajectory step
            last_step = rollout["trajectory"][-1]
            prompt_messages = last_step["prompt"]
            completion_messages = last_step["completion"]
            
            # Serialize full trajectory array (excluding large response objects and token arrays)
            trajectory_data = []
            for traj_step in rollout["trajectory"]:
                trajectory_data.append({
                    "prompt": traj_step["prompt"],
                    "completion": traj_step["completion"],
                    "reward": traj_step.get("reward"),
                    "advantage": traj_step.get("advantage"),
                    "extras": traj_step.get("extras", {}),
                    "num_input_tokens": len(traj_step.get("tokens", {}).get("prompt_ids", [])) if traj_step.get("tokens") else None,
                    "num_output_tokens": len(traj_step.get("tokens", {}).get("completion_ids", [])) if traj_step.get("tokens") else None,
                })
            
            # Get info, timing, and metrics fields - send raw data, backend will serialize
            info = rollout.get("info")
            timing = rollout.get("timing")
            metrics = rollout.get("metrics")

            sample = {
                "step": step,
                "example_id": rollout.get("example_id"),
                "prompt": prompt_messages,
                "completion": completion_messages,
                "trajectory": trajectory_data,
                "reward": rollout.get("reward"),
                "advantage": rollout.get("advantage"),
                "answer": rollout.get("answer"),
                "task": rollout.get("task"),
                "info": info,
                "metrics": metrics,
                "timing": timing,
            }
            samples.append(sample)

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

    def log_final_samples(self) -> None:
        """Log final samples (no-op - samples are logged per-step only)."""
        pass

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        """Log distributions to Prime Intellect API."""
        if not self.is_master:
            return
        if not self.enabled:
            return
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.distributions
            or step % self.config.log_extras.interval != 0
        ):
            # Do not log distributions if not enabled or not log interval step
            return

        assert self.last_log_distributions_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for distribution logging"

        self.logger.info(f"Logging distributions to Prime Intellect API at step {step}")
        start_time = time.perf_counter()

        # Upload distributions
        self._make_request(
            "distributions",
            {
                "run_id": self.run_id,
                "step": step,
                "distributions": distributions,
            },
        )
        self.last_log_distributions_step = step
        self.logger.debug(f"Logged distributions at step {step} to Prime Intellect API in {time.perf_counter() - start_time:.2f}s")

    def log_final_distributions(self) -> None:
        """Log final distributions (no-op - distributions are logged per-step only)."""
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to Prime Intellect API."""
        if not self.is_master or not self.enabled:
            return

        self.logger.info("Saving final summary to Prime Intellect API")
        self._make_request(
            "finalize",
            {
                "run_id": self.run_id,
                "summary": self.history[-1] if self.history else {},
            },
        )

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
