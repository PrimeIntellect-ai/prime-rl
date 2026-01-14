import asyncio
import os
import time
from pathlib import Path
from threading import Thread
from typing import Any

import httpx
import verifiers as vf
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import PrimeMonitorConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor
from prime_rl.utils.pydantic_config import BaseSettings


class PrimeMonitor(Monitor):
    """Logs to Prime Intellect API."""

    def __init__(
        self,
        config: PrimeMonitorConfig | None,
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

        # Get API key from environment variable
        api_key = os.getenv(config.api_key_var)
        if not api_key:
            self.logger.warning(
                f"API key not found. Set {config.api_key_var} environment variable. PrimeMonitor will not be able to upload data."
            )
            self.enabled = False
            return

        self.api_key = api_key
        self.base_url = config.base_url

        # Get run_id from environment variable (check before allocating resources)
        run_id = os.getenv("RUN_ID")
        if not run_id:
            self.logger.warning("RUN_ID environment variable not set. PrimeMonitor will not be able to upload data.")
            self.enabled = False
            return
        self.run_id = run_id

        # Set up async HTTP client with background event loop.
        # Evals can run in a forked subprocess (see run_evals_subprocess in eval/utils.py). When a
        # process forks, only the calling thread survives - our background thread running the
        # event loop is not copied. The Thread object still exists but the OS thread is gone,
        # so asyncio.run_coroutine_threadsafe() silently fails. We use register_at_fork to
        # recreate the thread, event loop, and HTTP client in the child process.
        self._init_async_client()
        os.register_at_fork(after_in_child=self._reinit_after_fork)

        # Optionally, initialize sample logging attributes
        if config is not None and config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.tokenizer = tokenizer
            if config.log_extras.distributions:
                self.last_log_distributions_step = -1

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
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
            trajectory = rollout["trajectory"]
            if not trajectory:
                continue
            last_step = trajectory[-1]
            prompt_messages = last_step["prompt"]
            completion_messages = last_step["completion"]

            # Serialize full trajectory array (excluding large response objects and token arrays)
            trajectory_data = []
            for traj_step in rollout["trajectory"]:
                trajectory_data.append(
                    {
                        "prompt": traj_step["prompt"],
                        "completion": traj_step["completion"],
                        "reward": traj_step.get("reward"),
                        "advantage": traj_step.get("advantage"),
                        "extras": traj_step.get("extras", {}),
                        "num_input_tokens": len(traj_step.get("tokens", {}).get("prompt_ids", []))
                        if traj_step.get("tokens")
                        else None,
                        "num_output_tokens": len(traj_step.get("tokens", {}).get("completion_ids", []))
                        if traj_step.get("tokens")
                        else None,
                    }
                )

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
        self.logger.debug(
            f"Logged samples at step {step} to Prime Intellect API in {time.perf_counter() - start_time:.2f}s"
        )

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
        self.logger.debug(
            f"Logged distributions at step {step} to Prime Intellect API in {time.perf_counter() - start_time:.2f}s"
        )

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

    def close(self) -> None:
        """Close the HTTP client and stop the background event loop."""
        if not hasattr(self, "_client"):
            return

        # Close the async client within the event loop
        async def _close_client() -> None:
            await self._client.aclose()

        try:
            future = asyncio.run_coroutine_threadsafe(_close_client(), self._loop)
            future.result(timeout=5.0)  # Wait up to 5 seconds for client to close
        except Exception as e:
            self.logger.debug(f"Error closing HTTP client: {e}")

        # Stop the event loop and wait for thread to finish
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()

    def _init_async_client(self) -> None:
        """Initialize the event loop, background thread, and HTTP client."""
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        self._client = httpx.AsyncClient(timeout=30)

    def _reinit_after_fork(self) -> None:
        """Reinitialize thread and event loop after fork."""
        self._init_async_client()

    def _run_event_loop(self) -> None:
        """Run the async event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _make_request_async(self, endpoint: str, data: dict[str, Any], max_retries: int = 3) -> None:
        """Make an async POST request to the Prime Intellect API with retries."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        full_endpoint = f"{self.base_url}/{endpoint}"

        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    full_endpoint,
                    headers=headers,
                    json=data,
                )
                response.raise_for_status()
                return  # Success
            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                if is_last_attempt:
                    self.logger.warning(
                        f"Failed to upload to Prime Intellect API ({endpoint}) after {max_retries} attempts: {type(e).__name__}: {e}"
                    )
                else:
                    # Exponential backoff: 1s, 2s, 4s...
                    delay = 2**attempt
                    self.logger.debug(
                        f"Retrying {endpoint} upload in {delay}s (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(delay)

    def _make_request(self, endpoint: str, data: dict[str, Any]) -> None:
        """Submit a request to the async queue (fire-and-forget)."""
        if not self.enabled:
            return

        asyncio.run_coroutine_threadsafe(
            self._make_request_async(endpoint, data),
            self._loop,
        )
