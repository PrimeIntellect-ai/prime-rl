import asyncio
import json
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
                self._pending_sample_steps: set[int] = set()
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

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        """Logs rollouts to Prime Intellect API using presigned URLs for direct R2 upload."""
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
        assert step not in self._pending_sample_steps, f"Step {step} upload already in progress"
        assert self.logger is not None, "Logger is required for sample logging"

        self.logger.info(f"Logging samples to Prime Intellect API at step {step}")
        start_time = time.perf_counter()

        # Prepare samples for API
        samples = self._prepare_samples(rollouts, step)

        if not samples:
            self.logger.warning(f"No samples to log at step {step}")
            return

        self._pending_sample_steps.add(step)

        # Use presigned URL flow for uploading samples
        self._upload_samples_via_presigned_url(samples, step)

        self.logger.debug(
            f"Initiated samples upload at step {step} to Prime Intellect API in {time.perf_counter() - start_time:.2f}s"
        )

    def _prepare_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> list[dict[str, Any]]:
        """Prepare samples from rollouts for upload."""
        samples = []
        for rollout in rollouts:
            # Extract prompt and completion from the rollout state, which includes final_env_response
            prompt_messages = rollout.get("prompt")
            completion_messages = rollout.get("completion")
            trajectory = rollout.get("trajectory") or []
            if prompt_messages is None or completion_messages is None or not trajectory:
                continue

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

        return samples

    def _upload_samples_via_presigned_url(self, samples: list[dict[str, Any]], step: int) -> None:
        """Upload samples using presigned URL flow (fire-and-forget)."""
        future = asyncio.run_coroutine_threadsafe(
            self._upload_samples_via_presigned_url_async(samples, step),
            self._loop,
        )
        self._pending_futures.append(future)
        # Clean up completed futures to avoid memory growth
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    async def _upload_samples_via_presigned_url_async(
        self, samples: list[dict[str, Any]], step: int, max_retries: int = 3
    ) -> None:
        """Upload samples via presigned URL flow."""
        try:
            presign_data = await self._request_presigned_url(step, len(samples))
            if not presign_data:
                self.logger.warning(f"Failed to get presigned URL for samples at step {step}")
                return

            if "presigned_url" not in presign_data or "s3_key" not in presign_data:
                self.logger.warning(f"Invalid presign response at step {step}")
                return

            presigned_url = presign_data["presigned_url"]
            s3_key = presign_data["s3_key"]
            json_bytes = json.dumps(samples).encode("utf-8")

            upload_success = await self._upload_to_r2(
                presigned_url, json_bytes, content_type="application/json", max_retries=max_retries
            )
            if not upload_success:
                self.logger.warning(f"Failed to upload samples to R2 at step {step}")
                return

            confirm_success = await self._confirm_samples_upload(step, s3_key, len(samples))
            if not confirm_success:
                self.logger.warning(f"Failed to confirm samples upload at step {step}")
                return

            self.last_log_samples_step = step
            self.logger.debug(f"Successfully completed samples upload at step {step}")

        except Exception as e:
            self.logger.warning(f"Failed to upload samples via presigned URL at step {step}: {type(e).__name__}: {e}")
        finally:
            self._pending_sample_steps.discard(step)

    async def _request_presigned_url(self, step: int, sample_count: int) -> dict[str, Any] | None:
        """Request a presigned URL from the backend."""
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        try:
            response = await self._client.post(
                f"{self.base_url}/samples/presign",
                headers=headers,
                json={"run_id": self.run_id, "step": step, "sample_count": sample_count},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.warning(f"Failed to request presigned URL: {type(e).__name__}: {e}")
            return None

    async def _upload_to_r2(
        self, presigned_url: str, data: bytes, content_type: str = "application/json", max_retries: int = 3
    ) -> bool:
        """Upload data to R2 using presigned URL."""
        for attempt in range(max_retries):
            try:
                response = await self._client.put(presigned_url, content=data, headers={"Content-Type": content_type})
                response.raise_for_status()
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.warning(f"Failed to upload to R2 after {max_retries} attempts: {type(e).__name__}: {e}")
                    return False
                delay = 2**attempt
                self.logger.debug(f"Retrying R2 upload in {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)

    async def _confirm_samples_upload(self, step: int, s3_key: str, sample_count: int, max_retries: int = 3) -> bool:
        """Confirm samples upload with the backend. Returns True on success."""
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    f"{self.base_url}/samples/confirm",
                    headers=headers,
                    json={"run_id": self.run_id, "step": step, "s3_key": s3_key, "sample_count": sample_count},
                )
                response.raise_for_status()
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.warning(
                        f"Failed to confirm samples upload after {max_retries} attempts: {type(e).__name__}: {e}"
                    )
                    return False
                delay = 2**attempt
                self.logger.debug(f"Retrying samples confirm in {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
        return False

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

        self._flush()

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
        self._pending_futures: list[asyncio.Future] = []
        if hasattr(self, "_pending_sample_steps") and self._pending_sample_steps:
            self._pending_sample_steps.clear()

    def _reinit_after_fork(self) -> None:
        """Reinitialize thread and event loop after fork."""
        self._init_async_client()

    def _run_event_loop(self) -> None:
        """Run the async event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _flush(self, timeout: float = 30.0) -> None:
        """Wait for all pending async requests to complete."""
        if not self.enabled or not hasattr(self, "_loop"):
            return

        if not self._pending_futures:
            return

        self.logger.debug(f"Flushing {len(self._pending_futures)} pending request(s)")
        for future in self._pending_futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                self.logger.debug(f"Pending request completed with error: {e}")

        self._pending_futures.clear()

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

        future = asyncio.run_coroutine_threadsafe(
            self._make_request_async(endpoint, data),
            self._loop,
        )
        self._pending_futures.append(future)
        # Clean up completed futures to avoid memory growth
        self._pending_futures = [f for f in self._pending_futures if not f.done()]
