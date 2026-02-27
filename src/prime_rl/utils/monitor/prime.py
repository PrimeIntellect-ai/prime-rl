import asyncio
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlparse

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import verifiers as vf
from prime_cli.core.config import Config as PrimeConfig
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.shared import PrimeMonitorConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor
from prime_rl.utils.pydantic_config import BaseSettings


def _json(val: Any) -> str:
    """JSON-serialize dicts/lists, pass strings through, default to empty string for None."""
    if isinstance(val, str):
        return val
    if val is None:
        return ""
    return json.dumps(val)


_SAMPLE_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("step", pa.int64()),
        ("tag", pa.string()),
        ("problem_id", pa.int64()),
        ("sample_id", pa.int64()),
        ("prompt", pa.string()),
        ("completion", pa.string()),
        ("trajectory", pa.string()),
        ("answer", pa.string()),
        ("task", pa.string()),
        ("info", pa.string()),
        ("reward", pa.float64()),
        ("advantage", pa.float64()),
        ("metrics", pa.string()),
        ("timing", pa.string()),
        ("num_input_tokens", pa.int64()),
        ("num_output_tokens", pa.int64()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ]
)


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
        self._registered = False
        self._finalized = False

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

        run_id = os.getenv("RUN_ID")
        if not run_id:
            run_id = self._register_run(config, run_config)
            if run_id:
                os.environ["RUN_ID"] = run_id
            else:
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

    def _register_run(self, config: PrimeMonitorConfig, run_config: BaseSettings | None) -> str | None:
        """Register an external run with the platform. Returns run_id on success, None on failure."""
        prime_config = PrimeConfig()
        registration_api_key = prime_config.api_key or None
        if not registration_api_key:
            self.logger.warning(
                "Prime Intellect API key not found. Either set PRIME_API_KEY or run `prime login`. "
                "PrimeMonitor will not be able to register or upload data."
            )
            return None

        team_id = config.team_id or prime_config.team_id

        model = getattr(run_config, "model", None) if run_config else None
        environments = getattr(run_config, "env", None) if run_config else None
        wandb = getattr(run_config, "wandb", None) if run_config else None

        payload: dict = {
            "base_model": model.name if model else "unknown",
            "max_steps": getattr(run_config, "max_steps", None) or 0,
        }
        if config.run_name:
            payload["name"] = config.run_name
        if team_id:
            payload["team_id"] = team_id
        if environments:
            payload["environments"] = [{"id": env.id} for env in environments if hasattr(env, "id")]
        if wandb and getattr(wandb, "project", None):
            payload["wandb_project"] = wandb.project

        self.logger.info(f"Registering external training run with platform at {config.base_url}")

        response = httpx.post(
            f"{config.base_url}/external-runs",
            headers={"Authorization": f"Bearer {registration_api_key}"},
            json=payload,
            timeout=30,
        )

        if response.status_code != 201:
            self.logger.warning(
                f"Failed to create platform run (HTTP {response.status_code}): {response.text}. "
                "PrimeMonitor will not be able to upload data."
            )
            return None

        run_id = response.json()["run"]["id"]
        parsed = urlparse(config.base_url)
        dashboard_url = f"{parsed.scheme}://{parsed.netloc}/dashboard/training/{run_id}"
        self.logger.success(f"Monitor run at:\n  {dashboard_url}")
        self._registered = True
        return run_id

    def _finalize_run(self, success: bool) -> None:
        """Mark the run as completed or failed on the platform."""
        if not getattr(self, "_registered", False):
            return
        prime_config = PrimeConfig()
        registration_api_key = prime_config.api_key or None
        if not registration_api_key:
            return

        payload: dict = {"status": "completed" if success else "failed"}
        status_label = "completed" if success else "failed"
        self.logger.info(f"Finalizing platform run {self.run_id} as {status_label}")

        response = httpx.put(
            f"{self.base_url}/external-runs/{self.run_id}/status",
            headers={"Authorization": f"Bearer {registration_api_key}"},
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            self.logger.warning(
                f"Failed to finalize platform run {self.run_id} (HTTP {response.status_code}): {response.text}"
            )
            return
        self.logger.info(f"Platform run {self.run_id} marked as {status_label}")

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

        parquet_bytes = self._rollouts_to_parquet_bytes(rollouts, step)

        if not parquet_bytes:
            self.logger.warning(f"No samples to log at step {step}")
            return

        self._pending_sample_steps.add(step)

        # Use presigned URL flow for uploading samples
        self._upload_samples_via_presigned_url(parquet_bytes, step)

        self.logger.debug(
            f"Initiated samples upload at step {step} to Prime Intellect API in {time.perf_counter() - start_time:.2f}s"
        )

    def _rollouts_to_parquet_bytes(self, rollouts: list[vf.RolloutOutput], step: int) -> bytes | None:
        """Convert rollouts directly to Parquet bytes for upload."""
        now = datetime.now(timezone.utc)
        rows = []

        for rollout in rollouts:
            prompt = rollout.get("prompt")
            completion = rollout.get("completion")
            trajectory = rollout.get("trajectory") or []
            if prompt is None or completion is None or not trajectory:
                continue

            trajectory_data = [
                {
                    "prompt": ts["prompt"],
                    "completion": ts["completion"],
                    "reward": ts.get("reward"),
                    "advantage": ts.get("advantage"),
                    "extras": ts.get("extras", {}),
                    "num_input_tokens": len(ts["tokens"]["prompt_ids"]) if ts.get("tokens") else None,
                    "num_output_tokens": len(ts["tokens"]["completion_ids"]) if ts.get("tokens") else None,
                }
                for ts in trajectory
            ]

            rows.append(
                {
                    "run_id": self.run_id,
                    "step": step,
                    "tag": "",
                    "problem_id": 0,
                    "sample_id": 0,
                    "prompt": json.dumps(prompt),
                    "completion": json.dumps(completion),
                    "trajectory": json.dumps(trajectory_data),
                    "answer": rollout.get("answer") or "",
                    "task": rollout.get("task") or "",
                    "info": _json(rollout.get("info")),
                    "reward": rollout.get("reward"),
                    "advantage": rollout.get("advantage"),
                    "metrics": _json(rollout.get("metrics")),
                    "timing": _json(rollout.get("timing")),
                    "num_input_tokens": 0,
                    "num_output_tokens": 0,
                    "created_at": now,
                }
            )

        if not rows:
            return None

        table = pa.Table.from_pylist(rows, schema=_SAMPLE_SCHEMA)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy", use_dictionary=True, write_statistics=True)
        return buf.getvalue()

    def _upload_samples_via_presigned_url(self, parquet_bytes: bytes, step: int) -> None:
        """Upload Parquet samples using presigned URL flow (fire-and-forget)."""
        future = asyncio.run_coroutine_threadsafe(
            self._upload_samples_via_presigned_url_async(parquet_bytes, step),
            self._loop,
        )
        self._pending_futures.append(future)
        # Clean up completed futures to avoid memory growth
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    async def _upload_samples_via_presigned_url_async(
        self, parquet_bytes: bytes, step: int, max_retries: int = 3
    ) -> None:
        """Upload Parquet bytes via presigned URL flow."""
        try:
            presign_data = await self._request_presigned_url(step)
            if not presign_data:
                self.logger.warning(f"Failed to get presigned URL for samples at step {step}")
                return

            if "presigned_url" not in presign_data or "s3_key" not in presign_data:
                self.logger.warning(f"Invalid presign response at step {step}")
                return

            presigned_url = presign_data["presigned_url"]
            s3_key = presign_data["s3_key"]

            upload_success = await self._upload_to_r2(
                presigned_url, parquet_bytes, content_type="application/parquet", max_retries=max_retries
            )
            if not upload_success:
                self.logger.warning(f"Failed to upload samples to R2 at step {step}")
                return

            confirm_success = await self._confirm_samples_upload(step, s3_key)
            if not confirm_success:
                self.logger.warning(f"Failed to confirm samples upload at step {step}")
                return

            self.last_log_samples_step = step
            self.logger.debug(f"Successfully completed samples upload at step {step}")

        except Exception as e:
            self.logger.warning(f"Failed to upload samples via presigned URL at step {step}: {type(e).__name__}: {e}")
        finally:
            self._pending_sample_steps.discard(step)

    async def _request_presigned_url(self, step: int) -> dict[str, Any] | None:
        """Request a presigned URL from the backend."""
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        try:
            response = await self._client.post(
                f"{self.base_url}/samples/presign",
                headers=headers,
                json={"run_id": self.run_id, "step": step},
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

    async def _confirm_samples_upload(self, step: int, s3_key: str, max_retries: int = 3) -> bool:
        """Confirm samples upload with the backend. Returns True on success."""
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    f"{self.base_url}/samples/confirm",
                    headers=headers,
                    json={"run_id": self.run_id, "step": step, "s3_key": s3_key},
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
        self._finalize_run(success=True)
        self._finalized = True

    def close(self) -> None:
        """Close the HTTP client and stop the background event loop."""
        if not hasattr(self, "_client"):
            return

        if self.is_master and self.enabled and not self._finalized:
            self._finalize_run(success=False)

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
