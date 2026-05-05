import asyncio
import io
import json
import math
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import verifiers as vf
from prime_cli.core.config import Config as PrimeConfig
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.shared import PrimeMonitorConfig
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor, sample_items_for_logging

_RETRYABLE_HTTPX_EXC = (httpx.TransportError, httpx.RemoteProtocolError)


def _is_retryable_upload_error(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE_HTTPX_EXC):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code in (408, 429) or 500 <= code < 600
    return False


# Per-HTTP-call retry budget. Wall-clock cap dominates so a single call cannot
# stall the upload pipeline indefinitely.
_UPLOAD_MAX_ATTEMPTS = 6
_UPLOAD_MAX_DELAY_S = 120.0
_UPLOAD_BACKOFF_MAX_S = 30.0

# Bounded backlog of (step, parquet_bytes) waiting to be uploaded. Sized to
# survive a few intervals of R2 unavailability without unbounded memory growth.
_MAX_PENDING_SAMPLE_UPLOADS = 5


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
        ("env_name", pa.string()),
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


_DROPPED_JSON_VALUE = object()


def _drop_non_finite_json_values(value: Any, dropped_paths: list[str], path: str = "") -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        dropped_paths.append(path)
        return _DROPPED_JSON_VALUE

    if isinstance(value, dict):
        return {
            key: sanitized_item
            for key, item in value.items()
            if (
                sanitized_item := _drop_non_finite_json_values(
                    item,
                    dropped_paths,
                    f"{path}.{key}" if path else str(key),
                )
            )
            is not _DROPPED_JSON_VALUE
        }

    if isinstance(value, list):
        return [
            sanitized_item
            for idx, item in enumerate(value)
            if (sanitized_item := _drop_non_finite_json_values(item, dropped_paths, f"{path}[{idx}]"))
            is not _DROPPED_JSON_VALUE
        ]

    return value


class PrimeMonitor(Monitor):
    """Logs to Prime Intellect API."""

    def _sanitize_json_payload(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Drop non-finite floats before sending JSON payloads to the public API."""
        dropped_paths: list[str] = []
        sanitized_payload = _drop_non_finite_json_values(payload, dropped_paths)
        if not dropped_paths:
            return payload

        preview = ", ".join(dropped_paths[:5])
        suffix = " ..." if len(dropped_paths) > 5 else ""
        self.logger.warning(
            f"Dropping {len(dropped_paths)} non-finite value(s) from Prime monitor {endpoint} payload: "
            f"{preview}{suffix}"
        )
        return sanitized_payload

    def __init__(
        self,
        config: PrimeMonitorConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseConfig | None = None,
        keep_full_history: bool = True,
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self._keep_full_history = keep_full_history
        self.output_dir = output_dir
        self._registered = False
        self._finalized = False
        self._closed = False
        self._owner_pid = os.getpid()

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0
        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return

        assert config is not None
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")

        api_key = os.getenv(config.api_key_var)
        if api_key is None:
            prime_config = PrimeConfig()
            api_key = prime_config.api_key

        if not api_key:
            self.logger.warning(
                f"API key not found. Set {config.api_key_var} environment variable or run `prime login`. "
                "PrimeMonitor will not be able to upload data."
            )
            self.enabled = False
            return

        self.api_key = api_key
        self.base_url = config.base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }

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
        if config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self._pending_sample_steps: set[int] = set()
                self.tokenizer = tokenizer
            if config.log_extras.distributions:
                self.last_log_distributions_step = -1

    def _register_run(self, config: PrimeMonitorConfig, run_config: BaseConfig | None) -> str | None:
        """Register an external run with the platform. Returns run_id on success, None on failure."""
        prime_config = None
        team_id = config.team_id
        frontend_url = config.frontend_url
        if team_id is None or frontend_url is None:
            prime_config = PrimeConfig()
        if team_id is None:
            team_id = prime_config.team_id
        if frontend_url is None:
            frontend_url = prime_config.frontend_url

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

        try:
            response = httpx.post(
                f"{self.base_url}/external-runs",
                headers=self._headers,
                json=payload,
                timeout=30,
            )
        except httpx.HTTPError as e:
            self.logger.warning(f"Failed to register platform run: {e}. PrimeMonitor will not be able to upload data.")
            return None

        if response.status_code != 201:
            self.logger.warning(
                f"Failed to create platform run (HTTP {response.status_code}): {response.text}. "
                "PrimeMonitor will not be able to upload data."
            )
            return None

        run_id = response.json()["run"]["id"]
        if frontend_url:
            dashboard_url = f"{frontend_url.rstrip('/')}/dashboard/training/{run_id}"
            self.logger.success(f"Monitor run at: {dashboard_url}")
        else:
            self.logger.success(f"Registered platform run {run_id}")
        self._registered = True
        return run_id

    def _finalize_run(self, success: bool) -> None:
        """Mark the run as completed or failed on the platform."""
        if not self._registered:
            return

        payload: dict = {"status": "completed" if success else "failed"}
        status_label = "completed" if success else "failed"
        self.logger.info(f"Finalizing platform run {self.run_id} as {status_label}")

        try:
            response = httpx.put(
                f"{self.base_url}/external-runs/{self.run_id}/status",
                headers=self._headers,
                json=payload,
                timeout=30,
            )
        except httpx.HTTPError as e:
            self.logger.warning(f"Failed to finalize platform run {self.run_id}: {e}")
            return

        if response.status_code != 200:
            self.logger.warning(
                f"Failed to finalize platform run {self.run_id} (HTTP {response.status_code}): {response.text}"
            )
            return
        self.logger.info(f"Platform run {self.run_id} marked as {status_label}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._keep_full_history:
            self.history.append(metrics)
        else:
            self.history = [metrics]
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
        """Logs rollouts to Prime Intellect API using presigned URLs for direct R2 upload.

        Adds the new step to a bounded backlog and schedules a drain on the background
        event loop. Failed uploads stay in the backlog and are retried on the next
        log_samples call, so a transient R2 outage does not silently drop data.
        """
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
            return

        rollouts = sample_items_for_logging(
            rollouts,
            self.config.log_extras.sample_ratio,
        )
        if not rollouts:
            return

        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert step not in self._pending_sample_steps, f"Step {step} upload already in progress"
        assert self.logger is not None, "Logger is required for sample logging"

        self.logger.info(f"Logging {len(rollouts)} samples to Prime Intellect API at step {step}")
        start_time = time.perf_counter()

        parquet_bytes = self._rollouts_to_parquet_bytes(rollouts, step)

        if not parquet_bytes:
            self.logger.warning(f"No samples to log at step {step}")
            return

        self._pending_sample_steps.add(step)
        self._enqueue_and_drain_samples(step, parquet_bytes)

        self.logger.debug(
            f"Queued samples upload for step {step} in {time.perf_counter() - start_time:.2f}s "
            f"(backlog={len(self._sample_upload_queue) + 1})"
        )

    def _rollouts_to_parquet_bytes(self, rollouts: list[vf.RolloutOutput], step: int) -> bytes | None:
        """Convert rollouts directly to Parquet bytes for upload."""
        now = datetime.now(timezone.utc)
        rows = []

        for sample_id, rollout in enumerate(rollouts):
            prompt = rollout.get("prompt")
            completion = rollout.get("completion")
            trajectory = rollout.get("trajectory") or []
            if prompt is None or completion is None or not trajectory:
                continue

            example_id = rollout.get("example_id")
            try:
                problem_id = int(example_id) if example_id is not None else sample_id
            except (TypeError, ValueError):
                problem_id = sample_id

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
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "prompt": json.dumps(prompt),
                    "completion": json.dumps(completion),
                    "trajectory": json.dumps(trajectory_data),
                    "answer": rollout.get("answer") or "",
                    "env_name": rollout.get("env_name") or "",
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

    def _enqueue_and_drain_samples(self, step: int, parquet_bytes: bytes) -> None:
        """Append a step's parquet to the backlog and trigger a drain on the bg loop."""
        future = asyncio.run_coroutine_threadsafe(
            self._enqueue_and_drain_samples_async(step, parquet_bytes),
            self._loop,
        )
        self._pending_futures.append(future)
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    async def _enqueue_and_drain_samples_async(self, step: int, parquet_bytes: bytes) -> None:
        """Backlog-aware upload: enqueue, then drain oldest-first under a lock."""
        if self._sample_upload_lock is None:
            self._sample_upload_lock = asyncio.Lock()

        self._sample_upload_queue.append((step, parquet_bytes))
        while len(self._sample_upload_queue) > _MAX_PENDING_SAMPLE_UPLOADS:
            dropped_step, _ = self._sample_upload_queue.popleft()
            self._pending_sample_steps.discard(dropped_step)
            self.logger.warning(
                f"Sample upload backlog exceeded {_MAX_PENDING_SAMPLE_UPLOADS}, "
                f"dropping oldest queued step {dropped_step}"
            )

        async with self._sample_upload_lock:
            while self._sample_upload_queue:
                # Pop the in-flight item out of the queue entirely before awaiting,
                # so concurrent eviction (which runs without the lock) cannot remove
                # the head we are uploading. On failure we appendleft to preserve
                # ordering; on success the item simply stays gone.
                pending_step, pending_bytes = self._sample_upload_queue.popleft()
                start = time.perf_counter()
                try:
                    await self._upload_one_sample_step(pending_step, pending_bytes)
                except Exception as e:
                    self._sample_upload_queue.appendleft((pending_step, pending_bytes))
                    self.logger.opt(exception=True).warning(
                        f"Sample upload for step {pending_step} failed after retries; "
                        f"keeping in backlog (size={len(self._sample_upload_queue)}): "
                        f"{type(e).__name__}: {e}"
                    )
                    # Stop draining; will retry on next log_samples tick.
                    return

                # Success: bookkeeping (item is already popped).
                self._pending_sample_steps.discard(pending_step)
                self.last_log_samples_step = pending_step
                self.logger.debug(f"Uploaded samples for step {pending_step} in {time.perf_counter() - start:.2f}s")

    async def _upload_one_sample_step(self, step: int, parquet_bytes: bytes) -> None:
        """Run presign → R2 PUT → confirm for a single step. Raises on final failure."""
        presign_data = await self._request_presigned_url(step)
        presigned_url = presign_data["presigned_url"]
        s3_key = presign_data["s3_key"]

        await self._upload_to_r2(presigned_url, parquet_bytes, step, content_type="application/parquet")
        await self._confirm_samples_upload(step, s3_key)

    def _retry_policy(self, op_name: str, step: int) -> AsyncRetrying:
        """Shared tenacity policy for sample-upload HTTP calls."""

        def _log_retry(retry_state) -> None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            self.logger.warning(
                f"Retrying {op_name} for step {step} "
                f"(attempt {retry_state.attempt_number}/{_UPLOAD_MAX_ATTEMPTS}): "
                f"{type(exc).__name__ if exc else 'unknown'}: {exc}"
            )

        return AsyncRetrying(
            retry=retry_if_exception(_is_retryable_upload_error),
            stop=stop_after_attempt(_UPLOAD_MAX_ATTEMPTS) | stop_after_delay(_UPLOAD_MAX_DELAY_S),
            wait=wait_exponential(multiplier=1, min=1, max=_UPLOAD_BACKOFF_MAX_S) + wait_random(0, 1),
            before_sleep=_log_retry,
            reraise=True,
        )

    async def _request_presigned_url(self, step: int) -> dict[str, Any]:
        """Request a presigned URL from the backend. Raises on final failure."""
        async for attempt in self._retry_policy("samples/presign", step):
            with attempt:
                response = await self._client.post(
                    f"{self.base_url}/samples/presign",
                    headers=self._headers,
                    json={"run_id": self.run_id, "step": step},
                )
                response.raise_for_status()
                response_data = response.json()["data"]
                return {
                    "presigned_url": response_data["presignedUrl"],
                    "s3_key": response_data["s3Key"],
                }
        raise RuntimeError("retry loop exited without returning")

    async def _upload_to_r2(
        self, presigned_url: str, data: bytes, step: int, content_type: str = "application/json"
    ) -> None:
        """Upload data to R2 via presigned URL. Raises on final failure."""
        async for attempt in self._retry_policy("R2 PUT", step):
            with attempt:
                response = await self._client.put(presigned_url, content=data, headers={"Content-Type": content_type})
                response.raise_for_status()
                return

    async def _confirm_samples_upload(self, step: int, s3_key: str) -> None:
        """Confirm samples upload with the backend. Raises on final failure."""
        async for attempt in self._retry_policy("samples/confirm", step):
            with attempt:
                response = await self._client.post(
                    f"{self.base_url}/samples/confirm",
                    headers=self._headers,
                    json={"run_id": self.run_id, "step": step, "s3_key": s3_key},
                )
                response.raise_for_status()
                return

    def log_eval_samples(self, rollouts: list[vf.RolloutOutput], env_name: str, step: int) -> None:
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

    def _submit_final_summary(self, summary: dict[str, Any]) -> bool:
        """Submit the final summary/finalize request synchronously."""
        payload = self._sanitize_json_payload(
            "finalize",
            {"run_id": self.run_id, "summary": summary},
        )

        try:
            response = httpx.post(
                f"{self.base_url}/finalize",
                headers=self._headers,
                json=payload,
                timeout=30,
            )
        except httpx.HTTPError as e:
            self.logger.warning(f"Failed to submit final summary for platform run {self.run_id}: {e}")
            return False

        if response.status_code != 200:
            self.logger.warning(
                f"Failed to submit final summary for platform run {self.run_id} "
                f"(HTTP {response.status_code}): {response.text}"
            )
            return False

        return True

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to Prime Intellect API."""
        if not self.is_master or not self.enabled:
            return

        self.logger.info("Saving final summary to Prime Intellect API")
        summary = self.history[-1] if self.history else {}
        finalized_via_summary = self._submit_final_summary(summary)

        if os.getpid() != self._owner_pid:
            return

        if finalized_via_summary:
            self._finalized = True
            return

        self._finalize_run(success=True)
        self._finalized = True

    def close(self) -> None:
        """Close the HTTP client and stop the background event loop."""
        if self._closed or not hasattr(self, "_client"):
            return

        self._closed = True

        should_finalize = self.is_master and self.enabled and not self._finalized and os.getpid() == self._owner_pid
        if should_finalize:
            self._finalize_run(success=False)
            self._finalized = True

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
        # Sample-upload backlog. Lock is constructed lazily on the bg loop to bind
        # to the right asyncio loop after fork.
        self._sample_upload_queue: deque[tuple[int, bytes]] = deque()
        self._sample_upload_lock: asyncio.Lock | None = None
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
        full_endpoint = f"{self.base_url}/{endpoint}"
        sanitized_data = self._sanitize_json_payload(endpoint, data)

        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    full_endpoint,
                    headers=self._headers,
                    json=sanitized_data,
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
