from __future__ import annotations

import asyncio
import atexit
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Coroutine

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import verifiers.v1 as vf
from prime_cli.core.config import Config as PrimeConfig
from verifiers.v1.utils.platform import build_samples

from prime_rl.configs.monitors import PrimeMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import sanitize

BASE_URL = "https://api.primeintellect.ai/api/v1/rft"
API_KEY_VAR = "PRIME_API_KEY"

SAMPLE_SCHEMA = pa.schema(
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


class PrimeMonitor(Monitor):
    """Logs metrics and episodes to the Prime platform.

    Uploads are fire-and-forget tasks on the caller's event loop — the prime
    monitor only runs in the orchestrator, whose call sites are all async. The
    platform calls and the episode serialization run in worker threads via
    ``asyncio.to_thread``, so they never stall the loop.
    """

    config: PrimeMonitorConfig

    def init(self, config: BaseConfig | None = None) -> None:
        api_key = os.getenv(API_KEY_VAR) or PrimeConfig().api_key
        if not api_key:
            raise RuntimeError(f"API key not found - set {API_KEY_VAR} or run `prime login`")
        self._owner_pid = os.getpid()
        self._loop = asyncio.get_running_loop()
        self._tasks: set[Any] = set()

        self.run = TrainRun(api_key)
        run_fields: dict[str, Any] = {}
        if config is not None:
            run_fields = dict(
                base_model=config.model.name,
                max_steps=config.max_steps or 0,
                batch_size=config.batch_size,
                rollouts_per_example=config.group_size,
                seq_len=config.seq_len,
                environments=[env.env_id for env in config.train.source],
                run_config=config.model_dump(exclude_none=True, mode="json"),
                wandb_project=config.monitors.wandb.project if config.monitors.wandb else None,
            )
        self.run.create(name=self.config.name, team_id=self.config.team_id, **run_fields)
        # A run that was created but never finalized did not exit cleanly; finalize
        # unregisters the hook. atexit rather than __del__ because it runs before
        # interpreter teardown, while httpx can still send.
        atexit.register(self._mark_failed)

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        metrics, dropped = sanitize(metrics)
        if dropped:
            self.logger.warning(f"Dropping {len(dropped)} non-finite metric value(s): {', '.join(dropped[:5])}")
        self._submit("metrics upload", asyncio.to_thread(self.run.log_metrics, metrics))

    def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        """Upload one platform sample per episode via the presigned-URL Parquet flow.
        Only the trained cohort ships to the platform."""
        if kind != "train" or subset != "effective" or not episodes:
            return

        def upload() -> None:
            parquet_bytes = self._episodes_to_parquet_bytes(episodes, step)
            if parquet_bytes is not None:
                self.run.upload_samples(parquet_bytes, step)

        self.logger.info(f"Logging {len(episodes)} episodes to Prime Intellect API at step {step}")
        self._submit(f"episodes upload at step {step}", asyncio.to_thread(upload))

    def finalize(self) -> None:
        self.run.finalize()
        atexit.unregister(self._mark_failed)

    def _mark_failed(self) -> None:
        # Forked children inherit the atexit table; only the creating process
        # may flip the run's status.
        if os.getpid() == self._owner_pid:
            self.run.set_status(success=False)

    def _submit(self, what: str, request: Coroutine[Any, Any, None]) -> None:
        """Run a request as a fire-and-forget task on the orchestrator's loop; a failure
        only warns. Thread-safe, since the episode fan-out runs in a worker thread."""

        async def guarded() -> None:
            try:
                await request
            except Exception as e:
                self.logger.warning(f"Failed {what} to Prime Intellect API: {type(e).__name__}: {e}")

        task = asyncio.run_coroutine_threadsafe(guarded(), self._loop)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _episodes_to_parquet_bytes(self, episodes: list[vf.Episode], step: int) -> bytes | None:
        """One row per episode. Sample construction is shared with verifiers' eval
        ``--push`` (``build_samples``: complete native episode in ``info.native_wrapper``,
        flat summary from one trainable trace), so a training episode and an eval sample
        land on the platform identically; the RFT-only columns (run/step/advantage/
        problem_id/env_name) are layered on here."""
        advantages: dict[str, float | None] = {}
        env_names: dict[str, str] = {}
        for episode in episodes:
            summary_trace = next((trace for trace in episode.traces if trace.agent.trainable), episode.traces[0])
            advantages[episode.id] = summary_trace.scalar_advantage()
            env_names[episode.id] = episode.env.id

        now = datetime.now(timezone.utc)
        rows = []
        for sample_id, sample in enumerate(build_samples(episodes)):
            trajectory = sample["trajectory"]
            if not trajectory:  # no branches (e.g. an episode that errored before any message)
                continue
            advantage = advantages.get(sample["episode_id"])
            trajectory = [{**branch, "advantage": advantage} for branch in trajectory]

            try:
                problem_id = int(sample["example_id"]) if sample["example_id"] is not None else sample_id
            except (TypeError, ValueError):
                problem_id = sample_id

            rows.append(
                {
                    "run_id": self.run.id,
                    "step": step,
                    "tag": "",
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "prompt": "",
                    "completion": json.dumps(sample["completion"]),
                    "trajectory": json.dumps(trajectory),
                    "answer": "",
                    "env_name": env_names.get(sample["episode_id"], ""),
                    "task": json.dumps(sample["task"]),
                    "info": json.dumps(sample["info"]),
                    "reward": sample["reward"],
                    "advantage": advantage,
                    "metrics": json.dumps(sample["metrics"]),
                    "timing": json.dumps(sample["timing"]),
                    "num_input_tokens": trajectory[-1]["num_input_tokens"],
                    "num_output_tokens": trajectory[-1]["num_output_tokens"],
                    "created_at": now,
                }
            )

        if not rows:
            return None

        table = pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy", use_dictionary=True, write_statistics=True)
        return buf.getvalue()


class TrainRun:
    """A training run on the Prime platform's RFT API.

    Owns the HTTP client and the run lifecycle (create, log, finalize) — the
    natural seam to be subsumed by the train SDK, the way ``wandb.Run`` backs
    the W&B monitor. All calls are synchronous; the client is thread-safe.
    """

    def __init__(self, api_key: str):
        self.id: str | None = None
        self.logger = get_logger()
        self.client = httpx.Client(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=30,
            transport=httpx.HTTPTransport(retries=3),
        )

    def create(
        self,
        name: str | None = None,
        team_id: str | None = None,
        base_model: str = "unknown",
        max_steps: int = 0,
        batch_size: int | None = None,
        rollouts_per_example: int | None = None,
        seq_len: int | None = None,
        environments: list[str] | None = None,
        run_config: dict[str, Any] | None = None,
        wandb_project: str | None = None,
    ) -> str:
        """Register the run with the platform and return its id."""
        prime_config = PrimeConfig()
        team_id = team_id or prime_config.team_id

        payload: dict[str, Any] = {"base_model": base_model, "max_steps": max_steps}
        if batch_size is not None:
            payload["batch_size"] = batch_size
        if rollouts_per_example is not None:
            payload["rollouts_per_example"] = rollouts_per_example
        if seq_len is not None:
            payload["seq_len"] = seq_len
        if environments is not None:
            payload["environments"] = [{"id": env_id} for env_id in environments]
        if run_config is not None:
            payload["run_config"] = run_config
        if wandb_project is not None:
            payload["wandb_project"] = wandb_project
        if name:
            payload["name"] = name
        if team_id:
            payload["team_id"] = team_id

        response = self.client.post("/external-runs", json=payload)
        if response.status_code != 201:
            raise RuntimeError(f"Failed to create platform run (HTTP {response.status_code}): {response.text}")

        self.id = response.json()["run"]["id"]
        if prime_config.frontend_url:
            self.logger.success(f"Monitor run at: {prime_config.frontend_url.rstrip('/')}/dashboard/training/{self.id}")
        else:
            self.logger.success(f"Registered platform run {self.id}")
        return self.id

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self.client.post("/metrics", json={"run_id": self.id, "metrics": metrics}).raise_for_status()

    def upload_samples(self, parquet_bytes: bytes, step: int) -> None:
        """Presigned-URL flow: presign -> R2 PUT -> confirm."""
        presign = self.client.post("/samples/presign", json={"run_id": self.id, "step": step})
        presign.raise_for_status()
        data = presign.json()["data"]
        # Bare request - the presigned URL rejects the client's auth headers.
        httpx.put(
            data["presignedUrl"], content=parquet_bytes, headers={"Content-Type": "application/parquet"}, timeout=30
        ).raise_for_status()
        self.client.post(
            "/samples/confirm", json={"run_id": self.id, "step": step, "s3_key": data["s3Key"]}
        ).raise_for_status()

    def finalize(self) -> None:
        """Finalize the run as completed."""
        self.logger.info(f"Finalizing platform run {self.id}")
        try:
            self.client.post("/finalize", json={"run_id": self.id, "summary": {}}).raise_for_status()
        except httpx.HTTPError as e:
            self.logger.warning(f"Failed to finalize platform run {self.id}: {e}")
            self.set_status(success=True)

    def set_status(self, success: bool) -> None:
        """Mark the run as completed or failed."""
        status = "completed" if success else "failed"
        self.logger.info(f"Marking platform run {self.id} as {status}")
        try:
            self.client.put(f"/external-runs/{self.id}/status", json={"status": status}).raise_for_status()
        except httpx.HTTPError as e:
            self.logger.warning(f"Failed to mark platform run {self.id} as {status}: {e}")
