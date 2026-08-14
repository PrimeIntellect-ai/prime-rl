from __future__ import annotations

import asyncio
import atexit
import io
import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Coroutine

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import verifiers.v1 as vf
from prime_cli.core.config import Config as PrimeConfig
from verifiers.v1.episode import EnvInfo
from verifiers.v1.utils.platform import build_samples

from prime_rl.configs.monitors import PrimeMonitorConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.monitors.base import Monitor
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


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


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=30, transport=httpx.AsyncHTTPTransport(retries=3))


def group_episodes(rollouts: list[Rollout]) -> list[vf.Episode]:
    """Regroup rollouts into their episodes. The dispatcher unwraps every
    ``vf.Episode`` into its traces on arrival; ``episode_id`` links them back
    together (a rollout without one forms a single-trace episode)."""
    groups: dict[str, list[Rollout]] = {}
    for rollout in rollouts:
        groups.setdefault(rollout.episode_id or rollout.id, []).append(rollout)
    return [
        vf.Episode(
            id=episode_id,
            env=EnvInfo(id=group[0].env_name),
            traces=group,
            ok=all(trace.ok for trace in group),
        )
        for episode_id, group in groups.items()
    ]


class PrimeMonitor(Monitor):
    """Logs metrics and episodes to the Prime platform.

    Uploads are fire-and-forget tasks on the caller's event loop — the prime
    monitor only runs in the orchestrator, whose call sites are all async.
    Each request uses a short-lived client (connect failures retry inside the
    transport), so forked processes (subprocess evals) need no special handling.
    """

    config: PrimeMonitorConfig

    def init(self, run_config: OrchestratorConfig | None = None) -> None:
        api_key = os.getenv(API_KEY_VAR) or PrimeConfig().api_key
        if not api_key:
            raise RuntimeError(f"API key not found - set {API_KEY_VAR} or run `prime login`")
        self.base_url = BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }
        self._last_metrics: dict[str, Any] = {}
        self._registered = False
        self._finalized = False
        self._owner_pid = os.getpid()
        self._tasks: set[asyncio.Task] = set()

        run_id = os.getenv("RUN_ID")
        if run_id is None:
            run_id = self._register_run(run_config)
            self._registered = True
        os.environ["RUN_ID"] = run_id
        self.run_id = run_id

        # atexit rather than __del__: it runs before interpreter teardown, while httpx can
        # still create a client. A run registered but never finalized did not exit cleanly.
        def mark_failed() -> None:
            if self._registered and not self._finalized and os.getpid() == self._owner_pid:
                self._set_run_status(success=False)

        atexit.register(mark_failed)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._last_metrics = metrics
        payload = self._sanitize({"run_id": self.run_id, "metrics": metrics})

        async def post() -> None:
            async with _client() as client:
                response = await client.post(f"{self.base_url}/metrics", headers=self.headers, json=payload)
                response.raise_for_status()

        self._submit("metrics upload", post())

    def log_episodes(self, rollouts: list[Rollout], step: int) -> None:
        """Upload one platform sample per episode via the presigned-URL Parquet flow."""
        episodes = group_episodes(rollouts)
        if not episodes:
            return

        async def upload() -> None:
            # Serialization dumps every episode's full model - heavy pure-Python work that
            # would stall the event loop (and with it dispatch) if run inline.
            parquet_bytes = await asyncio.to_thread(self._episodes_to_parquet_bytes, episodes, step)
            if parquet_bytes is None:
                return
            # Presigned-URL flow: presign -> R2 PUT -> confirm. The PUT carries no API
            # headers - extra auth breaks the presigned signature.
            async with _client() as client:
                presign = await client.post(
                    f"{self.base_url}/samples/presign",
                    headers=self.headers,
                    json={"run_id": self.run_id, "step": step},
                )
                presign.raise_for_status()
                data = presign.json()["data"]
                put = await client.put(
                    data["presignedUrl"], content=parquet_bytes, headers={"Content-Type": "application/parquet"}
                )
                put.raise_for_status()
                confirm = await client.post(
                    f"{self.base_url}/samples/confirm",
                    headers=self.headers,
                    json={"run_id": self.run_id, "step": step, "s3_key": data["s3Key"]},
                )
                confirm.raise_for_status()

        self.logger.info(f"Logging {len(episodes)} episodes to Prime Intellect API at step {step}")
        self._submit(f"episodes upload at step {step}", upload())

    def finalize(self) -> None:
        """Finalize the platform run as completed, submitting the last metrics as its summary."""
        self.logger.info(f"Finalizing platform run {self.run_id}")
        payload = self._sanitize({"run_id": self.run_id, "summary": self._last_metrics})
        try:
            httpx.post(f"{self.base_url}/finalize", headers=self.headers, json=payload, timeout=30).raise_for_status()
        except httpx.HTTPError as e:
            self.logger.warning(f"Failed to finalize platform run {self.run_id}: {e}")
            self._set_run_status(success=True)
        if os.getpid() == self._owner_pid:
            self._finalized = True

    def _register_run(self, run_config: OrchestratorConfig | None) -> str:
        """Register an external run with the platform and return its run id."""
        prime_config = PrimeConfig()
        team_id = self.config.team_id or prime_config.team_id
        frontend_url = prime_config.frontend_url

        payload: dict[str, Any] = {
            "base_model": run_config.model.name if run_config else "unknown",
            "max_steps": (run_config.max_steps if run_config else None) or 0,
        }
        if run_config:
            if run_config.batch_size is not None:
                payload["batch_size"] = run_config.batch_size
            payload["rollouts_per_example"] = run_config.group_size
            payload["seq_len"] = run_config.seq_len
            payload["environments"] = [{"id": env.env_id} for env in run_config.train.source]
            payload["run_config"] = run_config.model_dump(exclude_none=True, mode="json")
            if run_config.monitors.wandb:
                payload["wandb_project"] = run_config.monitors.wandb.project
        if self.config.name:
            payload["name"] = self.config.name
        if team_id:
            payload["team_id"] = team_id

        response = httpx.post(f"{self.base_url}/external-runs", headers=self.headers, json=payload, timeout=30)
        if response.status_code != 201:
            raise RuntimeError(f"Failed to create platform run (HTTP {response.status_code}): {response.text}")

        run_id = response.json()["run"]["id"]
        if frontend_url:
            self.logger.success(f"Monitor run at: {frontend_url.rstrip('/')}/dashboard/training/{run_id}")
        else:
            self.logger.success(f"Registered platform run {run_id}")
        return run_id

    def _set_run_status(self, success: bool) -> None:
        """Mark the run as completed or failed on the platform."""
        status = "completed" if success else "failed"
        self.logger.info(f"Marking platform run {self.run_id} as {status}")
        try:
            httpx.put(
                f"{self.base_url}/external-runs/{self.run_id}/status",
                headers=self.headers,
                json={"status": status},
                timeout=30,
            ).raise_for_status()
        except httpx.HTTPError as e:
            self.logger.warning(f"Failed to mark platform run {self.run_id} as {status}: {e}")

    def _sanitize(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Drop non-finite floats (invalid JSON) before sending payloads to the public API."""
        sanitized, dropped = sanitize(payload)
        if dropped:
            self.logger.warning(
                f"Dropping {len(dropped)} non-finite value(s) from Prime monitor payload: {', '.join(dropped[:5])}"
            )
        return sanitized

    def _submit(self, what: str, request: Coroutine[Any, Any, None]) -> None:
        """Run a request as a fire-and-forget task; a failure only warns."""

        async def guarded() -> None:
            try:
                await request
            except Exception as e:
                self.logger.warning(f"Failed {what} to Prime Intellect API: {type(e).__name__}: {e}")

        task = asyncio.get_running_loop().create_task(guarded())
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
                    "run_id": self.run_id,
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
