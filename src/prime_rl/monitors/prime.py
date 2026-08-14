from __future__ import annotations

import asyncio
import io
import json
import os
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import verifiers.v1 as vf
from prime_cli.core.config import Config as PrimeConfig
from verifiers.v1.episode import EnvInfo
from verifiers.v1.utils.platform import build_samples

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.shared import PrimeMonitorConfig
from prime_rl.monitors.base import Monitor, drop_non_finite_json_values
from prime_rl.utils.background_async import BackgroundAsync
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


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
    """Logs metrics and episodes to the Prime Intellect platform."""

    def __init__(self, config: PrimeMonitorConfig, run_config: OrchestratorConfig | None = None):
        self.config = config
        self.run_config = run_config
        self.logger = get_logger()
        self.base_url = config.base_url.rstrip("/")
        self._last_metrics: dict[str, Any] = {}
        self._registered = False
        self._finalized = False
        self._owner_pid = os.getpid()

    def init(self) -> None:
        api_key = os.getenv(self.config.api_key_var) or PrimeConfig().api_key
        if not api_key:
            raise RuntimeError(f"API key not found - set {self.config.api_key_var} or run `prime login`")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }

        run_id = os.getenv("RUN_ID") or self._register_run()
        os.environ["RUN_ID"] = run_id
        self.run_id = run_id
        self._background = BackgroundAsync()

    def _register_run(self) -> str:
        """Register an external run with the platform and return its run id."""
        config, run_config = self.config, self.run_config
        team_id = config.team_id
        frontend_url = config.frontend_url
        if team_id is None or frontend_url is None:
            prime_config = PrimeConfig()
            team_id = team_id or prime_config.team_id
            frontend_url = frontend_url or prime_config.frontend_url

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
        if config.run_name:
            payload["name"] = config.run_name
        if team_id:
            payload["team_id"] = team_id

        response = httpx.post(f"{self.base_url}/external-runs", headers=self._headers, json=payload, timeout=30)
        if response.status_code != 201:
            raise RuntimeError(f"Failed to create platform run (HTTP {response.status_code}): {response.text}")

        run_id = response.json()["run"]["id"]
        if frontend_url:
            self.logger.success(f"Monitor run at: {frontend_url.rstrip('/')}/dashboard/training/{run_id}")
        else:
            self.logger.success(f"Registered platform run {run_id}")
        self._registered = True
        return run_id

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._last_metrics = metrics
        payload = self._sanitize("metrics", {"run_id": self.run_id, "metrics": metrics})
        self._submit("metrics upload", lambda: self._post_async("metrics", payload))

    def log_episodes(self, rollouts: list[Rollout], step: int) -> None:
        """Upload one platform sample per episode via the presigned-URL Parquet flow."""
        config = self.config.log_episodes
        if config is None or step % config.interval != 0:
            return

        episodes = group_episodes(rollouts)
        if config.sample_ratio is not None and config.sample_ratio < 1.0:
            num_samples = max(1, int(len(episodes) * config.sample_ratio)) if config.sample_ratio > 0 else 0
            episodes = random.sample(episodes, min(num_samples, len(episodes)))
        if not episodes:
            return

        parquet_bytes = self._episodes_to_parquet_bytes(episodes, step)
        if parquet_bytes is None:
            return

        self.logger.info(f"Logging {len(episodes)} episodes to Prime Intellect API at step {step}")
        self._submit(f"episodes upload at step {step}", lambda: self._upload_samples_async(parquet_bytes, step))

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

        table = pa.Table.from_pylist(rows, schema=_SAMPLE_SCHEMA)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy", use_dictionary=True, write_statistics=True)
        return buf.getvalue()

    def finalize(self) -> None:
        """Finalize the platform run as completed, submitting the last metrics as its summary."""
        self.logger.info(f"Finalizing platform run {self.run_id}")
        payload = self._sanitize("finalize", {"run_id": self.run_id, "summary": self._last_metrics})
        try:
            response = httpx.post(f"{self.base_url}/finalize", headers=self._headers, json=payload, timeout=30)
            response.raise_for_status()
        except httpx.HTTPError as e:
            self.logger.warning(f"Failed to finalize platform run {self.run_id}: {e}")
            if os.getpid() == self._owner_pid:
                self._set_run_status(success=True)
                self._finalized = True
            return
        if os.getpid() == self._owner_pid:
            self._finalized = True

    def __del__(self) -> None:
        if not hasattr(self, "_background"):  # init never ran or failed
            return
        # A run that was registered but never finalized did not exit cleanly.
        if self._registered and not self._finalized and os.getpid() == self._owner_pid:
            self._set_run_status(success=False)
            self._finalized = True
        self._background.close()

    def _set_run_status(self, success: bool) -> None:
        """Mark the run as completed or failed on the platform."""
        status = "completed" if success else "failed"
        self.logger.info(f"Marking platform run {self.run_id} as {status}")
        try:
            response = httpx.put(
                f"{self.base_url}/external-runs/{self.run_id}/status",
                headers=self._headers,
                json={"status": status},
                timeout=30,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            self.logger.warning(f"Failed to mark platform run {self.run_id} as {status}: {e}")

    def _sanitize(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Drop non-finite floats before sending JSON payloads to the public API."""
        dropped_paths: list[str] = []
        sanitized = drop_non_finite_json_values(payload, dropped_paths)
        if dropped_paths:
            preview = ", ".join(dropped_paths[:5])
            suffix = " ..." if len(dropped_paths) > 5 else ""
            self.logger.warning(
                f"Dropping {len(dropped_paths)} non-finite value(s) from Prime monitor {endpoint} payload: "
                f"{preview}{suffix}"
            )
        return sanitized

    def _submit(self, what: str, request: Callable[[], Awaitable[None]]) -> None:
        """Run a request on the background loop (fire-and-forget); a failure only warns."""

        async def guarded() -> None:
            try:
                await request()
            except Exception as e:
                self.logger.warning(f"Failed {what} to Prime Intellect API: {type(e).__name__}: {e}")

        self._background.submit(guarded)

    async def _retry(self, request: Callable[[], Awaitable[httpx.Response]], max_retries: int = 3) -> httpx.Response:
        for attempt in range(max_retries):
            try:
                response = await request()
                response.raise_for_status()
                return response
            except Exception:
                if attempt + 1 == max_retries:
                    raise
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def _post_async(self, endpoint: str, payload: dict[str, Any]) -> None:
        client = self._background.client
        await self._retry(lambda: client.post(f"{self.base_url}/{endpoint}", headers=self._headers, json=payload))

    async def _upload_samples_async(self, parquet_bytes: bytes, step: int) -> None:
        """Presigned-URL upload flow: presign -> R2 PUT -> confirm."""
        client = self._background.client
        response = await self._retry(
            lambda: client.post(
                f"{self.base_url}/samples/presign",
                headers=self._headers,
                json={"run_id": self.run_id, "step": step},
            )
        )
        data = response.json()["data"]
        await self._retry(
            lambda: client.put(
                data["presignedUrl"], content=parquet_bytes, headers={"Content-Type": "application/parquet"}
            )
        )
        await self._retry(
            lambda: client.post(
                f"{self.base_url}/samples/confirm",
                headers=self._headers,
                json={"run_id": self.run_id, "step": step, "s3_key": data["s3Key"]},
            )
        )
        self.logger.debug(f"Uploaded episode samples for step {step}")
