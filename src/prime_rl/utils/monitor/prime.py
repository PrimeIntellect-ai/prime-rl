"""PrimeMonitor: durable rollout logging to Prime Intellect API.

Architecture:
  Main thread → snapshot rollouts → queue.put((snapshots, step)) → returns immediately
  Serializer thread → queue.get() → JSON encode → parquet chunks → disk outbox
  Uploader thread → scan outbox → presign → PUT R2 → confirm → delete local

The snapshot extracts lightweight fields from rollouts so the orchestrator can
free the heavy token arrays immediately. The disk outbox is the durability
primitive: files survive crashes and are picked up on restart. Deterministic
file keys (step_N_chunk_M.parquet) prevent duplicates on checkpoint resume.
"""

import asyncio
import io
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import verifiers as vf
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.shared import PrimeMonitorConfig
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor

_SAMPLE_SCHEMA_VERSION = 2

_SAMPLE_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("step", pa.int64()),
        ("schema_version", pa.int64()),
        ("tag", pa.string()),
        ("problem_id", pa.int64()),
        ("sample_id", pa.int64()),
        ("prompt", pa.string()),
        ("completion", pa.string()),
        ("completion_text", pa.string()),
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
        ("num_turns", pa.int64()),
        ("num_tool_calls", pa.int64()),
        ("tools_used", pa.string()),
        ("is_completed", pa.bool_()),
        ("is_truncated", pa.bool_()),
        ("error", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ]
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _json(val: Any) -> str:
    """JSON-serialize dicts/lists, pass strings through, default to empty string for None."""
    if isinstance(val, str):
        return val
    if val is None:
        return ""
    return json.dumps(val)


def _flatten_completion_text(completion: list | str | None) -> str:
    """Extract plain text from completion messages for search indexing."""
    if not completion:
        return ""
    if isinstance(completion, str):
        return completion
    parts: list[str] = []
    for msg in completion:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
        elif isinstance(msg, str):
            parts.append(msg)
    return "\n".join(parts)


def _extract_tool_info(trajectory: list) -> tuple[int, list[str]]:
    """Count tool calls and collect tool names from trajectory steps."""
    tool_count = 0
    tool_names: set[str] = set()
    for ts in trajectory:
        completion = ts.get("completion", [])
        if not isinstance(completion, list):
            continue
        for msg in completion:
            if not isinstance(msg, dict):
                continue
            for tc in msg.get("tool_calls", []) or []:
                if isinstance(tc, dict) and "name" in tc:
                    tool_count += 1
                    tool_names.add(tc["name"])
    return tool_count, sorted(tool_names)


def _snapshot_rollouts(rollouts: list[vf.RolloutOutput]) -> list[dict[str, Any]]:
    """Extract fields needed for parquet rows without holding heavy token arrays.

    Runs on the main thread so must be fast — only shallow dict/list copies,
    no JSON encoding. This lets the orchestrator free the full rollout objects
    (with their large token arrays) while the serializer thread does the
    expensive work later.
    """
    snapshots: list[dict[str, Any]] = []
    for rollout in rollouts:
        prompt = rollout.get("prompt")
        completion = rollout.get("completion")
        trajectory = rollout.get("trajectory") or []
        if prompt is None or completion is None or not trajectory:
            continue
        snapshots.append(
            {
                "prompt": prompt,
                "completion": completion,
                "trajectory": [
                    {
                        "prompt": ts["prompt"],
                        "completion": ts["completion"],
                        "reward": ts.get("reward"),
                        "advantage": ts.get("advantage"),
                        "extras": ts.get("extras", {}),
                        "num_input_tokens": len(ts["tokens"]["prompt_ids"]) if ts.get("tokens") else 0,
                        "num_output_tokens": len(ts["tokens"]["completion_ids"]) if ts.get("tokens") else 0,
                    }
                    for ts in trajectory
                ],
                "example_id": rollout.get("example_id", 0),
                "answer": rollout.get("answer") or "",
                "task": rollout.get("task") or "",
                "reward": rollout.get("reward"),
                "advantage": rollout.get("advantage"),
                "info": rollout.get("info"),
                "metrics": rollout.get("metrics"),
                "timing": rollout.get("timing"),
                "error": rollout.get("error"),
                "is_completed": rollout.get("is_completed", False),
                "is_truncated": rollout.get("is_truncated", False),
            }
        )
    return snapshots


def _write_atomic(path: Path, data: bytes) -> None:
    """Write bytes to path atomically via .tmp rename."""
    tmp_path = path.with_suffix(".parquet.tmp")
    tmp_path.write_bytes(data)
    tmp_path.rename(path)


def _serialize_rows_to_parquet(rows: list[dict[str, Any]]) -> bytes:
    """Serialize row dicts to Parquet bytes."""
    table = pa.Table.from_pylist(rows, schema=_SAMPLE_SCHEMA)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy", use_dictionary=True, write_statistics=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PrimeMonitor
# ---------------------------------------------------------------------------


class PrimeMonitor(Monitor):
    """Durable rollout logging to Prime Intellect API.

    Uses a disk outbox and two background threads (serializer + uploader)
    so the training loop is never blocked by monitoring I/O.
    """

    def __init__(
        self,
        config: PrimeMonitorConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseConfig | None = None,
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

        api_key = os.getenv(config.api_key_var)
        if not api_key:
            self.logger.warning(
                f"API key not found. Set {config.api_key_var} environment variable. "
                "PrimeMonitor will not be able to upload data."
            )
            self.enabled = False
            return

        self.api_key = api_key
        self.base_url = config.base_url

        run_id = os.getenv("RUN_ID")
        if not run_id:
            self.logger.warning("RUN_ID environment variable not set. PrimeMonitor will not be able to upload data.")
            self.enabled = False
            return
        self.run_id = run_id

        # Async HTTP client for metrics/distributions (fire-and-forget)
        self._init_async_client()
        os.register_at_fork(after_in_child=self._reinit_after_fork)

        # Sample logging setup
        if config.log_extras and config.log_extras.samples:
            self.last_log_samples_step = -1
            self.tokenizer = tokenizer
            self._init_sample_pipeline(config)
        if config.log_extras and config.log_extras.distributions:
            self.last_log_distributions_step = -1

    # -------------------------------------------------------------------
    # Sample pipeline: outbox + serializer + uploader
    # -------------------------------------------------------------------

    def _init_sample_pipeline(self, config: PrimeMonitorConfig) -> None:
        """Set up the outbox directory, work queue, and background threads."""
        assert config.log_extras is not None

        if self.output_dir:
            self._outbox_dir = self.output_dir / ".monitor_outbox" / "samples"
        else:
            self._outbox_dir = Path("/tmp") / f"prime_monitor_outbox_{self.run_id}" / "samples"
        self._outbox_dir.mkdir(parents=True, exist_ok=True)

        # Clean orphaned .tmp files from previous crash before starting threads
        for tmp_file in self._outbox_dir.glob("*.parquet.tmp"):
            tmp_file.unlink(missing_ok=True)

        leftover_count = len(list(self._outbox_dir.glob("*.parquet")))
        if leftover_count > 0:
            self.logger.info(
                f"Outbox has {leftover_count} leftover file(s) from previous run, uploader will pick them up"
            )

        self._chunk_size = config.log_extras.chunk_size
        self._stop_event = Event()
        self._work_queue: Queue[tuple[list[dict[str, Any]], int]] = Queue(maxsize=8)

        self._upload_client = httpx.Client(
            timeout=60,
            transport=httpx.HTTPTransport(retries=2),
        )

        self._serializer_thread = Thread(target=self._serializer_loop, daemon=True, name="prime-monitor-serializer")
        self._serializer_thread.start()

        self._uploader_thread = Thread(target=self._uploader_loop, daemon=True, name="prime-monitor-uploader")
        self._uploader_thread.start()

    def _serializer_loop(self) -> None:
        """Pull (snapshots, step) from queue, build rows, serialize to parquet, write to outbox."""
        while not self._stop_event.is_set():
            try:
                snapshots, step = self._work_queue.get(timeout=1.0)
            except Empty:
                continue
            try:
                rows = self._snapshots_to_rows(snapshots, step)
                for chunk_idx, chunk_start in enumerate(range(0, len(rows), self._chunk_size)):
                    chunk_rows = rows[chunk_start : chunk_start + self._chunk_size]
                    parquet_bytes = _serialize_rows_to_parquet(chunk_rows)
                    filename = f"step_{step}_chunk_{chunk_idx}.parquet"
                    _write_atomic(self._outbox_dir / filename, parquet_bytes)
                    self.logger.debug(
                        f"Serialized {len(chunk_rows)} rows to outbox: {filename} ({len(parquet_bytes) / 1024:.1f} KB)"
                    )
            except Exception as e:
                self.logger.warning(f"Serializer error at step {step}: {e}")
            finally:
                del snapshots
                self._work_queue.task_done()

    def _uploader_loop(self) -> None:
        """Scan outbox for .parquet files, upload each, delete on success."""
        fail_counts: dict[str, int] = {}
        max_retries = 5

        while not self._stop_event.is_set():
            self._upload_pending_files(fail_counts, max_retries)
            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=1.0)

        # Final drain after stop signal — upload remaining outbox files
        self._upload_pending_files(fail_counts, max_retries)

    _OUTBOX_FILENAME_PATTERN = re.compile(r"^step_(\d+)_chunk_(\d+)\.parquet$")

    def _upload_pending_files(self, fail_counts: dict[str, int], max_retries: int) -> None:
        """Upload all .parquet files in the outbox."""
        pattern = self._OUTBOX_FILENAME_PATTERN

        for filepath in sorted(self._outbox_dir.glob("*.parquet")):
            name = filepath.name
            if not pattern.match(name):
                self.logger.warning(f"Skipping unexpected file in outbox: {name}")
                continue

            if fail_counts.get(name, 0) >= max_retries:
                continue

            try:
                self._upload_one_file(filepath)
                fail_counts.pop(name, None)
            except Exception as e:
                fail_counts[name] = fail_counts.get(name, 0) + 1
                count = fail_counts[name]
                if count >= max_retries:
                    self.logger.error(f"Giving up on {name} after {max_retries} failures: {e}")
                else:
                    self.logger.warning(f"Uploader error for {name} (attempt {count}/{max_retries}): {e}")
                    time.sleep(min(2.0**count, 30.0))

    def _upload_one_file(self, filepath: Path) -> None:
        """Upload a single outbox file: presign → PUT → confirm → delete."""
        filename = filepath.name
        match = self._OUTBOX_FILENAME_PATTERN.match(filename)
        if not match:
            raise ValueError(f"Unexpected filename format: {filename}")

        step = int(match.group(1))
        parquet_bytes = filepath.read_bytes()
        s3_key = f"rft/runs/{self.run_id}/samples/{filename}"

        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        client = self._upload_client

        presign_resp = client.post(
            f"{self.base_url}/samples/presign",
            headers=headers,
            json={"run_id": self.run_id, "step": step, "s3_key": s3_key},
        )
        presign_resp.raise_for_status()
        presigned_url = presign_resp.json()["presigned_url"]

        put_resp = client.put(
            presigned_url,
            content=parquet_bytes,
            headers={"Content-Type": "application/parquet"},
        )
        put_resp.raise_for_status()

        file_stats = self._compute_file_stats(parquet_bytes)
        confirm_payload: dict[str, Any] = {
            "run_id": self.run_id,
            "step": step,
            "s3_key": s3_key,
            "size_bytes": len(parquet_bytes),
            **file_stats,
        }
        confirm_resp = client.post(
            f"{self.base_url}/samples/confirm",
            headers=headers,
            json=confirm_payload,
        )
        confirm_resp.raise_for_status()

        filepath.unlink(missing_ok=True)
        self.logger.debug(f"Uploaded and confirmed: {filename}")

    @staticmethod
    def _compute_file_stats(parquet_bytes: bytes) -> dict[str, Any]:
        """Extract metadata from parquet bytes for the manifest."""
        try:
            buf = io.BytesIO(parquet_bytes)
            table = pq.read_table(
                buf,
                columns=[
                    "reward",
                    "task",
                    "error",
                    "num_tool_calls",
                    "schema_version",
                ],
            )
            stats: dict[str, Any] = {"row_count": table.num_rows}

            if "schema_version" in table.column_names:
                sv = table.column("schema_version")
                if sv.null_count < len(sv):
                    stats["schema_version"] = sv.drop_null()[0].as_py()

            if "reward" in table.column_names:
                rewards = table.column("reward").drop_null()
                if len(rewards) > 0:
                    stats["reward_min"] = pc.min(rewards).as_py()
                    stats["reward_max"] = pc.max(rewards).as_py()

            if "task" in table.column_names:
                tasks = table.column("task").drop_null().to_pylist()
                unique_tasks = sorted(set(t for t in tasks if t))
                if unique_tasks:
                    stats["tasks"] = unique_tasks

            if "error" in table.column_names:
                errors = table.column("error").to_pylist()
                stats["has_errors"] = any(e and e != "" for e in errors)

            if "num_tool_calls" in table.column_names:
                tool_calls = table.column("num_tool_calls").drop_null().to_pylist()
                stats["has_tool_calls"] = any(tc > 0 for tc in tool_calls)

            return stats
        except Exception:
            return {}

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.history.append(metrics)
        if not self.is_master or not self.enabled:
            return
        self._make_request("metrics", {"run_id": self.run_id, "metrics": metrics})

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        """Snapshot rollout data and enqueue for background serialization.

        Only lightweight field extraction runs on the main thread (~microseconds).
        All JSON encoding, parquet serialization, and I/O happen in background threads.
        """
        if not self.is_master or not self.enabled:
            return
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            return

        max_samples = self.config.log_extras.max_samples
        if max_samples is not None and len(rollouts) > max_samples:
            rollouts = random.sample(rollouts, max_samples)

        snapshots = _snapshot_rollouts(rollouts)
        if not snapshots:
            return

        self.logger.info(f"Enqueuing {len(snapshots)} samples for step {step}")
        try:
            self._work_queue.put_nowait((snapshots, step))
        except Exception:
            self.logger.warning(
                f"Sample queue full at step {step}, dropping {len(snapshots)} samples "
                "(serializer may be falling behind)"
            )
            return
        self.last_log_samples_step = step

    def log_final_samples(self) -> None:
        pass

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if not self.is_master or not self.enabled:
            return
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.distributions
            or step % self.config.log_extras.interval != 0
        ):
            return
        self.last_log_distributions_step = step
        self._make_request(
            "distributions",
            {"run_id": self.run_id, "step": step, "distributions": distributions},
        )

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        if not self.is_master or not self.enabled:
            return
        self._make_request(
            "finalize",
            {"run_id": self.run_id, "summary": self.history[-1] if self.history else {}},
        )

    def close(self) -> None:
        """Drain queue, wait for uploads to finish, then shut down."""
        if getattr(self, "_closed", False):
            return
        self._closed = True

        if not hasattr(self, "_stop_event"):
            self._close_async_client()
            return

        self.logger.info("Closing PrimeMonitor: draining queue and outbox...")

        # Wait for serializer to finish current work (bounded to avoid hung shutdown)
        self._work_queue.join()

        # Signal threads to stop after serializer is drained
        self._stop_event.set()

        # Wait for serializer to exit (should be fast since queue is empty)
        self._serializer_thread.join(timeout=15.0)
        if self._serializer_thread.is_alive():
            self.logger.warning("Serializer thread did not exit in time")

        # Give uploader time to flush remaining outbox files to R2
        self._uploader_thread.join(timeout=120.0)
        if self._uploader_thread.is_alive():
            self.logger.warning("Uploader thread did not exit in time, some files may remain in outbox")

        remaining = list(self._outbox_dir.glob("*.parquet"))
        if remaining:
            self.logger.warning(f"{len(remaining)} file(s) still in outbox after shutdown")

        if hasattr(self, "_upload_client"):
            try:
                self._upload_client.close()
            except Exception:
                pass

        self._close_async_client()
        self.logger.info("PrimeMonitor closed")

    def __del__(self) -> None:
        self.close()

    # -------------------------------------------------------------------
    # Row building (shared between old and new path)
    # -------------------------------------------------------------------

    def _snapshots_to_rows(self, snapshots: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
        """Convert lightweight snapshots to parquet-ready row dicts.

        Runs in the serializer thread — all JSON encoding happens here,
        not on the main orchestrator thread.
        """
        now = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []

        for idx, snap in enumerate(snapshots):
            trajectory = snap["trajectory"]

            num_tool_calls, tool_names = _extract_tool_info(trajectory)
            total_input_tokens = sum(ts.get("num_input_tokens", 0) for ts in trajectory)
            total_output_tokens = sum(ts.get("num_output_tokens", 0) for ts in trajectory)

            error_info = snap.get("error")
            if isinstance(error_info, dict):
                error_str = json.dumps(error_info)
            elif isinstance(error_info, str):
                error_str = error_info
            else:
                error_str = ""

            rows.append(
                {
                    "run_id": self.run_id,
                    "step": step,
                    "schema_version": _SAMPLE_SCHEMA_VERSION,
                    "tag": "",
                    "problem_id": snap.get("example_id", 0),
                    "sample_id": idx,
                    "prompt": json.dumps(snap["prompt"]),
                    "completion": json.dumps(snap["completion"]),
                    "completion_text": _flatten_completion_text(snap["completion"]),
                    "trajectory": json.dumps(trajectory),
                    "answer": snap.get("answer") or "",
                    "task": snap.get("task") or "",
                    "info": _json(snap.get("info")),
                    "reward": snap.get("reward"),
                    "advantage": snap.get("advantage"),
                    "metrics": _json(snap.get("metrics")),
                    "timing": _json(snap.get("timing")),
                    "num_input_tokens": total_input_tokens,
                    "num_output_tokens": total_output_tokens,
                    "num_turns": len(trajectory),
                    "num_tool_calls": num_tool_calls,
                    "tools_used": json.dumps(tool_names),
                    "is_completed": bool(snap.get("is_completed", False)),
                    "is_truncated": bool(snap.get("is_truncated", False)),
                    "error": error_str,
                    "created_at": now,
                }
            )

        return rows

    # -------------------------------------------------------------------
    # Async HTTP client (for metrics/distributions — unchanged)
    # -------------------------------------------------------------------

    def _init_async_client(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        self._client = httpx.AsyncClient(timeout=30)
        self._pending_futures: list[asyncio.Future] = []

    def _reinit_after_fork(self) -> None:
        self._init_async_client()

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _close_async_client(self) -> None:
        if not hasattr(self, "_client"):
            return
        self._flush()
        try:
            future = asyncio.run_coroutine_threadsafe(self._client.aclose(), self._loop)
            future.result(timeout=5.0)
        except Exception as e:
            self.logger.debug(f"Error closing HTTP client: {e}")
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
        except Exception as e:
            self.logger.debug(f"Error stopping event loop: {e}")

    def _flush(self, timeout: float = 30.0) -> None:
        if not self.enabled or not hasattr(self, "_loop"):
            return
        if not self._pending_futures:
            return
        for future in self._pending_futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                self.logger.debug(f"Pending request completed with error: {e}")
        self._pending_futures.clear()

    async def _make_request_async(self, endpoint: str, data: dict[str, Any], max_retries: int = 3) -> None:
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(max_retries):
            try:
                resp = await self._client.post(url, headers=headers, json=data)
                resp.raise_for_status()
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.warning(f"Failed {endpoint} after {max_retries} attempts: {e}")
                else:
                    await asyncio.sleep(2**attempt)

    def _make_request(self, endpoint: str, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        future = asyncio.run_coroutine_threadsafe(self._make_request_async(endpoint, data), self._loop)
        self._pending_futures.append(future)
        self._pending_futures = [f for f in self._pending_futures if not f.done()]
