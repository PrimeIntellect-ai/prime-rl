"""Sync platform evaluations from Prime Traces into run dirs the dashboard serves
like local eval runs (``~/.cache/prime-rl/dashboard/platform/<evaluation id>/``)."""

import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import orjson
from prime_traces import PrimeTracesError, TracesClient, TransportError
from prime_traces.core.config import Config

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.entrypoints.dashboard import STATE_DIR
from prime_rl.monitors.file.traces import get_index_path, get_trace_stream
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl
from prime_rl.monitors.file.traces.index import index_row
from prime_rl.utils.pathing import get_eval_plan_path, get_platform_run_path

PLATFORM_DIR = STATE_DIR / "platform"
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
POLL_S = 15.0
# created_at is the producer's clock at trace start, so uploads can land after later-sorting episodes
FOLLOW_MARGIN = timedelta(hours=1)
FETCH_WORKERS = 8
FETCH_ATTEMPTS = 3
FINISHED_PASSES = 3
OPTS = orjson.OPT_APPEND_NEWLINE


class PlatformError(Exception):
    """A platform failure retrying won't fix (no credentials, no access)."""


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    tmp.replace(path)


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class Platform:
    def __init__(self) -> None:
        self.config = Config()
        if not self.config.api_key:
            raise PlatformError("not logged in - run `prime login`")
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        if self.config.team_id:
            headers["X-Prime-Team-ID"] = self.config.team_id
        self.http = httpx.Client(base_url=f"{self.config.base_url}/api/v1", headers=headers, timeout=30)
        self.traces = TracesClient()

    def get(self, path: str, **params: Any) -> dict:
        response = self.http.get(path, params={k: v for k, v in params.items() if v is not None})
        if response.status_code in (401, 403, 404):
            raise PlatformError(f"{response.status_code} from {path}: {response.text[:200]}")
        response.raise_for_status()
        return response.json()

    def evaluations(self, limit: int, skip: int) -> dict:
        return self.get("/evaluations/", team_id=self.config.team_id, limit=limit, skip=skip)

    def evaluation(self, evaluation_id: str) -> dict:
        return self.get(f"/evaluations/{evaluation_id}")

    def account(self) -> dict:
        return {"base_url": self.config.base_url, "team_id": self.config.team_id}


def eval_settings(evaluation: dict) -> dict:
    # hosted evaluations keep these in eval_config, ones opened by a prime-rl eval in metadata
    return {**(evaluation.get("metadata") or {}), **(evaluation.get("eval_config") or {})}


def env_name(evaluation: dict, run_dir: Path) -> str:
    """The env the written episodes are filed under (``env.name``, or ``env.id`` on older
    uploads), else the platform's name for it."""
    index = get_index_path(get_trace_stream(run_dir))
    if index.is_file():
        with index.open("rb") as f:
            if (first := f.readline()).strip():
                return orjson.loads(first)["env"]
    names = evaluation.get("environment_names") or []
    return names[0] if names else evaluation["evaluation_id"]


def expected_episodes(evaluation: dict) -> int | None:
    settings = eval_settings(evaluation)
    examples, rollouts = settings.get("num_examples"), settings.get("rollouts_per_example")
    if not isinstance(examples, int) or examples < 0 or not isinstance(rollouts, int):
        return None
    return examples * rollouts


def eval_config(evaluation: dict, name: str) -> dict:
    settings = eval_settings(evaluation)
    source: dict[str, Any] = {"name": name, "env": {"taskset": {"id": name}}}
    if isinstance(settings.get("rollouts_per_example"), int):
        source["group_size"] = settings["rollouts_per_example"]
    if expected_episodes(evaluation) is not None:
        source["num_examples"] = settings["num_examples"]
    return {
        "model": evaluation.get("inference_model") or evaluation.get("model_name"),
        "source": [source],
        "sampling": settings.get("sampling_args") or {},
    }


def write_run_files(run_dir: Path, evaluation: dict) -> None:
    step = eval_settings(evaluation).get("step") or 0
    name = env_name(evaluation, run_dir)
    configs = run_dir / "configs"
    write_json(configs / "eval.json", eval_config(evaluation, name))
    write_json(configs / "platform.json", evaluation)
    # the dashboard dates a run by its config dir's mtime
    created = timestamp(evaluation["created_at"])
    os.utime(configs, (created, created))
    if (expected := expected_episodes(evaluation)) is not None:
        write_json(get_eval_plan_path(run_dir), {name: {str(step): expected}})
    write_json(
        get_platform_run_path(run_dir),
        {
            "kind": "eval",
            "source": "traces",
            "name": evaluation.get("name"),
            "run_id": evaluation.get("run_id"),
            "evaluations": {
                name: {"step": step, "id": evaluation["evaluation_id"], "url": evaluation.get("viewer_url")}
            },
        },
    )


def written_episode_ids(run_dir: Path) -> list[str]:
    index = get_index_path(get_trace_stream(run_dir))
    if not index.is_file():
        return []
    return [orjson.loads(line)["id"] for line in index.read_bytes().splitlines() if line.strip()]


def list_new_episodes(client: TracesClient, run_id: str, seen: set[str], created_after: datetime | None) -> list:
    episodes, cursor = [], None
    while True:
        page = client.list_episodes(
            run_id=run_id,
            created_after=created_after.isoformat() if created_after else None,
            cursor=cursor,
            limit=100,
        )
        episodes.extend(episode for episode in page.items if episode.episode_id not in seen)
        cursor = page.next_cursor
        if not cursor:
            break
    return sorted(episodes, key=lambda episode: (episode.created_at, episode.episode_id))


def read_raw(read: Callable[[str], bytes], document_id: str) -> Any:
    # a stream dropped midway through a large trace is past the SDK's own retries
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            return orjson.loads(read(document_id))
        except TransportError:
            if attempt == FETCH_ATTEMPTS:
                raise


def episode_record(client: TracesClient, episode_id: str) -> dict:
    """The stored envelope with its member traces inlined, as the file monitor writes it."""
    envelope = read_raw(client.get_episode_raw, episode_id)
    envelope["traces"] = [read_raw(client.get_raw, trace_id) for trace_id in envelope.get("traces") or []]
    # older uploads have no group; an eval's group is the rollouts of one task
    if not envelope.get("group") and (key := (envelope.get("task") or {}).get("key")):
        envelope["group"] = {"id": key}
    return envelope


def fetch_records(client: TracesClient, episode_ids: list[str]) -> Iterator[tuple[str, dict | PrimeTracesError]]:
    def fetch(episode_id: str) -> tuple[str, dict | PrimeTracesError]:
        try:
            return episode_id, episode_record(client, episode_id)
        except PrimeTracesError as e:
            return episode_id, e

    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        yield from pool.map(fetch, episode_ids)


def append_episodes(
    run_dir: Path, records: Iterator[tuple[str, dict | PrimeTracesError]], line: int
) -> tuple[int, dict[str, str]]:
    """Returns how many records were written and the errors of those that weren't."""
    stream_dir = get_trace_stream(run_dir)
    stream = ChunkedJsonl(stream_dir, FileMonitorConfig().chunk_bytes, compress=False)
    written, failed = 0, {}
    with open(get_index_path(stream_dir), "ab") as index:
        for episode_id, record in records:
            if isinstance(record, PrimeTracesError):
                failed[episode_id] = f"{type(record).__name__}: {record}"
                continue
            chunk, offset = stream.append(orjson.dumps(record, default=str, option=OPTS))
            stream.flush()
            written += 1
            index.write(orjson.dumps(index_row(line + written, record, chunk, offset), default=str, option=OPTS))
            index.flush()
    stream.close()
    return written, failed


def seal(run_dir: Path, evaluation: dict) -> None:
    stream_dir = get_trace_stream(run_dir)
    ChunkedJsonl(stream_dir, FileMonitorConfig().chunk_bytes, compress=True).close()
    # a finished run's duration ends at its stream's mtime
    ended = timestamp(evaluation.get("completed_at") or evaluation["updated_at"])
    os.utime(stream_dir, (ended, ended))


@dataclass
class SyncJob:
    evaluation_id: str
    state: str = "starting"  # starting, syncing, following, done, empty, error
    status: str | None = None
    written: int = 0
    expected: int | None = None
    failed: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    thread: threading.Thread | None = None

    def view(self) -> dict:
        return {
            "state": self.state,
            "status": self.status,
            "written": self.written,
            "expected": self.expected,
            "failed": len(self.failed),
            "error": self.error or next(iter(self.failed.values()), None),
        }


class PlatformSync:
    """One background sync per evaluation, running until it has every episode of a
    finished evaluation."""

    def __init__(self, root: Path = PLATFORM_DIR) -> None:
        self.root = root
        self.jobs: dict[str, SyncJob] = {}
        self._lock = threading.Lock()
        self._platform: Platform | None = None

    def platform(self) -> Platform:
        if self._platform is None:
            self._platform = Platform()
        return self._platform

    def run_dir(self, evaluation_id: str) -> Path:
        return self.root / evaluation_id

    def start(self, evaluation_id: str) -> SyncJob:
        """Writes the run files before returning, so the run is listed right away."""
        with self._lock:
            job = self.jobs.get(evaluation_id)
            if job is not None and job.thread is not None and job.thread.is_alive():
                return job
            evaluation = self.platform().evaluation(evaluation_id)
            write_run_files(self.run_dir(evaluation_id), evaluation)
            job = SyncJob(evaluation_id, status=evaluation["status"], expected=expected_episodes(evaluation))
            job.thread = threading.Thread(target=self._run, args=(job,), daemon=True, name=f"sync-{evaluation_id}")
            self.jobs[evaluation_id] = job
            job.thread.start()
            return job

    def _run(self, job: SyncJob) -> None:
        newest: datetime | None = None
        finished_passes = 0
        while True:
            try:
                newest = self.sync_pass(job, newest)
            except (httpx.HTTPError, PrimeTracesError, PlatformError) as e:
                job.error = f"{type(e).__name__}: {e}"
                if isinstance(e, PlatformError):
                    job.state = "error"
                    return
                time.sleep(POLL_S)
                continue
            job.error = None
            if job.status in TERMINAL_STATUSES:
                finished_passes += 1
                if not job.failed:
                    # nothing in Traces for a finished evaluation: its samples went elsewhere
                    job.state = "done" if job.written else "empty"
                    return
                if finished_passes >= FINISHED_PASSES:
                    job.state = "error"
                    return
            time.sleep(POLL_S)

    def sync_pass(self, job: SyncJob, newest: datetime | None) -> datetime | None:
        """Appends the episodes not written yet; returns the newest ``created_at`` written."""
        platform = self.platform()
        run_dir = self.run_dir(job.evaluation_id)
        # read the status before listing: a terminal evaluation has uploaded everything
        evaluation = platform.evaluation(job.evaluation_id)
        job.status, job.expected = evaluation["status"], expected_episodes(evaluation)
        finished = job.status in TERMINAL_STATUSES
        job.state = "syncing" if finished or not job.written else "following"
        write_run_files(run_dir, evaluation)
        written_ids = written_episode_ids(run_dir)
        # re-list everything to retry failed episodes, and before sealing to catch late
        # uploads outside the follow window
        created_after = newest - FOLLOW_MARGIN if newest and not job.failed and not finished else None
        episodes = list_new_episodes(platform.traces, job.evaluation_id, set(written_ids), created_after)
        records = fetch_records(platform.traces, [episode.episode_id for episode in episodes])
        written, job.failed = append_episodes(run_dir, records, len(written_ids))
        job.written = len(written_ids) + written
        if written and not written_ids:
            write_run_files(run_dir, evaluation)  # the first episodes name the env
        if finished and not job.failed:
            seal(run_dir, evaluation)
        landed = [episode.created_at for episode in episodes if episode.episode_id not in job.failed]
        return max(filter(None, (newest, *landed)), default=None)

    def views(self) -> dict[str, dict]:
        return {evaluation_id: job.view() for evaluation_id, job in self.jobs.items()}
