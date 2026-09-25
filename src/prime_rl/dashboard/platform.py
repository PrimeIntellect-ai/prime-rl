"""Hosted evaluations in the local dashboard: ``dashboard-sync <evaluation id>`` pulls
a platform evaluation out of Prime Traces into a run directory laid out the way the
file monitor writes one, so the dashboard serves it like any local run.

    ~/.cache/prime-rl/dashboard/platform/<evaluation id>/
        configs/eval.json            what the overview reads: model, env, expected episodes
        configs/platform.json        the evaluation record as the platform returned it
        monitors/file/plan.json      the epoch's expected episode count
        monitors/file/traces/stream  every episode with its member traces inlined, and its index
        monitors/prime/run.json      the "view on platform" link

Traces stores an episode's envelope with ``traces`` reduced to member ids and each
trace on its own, so a record is the envelope with its members fetched back in. The
directory is only appended to: a rerun (or ``--follow``) fetches just the episodes it
has not written yet, and a finished evaluation's stream is sealed, which is what makes
the dashboard read it as completed.
"""

import argparse
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import orjson
from prime_traces import PrimeTracesError, TracesClient
from prime_traces.core.config import Config

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.entrypoints.dashboard import STATE_DIR, ensure_dashboard, log_dashboard_url
from prime_rl.monitors.file.traces import get_index_path, get_trace_stream
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl
from prime_rl.monitors.file.traces.index import index_row
from prime_rl.utils.pathing import get_eval_plan_path, get_platform_run_path

PLATFORM_DIR = STATE_DIR / "platform"
"""The output dir every synced run lives under, registered with the dashboard once."""

TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})

FOLLOW_MARGIN = timedelta(hours=1)
"""How far behind the newest episode a follow poll re-lists. An episode's ``created_at``
is its producer's clock when its first trace started, so an upload can land well after
episodes that sort later."""

FETCH_WORKERS = 8
"""Episodes fetched concurrently: each is one request plus one per member trace."""

OPTS = orjson.OPT_APPEND_NEWLINE


def write_json(path: Path, data: Any) -> None:
    """Atomic replace, so the dashboard never reads a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    tmp.replace(path)


def get_evaluation(config: Config, evaluation_id: str) -> dict:
    headers = {"Authorization": f"Bearer {config.api_key}"}
    if config.team_id:
        headers["X-Prime-Team-ID"] = config.team_id
    response = httpx.get(f"{config.base_url}/api/v1/evaluations/{evaluation_id}", headers=headers, timeout=30)
    if response.status_code == 404:
        raise SystemExit(
            f"{evaluation_id} is not a platform evaluation visible to this account (only evaluations sync)"
        )
    response.raise_for_status()
    return response.json()


def eval_settings(evaluation: dict) -> dict:
    """The run's sampling shape: hosted evaluations keep it in ``eval_config``,
    ones a prime-rl eval opened in ``metadata``."""
    return {**(evaluation.get("metadata") or {}), **(evaluation.get("eval_config") or {})}


def env_name(evaluation: dict, run_dir: Path) -> str:
    """The env the run's episodes are filed under, read off the first one written
    (``env.name``, or ``env.id`` from producers that recorded no name); before any
    lands, the platform's name for it. An evaluation covers one env."""
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
    """The slice of a prime-rl eval config the dashboard reads, rebuilt from the
    evaluation: one ``[[source]]`` named for its env, with the platform's group size
    and, when the evaluation fixed one, its example count."""
    settings = eval_settings(evaluation)
    source: dict[str, Any] = {"name": name, "env": {"taskset": {"id": name}}}
    if isinstance(settings.get("rollouts_per_example"), int):
        source["group_size"] = settings["rollouts_per_example"]
    # a negative count means the whole taskset, which the dashboard reads as unknown up front
    if expected_episodes(evaluation) is not None:
        source["num_examples"] = settings["num_examples"]
    return {
        "model": evaluation.get("inference_model") or evaluation.get("model_name"),
        "source": [source],
        "sampling": settings.get("sampling_args") or {},
    }


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def write_run_files(run_dir: Path, evaluation: dict, step: int) -> None:
    """Everything but the stream, rewritten each pass so a status change shows up."""
    name = env_name(evaluation, run_dir)
    configs = run_dir / "configs"
    write_json(configs / "eval.json", eval_config(evaluation, name))
    write_json(configs / "platform.json", evaluation)
    # the dashboard dates a run by its config dir: the evaluation's creation, not the sync
    created = timestamp(evaluation["created_at"])
    os.utime(configs, (created, created))
    if (expected := expected_episodes(evaluation)) is not None:
        write_json(get_eval_plan_path(run_dir), {name: {str(step): expected}})
    write_json(
        get_platform_run_path(run_dir),
        {
            "kind": "eval",
            "run_id": evaluation.get("run_id"),
            "evaluations": {
                name: {"step": step, "id": evaluation["evaluation_id"], "url": evaluation.get("viewer_url")}
            },
        },
    )


def written_episode_ids(run_dir: Path) -> list[str]:
    """The ids already in the stream, in line order."""
    index = get_index_path(get_trace_stream(run_dir))
    if not index.is_file():
        return []
    return [orjson.loads(line)["id"] for line in index.read_bytes().splitlines() if line.strip()]


def list_new_episodes(client: TracesClient, run_id: str, seen: set[str], created_after: datetime | None) -> list:
    """The run's episodes not written yet, oldest first (the listing is newest first)."""
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


def episode_record(client: TracesClient, episode_id: str) -> dict:
    """The episode as the file monitor would have written it: the stored envelope
    with each member trace in place of its id."""
    envelope = orjson.loads(client.get_episode_raw(episode_id))
    envelope["traces"] = [orjson.loads(client.get_raw(trace_id)) for trace_id in envelope.get("traces") or []]
    # older producers recorded no group; an eval's group is the rollouts of one task
    if not envelope.get("group") and (key := (envelope.get("task") or {}).get("key")):
        envelope["group"] = {"id": key}
    return envelope


def fetch_records(client: TracesClient, episode_ids: list[str]) -> Iterator[dict]:
    """Records in the order given, fetched concurrently."""
    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        yield from pool.map(lambda episode_id: episode_record(client, episode_id), episode_ids)


def append_episodes(run_dir: Path, records: Iterator[dict], line: int, seal: bool) -> int:
    """Append records to the stream and its index, flushing each so a dashboard
    watching the run sees it fill in. Sealing marks the stream finished."""
    stream_dir = get_trace_stream(run_dir)
    stream = ChunkedJsonl(stream_dir, FileMonitorConfig().chunk_bytes, compress=seal)
    written = 0
    with open(get_index_path(stream_dir), "ab") as index:
        for record in records:
            chunk, offset = stream.append(orjson.dumps(record, default=str, option=OPTS))
            stream.flush()
            written += 1
            index.write(orjson.dumps(index_row(line + written, record, chunk, offset), default=str, option=OPTS))
            index.flush()
    stream.close()
    return written


def sync(
    client: TracesClient, config: Config, evaluation_id: str, run_dir: Path, created_after: datetime | None
) -> tuple[dict, int, datetime | None]:
    """One pass: refresh the run files, append the episodes not written yet. Returns
    the evaluation, how many episodes were written and the newest ``created_at``
    among them."""
    # read before listing: an evaluation already terminal here has uploaded everything
    evaluation = get_evaluation(config, evaluation_id)
    finished = evaluation["status"] in TERMINAL_STATUSES
    # a prime-rl eval epoch records its policy step; a hosted evaluation has one epoch
    step = eval_settings(evaluation).get("step") or 0
    write_run_files(run_dir, evaluation, step)
    written_ids = written_episode_ids(run_dir)
    episodes = list_new_episodes(client, evaluation_id, set(written_ids), created_after)
    records = fetch_records(client, [episode.episode_id for episode in episodes])
    written = append_episodes(run_dir, records, len(written_ids), seal=finished)
    if written and not written_ids:
        write_run_files(run_dir, evaluation, step)  # the first episodes name the env
    if finished:
        # a finished run's duration ends at its stream's mtime: the evaluation's end, not the sync
        ended = timestamp(evaluation.get("completed_at") or evaluation["updated_at"])
        os.utime(get_trace_stream(run_dir), (ended, ended))
    return evaluation, written, max((episode.created_at for episode in episodes), default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a hosted evaluation from Prime Traces into the local dashboard")
    parser.add_argument("evaluation_id", help="the platform evaluation id (the last segment of its dashboard URL)")
    parser.add_argument("--follow", action="store_true", help="keep polling until the evaluation finishes")
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between polls with --follow")
    parser.add_argument("--dir", type=Path, default=PLATFORM_DIR, help="output dir the run directory is created in")
    parser.add_argument("--no-dashboard", action="store_true", help="don't register the dir or start a dashboard")
    args = parser.parse_args()

    from prime_rl.utils.logger import setup_logger

    logger = setup_logger()
    config = Config()
    if not config.api_key:
        raise SystemExit("API key not found - set PRIME_API_KEY or run `prime login`")
    run_dir = args.dir / args.evaluation_id
    dashboard_url = None if args.no_dashboard else ensure_dashboard(args.dir, logger)
    client = TracesClient()
    newest: datetime | None = None
    status = None
    while True:
        created_after = newest - FOLLOW_MARGIN if newest else None
        try:
            evaluation, written, pass_newest = sync(client, config, args.evaluation_id, run_dir, created_after)
        except (httpx.HTTPError, PrimeTracesError) as e:
            if not args.follow or status is None:
                raise
            # a follow outlives a failed poll: the next one resumes from what is on disk
            logger.warning(f"Poll failed, retrying in {args.interval:.0f}s: {type(e).__name__}: {e}")
            time.sleep(args.interval)
            continue
        newest = max(filter(None, (newest, pass_newest)), default=None)
        if written or evaluation["status"] != status:
            total = len(written_episode_ids(run_dir))
            logger.info(
                f"{evaluation['name']} ({evaluation['status']}): {written} new episodes, {total} total in {run_dir}"
            )
        if status is None:
            log_dashboard_url(logger, dashboard_url)
        status = evaluation["status"]
        if not args.follow or status in TERMINAL_STATUSES:
            break
        time.sleep(args.interval)
    client.close()
