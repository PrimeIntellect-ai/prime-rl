"""Resume an interrupted eval from its trace stream.

The stream records what landed, so it is what a resume continues from. Episodes the
environment accepts rejoin the epoch through the monitors as if they had just arrived,
so the rebuilt stream, the epoch's metrics and the platform upload cover the
whole epoch - and only the rollouts still owed run. Rejected episodes and the in-flight
ones the interruption cut off are owed again.

The shared Verifiers rollout planner matches these episodes to the current tasks
by content hash. The resumed config is not checked against the interrupted one:
any of it may be overridden.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import orjson
import verifiers.v1 as vf

from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from prime_rl.utils.pathing import get_file_monitor_dir

CONFIG_NAME = "eval.json"
"""The resolved config an attempt stamps into its file monitor directory once it is
running, beside the episodes it produces, recording the config those episodes were
measured with."""


def stamp_config(run_dir: Path, config: dict) -> None:
    directory = get_file_monitor_dir(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONFIG_NAME).write_bytes(orjson.dumps(config, option=orjson.OPT_INDENT_2))


def read_records(stream: Path) -> Iterator[dict]:
    """Every record of a trace stream, in order. A torn last line (the interrupted
    process died mid-append) ends the stream."""
    for number in sorted(chunk_numbers(stream)):
        with open_chunk(stream, number) as chunk:
            for line in chunk:
                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError:
                    return


def archives(run_dir: Path) -> list[Path]:
    """The file monitor directories of the run's earlier attempts, oldest first: a resume
    renames the one it finds to ``monitors/file.attempt_N`` and starts a fresh one."""
    monitors = get_file_monitor_dir(run_dir).parent
    return sorted(monitors.glob("file.attempt_*"), key=lambda path: int(path.name.rsplit("_", 1)[1]))


def take_landed(run_dir: Path, complete: Callable[[vf.WireEpisode], bool]) -> list[vf.WireEpisode]:
    """Environment-accepted episodes from every attempt; the planner deduplicates them.
    The current file monitor directory joins the archives so the resumed attempt writes a
    fresh stream, plan and metrics; nothing is deleted."""
    current = get_file_monitor_dir(run_dir)
    stream = get_trace_stream(run_dir).relative_to(current)
    landed: list[vf.WireEpisode] = []
    for directory in [*archives(run_dir), current]:
        if (directory / stream).is_dir():
            for record in read_records(directory / stream):
                try:
                    if "traces" not in record:
                        continue
                    episode = vf.WireEpisode.model_validate(record)
                    if not complete(episode):
                        continue
                except Exception:
                    # A malformed record or failed completion check does not satisfy a rollout.
                    continue
                landed.append(episode)
    if current.is_dir():
        current.rename(current.with_name(f"file.attempt_{len(archives(run_dir)) + 1}"))
    return landed
