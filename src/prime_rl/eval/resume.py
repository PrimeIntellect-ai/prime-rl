"""Resume an interrupted eval from its trace stream.

The stream records what landed, so it is what a resume continues from. Successful
episodes rejoin the epoch through the monitors as if they had just arrived,
so the rebuilt stream, the epoch's metrics and the platform upload cover the
whole epoch - and only the rollouts still owed run. Failed episodes and the in-flight
ones the interruption cut off are owed again.

Each attempt's saved experiment settings must match before its episodes are reused.
Operational settings such as concurrency and monitoring may change between attempts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import orjson
import verifiers.v1 as vf
from pydantic import TypeAdapter

from prime_rl.configs.eval import EvalConfig
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_file_monitor_dir

CONFIG_NAME = "eval.json"
"""The resolved config an attempt stamps into its file monitor directory once it is
running, beside the episodes it produces, recording the config those episodes were
measured with."""

# Operational settings and group defaults already resolved into each source.
IGNORED_FIELDS = {
    "env": True,
    "sampling": True,
    "select": True,
    "group_size": True,
    "resume": True,
    "output_dir": True,
    "clean": True,
    "dry_run": True,
    "log": True,
    "dashboard": True,
    "monitors": True,
    "heartbeat": True,
    "concurrency": True,
    "tasks_per_minute": True,
    "client": {"wait_for_ready_timeout", "skip_model_check"},
    "source": {"__all__": {"serve": {"pool", "max_concurrent"}}},
}


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


def take_landed(run_dir: Path, config: EvalConfig) -> list[vf.WireEpisode]:
    """Successful episodes from every attempt; the planner deduplicates them.
    The current file monitor directory joins the archives so the resumed attempt writes a
    fresh stream, plan and metrics; nothing is deleted."""
    current = get_file_monitor_dir(run_dir)
    stream = get_trace_stream(run_dir).relative_to(current)
    directories = archives(run_dir)
    if current.is_dir() or not directories:
        directories.append(current)
    expected = config.model_dump(mode="json", exclude=IGNORED_FIELDS)
    # Sources are identified by unique names; their declaration order does not affect resume.
    expected["source"].sort(key=lambda source: orjson.dumps(source, option=orjson.OPT_SORT_KEYS))
    snapshot = TypeAdapter(dict)
    skip_checks = config.resume is not None and config.resume.skip_checks
    if skip_checks:
        get_logger().warning(
            "Skipping resume compatibility checks; saved episodes may use different experiment settings."
        )
    landed: list[vf.WireEpisode] = []
    for directory in directories:
        if not skip_checks:
            saved_path = directory / CONFIG_NAME
            if not saved_path.is_file():
                raise ValueError(f"--resume: no saved experiment config at {saved_path}")
            # Snapshots are resolved configs: re-validating would fill missing fields with new defaults.
            previous = snapshot.dump_python(snapshot.validate_json(saved_path.read_bytes()), exclude=IGNORED_FIELDS)
            if "source" in previous:
                previous["source"].sort(key=lambda source: orjson.dumps(source, option=orjson.OPT_SORT_KEYS))
            changed = sorted(
                key
                for key in expected.keys() | previous.keys()
                if key not in expected or key not in previous or expected[key] != previous[key]
            )
            if changed:
                raise ValueError(
                    f"--resume: config differs from {saved_path} in [{', '.join(changed)}]. "
                    "Use the saved experiment settings, start a fresh run, or explicitly set --resume.skip-checks."
                )
        if (directory / stream).is_dir():
            for record in read_records(directory / stream):
                episode = vf.WireEpisode.model_validate(record)
                if episode.ok and "traces" in record:
                    landed.append(episode)
    if current.is_dir():
        current.rename(current.with_name(f"file.attempt_{len(archives(run_dir)) + 1}"))
    return landed
