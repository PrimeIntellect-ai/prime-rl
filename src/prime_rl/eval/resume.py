"""Resume an interrupted eval from its trace stream.

The stream records what landed, so it is what a resume continues from. Successful
episodes rejoin the epoch through the monitors as if they had just arrived,
so the rebuilt stream, the epoch's metrics and the platform upload cover the
whole epoch - and only the rollouts still owed run. Failed episodes and the in-flight
ones the interruption cut off are owed again.

The shared Verifiers rollout planner matches episodes to tasks by content hash.
Each attempt's saved resolved config must match, except for ``resume``, before its
episodes are reused. Intentional changes require ``--resume.skip-checks``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import orjson
import verifiers.v1 as vf
from verifiers.v1.task import WireTaskData

from prime_rl.configs.eval import EvalConfig
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_file_monitor_dir

CONFIG_NAME = "eval.json"
"""The resolved config an attempt stamps into its file monitor directory once it is
running, beside the episodes it produces, recording the config those episodes were
measured with."""


def stamp_config(run_dir: Path, config: dict) -> None:
    directory = get_file_monitor_dir(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONFIG_NAME).write_bytes(orjson.dumps(config, option=orjson.OPT_INDENT_2))


def indexed_records(stream: Path) -> Iterator[tuple[int, int, dict]]:
    """Every record of a trace stream, in order. A torn last line (the interrupted
    process died mid-append) ends the stream."""
    for number in sorted(chunk_numbers(stream)):
        with open_chunk(stream, number) as chunk:
            offset = 0
            for line in chunk:
                try:
                    yield number, offset, orjson.loads(line)
                except orjson.JSONDecodeError:
                    return
                offset += len(line)


def archives(run_dir: Path) -> list[Path]:
    """The file monitor directories of the run's earlier attempts, oldest first: a resume
    renames the one it finds to ``monitors/file.attempt_N`` and starts a fresh one."""
    monitors = get_file_monitor_dir(run_dir).parent
    return sorted(monitors.glob("file.attempt_*"), key=lambda path: int(path.name.rsplit("_", 1)[1]))


@dataclass
class SavedEpisode:
    header: vf.WireEpisode
    stream: Path
    chunk: int
    offset: int

    def load(self, group: vf.GroupInfo | None) -> vf.WireEpisode:
        with open_chunk(self.stream, self.chunk) as chunk:
            chunk.seek(self.offset)
            episode = vf.WireEpisode.model_validate(orjson.loads(chunk.readline()))
        episode.group = group
        return episode


def replay(saved: dict[tuple[str, str], SavedEpisode], planned: Iterable[vf.WireEpisode]) -> Iterator[vf.WireEpisode]:
    for header in planned:
        yield saved[header.env.name or header.env.id, header.id].load(header.group)


def take_landed(run_dir: Path, config: EvalConfig) -> dict[tuple[str, str], SavedEpisode]:
    """Validated, distinct successful episode locations, oldest attempt first.
    The current file monitor directory joins the archives so the resumed attempt writes a
    fresh stream, plan and metrics; nothing is deleted."""
    current = get_file_monitor_dir(run_dir)
    stream = get_trace_stream(run_dir).relative_to(current)
    directories = archives(run_dir)
    if current.is_dir() or not directories:
        directories.append(current)
    # Enabling resume necessarily differs from the original launch.
    expected = config.model_dump(mode="json", exclude={"resume"})
    skip_checks = config.resume is not None and config.resume.skip_checks
    if skip_checks:
        get_logger().warning(
            "Skipping resume compatibility checks; saved episodes may use different experiment settings."
        )
    landed: dict[tuple[str, str], SavedEpisode] = {}
    archive = current.with_name(f"file.attempt_{len(archives(run_dir)) + 1}")
    for directory in directories:
        if not skip_checks:
            saved_path = directory / CONFIG_NAME
            if not saved_path.is_file():
                raise ValueError(
                    f"--resume: no saved config at {saved_path}. "
                    "Set --resume.skip-checks to reuse episodes without a saved config."
                )
            # Snapshots are resolved configs: re-validating would fill missing fields with new defaults.
            previous = orjson.loads(saved_path.read_bytes())
            previous.pop("resume", None)
            changed = sorted(
                key
                for key in expected.keys() | previous.keys()
                if key not in expected or key not in previous or expected[key] != previous[key]
            )
            if changed:
                raise ValueError(
                    f"--resume: config differs from {saved_path} in [{', '.join(changed)}]. "
                    "Use the saved config, start a fresh run, or set --resume.skip-checks if the changes are intentional."
                )
        if (directory / stream).is_dir():
            saved_stream = (archive if directory == current else directory) / stream
            for number, offset, record in indexed_records(directory / stream):
                episode = vf.WireEpisode.model_validate(record)
                key = (episode.env.name or episode.env.id, episode.id)
                if episode.ok and "traces" in record and key not in landed:
                    task_hash = episode.task.hash if episode.task.hash is not None else vf.Task(episode.task.data).hash
                    task = episode.task.model_copy(update={"hash": task_hash, "data": WireTaskData()})
                    header = vf.WireEpisode(id=episode.id, env=episode.env, task=task)
                    landed[key] = SavedEpisode(header, saved_stream, number, offset)
    if current.is_dir():
        current.rename(archive)
    return landed
