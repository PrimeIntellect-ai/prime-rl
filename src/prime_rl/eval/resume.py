"""Resume an interrupted eval from its trace stream.

The stream records what landed, so it is what a resume continues from. The run's ok
episodes are read back and rejoin the epoch as if they had just arrived - through the
monitors, so the rebuilt stream, the epoch's metrics and the platform upload cover the
whole epoch - and only the rollouts still owed run. Errored episodes and the in-flight
ones the interruption cut off are owed again.

A landed episode counts toward the task with its ``task.key``, so a resumed run may
select more or fewer examples or rollouts per example than the interrupted one: the
kept episodes are matched to the new selection and the rest is owed. What defines the
measurement itself - the model, the sampling, each source's env - must not change
(``check_config``).
"""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path

import orjson
import verifiers.v1 as vf

from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.utils.pathing import get_config_dir, get_file_monitor_dir

RESUMABLE = (
    "resume",
    "clean",
    "dry_run",
    "dashboard",
    "num_examples",
    "group_size",
    "concurrency",
    "client",
    "log",
    "monitors",
    "source.*.num_examples",
    "source.*.group_size",
    "source.*.serve",
)
"""Config paths a resumed run may change: how many rollouts to run and how to run them,
never what is measured."""


def previous_config(run_dir: Path) -> dict:
    """The resolved config of the run's latest earlier attempt."""
    current = get_config_dir(run_dir)
    attempts = sorted(
        (path for path in (run_dir / "configs").glob("attempt_*/resolved/eval.json") if path.parent != current),
        key=lambda path: int(path.parts[-3].removeprefix("attempt_")),
    )
    if not attempts:
        raise FileNotFoundError(f"Nothing to resume: {run_dir} has no earlier eval attempt")
    return orjson.loads(attempts[-1].read_bytes())


def config_diff(previous, current, prefix: str = "") -> list[str]:
    """Dotted paths at which two resolved configs differ (list items by index)."""
    if isinstance(previous, dict) and isinstance(current, dict):
        return [
            path
            for key in sorted(set(previous) | set(current))
            for path in config_diff(previous.get(key), current.get(key), f"{prefix}.{key}" if prefix else key)
        ]
    if isinstance(previous, list) and isinstance(current, list) and len(previous) == len(current):
        return [
            path
            for index, (before, after) in enumerate(zip(previous, current, strict=True))
            for path in config_diff(before, after, f"{prefix}.{index}")
        ]
    return [] if previous == current else [prefix]


def check_config(previous: dict, current: dict) -> None:
    changed = [
        path
        for path in config_diff(previous, current)
        if not any(fnmatch(path, pattern) or fnmatch(path, f"{pattern}.*") for pattern in RESUMABLE)
    ]
    if changed:
        raise ValueError(
            f"The run cannot resume with a different {', '.join(changed)} - the landed episodes would not "
            "measure the same thing. Relaunch with --clean to start over."
        )


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


def take_landed(run_dir: Path) -> list[dict]:
    """The ok eval episodes the run has landed. The file monitor's directory is removed
    with them: the resumed attempt writes a fresh stream, plan and metrics."""
    stream = get_trace_stream(run_dir)
    landed = [record for record in read_records(stream) if record.get("ok")] if stream.is_dir() else []
    shutil.rmtree(get_file_monitor_dir(run_dir), ignore_errors=True)
    return landed


def plan(landed: list[dict], eval_envs: EvalEnvs) -> tuple[list[vf.WireEpisode], dict[str, dict[str, int]]]:
    """Match the landed episodes to the run's tasks: the episodes to keep, in stream
    order, and the rollouts still owed per env and task key."""
    targets: dict[str, Counter[str]] = {}
    for env in eval_envs:
        targets[env.name] = Counter(task.key for task in env.examples)
        for key in targets[env.name]:
            targets[env.name][key] *= env.config.group_size
    kept: list[vf.WireEpisode] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in landed:
        env_name = record["env"].get("name") or record["env"]["id"]
        key = record["task"]["key"]
        if counts[env_name][key] >= targets.get(env_name, Counter())[key]:
            continue
        kept.append(vf.WireEpisode.model_validate(record))
        counts[env_name][key] += 1
    owed = {
        env_name: {
            key: target - counts[env_name][key]
            for key, target in target_counts.items()
            if target > counts[env_name][key]
        }
        for env_name, target_counts in targets.items()
    }
    return kept, owed
