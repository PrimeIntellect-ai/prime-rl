"""Resume an interrupted eval from its trace stream.

The stream records what landed, so it is what a resume continues from. The run's ok
episodes are read back and rejoin the epoch as if they had just arrived - through the
monitors, so the rebuilt stream, the epoch's metrics and the platform upload cover the
whole epoch - and only the rollouts still owed run. Errored episodes and the in-flight
ones the interruption cut off are owed again.

A landed episode counts toward the task with its ``task.key``, so a resumed run may
select more or fewer examples or rollouts per example than the interrupted one: the
kept episodes are matched to the new selection and the rest is owed. The resumed
config is not checked against the interrupted one: any of it may be overridden.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import orjson
import verifiers.v1 as vf

from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from prime_rl.orchestrator.envs import EvalEnvs
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


def take_landed(run_dir: Path) -> list[dict]:
    """The ok eval episodes the run has landed, each once, from every attempt's stream.
    The current file monitor directory joins the archives so the resumed attempt writes a
    fresh stream, plan and metrics; nothing is deleted."""
    current = get_file_monitor_dir(run_dir)
    stream = get_trace_stream(run_dir).relative_to(current)
    landed: dict[str, dict] = {}
    for directory in [*archives(run_dir), current]:
        if (directory / stream).is_dir():
            for record in read_records(directory / stream):
                if record.get("ok"):
                    landed.setdefault(record["id"], record)
    if current.is_dir():
        current.rename(current.with_name(f"file.attempt_{len(archives(run_dir)) + 1}"))
    return list(landed.values())


def plan(
    landed: list[dict], eval_envs: EvalEnvs
) -> tuple[list[vf.WireEpisode], dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    """Match the landed episodes to the run's tasks: the episodes to keep, in stream
    order, the rollouts still owed per env and task key, and the group id a task's kept
    episodes carry, so the owed ones complete that group rather than open another."""
    targets: dict[str, Counter[str]] = {}
    for env in eval_envs:
        targets[env.name] = Counter(task.key for task in env.examples)
        for key in targets[env.name]:
            targets[env.name][key] *= env.config.group_size
    # a streaming env's tasks are not known up front: each landed key is owed one group
    streams = {env.name: env.config.group_size for env in eval_envs if env.feed is not None}
    kept: list[vf.WireEpisode] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for record in landed:
        env_name = record["env"].get("name") or record["env"]["id"]
        key = record["task"]["key"]
        if counts[env_name][key] >= streams.get(env_name, targets.get(env_name, Counter())[key]):
            continue
        kept.append(vf.WireEpisode.model_validate(record))
        counts[env_name][key] += 1
        if (group := record.get("group") or {}).get("id"):
            groups[env_name].setdefault(key, group["id"])
    owed = {
        env_name: {
            key: target - counts[env_name][key]
            for key, target in target_counts.items()
            if target > counts[env_name][key]
        }
        for env_name, target_counts in targets.items()
    }
    # complete keys stay at zero, so the stream skips them rather than run them fresh
    for env_name, group_size in streams.items():
        owed[env_name] = {key: group_size - count for key, count in counts[env_name].items()}
    return kept, owed, dict(groups)
