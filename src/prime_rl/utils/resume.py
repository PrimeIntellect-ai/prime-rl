"""Validate saved run configs and restore completed eval episodes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
import verifiers.v1 as vf

from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_file_monitor_dir

if TYPE_CHECKING:
    from prime_rl.configs.eval import EvalConfig
    from prime_rl.configs.rl import RLConfig
    from prime_rl.configs.sft import SFTConfig


EVAL_CONFIG_NAME = "eval.json"
"""Resolved eval config saved beside the episodes produced by each attempt."""


def check_resume_config(config: EvalConfig | RLConfig | SFTConfig, saved_paths: Iterable[Path]) -> None:
    """Require matching resolved configs before reusing saved run state."""
    if config.resume is not None and config.resume.skip_checks:
        get_logger().warning("Skipping resume compatibility checks; saved state may use different experiment settings.")
        return

    expected = config.model_dump(mode="json")
    for saved_path in saved_paths:
        if not saved_path.is_file():
            raise ValueError(
                f"--resume: no saved config at {saved_path}. "
                "Set --resume.skip-checks to reuse state without a saved config."
            )
        # Re-validating snapshots would fill missing fields with current defaults.
        previous = orjson.loads(saved_path.read_bytes())
        # Generated SLURM scripts run with dry_run=False and mark their attempt when started.
        if previous.get("dry_run"):
            if not saved_path.with_suffix(".started").is_file():
                continue
            previous["dry_run"] = False
        for snapshot in (expected, previous):
            # RL propagates resume into both components; enabling it must not cause a mismatch.
            snapshot.pop("resume", None)
            for component in ("trainer", "orchestrator"):
                if component in snapshot:
                    snapshot[component].pop("resume", None)
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


def stamp_config(run_dir: Path, config: dict) -> None:
    directory = get_file_monitor_dir(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / EVAL_CONFIG_NAME).write_bytes(orjson.dumps(config, option=orjson.OPT_INDENT_2))


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
    check_resume_config(config, (directory / EVAL_CONFIG_NAME for directory in directories))
    landed: list[vf.WireEpisode] = []
    for directory in directories:
        if (directory / stream).is_dir():
            for record in read_records(directory / stream):
                episode = vf.WireEpisode.model_validate(record)
                if episode.ok and "traces" in record:
                    landed.append(episode)
    if current.is_dir():
        current.rename(current.with_name(f"file.attempt_{len(archives(run_dir)) + 1}"))
    return landed
