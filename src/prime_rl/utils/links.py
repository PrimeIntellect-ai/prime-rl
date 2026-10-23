"""The run-page links each monitor opens (the W&B run, the Prime run or evaluations),
left in ``monitors/links.json`` for the launcher's startup banner.

A monitor registers when it initializes, with how many links it will publish - zero
when it opens no page, e.g. without credentials - and publishes each link once known.
The launcher waits until every monitor it configured has registered and published."""

import os
import time
from pathlib import Path
from typing import Any, Callable

from prime_rl.utils.pathing import get_links_path
from prime_rl.utils.record import read_record, update_record

# The launcher exports it to every component: entries stamped with another attempt are
# a previous launch's.
ATTEMPT_VAR = "PRL_ATTEMPT_LOG_DIR"
# Monitor keys in the links record.
PRIME_LINKS = "prime"
WANDB_LINKS = "wandb"
LINK_TIMEOUT_S = 300.0
LINK_POLL_S = 0.5


def _entry(record: dict[str, Any], monitor: str, expected: int) -> dict[str, Any]:
    """This attempt's entry for ``monitor``, started over when the record holds a
    previous attempt's."""
    attempt = os.getenv(ATTEMPT_VAR)
    entry = record.get(monitor)
    if entry is None or entry.get("attempt") != attempt:
        entry = record[monitor] = {"attempt": attempt, "expected": expected, "links": {}}
    return entry


def register_links(output_dir: Path, monitor: str, expected: int) -> None:
    """Announce that ``monitor`` publishes ``expected`` links in this attempt. Processes
    that share one monitor's page (a shared W&B run) register the same entry."""

    def merge(record: dict[str, Any]) -> dict[str, Any]:
        entry = _entry(record, monitor, expected)
        entry["expected"] = max(entry["expected"], expected)
        return record

    update_record(get_links_path(output_dir), merge)


def publish_link(output_dir: Path, monitor: str, label: str, url: str | None) -> None:
    """One of ``monitor``'s links; ``None`` when the page it was registered for did
    not open, so the launcher stops waiting for it. A publish before the monitor's
    registration counts as one expected link."""

    def merge(record: dict[str, Any]) -> dict[str, Any]:
        _entry(record, monitor, 1)["links"][label] = url
        return record

    update_record(get_links_path(output_dir), merge)


def wait_for_links(
    output_dir: Path,
    attempt_log_dir: Path,
    monitors: list[str],
    stop: Callable[[], bool],
    timeout: float = LINK_TIMEOUT_S,
) -> dict[str, dict[str, str]]:
    """``{monitor: {label: url}}`` for this attempt. Polls until each of ``monitors``
    has registered and published every link it registered for, ``stop()`` turns true
    (a component failed, or the run ended) or ``timeout`` passes, and returns what is
    in by then."""
    path = get_links_path(output_dir)
    attempt = str(attempt_log_dir)
    deadline = time.monotonic() + timeout
    while True:
        try:
            record = read_record(path) or {}
        except (OSError, ValueError):
            record = {}
        entries = {name: record.get(name) for name in monitors}
        entries = {name: entry for name, entry in entries.items() if entry and entry.get("attempt") == attempt}
        done = len(entries) == len(monitors) and all(len(e["links"]) >= e["expected"] for e in entries.values())
        if done or stop() or time.monotonic() >= deadline:
            return {name: {label: url for label, url in e["links"].items() if url} for name, e in entries.items()}
        time.sleep(LINK_POLL_S)
