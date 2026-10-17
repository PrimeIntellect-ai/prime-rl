"""Run-scoped cleanup and accounting of the Prime sandboxes a run creates.

Every sandbox carries ``prl-run:<id>`` (``set_base_sandbox_labels``), where the id is
minted on the run's first launch and kept in its run dir (``launcher/run_id``), so every resume and requeue
of the run shares it and a restarted run finds what its predecessor leaked (a SIGKILL
skips every teardown path). ``run.name`` is not unique (every run of one config shares
a name set in it), so it is only a readable label, never swept by. Sandboxes of a
launcher-managed env server also carry ``prl-env:<source name>``."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger

RUN_LABEL_PREFIX = "prl-run:"
ENV_LABEL_PREFIX = "prl-env:"
METRICS_INTERVAL_S = 60


def run_id_file(run_dir: Path) -> Path:
    # Under launcher/, which the run-dir reuse check does not count as a prior run's output.
    return run_dir / "launcher" / "run_id"


def stable_run_id(run_dir: Path) -> str:
    """The id kept in ``<run_dir>/launcher/run_id``, minted on the run's first launch."""
    path = run_id_file(run_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(uuid.uuid4().hex)
    return path.read_text().strip()


def sweep_previous_run(run_dir: Path) -> None:
    """Sweep the sandboxes of the run that last used ``run_dir``. A clean start deletes
    the run dir and its id with it, so its sandboxes can only be swept before."""
    path = run_id_file(run_dir)
    if path.exists():
        sweep_sandboxes(f"{RUN_LABEL_PREFIX}{path.read_text().strip()}")


def run_label() -> str | None:
    """``prl-run:$PRL_STABLE_RUN_ID``; None for a component launched outside a run."""
    run_id = os.environ.get("PRL_STABLE_RUN_ID")
    return f"{RUN_LABEL_PREFIX}{run_id}" if run_id else None


def _client():
    """A sync sandbox client, or None without Prime credentials (a run without Prime
    sandboxes has nothing to clean up)."""
    from prime_sandboxes import SandboxClient
    from prime_sandboxes.core import APIClient, Config

    if not Config().api_key:
        return None
    return SandboxClient(APIClient())


def sweep_sandboxes(label: str) -> None:
    """Delete every sandbox labeled ``label``. Best-effort: a failure is logged."""
    try:
        client = _client()
        if client is None:
            return
        response = client.bulk_delete(labels=[label], team_id=client.client.config.team_id)
    except Exception as e:
        get_logger().warning(f"Could not sweep the sandboxes labeled {label}: {e}")
        return
    if response.succeeded:
        get_logger().warning(f"Deleted {len(response.succeeded)} leftover sandboxes labeled {label}")
    if response.failed:
        get_logger().warning(f"Could not delete {len(response.failed)} sandboxes labeled {label}: {response.failed}")


@contextmanager
def sweep_on_exit(label: str) -> Iterator[None]:
    """Sweep the run's sandboxes when the block exits, however it exits."""
    try:
        yield
    finally:
        sweep_sandboxes(label)


class SandboxMetrics:
    """Polls the platform for the run's live sandboxes and logs them on wall time:
    ``sandboxes/live`` (and per env) and their age in minutes."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.client = _client()
        self.task: asyncio.Task | None = None

    def poll(self) -> dict[str, float]:
        sandboxes, page = [], 1
        while True:
            response = self.client.list(labels=[self.label], exclude_terminated=True, page=page, per_page=500)
            sandboxes += response.sandboxes
            if not response.has_next:
                break
            page += 1
        now = datetime.now(timezone.utc).replace(tzinfo=None)  # the API's timestamps are naive UTC
        ages = sorted((now - sandbox.created_at).total_seconds() / 60 for sandbox in sandboxes)
        metrics = {"sandboxes/live": float(len(sandboxes))}
        if ages:
            metrics |= {
                "sandboxes/age_min/p50": ages[len(ages) // 2],
                "sandboxes/age_min/p90": ages[int(len(ages) * 0.9)],
                "sandboxes/age_min/max": ages[-1],
            }
        for sandbox in sandboxes:
            for sandbox_label in sandbox.labels:
                if sandbox_label.startswith(ENV_LABEL_PREFIX):
                    key = f"sandboxes/live/{sandbox_label.removeprefix(ENV_LABEL_PREFIX)}"
                    metrics[key] = metrics.get(key, 0.0) + 1
        return metrics

    async def run(self) -> None:
        from prime_rl import monitors

        while True:
            start = time.monotonic()
            try:
                metrics = await asyncio.to_thread(self.poll)
                await monitors.log(metrics, step=None)
            except Exception as e:
                get_logger().warning(f"Could not poll the sandboxes labeled {self.label}: {e}")
            await asyncio.sleep(max(0.0, METRICS_INTERVAL_S - (time.monotonic() - start)))

    async def start(self) -> None:
        if self.client is not None:
            self.task = asyncio.create_task(self.run(), name="sandbox_metrics")

    async def stop(self) -> None:
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None
