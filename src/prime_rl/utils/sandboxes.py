"""Run-scoped cleanup and accounting of the Prime sandboxes a run creates.

Every sandbox carries the run name as a label (``set_base_sandbox_labels``), and the
sandboxes of a launcher-managed env server also carry ``prl-env:<source name>``. The
run name is stable across restarts of one run, so a restarted run finds what its
predecessor leaked (a SIGKILL skips every teardown path)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger

ENV_LABEL_PREFIX = "prl-env:"
METRICS_INTERVAL_S = 60


def _client():
    """A sync sandbox client, or None without Prime credentials (a run without Prime
    sandboxes has nothing to clean up)."""
    from prime_sandboxes import SandboxClient
    from prime_sandboxes.core import APIClient, Config

    if not Config().api_key:
        return None
    return SandboxClient(APIClient())


def sweep_sandboxes(run_name: str) -> None:
    """Delete every sandbox labeled with ``run_name``. Best-effort: a failure is logged."""
    try:
        client = _client()
        if client is None:
            return
        response = client.bulk_delete(labels=[run_name])
    except Exception as e:
        get_logger().warning(f"Could not sweep the sandboxes of run {run_name}: {e}")
        return
    if response.succeeded:
        get_logger().warning(f"Deleted {len(response.succeeded)} leftover sandboxes of run {run_name}")
    if response.failed:
        get_logger().warning(f"Could not delete {len(response.failed)} sandboxes of run {run_name}: {response.failed}")


@contextmanager
def sweep_on_exit(run_name: str) -> Iterator[None]:
    """Sweep the run's sandboxes when the block exits, however it exits."""
    try:
        yield
    finally:
        sweep_sandboxes(run_name)


class SandboxMetrics:
    """Polls the platform for the run's live sandboxes and logs them on wall time:
    ``sandboxes/live`` (and per env), their age in minutes, and how many ids appeared
    (``created``) and disappeared (``deleted``) since the run started. A sandbox
    created and deleted between two polls is counted in neither."""

    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        self.client = _client()
        self.seen: set[str] | None = None
        self.created = 0
        self.deleted = 0
        self.task: asyncio.Task | None = None

    def poll(self) -> dict[str, float]:
        sandboxes, page = [], 1
        while True:
            response = self.client.list(labels=[self.run_name], exclude_terminated=True, page=page, per_page=500)
            sandboxes += response.sandboxes
            if not response.has_next:
                break
            page += 1
        ids = {sandbox.id for sandbox in sandboxes}
        if self.seen is not None:
            self.created += len(ids - self.seen)
            self.deleted += len(self.seen - ids)
        self.seen = ids
        now = datetime.now(timezone.utc).replace(tzinfo=None)  # the API's timestamps are naive UTC
        ages = sorted((now - sandbox.created_at).total_seconds() / 60 for sandbox in sandboxes)
        metrics = {
            "sandboxes/live": float(len(sandboxes)),
            "sandboxes/created": float(self.created),
            "sandboxes/deleted": float(self.deleted),
        }
        if ages:
            metrics |= {
                "sandboxes/age_min/p50": ages[len(ages) // 2],
                "sandboxes/age_min/p90": ages[int(len(ages) * 0.9)],
                "sandboxes/age_min/max": ages[-1],
            }
        for sandbox in sandboxes:
            for label in sandbox.labels:
                if label.startswith(ENV_LABEL_PREFIX):
                    key = f"sandboxes/live/{label.removeprefix(ENV_LABEL_PREFIX)}"
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
                get_logger().warning(f"Could not poll the sandboxes of run {self.run_name}: {e}")
            await asyncio.sleep(max(0.0, METRICS_INTERVAL_S - (time.monotonic() - start)))

    async def start(self) -> None:
        if self.client is not None:
            self.task = asyncio.create_task(self.run(), name="sandbox_metrics")

    async def stop(self) -> None:
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None
