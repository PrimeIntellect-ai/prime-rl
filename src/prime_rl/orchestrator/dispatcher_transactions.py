"""Small transactional helpers for dispatcher output ownership."""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import verifiers.v1 as vf

from prime_rl.orchestrator.dispatcher_metrics import DispatcherMetrics
from prime_rl.orchestrator.types import GroupState, InflightRollout, Rollout
from prime_rl.utils.async_utils import gather_shielded, safe_cancel
from prime_rl.utils.logger import get_logger


@dataclass
class EmissionRecord:
    """A completion handler that still owns group-output accounting."""

    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: BaseException | None = None


class EmissionTracker:
    """Make completion handlers visible after their inflight task is popped."""

    def __init__(self) -> None:
        self._by_group: dict[uuid.UUID, list[EmissionRecord]] = defaultdict(list)

    @asynccontextmanager
    async def track(self, group_id: uuid.UUID) -> AsyncIterator[None]:
        record = EmissionRecord()
        self._by_group[group_id].append(record)
        try:
            yield
        except BaseException as exc:
            record.error = exc
            raise
        finally:
            record.done.set()
            records = self._by_group[group_id]
            records.remove(record)
            if not records:
                self._by_group.pop(group_id, None)

    def claim(self, group_ids: Iterable[uuid.UUID]) -> list[EmissionRecord]:
        return [record for group_id in group_ids for record in self._by_group.get(group_id, ())]


async def settle_transaction_cleanup(
    cleanup: Awaitable[object],
    primary_error: BaseException,
    description: str,
) -> None:
    """Settle cleanup despite repeated cancellation, preserving the primary error."""
    results, cancellation = await gather_shielded(cleanup)
    failures = [result for result in results if isinstance(result, BaseException)]
    for cleanup_error in failures:
        primary_error.add_note(f"{description} also failed: {cleanup_error!r}")
    if cancellation is not None:
        primary_error.add_note(f"{description} was cancelled again but settled before propagation")


async def _put_before_stop(
    out_q: asyncio.Queue[Rollout],
    stopped_event: asyncio.Event,
    rollout: Rollout,
) -> None:
    put = asyncio.create_task(out_q.put(rollout))
    stopped = asyncio.create_task(stopped_event.wait())
    try:
        await asyncio.wait((put, stopped), return_when=asyncio.FIRST_COMPLETED)
        if put.done():
            put.result()
        else:
            raise RuntimeError("Dispatcher stopped while emitting policy barrier cancellation markers")
    finally:
        if not put.done():
            await safe_cancel(put)
        if not stopped.done():
            await safe_cancel(stopped)


async def emit_policy_cancellation_markers(
    groups: dict[uuid.UUID, GroupState],
    metadata_by_group: dict[uuid.UUID, InflightRollout],
    *,
    out_q: asyncio.Queue[Rollout],
    stopped: asyncio.Event,
    metrics: DispatcherMetrics,
) -> None:
    """Finish every claimed group without hanging after dispatcher stop."""
    cancelled = 0
    for group_id, group in groups.items():
        owed = max(0, group.target_rollouts - group.emitted)
        if owed == 0:
            continue
        meta = metadata_by_group.get(group_id) or InflightRollout(
            kind=group.kind,
            env_name=group.env_name,
            group_id=group_id,
            policy_version=group.policy_version_at_start,
            rollout_count=1,
            eval_step=group.eval_step,
            uses_mutable_policy=group.uses_mutable_policy,
        )
        for _ in range(owed):
            rollout = Rollout(
                task=vf.TraceTask(type="Task", data=vf.TaskData(idx=group.task_idx, prompt=None)),
                errors=[vf.Error(type="Cancelled", message="Policy update barrier")],
                stop_condition="error",
                kind=meta.kind,
                env_name=meta.env_name,
                group_id=meta.group_id,
                policy_version=group.policy_version_at_start,
                off_policy_steps=meta.off_policy_steps,
                eval_step=group.eval_step if meta.kind == "eval" else None,
            )
            await _put_before_stop(out_q, stopped, rollout)
            group.emitted += 1
        metrics.record_cancellation(kind=meta.kind, env_name=meta.env_name, n=owed)
        cancelled += owed
    if cancelled:
        get_logger().debug(f"Policy update barrier cancelled {cancelled} mutable-policy rollout(s)")
