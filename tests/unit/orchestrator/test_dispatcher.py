import asyncio
import uuid
from types import SimpleNamespace

from prime_rl.orchestrator.dispatcher import RolloutDispatcher


def _bare_dispatcher(*, eval_source=None, groups=None, inflight=None) -> RolloutDispatcher:
    """A RolloutDispatcher with only the fields ``is_idle`` reads, bypassing the heavy __init__."""
    d = RolloutDispatcher.__new__(RolloutDispatcher)
    d.inflight = inflight or {}
    d.out_q = asyncio.Queue()
    d.eval_source = eval_source
    d.groups = groups or {}
    return d


def test_is_idle_waits_for_partly_scheduled_eval_group():
    # The drain race: an eval example has left the source queue (its group opened) but the group's
    # rollouts aren't all scheduled yet (rollouts_to_schedule > 0), and nothing is momentarily in
    # flight. is_idle must NOT report drained, or the tail eval rollouts are never scheduled and the
    # eval point is silently dropped.
    pending = SimpleNamespace(kind="eval", rollouts_to_schedule=1)
    d = _bare_dispatcher(eval_source=None, groups={uuid.uuid4(): pending})
    assert d.is_idle is False

    pending.rollouts_to_schedule = 0  # group fully dispatched → genuinely drained
    assert d.is_idle is True
