"""Tests for RolloutEngine — off-policy cancel, LoRA swap, group lifecycle.

We construct RolloutEngine directly with EngineInputs (stub scheduler +
stub Group) so tests don't need a real verifiers env or vLLM client.
"""

import asyncio

from prime_rl.orchestrator.engine import EngineInputs, GroupOutput, Inflight, RolloutEngine
from prime_rl.orchestrator.scheduler import Dispatch, Task


def _run(coro):
    return asyncio.run(coro)


class _StubGroup:
    """Stand-in for a Group (algorithm Protocol).

    `run()` returns a list of `task.rollouts_per_group` rollouts, each populated
    with `result`. If `block=True`, blocks forever (drives timeout/cancel
    paths). Tests can monkey-patch `.run` to inject custom behavior."""

    def __init__(self, *, result: dict | None = None, block: bool = False, model: str = "base-model"):
        self._result = result if result is not None else {"reward": 0.0, "trajectory": []}
        self._block = block
        self.model = model
        self.calls = 0

    async def run(self, task: Task, example: dict):
        self.calls += 1
        if self._block:
            await asyncio.Event().wait()  # blocks forever
        return [dict(self._result) for _ in range(task.rollouts_per_group)]


class _StubScheduler:
    """Yields a fixed list of dispatches, then None."""

    def __init__(self, dispatches: list[Dispatch]):
        self._iter = iter(dispatches)

    def next_task(self) -> Dispatch | None:
        return next(self._iter, None)


def _task(id: str = "t", kind: str = "train", rollouts_per_group: int = 1) -> Task:
    return Task(
        id=id,
        env=None,  # type: ignore[arg-type]  # stub group doesn't touch task.env
        sampling_args={},
        kind=kind,  # type: ignore[arg-type]
        rollouts_per_group=rollouts_per_group,
    )


def _build(
    *,
    scheduler: _StubScheduler | None = None,
    out_q: asyncio.Queue | None = None,
    group: _StubGroup | None = None,
    max_off_policy: int = 1,
    concurrency: int = 1,
    max_rollout_time_seconds: float | None = None,
    lora_name: str | None = None,
) -> RolloutEngine:
    return RolloutEngine(
        EngineInputs(
            scheduler=scheduler or _StubScheduler([]),  # type: ignore[arg-type]
            out_q=out_q or asyncio.Queue(),
            group=group or _StubGroup(),  # type: ignore[arg-type]
            max_off_policy=max_off_policy,
            concurrency=concurrency,
            max_rollout_time_seconds=max_rollout_time_seconds,
            lora_name=lora_name,
        )
    )


# --- max_off_policy_level ----------------------------------------------------


def test_max_off_policy_level_zero_when_no_inflight():
    eng = _build()
    assert eng.max_off_policy_level() == 0


def test_max_off_policy_level_max_over_train_inflight():
    eng = _build()
    eng.policy_version = 5
    eng._inflight = [
        Inflight(version=2, kind="train"),  # lag 3
        Inflight(version=4, kind="train"),  # lag 1
    ]
    assert eng.max_off_policy_level() == 3


def test_max_off_policy_level_excludes_eval():
    eng = _build()
    eng.policy_version = 10
    eng._inflight = [
        Inflight(version=0, kind="eval"),  # lag 10 — ignored
        Inflight(version=8, kind="train"),  # lag 2
    ]
    assert eng.max_off_policy_level() == 2


# --- on_new_version ----------------------------------------------------------


def test_on_new_version_bumps_policy_version():
    eng = _build()
    _run(eng.on_new_version(7))
    assert eng.policy_version == 7


def test_on_new_version_swaps_to_lora_on_first_call():
    group = _StubGroup(model="base-model")
    eng = _build(group=group, lora_name="my-adapter")
    assert eng.group.model == "base-model"
    _run(eng.on_new_version(1))
    assert eng.group.model == "my-adapter"


def test_on_new_version_lora_swap_is_idempotent():
    group = _StubGroup(model="base-model")
    eng = _build(group=group, lora_name="my-adapter")
    _run(eng.on_new_version(1))
    _run(eng.on_new_version(2))
    assert eng.group.model == "my-adapter"


def test_on_new_version_no_lora_leaves_model_unchanged():
    group = _StubGroup(model="base-model")
    eng = _build(group=group, lora_name=None)
    _run(eng.on_new_version(1))
    assert eng.group.model == "base-model"


# --- _run_group lifecycle ----------------------------------------------------


def test_run_group_happy_path_puts_group_to_out_q():
    out_q: asyncio.Queue = asyncio.Queue()
    group_impl = _StubGroup(result={"reward": 0.5, "trajectory": [{}]})
    task = _task(id="env_a", rollouts_per_group=3)
    eng = _build(out_q=out_q, group=group_impl)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {"x": 1}, sem, eval_step=None)
        return await out_q.get(), sem

    out, sem = _run(go())
    assert isinstance(out, GroupOutput)
    assert out.env_id == "env_a"
    assert out.kind == "train"
    assert out.policy_version == 0
    assert out.eval_step is None
    assert len(out.rollouts) == 3
    assert group_impl.calls == 1  # one call to Group.run produces all 3 rollouts
    assert sem.locked() is False


def test_run_group_train_dropped_if_off_policy_by_end():
    """If policy_version advanced past max_off_policy while we awaited, drop
    the group silently — no out_q.put."""
    out_q: asyncio.Queue = asyncio.Queue()
    group_impl = _StubGroup()
    task = _task()
    eng = _build(out_q=out_q, group=group_impl, max_off_policy=1)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()

        async def advance_then_run(_task, _example):
            eng.policy_version = 10  # blow past max_off_policy
            return [{"reward": 0.0}]

        group_impl.run = advance_then_run  # type: ignore[assignment]
        await eng._run_group(task, {"x": 1}, sem, eval_step=None)

    _run(go())
    assert out_q.empty()  # dropped silently


def test_run_group_eval_ships_even_when_off_policy():
    """Eval is tagged with trigger step, never dropped for staleness."""
    out_q: asyncio.Queue = asyncio.Queue()
    group_impl = _StubGroup()
    task = _task(kind="eval")
    eng = _build(out_q=out_q, group=group_impl, max_off_policy=1)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()

        async def advance_then_run(_task, _example):
            eng.policy_version = 99
            return [{"reward": 0.0}]

        group_impl.run = advance_then_run  # type: ignore[assignment]
        await eng._run_group(task, {"x": 1}, sem, eval_step=42)

    _run(go())
    out = out_q.get_nowait()
    assert out.kind == "eval"
    assert out.eval_step == 42


def test_run_group_train_timeout_drops_silently():
    out_q: asyncio.Queue = asyncio.Queue()
    group_impl = _StubGroup(block=True)  # blocks forever
    task = _task()
    eng = _build(out_q=out_q, group=group_impl, max_rollout_time_seconds=0.05)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {"x": 1}, sem, eval_step=None)

    _run(go())
    assert out_q.empty()


def test_run_group_eval_timeout_emits_empty_group():
    """Eval timeout emits an empty-rollouts GroupOutput so the batcher's
    expected-count check resolves and flushes the partial epoch."""
    out_q: asyncio.Queue = asyncio.Queue()
    group_impl = _StubGroup(block=True)
    task = _task(kind="eval")
    eng = _build(out_q=out_q, group=group_impl, max_rollout_time_seconds=0.05)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {"x": 1}, sem, eval_step=42)

    _run(go())
    out = out_q.get_nowait()
    assert out.kind == "eval"
    assert out.eval_step == 42
    assert out.rollouts == []  # empty


def test_run_group_releases_semaphore_on_timeout():
    group_impl = _StubGroup(block=True)
    task = _task()
    eng = _build(group=group_impl, max_rollout_time_seconds=0.05)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {}, sem, eval_step=None)
        return sem

    sem = _run(go())
    assert not sem.locked()


def test_run_group_removes_inflight_on_completion():
    group_impl = _StubGroup()
    task = _task()
    eng = _build(group=group_impl)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {}, sem, eval_step=None)

    _run(go())
    assert eng._inflight == []


# --- run() loop --------------------------------------------------------------


def test_run_drains_scheduler_until_none():
    out_q: asyncio.Queue = asyncio.Queue()
    group_impl = _StubGroup()
    task = _task()
    sched = _StubScheduler(
        [
            Dispatch(task=task, example={"a": 1}),
            Dispatch(task=task, example={"a": 2}),
            Dispatch(task=task, example={"a": 3}),
        ]
    )
    # concurrency=1 keeps the test simple: each spawned task completes
    # before the next dispatch is pulled, so by the time run() returns,
    # all groups have been put.
    eng = _build(scheduler=sched, out_q=out_q, group=group_impl, concurrency=1)

    async def go():
        await asyncio.wait_for(eng.run(), timeout=2.0)
        # Defensive: drain any in-flight spawned tasks before we read out_q.
        # With concurrency=1 this should be a no-op.
        for t in [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
            await t

    _run(go())
    outs = []
    while not out_q.empty():
        outs.append(out_q.get_nowait())
    assert len(outs) == 3
    assert group_impl.calls == 3
