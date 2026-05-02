"""Tests for RolloutEngine — off-policy cancel, LoRA swap, group lifecycle.

We construct RolloutEngine directly with EngineInputs (stub scheduler +
fake env) so tests don't need a real verifiers env or vLLM client.
"""

import asyncio

import verifiers as vf

from prime_rl.orchestrator.engine import EngineInputs, Group, Inflight, RolloutEngine
from prime_rl.orchestrator.scheduler import Dispatch, Task


def _run(coro):
    return asyncio.run(coro)


class _StubEnv:
    """Fake verifiers env. `run_rollout` returns a queued result, or blocks
    forever if `block=True` (so we can drive cancellation paths)."""

    def __init__(self, result: dict | None = None, block: bool = False):
        self._result = result if result is not None else {"reward": 0.0, "trajectory": []}
        self._block = block
        self.calls = 0

    async def run_rollout(self, *args, **kwargs):
        self.calls += 1
        if self._block:
            await asyncio.Event().wait()  # blocks forever
        return self._result


class _StubScheduler:
    """Yields a fixed list of dispatches, then None."""

    def __init__(self, dispatches: list[Dispatch]):
        self._iter = iter(dispatches)

    def next_task(self) -> Dispatch | None:
        return next(self._iter, None)


def _task(id: str = "t", env: _StubEnv | None = None, kind: str = "train", rollouts_per_group: int = 1) -> Task:
    return Task(
        id=id,
        env=env or _StubEnv(),  # type: ignore[arg-type]
        sampling_args={},
        kind=kind,  # type: ignore[arg-type]
        rollouts_per_group=rollouts_per_group,
    )


def _build(
    *,
    scheduler: _StubScheduler | None = None,
    out_q: asyncio.Queue | None = None,
    model: str = "base-model",
    max_off_policy: int = 1,
    concurrency: int = 1,
    max_rollout_time_seconds: float | None = None,
    lora_name: str | None = None,
) -> RolloutEngine:
    return RolloutEngine(
        EngineInputs(
            scheduler=scheduler or _StubScheduler([]),  # type: ignore[arg-type]
            out_q=out_q or asyncio.Queue(),
            client=vf.ClientConfig(client_type="openai_chat_completions", api_base_url="http://x"),
            model=model,
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
    async def go():
        eng = _build()
        eng.policy_version = 5
        eng._inflight = [
            Inflight(version=2, gather=asyncio.Future(), kind="train"),  # lag 3
            Inflight(version=4, gather=asyncio.Future(), kind="train"),  # lag 1
        ]
        return eng.max_off_policy_level()

    assert _run(go()) == 3


def test_max_off_policy_level_excludes_eval():
    async def go():
        eng = _build()
        eng.policy_version = 10
        eng._inflight = [
            Inflight(version=0, gather=asyncio.Future(), kind="eval"),  # lag 10 — ignored
            Inflight(version=8, gather=asyncio.Future(), kind="train"),  # lag 2
        ]
        return eng.max_off_policy_level()

    assert _run(go()) == 2


# --- on_new_version ----------------------------------------------------------


def test_on_new_version_bumps_policy_version():
    eng = _build()
    _run(eng.on_new_version(7))
    assert eng.policy_version == 7


def test_on_new_version_swaps_to_lora_on_first_call():
    eng = _build(model="base-model", lora_name="my-adapter")
    assert eng.model == "base-model"
    _run(eng.on_new_version(1))
    assert eng.model == "my-adapter"


def test_on_new_version_lora_swap_is_idempotent():
    eng = _build(model="base-model", lora_name="my-adapter")
    _run(eng.on_new_version(1))
    _run(eng.on_new_version(2))
    assert eng.model == "my-adapter"


def test_on_new_version_no_lora_leaves_model_unchanged():
    eng = _build(model="base-model", lora_name=None)
    _run(eng.on_new_version(1))
    assert eng.model == "base-model"


def test_on_new_version_cancels_stale_train_inflight():
    eng = _build(max_off_policy=1)

    async def go():
        f = asyncio.Future()
        eng._inflight = [Inflight(version=0, gather=f, kind="train")]
        await eng.on_new_version(5)  # lag 5 > max_off_policy 1
        return f

    f = _run(go())
    assert f.cancelled()


def test_on_new_version_does_not_cancel_within_policy_inflight():
    eng = _build(max_off_policy=2)

    async def go():
        f = asyncio.Future()
        eng._inflight = [Inflight(version=3, gather=f, kind="train")]
        await eng.on_new_version(5)  # lag 2 == max_off_policy
        return f

    f = _run(go())
    assert not f.cancelled()


def test_on_new_version_never_cancels_eval_inflight():
    """Eval is always tagged with its trigger step, so it's exempt from
    off-policy cancellation regardless of how stale it gets."""
    eng = _build(max_off_policy=1)

    async def go():
        f = asyncio.Future()
        eng._inflight = [Inflight(version=0, gather=f, kind="eval")]
        await eng.on_new_version(99)  # very stale
        return f

    f = _run(go())
    assert not f.cancelled()


# --- _run_group lifecycle ----------------------------------------------------


def test_run_group_happy_path_puts_group_to_out_q():
    out_q: asyncio.Queue = asyncio.Queue()
    env = _StubEnv(result={"reward": 0.5, "trajectory": [{}]})
    task = _task(id="env_a", env=env, rollouts_per_group=3)
    eng = _build(out_q=out_q)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {"x": 1}, sem, eval_step=None)
        return await out_q.get(), sem

    group, sem = _run(go())
    assert isinstance(group, Group)
    assert group.env_id == "env_a"
    assert group.kind == "train"
    assert group.policy_version == 0
    assert group.eval_step is None
    assert len(group.rollouts) == 3
    assert env.calls == 3
    # sem released
    assert sem.locked() is False


def test_run_group_train_dropped_if_off_policy_by_end():
    """If policy_version advanced past max_off_policy while we awaited, drop
    the group silently — no out_q.put."""
    out_q: asyncio.Queue = asyncio.Queue()
    env = _StubEnv()
    task = _task(env=env)
    eng = _build(out_q=out_q, max_off_policy=1)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()

        # Patch _run_group's await target to advance policy_version mid-flight
        async def advance_then_rollout(*a, **k):
            eng.policy_version = 10  # blow past max_off_policy
            return {"reward": 0.0}

        env.run_rollout = advance_then_rollout  # type: ignore[assignment]
        await eng._run_group(task, {"x": 1}, sem, eval_step=None)

    _run(go())
    assert out_q.empty()  # dropped silently


def test_run_group_eval_ships_even_when_off_policy():
    """Eval is tagged with trigger step, never dropped for staleness."""
    out_q: asyncio.Queue = asyncio.Queue()
    env = _StubEnv()
    task = _task(env=env, kind="eval")
    eng = _build(out_q=out_q, max_off_policy=1)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()

        async def advance_then_rollout(*a, **k):
            eng.policy_version = 99
            return {"reward": 0.0}

        env.run_rollout = advance_then_rollout  # type: ignore[assignment]
        await eng._run_group(task, {"x": 1}, sem, eval_step=42)

    _run(go())
    group = out_q.get_nowait()
    assert group.kind == "eval"
    assert group.eval_step == 42


def test_run_group_train_timeout_drops_silently():
    out_q: asyncio.Queue = asyncio.Queue()
    env = _StubEnv(block=True)  # blocks forever
    task = _task(env=env)
    eng = _build(out_q=out_q, max_rollout_time_seconds=0.05)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {"x": 1}, sem, eval_step=None)

    _run(go())
    assert out_q.empty()


def test_run_group_eval_timeout_emits_empty_group():
    """Eval timeout emits an empty-rollouts Group so the batcher's
    expected-count check resolves and flushes the partial epoch."""
    out_q: asyncio.Queue = asyncio.Queue()
    env = _StubEnv(block=True)
    task = _task(env=env, kind="eval")
    eng = _build(out_q=out_q, max_rollout_time_seconds=0.05)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {"x": 1}, sem, eval_step=42)

    _run(go())
    group = out_q.get_nowait()
    assert group.kind == "eval"
    assert group.eval_step == 42
    assert group.rollouts == []  # empty


def test_run_group_releases_semaphore_on_timeout():
    env = _StubEnv(block=True)
    task = _task(env=env)
    eng = _build(max_rollout_time_seconds=0.05)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {}, sem, eval_step=None)
        return sem

    sem = _run(go())
    assert not sem.locked()


def test_run_group_removes_inflight_on_completion():
    env = _StubEnv()
    task = _task(env=env)
    eng = _build()

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        await eng._run_group(task, {}, sem, eval_step=None)

    _run(go())
    assert eng._inflight == []


# --- run() loop --------------------------------------------------------------


def test_run_drains_scheduler_until_none():
    out_q: asyncio.Queue = asyncio.Queue()
    env = _StubEnv()
    task = _task(env=env)
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
    eng = _build(scheduler=sched, out_q=out_q, concurrency=1)

    async def go():
        await asyncio.wait_for(eng.run(), timeout=2.0)
        # Defensive: drain any in-flight spawned tasks before we read out_q.
        # With concurrency=1 this should be a no-op.
        for t in [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
            await t

    _run(go())
    groups = []
    while not out_q.empty():
        groups.append(out_q.get_nowait())
    assert len(groups) == 3
    assert env.calls == 3
