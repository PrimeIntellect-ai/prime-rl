import asyncio
from types import SimpleNamespace

from prime_rl.orchestrator.orchestrator import Orchestrator


def make_draining_orchestrator(*, broadcast_type: str, max_steps: int | None, policy_version: int) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = SimpleNamespace(
        max_steps=max_steps,
        weight_broadcast=SimpleNamespace(type=broadcast_type),
    )
    orchestrator.policy = SimpleNamespace(version=policy_version)
    orchestrator.dispatcher = SimpleNamespace(is_idle=True)
    orchestrator.draining = True
    orchestrator.stopped = asyncio.Event()
    return orchestrator


def test_nccl_drain_waits_for_last_required_broadcast():
    async def run() -> None:
        orchestrator = make_draining_orchestrator(
            broadcast_type="nccl",
            max_steps=20,
            policy_version=17,
        )

        main_loop = asyncio.create_task(orchestrator.main_loop())
        await asyncio.sleep(0.05)
        assert not main_loop.done()
        assert not orchestrator.stopped.is_set()

        orchestrator.policy.version = 18
        await asyncio.wait_for(main_loop, timeout=1.0)
        assert orchestrator.stopped.is_set()

    asyncio.run(run())


def test_non_nccl_drain_exits_without_waiting_for_policy_update():
    async def run() -> None:
        orchestrator = make_draining_orchestrator(
            broadcast_type="filesystem",
            max_steps=20,
            policy_version=0,
        )

        await asyncio.wait_for(orchestrator.main_loop(), timeout=0.1)
        assert orchestrator.stopped.is_set()

    asyncio.run(run())


def test_dispatch_gate_remains_active_for_final_batches():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.progress = SimpleNamespace(step=20)
    orchestrator.policy = SimpleNamespace(version=17)
    orchestrator.dispatcher = SimpleNamespace(dispatch_allowed=asyncio.Event())
    orchestrator.dispatcher.dispatch_allowed.set()
    orchestrator.gate_closed_at = None
    orchestrator.wait_for_policy_time = 0.0

    orchestrator.update_dispatch_gate()
    assert not orchestrator.dispatcher.dispatch_allowed.is_set()

    orchestrator.policy.version = 18
    orchestrator.update_dispatch_gate()
    assert orchestrator.dispatcher.dispatch_allowed.is_set()

