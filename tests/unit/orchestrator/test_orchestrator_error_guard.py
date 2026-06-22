import asyncio
import uuid
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.types import EvalRollout, TrainRollout


def _make_orchestrator(*, limit: int | None) -> Orchestrator:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.config = SimpleNamespace(max_consecutive_errored_rollouts=limit)
    orchestrator.consecutive_errored_rollouts = 0
    return orchestrator


def _make_raw(error_type: str | None = None) -> dict:
    raw = {
        "trajectory": [],
        "reward": 0.0,
        "metrics": {},
    }
    if error_type is not None:
        raw["error"] = {
            "error": error_type,
            "error_chain_repr": error_type,
            "error_chain_str": error_type,
        }
    return raw


def _make_train_rollout(error_type: str | None = None) -> TrainRollout:
    return TrainRollout(
        raw=_make_raw(error_type),
        env_name="test-env",
        example_id=123,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
    )


def _make_eval_rollout(error_type: str | None = None) -> EvalRollout:
    return EvalRollout(
        raw=_make_raw(error_type),
        env_name="eval-env",
        example_id=456,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
        eval_step=0,
    )


def test_consecutive_errored_train_rollouts_raise_at_configured_threshold():
    orchestrator = _make_orchestrator(limit=2)

    orchestrator.update_consecutive_errored_rollouts(_make_train_rollout("TimeoutError"))
    assert orchestrator.consecutive_errored_rollouts == 1

    with pytest.raises(RuntimeError, match="2 consecutive terminal errored rollouts"):
        orchestrator.update_consecutive_errored_rollouts(_make_train_rollout("TimeoutError"))


def test_successful_train_rollout_resets_errored_rollout_streak():
    orchestrator = _make_orchestrator(limit=10)
    orchestrator.update_consecutive_errored_rollouts(_make_train_rollout("TimeoutError"))

    orchestrator.update_consecutive_errored_rollouts(_make_train_rollout())

    assert orchestrator.consecutive_errored_rollouts == 0


def test_successful_eval_rollout_resets_errored_rollout_streak():
    orchestrator = _make_orchestrator(limit=10)
    orchestrator.update_consecutive_errored_rollouts(_make_train_rollout("TimeoutError"))

    orchestrator.update_consecutive_errored_rollouts(_make_eval_rollout())

    assert orchestrator.consecutive_errored_rollouts == 0


def test_cancelled_train_rollout_markers_do_not_increment_errored_rollout_streak():
    orchestrator = _make_orchestrator(limit=10)
    orchestrator.consecutive_errored_rollouts = 1

    orchestrator.update_consecutive_errored_rollouts(_make_train_rollout("Cancelled"))

    assert orchestrator.consecutive_errored_rollouts == 1


def test_none_disables_errored_rollout_guard():
    orchestrator = _make_orchestrator(limit=None)

    for _ in range(3):
        orchestrator.update_consecutive_errored_rollouts(_make_train_rollout("TimeoutError"))

    assert orchestrator.consecutive_errored_rollouts == 0


def test_main_loop_raises_on_errored_train_rollout_before_sink_add():
    async def run() -> None:
        orchestrator = _make_orchestrator(limit=1)
        orchestrator.stopped = asyncio.Event()
        orchestrator.draining = False
        orchestrator.dispatcher = SimpleNamespace(out_q=asyncio.Queue())

        class TrainSinkShouldNotBeCalled:
            async def add(self, rollout: TrainRollout):
                raise AssertionError("TrainSink.add should not be called after the guard trips")

        orchestrator.train_sink = TrainSinkShouldNotBeCalled()
        await orchestrator.dispatcher.out_q.put(_make_train_rollout("TimeoutError"))

        with pytest.raises(RuntimeError, match="1 consecutive terminal errored rollouts"):
            await orchestrator.main_loop()

    asyncio.run(run())


def test_main_loop_raises_on_errored_eval_rollout_before_sink_add():
    async def run() -> None:
        orchestrator = _make_orchestrator(limit=1)
        orchestrator.stopped = asyncio.Event()
        orchestrator.draining = False
        orchestrator.dispatcher = SimpleNamespace(out_q=asyncio.Queue())

        class EvalSinkShouldNotBeCalled:
            def add(self, rollout: EvalRollout):
                raise AssertionError("EvalSink.add should not be called after the guard trips")

        orchestrator.eval_sink = EvalSinkShouldNotBeCalled()
        await orchestrator.dispatcher.out_q.put(_make_eval_rollout("TimeoutError"))

        with pytest.raises(RuntimeError, match="1 consecutive terminal errored rollouts"):
            await orchestrator.main_loop()

    asyncio.run(run())
