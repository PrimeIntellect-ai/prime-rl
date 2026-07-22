import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.transport import TrainingSample


class _Metrics:
    def to_wandb(self, **_kwargs) -> dict[str, float]:
        return {}


class _Rollout:
    is_trainable = True
    num_total_tokens = 1
    num_input_tokens = 1
    num_output_tokens = 0
    group_id = "group"
    reward = 1.0

    def to_record(self) -> dict:
        return {}

    def scalar_advantage(self) -> float:
        return 1.0


class _Rollouts(list[_Rollout]):
    @property
    def effective(self) -> "_Rollouts":
        return self

    @property
    def rollouts(self) -> list[_Rollout]:
        return list(self)

    @property
    def metrics(self) -> _Metrics:
        return _Metrics()

    def by_env(self) -> dict[str, "_Rollouts"]:
        return {"test": self}


def _make_orchestrator(output_dir: Path, *, max_steps: int, step: int) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = SimpleNamespace(max_steps=max_steps, output_dir=output_dir)
    orchestrator.progress = SimpleNamespace(step=step, total_tokens=0, total_samples=0, total_problems=0)
    orchestrator.last_batch_at = None
    orchestrator.consecutive_empty_batches = 0
    orchestrator.draining = False
    orchestrator.dispatcher = SimpleNamespace(
        disable_train_scheduling=MagicMock(),
        cancel_inflight_train_rollouts=AsyncMock(return_value=3),
    )
    orchestrator.sender = SimpleNamespace(send=AsyncMock())
    orchestrator.maybe_save_ckpt = AsyncMock(return_value=0.0)
    orchestrator.update_dispatch_gate = MagicMock()
    orchestrator.monitor = MagicMock()
    orchestrator.wait_for_policy_time = 0.0
    orchestrator.train_sink = SimpleNamespace(pre_filter_seen=0, reset_pre_filter_stats=MagicMock())
    orchestrator.usage_reporter = None
    orchestrator.heart = None
    orchestrator.log_train_batch = MagicMock()
    orchestrator.maybe_trigger_eval = MagicMock()
    return orchestrator


def _make_batch(*, with_samples: bool = True) -> SimpleNamespace:
    samples = (
        [TrainingSample(token_ids=[1], mask=[True], logprobs=[0.0], temperatures=[1.0], env_name="test")]
        if with_samples
        else []
    )
    return SimpleNamespace(rollouts=_Rollouts([_Rollout()]), samples=samples)


def test_final_train_batch_starts_draining_immediately(tmp_path: Path) -> None:
    orchestrator = _make_orchestrator(tmp_path, max_steps=1, step=1)

    with (
        patch("prime_rl.orchestrator.orchestrator.save_rollouts"),
        patch("prime_rl.orchestrator.orchestrator.trim_process_memory"),
    ):
        asyncio.run(orchestrator.finalize_train_batch(_make_batch()))

    assert orchestrator.draining
    orchestrator.sender.send.assert_awaited_once()
    orchestrator.dispatcher.disable_train_scheduling.assert_called_once_with()
    orchestrator.dispatcher.cancel_inflight_train_rollouts.assert_awaited_once_with()
    assert orchestrator.progress.step == 2
    orchestrator.maybe_trigger_eval.assert_called_once_with(2)


def test_non_final_train_batch_keeps_dispatching(tmp_path: Path) -> None:
    orchestrator = _make_orchestrator(tmp_path, max_steps=2, step=1)

    with (
        patch("prime_rl.orchestrator.orchestrator.save_rollouts"),
        patch("prime_rl.orchestrator.orchestrator.trim_process_memory"),
    ):
        asyncio.run(orchestrator.finalize_train_batch(_make_batch()))

    assert not orchestrator.draining
    orchestrator.dispatcher.disable_train_scheduling.assert_not_called()
    orchestrator.dispatcher.cancel_inflight_train_rollouts.assert_not_awaited()


def test_empty_final_train_batch_keeps_dispatching(tmp_path: Path) -> None:
    orchestrator = _make_orchestrator(tmp_path, max_steps=1, step=1)

    asyncio.run(orchestrator.finalize_train_batch(_make_batch(with_samples=False)))

    assert not orchestrator.draining
    orchestrator.sender.send.assert_not_awaited()
    orchestrator.dispatcher.disable_train_scheduling.assert_not_called()
    orchestrator.dispatcher.cancel_inflight_train_rollouts.assert_not_awaited()
    assert orchestrator.progress.step == 1
