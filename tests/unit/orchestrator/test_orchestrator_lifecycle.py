import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.transport import TrainingSample


def test_final_train_batch_starts_draining_immediately(tmp_path: Path) -> None:
    rollout = SimpleNamespace(
        is_trainable=True,
        num_total_tokens=1,
        num_input_tokens=1,
        num_output_tokens=0,
        group_id="group",
        reward=1.0,
        to_record=MagicMock(return_value={}),
        scalar_advantage=MagicMock(return_value=1.0),
    )
    rollouts = MagicMock()
    rollouts.__len__.return_value = 1
    rollouts.__iter__.side_effect = lambda: iter([rollout])
    rollouts.effective = rollouts
    rollouts.rollouts = [rollout]
    rollouts.metrics = SimpleNamespace(to_wandb=MagicMock(return_value={}))
    rollouts.by_env.return_value = {"test": rollouts}
    batch = SimpleNamespace(
        rollouts=rollouts,
        samples=[TrainingSample(token_ids=[1], mask=[True], logprobs=[0.0], temperatures=[1.0], env_name="test")],
    )

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = SimpleNamespace(max_steps=1, output_dir=tmp_path)
    orchestrator.progress = SimpleNamespace(step=1, total_tokens=0, total_samples=0, total_problems=0)
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

    with (
        patch("prime_rl.orchestrator.orchestrator.save_rollouts"),
        patch("prime_rl.orchestrator.orchestrator.trim_process_memory"),
    ):
        asyncio.run(orchestrator.finalize_train_batch(batch))

    assert orchestrator.draining
    orchestrator.sender.send.assert_awaited_once()
    orchestrator.dispatcher.disable_train_scheduling.assert_called_once_with()
    orchestrator.dispatcher.cancel_inflight_train_rollouts.assert_awaited_once_with()
    assert orchestrator.progress.step == 2
    orchestrator.maybe_trigger_eval.assert_called_once_with(2)
