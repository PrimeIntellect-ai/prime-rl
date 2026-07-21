import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from renderers import Qwen3VLRendererConfig

from prime_rl.orchestrator.dispatcher import DispatcherMode, RolloutDispatcher
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.types import Policy
from prime_rl.orchestrator.utils import setup_policy_inference_pool
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path


def test_setup_policy_inference_pool_uses_renderer_when_enabled():
    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                name="policy-model",
            ),
            renderer=renderer_settings,
            pool_size=None,
            any_policy_sourced=True,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_awaited_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
            pool_size=None,
        )

    asyncio.run(run())


def test_setup_policy_inference_pool_keeps_renderer_without_policy_sampling():
    """Frozen-sourced runs (e.g. sft) have no train env sampling from the live
    policy, but training is renderer-only: the renderer is still built and the
    pool is wired with the renderer train client. ``any_policy_sourced`` only
    flips the log line, not the pool setup."""

    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                name="policy-model",
            ),
            renderer=renderer_settings,
            pool_size=None,
            any_policy_sourced=False,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_awaited_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
            pool_size=None,
        )

    asyncio.run(run())


def test_eval_source_eligibility_does_not_consume_startup_trigger():
    source = EvalSource.__new__(EvalSource)
    source.eval_config = SimpleNamespace(skip_first_step=True)
    source.first_trigger = True
    source.intervals = {"eval": 2}
    source.examples_by_env = {"eval": []}
    source.queue = deque()

    assert source.eligible_envs(2) == []
    assert source.first_trigger
    assert source.trigger(2) == []
    assert not source.first_trigger
    assert source.eligible_envs(2) == ["eval"]


def test_weight_watcher_coalesces_after_eval_policy_lock(tmp_path):
    async def run() -> None:
        broadcast_dir = get_broadcast_dir(tmp_path)
        first_weights_path = get_step_path(broadcast_dir, 1)
        first_weights_path.mkdir(parents=True)
        (first_weights_path / "STABLE").touch()
        policy = Policy(model_name="policy", version=0)
        inference = SimpleNamespace(
            update_weights=AsyncMock(),
            update_model_name=lambda _name: None,
        )
        observer = SimpleNamespace(
            on_version_pending=AsyncMock(),
            on_new_version=AsyncMock(),
        )
        watcher = WeightWatcher(
            SimpleNamespace(output_dir=tmp_path),
            policy=policy,
            inference=inference,
            observers=[observer],
            lora_name=None,
        )

        await watcher.update_lock.acquire()
        update = asyncio.create_task(watcher.apply_policy_update(1))
        await asyncio.sleep(0)

        inference.update_weights.assert_not_awaited()
        assert policy.version == 0

        latest_weights_path = get_step_path(broadcast_dir, 2)
        latest_weights_path.mkdir(parents=True)
        (latest_weights_path / "STABLE").touch()
        (first_weights_path / "STABLE").unlink()
        first_weights_path.rmdir()

        watcher.update_lock.release()
        await update

        inference.update_weights.assert_awaited_once_with(
            latest_weights_path,
            lora_name=None,
            step=2,
        )
        observer.on_version_pending.assert_awaited_once_with(2)
        observer.on_new_version.assert_awaited_once_with(2)
        assert watcher.ckpt_step == 2
        assert policy.version == 2

    asyncio.run(run())


def test_weight_watcher_waits_for_stable_marker_without_holding_policy_lock(tmp_path):
    async def run() -> None:
        weights_path = get_step_path(get_broadcast_dir(tmp_path), 1)
        weights_path.mkdir(parents=True)
        wait_started = asyncio.Event()
        finish_wait = asyncio.Event()

        async def controlled_wait(_path) -> None:
            wait_started.set()
            await finish_wait.wait()

        policy = Policy(model_name="policy", version=0)
        inference = SimpleNamespace(
            update_weights=AsyncMock(),
            update_model_name=lambda _name: None,
        )
        watcher = WeightWatcher(
            SimpleNamespace(output_dir=tmp_path),
            policy=policy,
            inference=inference,
            observers=[],
            lora_name=None,
        )

        with patch("prime_rl.orchestrator.watcher.wait_for_path", side_effect=controlled_wait):
            update = asyncio.create_task(watcher.apply_policy_update(1))
            await wait_started.wait()
            await asyncio.wait_for(watcher.update_lock.acquire(), timeout=0.1)
            watcher.update_lock.release()
            (weights_path / "STABLE").touch()
            finish_wait.set()
            await update

        inference.update_weights.assert_awaited_once_with(weights_path, lora_name=None, step=1)
        assert policy.version == 1

    asyncio.run(run())


def test_weight_watcher_retargets_when_requested_checkpoint_was_deleted(tmp_path):
    async def run() -> None:
        latest_weights_path = get_step_path(get_broadcast_dir(tmp_path), 2)
        latest_weights_path.mkdir(parents=True)
        (latest_weights_path / "STABLE").touch()
        policy = Policy(model_name="policy", version=0)
        inference = SimpleNamespace(
            update_weights=AsyncMock(),
            update_model_name=lambda _name: None,
        )
        watcher = WeightWatcher(
            SimpleNamespace(output_dir=tmp_path),
            policy=policy,
            inference=inference,
            observers=[],
            lora_name=None,
        )

        with patch("prime_rl.orchestrator.watcher.wait_for_path", new=AsyncMock()) as wait_mock:
            await watcher.apply_policy_update(1)

        wait_mock.assert_not_awaited()
        inference.update_weights.assert_awaited_once_with(latest_weights_path, lora_name=None, step=2)
        assert policy.version == 2

    asyncio.run(run())


def test_eval_trigger_waits_off_consumer_and_holds_lease_for_all_envs():
    async def run() -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.stopped = asyncio.Event()
        orchestrator.active_eval_epochs = set()
        orchestrator.pending_eval_steps = deque()
        orchestrator.eval_trigger_task = None
        orchestrator.eval_trigger_error = None
        orchestrator.eval_policy_lock_held = False
        orchestrator.eval_triggered_at = {}
        orchestrator.policy = Policy(version=3, model_name="policy")
        orchestrator.watcher = SimpleNamespace(update_lock=asyncio.Lock())
        orchestrator.dispatcher = SimpleNamespace(switch_mode=MagicMock())
        orchestrator.eval_source = SimpleNamespace(
            eligible_envs=lambda _step: ["a", "b"],
            trigger=lambda _step: ["a", "b"],
        )
        env = SimpleNamespace(config=SimpleNamespace(group_size=1), examples=[{"task_idx": 0}])
        orchestrator.eval_envs = SimpleNamespace(get=lambda _name: env)

        await orchestrator.watcher.update_lock.acquire()
        orchestrator.maybe_trigger_eval(10)
        trigger = orchestrator.eval_trigger_task
        assert trigger is not None
        await asyncio.sleep(0)

        assert not trigger.done()
        assert not orchestrator.active_eval_epochs

        orchestrator.watcher.update_lock.release()
        await trigger

        assert orchestrator.watcher.update_lock.locked()
        assert orchestrator.active_eval_epochs == {("a", 10), ("b", 10)}

        orchestrator._finalize_eval_batch = AsyncMock()
        await orchestrator.finalize_eval_batch(SimpleNamespace(env_name="a", step=10))
        assert orchestrator.watcher.update_lock.locked()
        await orchestrator.finalize_eval_batch(SimpleNamespace(env_name="b", step=10))
        assert not orchestrator.watcher.update_lock.locked()
        assert orchestrator.eval_pipeline_idle

    asyncio.run(run())


def test_eval_trigger_failure_releases_policy_lease():
    async def run() -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.stopped = asyncio.Event()
        orchestrator.active_eval_epochs = set()
        orchestrator.pending_eval_steps = deque()
        orchestrator.eval_trigger_task = None
        orchestrator.eval_trigger_error = None
        orchestrator.eval_policy_lock_held = False
        orchestrator.eval_triggered_at = {}
        orchestrator.watcher = SimpleNamespace(update_lock=asyncio.Lock())
        orchestrator.eval_source = SimpleNamespace(
            eligible_envs=lambda _step: ["eval"],
            trigger=MagicMock(side_effect=ValueError("invalid eval")),
        )

        orchestrator.maybe_trigger_eval(10)
        trigger = orchestrator.eval_trigger_task
        assert trigger is not None
        with pytest.raises(ValueError, match="invalid eval"):
            await trigger
        await asyncio.sleep(0)

        assert not orchestrator.watcher.update_lock.locked()
        assert not orchestrator.eval_policy_lock_held
        with pytest.raises(RuntimeError, match="Eval trigger failed") as exc_info:
            orchestrator._raise_if_eval_trigger_failed()
        assert isinstance(exc_info.value.__cause__, ValueError)

    asyncio.run(run())


def test_pending_eval_does_not_resume_train_after_scheduling_is_disabled():
    async def run() -> None:
        dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
        dispatcher.mode = DispatcherMode.PREFER_EVAL
        dispatcher.max_inflight = 1
        dispatcher.inflight_permits = 0
        dispatcher.inflight = {}
        dispatcher.groups = {object(): SimpleNamespace(kind="train", rollouts_to_schedule=1)}
        dispatcher.eval_source = deque([object()])
        dispatcher.train_scheduling_disabled = True
        dispatcher.try_schedule_existing = AsyncMock()
        dispatcher.try_schedule = AsyncMock(return_value=False)

        await dispatcher.fill_inflight()

        dispatcher.try_schedule_existing.assert_not_awaited()
        dispatcher.try_schedule.assert_awaited_once_with("eval")

    asyncio.run(run())


def test_eval_scheduling_drains_train_then_refills_during_eval_tail():
    async def run() -> None:
        dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
        dispatcher.mode = DispatcherMode.PREFER_EVAL
        dispatcher.max_inflight = 2
        dispatcher.inflight_permits = 0
        dispatcher.inflight = {}
        train_group = SimpleNamespace(kind="train", rollouts_to_schedule=1)
        dispatcher.groups = {object(): train_group}
        dispatcher.eval_source = deque([object(), object()])
        dispatcher.train_scheduling_disabled = False
        dispatcher.dispatch_allowed = asyncio.Event()
        dispatcher.dispatch_allowed.set()

        async def finish_open_train(_kind: str) -> bool:
            train_group.rollouts_to_schedule = 0
            dispatcher.inflight[object()] = SimpleNamespace(kind="train", rollout_count=1)
            dispatcher.inflight_permits += 1
            return True

        dispatcher.try_schedule_existing = AsyncMock(side_effect=finish_open_train)
        await dispatcher.fill_inflight()
        dispatcher.try_schedule_existing.assert_awaited_once_with("train")

        dispatcher.inflight.clear()
        dispatcher.inflight_permits = 0

        async def schedule_eval(_kind: str) -> bool:
            dispatcher.eval_source.popleft()
            dispatcher.inflight[object()] = SimpleNamespace(kind="eval", rollout_count=1)
            dispatcher.inflight_permits += 1
            return True

        dispatcher.try_schedule = AsyncMock(side_effect=schedule_eval)
        await dispatcher.fill_inflight()
        assert dispatcher.try_schedule.await_count == 2
        assert not dispatcher.eval_source
        assert dispatcher.inflight_eval_count == 2

        completed = next(iter(dispatcher.inflight))
        dispatcher.inflight.pop(completed)
        dispatcher.inflight_permits -= 1
        dispatcher.try_schedule = AsyncMock(return_value=False)
        await dispatcher.fill_inflight()

        assert dispatcher.mode == DispatcherMode.PREFER_TRAIN
        dispatcher.try_schedule.assert_awaited_once_with("train")
        assert dispatcher.inflight_eval_count == 1

    asyncio.run(run())
