from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import prime_rl.orchestrator.hf_export as hf_export
import prime_rl.orchestrator.orchestrator as orchestrator_module
from prime_rl.orchestrator.orchestrator import Orchestrator


@pytest.fixture
def finalizer(tmp_path, monkeypatch):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = SimpleNamespace(
        output_dir=tmp_path,
        ckpt=SimpleNamespace(
            best_metrics=("training_reward", "continuous_reward"),
            hf_repo_prefix="owner/model",
        ),
        weight_broadcast=SimpleNamespace(timeout=1),
    )
    orchestrator.ckpt_manager = SimpleNamespace(ckpt_dir=tmp_path / "checkpoints", generation="g1", maybe_clean=Mock())
    orchestrator.retention = SimpleNamespace(
        best_steps={"training_reward": 40, "continuous_reward": 60}, reconcile=Mock()
    )
    monkeypatch.setattr(orchestrator_module, "is_joint_ckpt_step", lambda *args, **kwargs: True)
    monkeypatch.setattr(orchestrator_module, "get_resume_ckpt_steps", lambda _: [40, 60, 80])
    export = Mock(return_value=True)
    monkeypatch.setattr(hf_export, "export_checkpoints", export)
    return orchestrator, export


@pytest.mark.asyncio
async def test_final_export_passes_both_best_roles_and_latest(finalizer):
    orchestrator, export = finalizer
    await orchestrator.finalize_exports(80)
    export.assert_called_once_with(
        output_dir=orchestrator.config.output_dir,
        roles={"latest": 80, "best_training_reward": 40, "best_continuous_reward": 60},
        repo_prefix="owner/model",
    )


@pytest.mark.asyncio
async def test_final_export_requires_all_requested_best_roles(finalizer):
    orchestrator, export = finalizer
    orchestrator.retention.best_steps["training_reward"] = None
    with pytest.raises(RuntimeError, match="No eligible held-out BEST"):
        await orchestrator.finalize_exports(80)
    export.assert_not_called()


@pytest.mark.asyncio
async def test_final_export_rejects_nonjoint_best_and_failed_export(finalizer):
    orchestrator, export = finalizer
    orchestrator.retention.best_steps["training_reward"] = 20
    with pytest.raises(RuntimeError, match="invalid joint checkpoint"):
        await orchestrator.finalize_exports(80)
    export.assert_not_called()
    orchestrator.retention.best_steps["training_reward"] = 40
    export.return_value = False
    with pytest.raises(RuntimeError, match="did not validate"):
        await orchestrator.finalize_exports(80)


@pytest.mark.asyncio
async def test_legacy_final_export_preserves_best_alias_and_latest_only(finalizer):
    orchestrator, export = finalizer
    orchestrator.config.ckpt.best_metrics = ("continuous_reward",)
    orchestrator.retention.best_steps = {"continuous_reward": 60}
    await orchestrator.finalize_exports(80)
    assert export.call_args.kwargs["roles"] == {"latest": 80, "best": 60}
    orchestrator.retention.best_steps = {"continuous_reward": None}
    await orchestrator.finalize_exports(80)
    assert export.call_args.kwargs["roles"] == {"latest": 80}


@pytest.mark.asyncio
async def test_final_drain_waits_for_final_broadcast_and_eval_trigger(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.stopped = asyncio.Event()
    orchestrator.draining = True
    orchestrator.config = SimpleNamespace(max_steps=80)
    orchestrator.progress = SimpleNamespace(step=81)
    orchestrator.policy = SimpleNamespace(version=79)
    orchestrator.watcher = SimpleNamespace(update_lock=asyncio.Lock())
    orchestrator.dispatcher = SimpleNamespace(is_idle=True, out_q=asyncio.Queue())
    orchestrator._raise_if_component_stopped = Mock()

    async def apply_final():
        orchestrator.policy.version = 80
        orchestrator.dispatcher.is_idle = False
        orchestrator.dispatcher.out_q.put_nowait(SimpleNamespace(run=None))

    orchestrator.wait_for_final_broadcast = AsyncMock(side_effect=apply_final)
    # The deliberately invalid item proves the loop resumed consuming the final eval queue.
    with pytest.raises(ValueError, match="missing run identity"):
        await orchestrator.main_loop()
    orchestrator.wait_for_final_broadcast.assert_awaited_once()
    assert not orchestrator.stopped.is_set()


@pytest.mark.asyncio
async def test_final_drain_rejects_unapplied_policy():
    import asyncio
    from unittest.mock import AsyncMock

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.stopped = asyncio.Event()
    orchestrator.draining = True
    orchestrator.config = SimpleNamespace(max_steps=80)
    orchestrator.progress = SimpleNamespace(step=81)
    orchestrator.policy = SimpleNamespace(version=79)
    orchestrator.dispatcher = SimpleNamespace(is_idle=True)
    orchestrator._raise_if_component_stopped = Mock()
    orchestrator.wait_for_final_broadcast = AsyncMock()
    with pytest.raises(RuntimeError, match="Final policy was not applied"):
        await orchestrator.main_loop()
    assert not orchestrator.stopped.is_set()
