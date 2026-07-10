"""Cancellation and partial-failure contracts for stateful Dynamo admin fanouts."""

import asyncio
from pathlib import Path

import pytest

from prime_rl.inference.dynamo_admin import DynamoAdminAPI


@pytest.mark.asyncio
async def test_nccl_initialization_settles_siblings_and_fails_closed_after_partial_error(
    monkeypatch: pytest.MonkeyPatch,
):
    admin = DynamoAdminAPI()
    delayed_started = asyncio.Event()
    release_delayed = asyncio.Event()
    delayed_finished = asyncio.Event()
    calls = 0

    async def post(client, method, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        assert method == "init_weights_update_group"
        if client == "failed":
            raise RuntimeError("init failed")
        delayed_started.set()
        await release_delayed.wait()
        delayed_finished.set()
        return {}

    monkeypatch.setattr(admin, "_post", post)
    initialize = asyncio.create_task(
        admin.initialize_nccl(
            ["failed", "delayed"],
            host="localhost",
            port=29511,
            timeout=12000,
            inference_world_size=2,
            gpus_per_worker=1,
            quantize_in_weight_transfer=False,
        )
    )
    await delayed_started.wait()
    await asyncio.sleep(0)
    assert not initialize.done()

    release_delayed.set()
    with pytest.raises(RuntimeError, match="init failed"):
        await initialize
    assert delayed_finished.is_set()

    with pytest.raises(RuntimeError, match="indeterminate"):
        await admin.initialize_nccl(
            ["failed", "delayed"],
            host="localhost",
            port=29511,
            timeout=12000,
            inference_world_size=2,
            gpus_per_worker=1,
            quantize_in_weight_transfer=False,
        )
    with pytest.raises(RuntimeError, match="indeterminate"):
        await admin.update_weights(["failed", "delayed"], Path("weights"), step=1)
    assert calls == 2


@pytest.mark.asyncio
async def test_nccl_initialization_settles_before_propagating_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch,
):
    admin = DynamoAdminAPI()
    init_started = asyncio.Event()
    release_init = asyncio.Event()
    init_finished = asyncio.Event()

    async def post(_client, method, *_args, **_kwargs):
        assert method == "init_weights_update_group"
        init_started.set()
        await release_init.wait()
        init_finished.set()
        return {}

    monkeypatch.setattr(admin, "_post", post)
    initialize = asyncio.create_task(
        admin.initialize_nccl(
            ["worker"],
            host="localhost",
            port=29511,
            timeout=12000,
            inference_world_size=1,
            gpus_per_worker=1,
            quantize_in_weight_transfer=False,
        )
    )
    await init_started.wait()
    initialize.cancel()
    await asyncio.sleep(0)
    initialize.cancel()
    await asyncio.sleep(0)
    assert not initialize.done()

    release_init.set()
    with pytest.raises(asyncio.CancelledError):
        await initialize
    assert init_finished.is_set()

    with pytest.raises(RuntimeError, match="indeterminate"):
        await admin.update_weights(["worker"], Path("weights"), step=1)


@pytest.mark.asyncio
async def test_pause_fanout_settles_delayed_sibling_before_resume(monkeypatch: pytest.MonkeyPatch):
    admin = DynamoAdminAPI()
    delayed_started = asyncio.Event()
    release_delayed = asyncio.Event()
    delayed_finished = asyncio.Event()
    resume_started = asyncio.Event()

    async def post(client, method, *_args, **_kwargs):
        if method == "pause_generation":
            if client == "fast-failure":
                raise RuntimeError("pause failed")
            delayed_started.set()
            await release_delayed.wait()
            delayed_finished.set()
            return {}
        if method == "resume_generation":
            resume_started.set()
            assert delayed_finished.is_set()
            return {}
        raise AssertionError(method)

    monkeypatch.setattr(admin, "_post", post)
    update = asyncio.create_task(admin.update_weights(["fast-failure", "delayed"], Path("weights"), step=1))
    await delayed_started.wait()
    await asyncio.sleep(0)
    assert not resume_started.is_set()

    release_delayed.set()
    with pytest.raises(RuntimeError, match="pause failed"):
        await update

    assert resume_started.is_set()


@pytest.mark.asyncio
async def test_collective_update_settles_delayed_sibling_before_resume(monkeypatch: pytest.MonkeyPatch):
    admin = DynamoAdminAPI()
    delayed_started = asyncio.Event()
    release_delayed = asyncio.Event()
    delayed_finished = asyncio.Event()
    resume_started = asyncio.Event()

    async def post(client, method, *_args, **_kwargs):
        if method == "pause_generation":
            return {}
        if method == "update_weights_from_disk":
            if client == "fast-failure":
                raise RuntimeError("collective failed")
            delayed_started.set()
            await release_delayed.wait()
            delayed_finished.set()
            return {}
        if method == "resume_generation":
            resume_started.set()
            assert delayed_finished.is_set()
            return {}
        raise AssertionError(method)

    monkeypatch.setattr(admin, "_post", post)
    update = asyncio.create_task(admin.update_weights(["fast-failure", "delayed"], Path("weights"), step=1))
    await delayed_started.wait()
    await asyncio.sleep(0)
    assert not resume_started.is_set()

    release_delayed.set()
    with pytest.raises(RuntimeError, match="collective failed"):
        await update

    assert resume_started.is_set()


@pytest.mark.asyncio
async def test_collective_update_is_settled_before_propagating_cancellation(monkeypatch: pytest.MonkeyPatch):
    admin = DynamoAdminAPI()
    update_started = asyncio.Event()
    release_update = asyncio.Event()
    update_finished = asyncio.Event()
    resume_started = asyncio.Event()

    async def post(_client, method, *_args, **_kwargs):
        if method == "pause_generation":
            return {}
        if method == "update_weights_from_disk":
            update_started.set()
            await release_update.wait()
            update_finished.set()
            return {}
        if method == "resume_generation":
            resume_started.set()
            assert update_finished.is_set()
            return {}
        raise AssertionError(method)

    monkeypatch.setattr(admin, "_post", post)
    update = asyncio.create_task(admin.update_weights(["worker"], Path("weights"), step=1))
    await update_started.wait()
    update.cancel()
    await asyncio.sleep(0)
    update.cancel()
    await asyncio.sleep(0)

    assert not update.done()
    assert not resume_started.is_set()

    release_update.set()
    with pytest.raises(asyncio.CancelledError):
        await update

    assert update_finished.is_set()
    assert resume_started.is_set()


@pytest.mark.asyncio
async def test_resume_failure_after_successful_mutation_is_reported(monkeypatch: pytest.MonkeyPatch):
    admin = DynamoAdminAPI()
    operations: list[str] = []

    async def post(_client, method, *_args, **_kwargs):
        operations.append(method)
        if method == "resume_generation":
            raise RuntimeError("resume failed after mutation")
        return {}

    monkeypatch.setattr(admin, "_post", post)

    with pytest.raises(RuntimeError, match="resume failed after mutation"):
        await admin.update_weights(["worker"], Path("weights"), step=1)

    assert operations == ["pause_generation", "update_weights_from_disk", "resume_generation"]
    with pytest.raises(RuntimeError, match="weight state is indeterminate"):
        await admin.update_weights(["worker"], Path("weights"), step=2)
    assert operations == ["pause_generation", "update_weights_from_disk", "resume_generation"]


@pytest.mark.asyncio
async def test_fanout_preserves_primary_error_and_annotates_siblings(monkeypatch: pytest.MonkeyPatch):
    admin = DynamoAdminAPI()

    async def post(client, method, *_args, **_kwargs):
        if method == "pause_generation":
            raise RuntimeError(f"{client} pause failed")
        if method == "resume_generation":
            return {}
        raise AssertionError(method)

    monkeypatch.setattr(admin, "_post", post)

    with pytest.raises(RuntimeError, match="primary pause failed") as exc:
        await admin.update_weights(["primary", "sibling"], Path("weights"), step=1)

    assert exc.value.__notes__ == ["Dynamo pause_generation sibling also failed: RuntimeError('sibling pause failed')"]
