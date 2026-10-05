import asyncio
import hashlib
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi import FastAPI

from prime_rl.inference.vllm.delta_sync import GENERATION_PATHS, WeightServingMiddleware
from prime_rl.inference.vllm.delta_sync import router as delta_sync_router
from prime_rl.inference.vllm.server import router as server_router
from prime_rl.orchestrator.delta_sync import (
    DeltaEndpointPool,
    commit_weights,
    reload_weights,
    set_weight_serving,
    stage_weights,
)
from prime_rl.utils.weight_sync import WEIGHT_VERSION_HEADER


class FakeEngineClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.pause_modes: list[str] = []
        self.paused = False

    async def collective_rpc(self, method: str, args: tuple = ()) -> None:
        self.calls.append((method, args))
        await asyncio.sleep(0)

    async def pause_generation(self, mode: str = "keep", clear_cache: bool = False) -> None:
        self.pause_modes.append(mode)
        self.paused = True

    async def resume_generation(self) -> None:
        self.paused = False

    async def check_health(self) -> None:
        pass


def make_app(
    tmp_path,
    *,
    relay_peers: list[str] | None = None,
    relay_transports: dict[str, httpx.AsyncBaseTransport] | None = None,
    relay_fail_on_peer_error: bool = False,
    stage_chunk_delay_s: float = 0,
):
    app = FastAPI()
    app.include_router(delta_sync_router)
    app.include_router(server_router)
    app.add_middleware(WeightServingMiddleware)

    @app.post("/v1/chat/completions")
    async def generate():
        app.state.generation_requests += 1
        return {"version": app.state.active_version}

    app.state.engine_client = FakeEngineClient()
    app.state.staging_dir = tmp_path / "staging"
    app.state.staging_dir.mkdir(parents=True)
    app.state.staged_versions = {}
    app.state.stage_uploads = {}
    app.state.active_version = "base"
    app.state.active_weight_sha256 = None
    app.state.weight_serving_ready = True
    app.state.require_weight_version = False
    app.state.generation_requests = 0
    app.state.api_server_count = 1
    app.state.relay_enabled = bool(relay_peers)
    app.state.relay_peers = relay_peers or []
    app.state.relay_transports = relay_transports or {}
    app.state.relay_fail_on_peer_error = relay_fail_on_peer_error
    app.state.relay_stage_timeout_s = 30.0
    app.state.relay_commit_timeout_s = 30.0
    app.state.relay_reload_timeout_s = 30.0
    app.state.active_stage_chunk_requests = 0
    app.state.max_active_stage_chunk_requests = 0

    if stage_chunk_delay_s:

        @app.middleware("http")
        async def delay_stage_chunks(request, call_next):
            if request.url.path not in {"/stage_chunk", "/stage_stream_chunk"}:
                return await call_next(request)
            app.state.active_stage_chunk_requests += 1
            app.state.max_active_stage_chunk_requests = max(
                app.state.max_active_stage_chunk_requests,
                app.state.active_stage_chunk_requests,
            )
            try:
                await asyncio.sleep(stage_chunk_delay_s)
                return await call_next(request)
            finally:
                app.state.active_stage_chunk_requests -= 1

    return app


def test_stage_commit_delta_path_protocol(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    (delta_dir / "delta.safetensors").write_bytes(b"placeholder")
    app = make_app(tmp_path)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            stage_response = await client.post(
                "/stage",
                data={"version": "1", "mode": "delta", "base_version": "base", "path": delta_dir.as_posix()},
            )
            assert stage_response.status_code == 200

            responses = await asyncio.gather(
                *(client.post("/commit", data={"version": "1", "mode": "delta"}) for _ in range(3))
            )
            assert all(response.status_code == 200 for response in responses)

    asyncio.run(run())

    assert app.state.active_version == "1"
    assert app.state.engine_client.calls == [("update_weights_from_delta_path", (delta_dir.as_posix(),))]


@pytest.mark.parametrize("active_version", ["base", "1"])
def test_stage_delta_rejects_base_version_mismatch(tmp_path, active_version) -> None:
    delta_dir = tmp_path / "step_2"
    delta_dir.mkdir()
    (delta_dir / "delta.safetensors").write_bytes(b"placeholder")
    app = make_app(tmp_path)
    app.state.active_version = active_version
    correct_base = active_version

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            mismatch_response = await client.post(
                "/stage",
                data={"version": "2", "mode": "delta", "base_version": "wrong", "path": delta_dir.as_posix()},
            )
            assert mismatch_response.status_code == 409
            assert app.state.staged_versions == {}

            stage_response = await client.post(
                "/stage",
                data={"version": "2", "mode": "delta", "base_version": correct_base, "path": delta_dir.as_posix()},
            )
            assert stage_response.status_code == 200
            app.state.active_version = "changed-after-staging"
            commit_response = await client.post("/commit", data={"version": "2", "mode": "delta"})
            assert commit_response.status_code == 409
            assert app.state.engine_client.calls == []

    asyncio.run(run())

    assert app.state.staged_versions["2"]["path"] == delta_dir


def test_reload_weights_clears_staging_state(tmp_path) -> None:
    app = make_app(tmp_path)
    staged_file = app.state.staging_dir / "1_delta_delta.safetensors"
    staged_file.write_bytes(b"placeholder")
    app.state.staged_versions = {"1": {"path": staged_file, "mode": "delta", "owned": True}}
    app.state.active_version = "1"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/reload_weights")
            assert response.status_code == 200

    asyncio.run(run())

    assert app.state.active_version == "base"
    assert app.state.staged_versions == {}
    assert not staged_file.exists()
    assert app.state.engine_client.calls == [("reload_weights", ())]


def test_failed_commit_requires_reload_before_retry(tmp_path) -> None:
    app = make_app(tmp_path)
    delta = tmp_path / "delta.safetensors"
    delta.write_bytes(b"delta")

    async def fail_update(method, args=()):
        raise RuntimeError("worker failed during partial application")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/stage", data={"version": "1", "mode": "delta", "base_version": "base", "path": str(delta)}
            )
            assert response.status_code == 200
            app.state.engine_client.collective_rpc = fail_update
            with pytest.raises(RuntimeError, match="partial application"):
                await client.post("/commit", data={"version": "1"})
            response = await client.post("/commit", data={"version": "1"})
            assert response.status_code == 409
            assert (await client.post("/resume")).status_code == 409
            app.state.engine_client = FakeEngineClient()
            assert (await client.post("/reload_weights")).status_code == 200
            assert not app.state.weights_dirty

    asyncio.run(run())


def test_weight_health_rejects_dirty_worker() -> None:
    app = FastAPI()
    app.include_router(server_router)
    app.state.engine_client = FakeEngineClient()
    app.state.weights_dirty = False

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/weight_health")).status_code == 200
            app.state.weights_dirty = True
            assert (await client.get("/weight_health")).status_code == 503

    asyncio.run(run())


@pytest.mark.gpu
def test_weight_guard_registers_before_vllm_builds_middleware_stack() -> None:
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from prime_rl.inference.vllm.server import custom_build_app

    args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
    app = custom_build_app(args, ("generate",))
    app.state.weight_serving_ready = False
    assert args.middleware == []

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/completions", json={"model": "test", "prompt": "hello"})
            assert response.status_code == 503
            assert response.json()["error"] == "weights are not ready for generation"

    asyncio.run(run())


def test_generation_is_fenced_until_exact_serving_version_is_ready(tmp_path) -> None:
    app = make_app(tmp_path)
    app.state.active_version = "2"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/v1/chat/completions")).status_code == 200
            await set_weight_serving(client, enabled=False)
            assert app.state.engine_client.pause_modes[-1] == "abort"
            assert (await client.get("/weight_health")).status_code == 503
            assert (await client.post("/resume")).status_code == 409
            assert (await client.post("/v1/chat/completions", headers={WEIGHT_VERSION_HEADER: "2"})).status_code == 503
            assert (await client.post("/reload_weights", data={"resume": "false"})).status_code == 200
            assert not app.state.weight_serving_ready
            app.state.active_version = "2"
            response = await client.post("/weight_serving", json={"enabled": True, "version": "3"})
            assert response.status_code == 409
            assert not app.state.weight_serving_ready
            await set_weight_serving(client, enabled=True, version="2")
            assert (await client.post("/v1/chat/completions")).status_code == 400
            assert (await client.post("/v1/chat/completions", headers={WEIGHT_VERSION_HEADER: "3"})).status_code == 503
            assert (await client.post("/v1/chat/completions", headers={WEIGHT_VERSION_HEADER: "2"})).status_code == 200
            assert app.state.generation_requests == 2

    asyncio.run(run())


@pytest.mark.parametrize("root_path", ["", "/api"])
def test_quarantine_covers_model_routes_with_application_root_path(tmp_path, root_path) -> None:
    app = make_app(tmp_path)
    app.root_path = root_path
    app.state.weight_serving_ready = False

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for path in GENERATION_PATHS:
                response = await client.post(root_path + path, json={})
                assert response.status_code == 503, path
            assert (await client.get(root_path + "/v1/models")).status_code == 404

    asyncio.run(run())


@pytest.mark.parametrize("transport", ["multipart", "chunked", "streaming"])
@pytest.mark.parametrize("trust", ["clean", "dirty", "wrong_hash"])
def test_background_recovery_stays_fenced_and_catches_up_without_blocking(tmp_path, transport, trust) -> None:
    healthy = make_app(tmp_path / "healthy")
    lagging = make_app(tmp_path / "lagging")

    async def run() -> None:
        fail_stage = False
        delay_recovery = False
        started = asyncio.Event()
        release = asyncio.Event()

        async def lagging_transport(request):
            if request.url.path.startswith("/stage"):
                if fail_stage:
                    return httpx.Response(503)
                if delay_recovery:
                    started.set()
                    await release.wait()
            return await httpx.ASGITransport(app=lagging).handle_async_request(request)

        async with (
            httpx.AsyncClient(transport=httpx.ASGITransport(app=healthy), base_url="http://healthy") as first,
            httpx.AsyncClient(transport=httpx.MockTransport(lagging_transport), base_url="http://lagging") as second,
        ):
            pool = DeltaEndpointPool(
                [first, second],
                lease_enabled=True,
                recovery_enabled=True,
                cooldown_s=0.01,
                health_timeout_s=0.02,
                stage_num_streams=2,
                stage_chunk_size_bytes=4,
                stage_chunk_retries=0,
                stage_retries=0,
            )

            async def advance(version):
                path = tmp_path / f"delta-{version}.safetensors"
                path.write_bytes(f"delta-{version}".encode())
                await pool.stage(
                    path,
                    version=str(version),
                    base_version=pool.active_version,
                    upload=True,
                    upload_method=transport,
                    done_path=None,
                )
                await pool.commit(str(version))

            try:
                await advance(0)
                await advance(1)
                fail_stage = True
                await advance(2)
                assert lagging.state.active_version == "1"
                assert not lagging.state.weight_serving_ready
                assert (await second.get("/weight_health")).status_code == 503
                if trust == "dirty":
                    lagging.state.weights_dirty = True
                elif trust == "wrong_hash":
                    lagging.state.active_weight_sha256 = "wrong"
                fail_stage = False
                delay_recovery = True
                await asyncio.wait_for(started.wait(), timeout=2)
                await asyncio.wait_for(advance(3), timeout=1)
                assert healthy.state.active_version == "3"
                assert not lagging.state.weight_serving_ready
                response = await second.post("/v1/chat/completions", headers={WEIGHT_VERSION_HEADER: "3"})
                assert response.status_code == 503
                assert lagging.state.generation_requests == 0
                task = pool._recovery_tasks["http://lagging"]
                release.set()
                await asyncio.wait_for(task, timeout=2)
                assert lagging.state.active_version == "3"
                assert lagging.state.weight_serving_ready
                runtime = pool.runtime["http://lagging"]
                assert runtime.state == "healthy"
                assert runtime.reload_count == (0 if trust == "clean" else 1)
                assert runtime.replayed_delta_count == (2 if trust == "clean" else 4)
                assert (
                    await second.post("/v1/chat/completions", headers={WEIGHT_VERSION_HEADER: "3"})
                ).status_code == 200
            finally:
                release.set()
                await pool.aclose()

    asyncio.run(run())


def test_stale_worker_rejects_new_policy_even_when_admin_quarantine_is_unreachable(tmp_path) -> None:
    healthy = make_app(tmp_path / "healthy")
    lagging = make_app(tmp_path / "lagging")

    async def run() -> None:
        failed = False

        async def partial_outage(request):
            if failed and (request.url.path.startswith("/stage") or request.url.path == "/weight_serving"):
                return httpx.Response(503)
            return await httpx.ASGITransport(app=lagging).handle_async_request(request)

        async with (
            httpx.AsyncClient(transport=httpx.ASGITransport(app=healthy), base_url="http://healthy") as first,
            httpx.AsyncClient(transport=httpx.MockTransport(partial_outage), base_url="http://lagging") as second,
        ):
            pool = DeltaEndpointPool(
                [first, second],
                lease_enabled=True,
                recovery_enabled=False,
                cooldown_s=0,
                health_timeout_s=1,
                stage_chunk_retries=0,
                stage_retries=0,
            )
            try:
                for version in range(2):
                    path = tmp_path / f"delta-{version}.safetensors"
                    path.write_bytes(f"delta-{version}".encode())
                    failed = version == 1
                    await pool.stage(
                        path,
                        version=str(version),
                        base_version=pool.active_version,
                        upload=True,
                        upload_method="multipart",
                        done_path=None,
                    )
                    await pool.commit(str(version))
                assert lagging.state.active_version == "0"
                assert lagging.state.weight_serving_ready
                response = await second.post("/v1/chat/completions", headers={WEIGHT_VERSION_HEADER: "1"})
                assert response.status_code == 503
                assert lagging.state.generation_requests == 0
            finally:
                await pool.aclose()

    asyncio.run(run())


def test_newer_quarantine_supersedes_pending_enable(tmp_path) -> None:
    app = make_app(tmp_path)
    app.state.active_version = "0"

    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_resume():
            entered.set()
            await release.wait()
            app.state.engine_client.paused = False

        app.state.engine_client.resume_generation = delayed_resume
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            enable = asyncio.create_task(client.post("/weight_serving", json={"enabled": True, "version": "0"}))
            await asyncio.wait_for(entered.wait(), timeout=1)
            disable = asyncio.create_task(client.post("/weight_serving", json={"enabled": False}))
            await asyncio.sleep(0)
            assert not app.state.weight_serving_ready
            assert (await client.post("/v1/chat/completions", headers={WEIGHT_VERSION_HEADER: "0"})).status_code == 503
            release.set()
            assert (await asyncio.wait_for(enable, timeout=1)).status_code == 409
            assert (await asyncio.wait_for(disable, timeout=1)).status_code == 200
            assert app.state.engine_client.paused
            assert not app.state.weight_serving_ready

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_recovery_admission_does_not_hold_pool_lock_and_shutdown_fences_worker(tmp_path, cancel) -> None:
    healthy = make_app(tmp_path / "healthy")
    lagging = make_app(tmp_path / "lagging")

    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        delay = False

        async def delayed_control(request):
            response = await httpx.ASGITransport(app=lagging).handle_async_request(request)
            if delay and request.url.path == "/weight_serving" and b'"enabled":true' in request.content:
                entered.set()
                await release.wait()
            return response

        async with (
            httpx.AsyncClient(transport=httpx.ASGITransport(app=healthy), base_url="http://healthy") as first,
            httpx.AsyncClient(transport=httpx.MockTransport(delayed_control), base_url="http://lagging") as second,
        ):
            pool = DeltaEndpointPool(
                [first, second], lease_enabled=True, recovery_enabled=True, cooldown_s=0, health_timeout_s=1
            )

            async def advance(version):
                path = tmp_path / f"delta-{version}.safetensors"
                path.write_bytes(f"delta-{version}".encode())
                await pool.stage(
                    path,
                    version=str(version),
                    base_version=pool.active_version,
                    upload=True,
                    upload_method="multipart",
                    done_path=None,
                )
                await pool.commit(str(version))

            try:
                await advance(0)
                pool._retire("http://lagging", "lost response")
                delay = True
                pool._schedule_recoveries()
                await asyncio.wait_for(entered.wait(), timeout=1)
                await asyncio.wait_for(advance(1), timeout=1)
                assert healthy.state.active_version == "1"
                assert lagging.state.active_version == "0"
                assert (
                    await second.post("/v1/chat/completions", headers={WEIGHT_VERSION_HEADER: "1"})
                ).status_code == 503
                task = pool._recovery_tasks["http://lagging"]
                delay = False
                if cancel:
                    await asyncio.wait_for(pool.aclose(), timeout=1)
                    assert task.cancelled()
                    assert not lagging.state.weight_serving_ready
                    assert lagging.state.engine_client.paused
                    assert not pool._recovery_tasks
                else:
                    release.set()
                    await asyncio.wait_for(task, timeout=1)
                    assert lagging.state.active_version == "1"
                    assert lagging.state.weight_serving_ready
                    assert pool.runtime["http://lagging"].state == "healthy"
            finally:
                release.set()
                await pool.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("dirty_peer", [False, True])
def test_relay_recovery_replays_each_peer_from_its_own_trusted_version(tmp_path, dirty_peer) -> None:
    healthy = make_app(tmp_path / "healthy")
    peer = make_app(tmp_path / "peer")
    fail_commit = False

    async def peer_transport(request):
        if fail_commit and request.url.path == "/commit":
            return httpx.Response(503)
        return await httpx.ASGITransport(app=peer).handle_async_request(request)

    seed = make_app(
        tmp_path / "seed",
        relay_peers=["http://peer"],
        relay_transports={"http://peer": httpx.MockTransport(peer_transport)},
        relay_fail_on_peer_error=True,
    )

    async def run() -> None:
        nonlocal fail_commit
        async with (
            httpx.AsyncClient(transport=httpx.ASGITransport(app=healthy), base_url="http://healthy") as first,
            httpx.AsyncClient(transport=httpx.ASGITransport(app=seed), base_url="http://seed") as second,
        ):
            pool = DeltaEndpointPool(
                [first, second], lease_enabled=True, recovery_enabled=False, cooldown_s=0, health_timeout_s=1
            )
            try:
                for version in range(3):
                    path = tmp_path / f"delta-{version}.safetensors"
                    path.write_bytes(f"delta-{version}".encode())
                    fail_commit = version == 1
                    await pool.stage(
                        path,
                        version=str(version),
                        base_version=pool.active_version,
                        upload=True,
                        upload_method="multipart",
                        done_path=None,
                    )
                    await pool.commit(str(version))
                assert seed.state.active_version == "1"
                assert peer.state.active_version == "0"
                assert not seed.state.weight_serving_ready and not peer.state.weight_serving_ready
                peer.state.weights_dirty = dirty_peer
                fail_commit = False
                pool.recovery_enabled = True
                pool._schedule_recoveries()
                await asyncio.wait_for(pool._recovery_tasks["http://seed"], timeout=2)
                assert seed.state.active_version == peer.state.active_version == "2"
                assert seed.state.weight_serving_ready and peer.state.weight_serving_ready
                assert len(seed.state.engine_client.calls) == 3
                peer_methods = [method for method, _ in peer.state.engine_client.calls]
                assert peer_methods.count("reload_weights") == int(dirty_peer)
                runtime = pool.runtime["http://seed"]
                assert runtime.reload_count == int(dirty_peer)
                assert runtime.replayed_delta_count == (4 if dirty_peer else 3)
                assert (await second.get("/weight_peer/0/v1/chat/completions")).status_code == 404
                assert (await second.get("/weight_peer/1/weight_status")).status_code == 404
            finally:
                await pool.aclose()

    asyncio.run(run())


def test_relay_commit_retry_does_not_reapply_seed_delta(tmp_path) -> None:
    peer = make_app(tmp_path / "peer")
    attempts = 0

    async def fail_first_commit(request):
        nonlocal attempts
        if request.url.path == "/commit":
            attempts += 1
            if attempts == 1:
                return httpx.Response(503)
        return await httpx.ASGITransport(app=peer).handle_async_request(request)

    seed = make_app(
        tmp_path / "seed",
        relay_peers=["http://peer"],
        relay_transports={"http://peer": httpx.MockTransport(fail_first_commit)},
        relay_fail_on_peer_error=True,
    )
    delta = tmp_path / "delta.safetensors"
    delta.write_bytes(b"delta")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=seed), base_url="http://seed") as client:
            await stage_weights([client], delta, version="1", mode="delta", base_version="base", upload=True)
            assert (await client.post("/commit", data={"version": "1"})).status_code == 502
            assert (await client.post("/commit", data={"version": "1"})).status_code == 200
            assert seed.state.active_version == peer.state.active_version == "1"
            assert len(seed.state.engine_client.calls) == len(peer.state.engine_client.calls) == 1

    asyncio.run(run())


def test_streaming_upload_sessions_do_not_share_files(tmp_path) -> None:
    app = make_app(tmp_path)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            uploads = []
            for _ in range(2):
                response = await client.post(
                    "/stage_stream_init",
                    json={"version": "1", "mode": "delta", "base_version": "base", "filename": "delta.stream"},
                )
                assert response.status_code == 200
                uploads.append(response.json()["upload_id"])
            for upload_id, content in zip(uploads, [b"first", b"second"]):
                response = await client.post(
                    "/stage_stream_chunk", params={"upload_id": upload_id, "offset": 0}, content=content
                )
                assert response.status_code == 200
            for upload_id, content in zip(uploads, [b"first", b"second"]):
                response = await client.post(
                    "/stage_stream_finalize",
                    data={
                        "upload_id": upload_id,
                        "final_size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    },
                )
                assert response.status_code == 200
                assert app.state.staged_versions["1"]["path"].read_bytes() == content

    asyncio.run(run())


def test_pool_restages_pending_delta_after_recovery(tmp_path) -> None:
    app = make_app(tmp_path)
    app.state.active_version = "base"
    delta = tmp_path / "delta.safetensors"
    delta.write_bytes(b"delta")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            pool = DeltaEndpointPool(
                [client],
                lease_enabled=True,
                recovery_enabled=True,
                cooldown_s=0,
                health_timeout_s=1,
            )
            await pool.stage(
                delta,
                version="1",
                base_version="base",
                upload=True,
                upload_method="multipart",
                done_path=None,
            )
            pool._retire("http://test", "simulated rollout failure")
            await pool._quarantine(client)
            pool._schedule_recoveries()
            await asyncio.wait_for(pool._recovery_tasks["http://test"], timeout=1)
            assert pool.runtime["http://test"].state == "healthy"
            assert pool.runtime["http://test"].reload_count == 0
            await pool.commit("1")
            assert app.state.active_version == "1"
            assert [method for method, _ in app.state.engine_client.calls] == ["update_weights_from_delta_path"]
            await pool.aclose()

    asyncio.run(run())


def test_client_stage_uploads_delta_file(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"uploaded-delta")
    app = make_app(tmp_path)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await stage_weights([client], delta_dir, version="1", mode="delta", base_version="base", upload=True)

    asyncio.run(run())

    staged = app.state.staged_versions["1"]
    staged_path = staged["path"]
    assert staged["owned"] is True
    assert staged["mode"] == "delta"
    assert staged_path.parent == app.state.staging_dir
    assert staged_path.read_bytes() == b"uploaded-delta"


def test_client_stage_chunk_uploads_delta_file(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"chunked-delta-upload")
    app = make_app(tmp_path)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await stage_weights(
                [client],
                delta_dir,
                version="1",
                mode="delta",
                base_version="base",
                upload=True,
                upload_method="chunked",
                chunk_size_bytes=5,
            )

    results = asyncio.run(run())

    staged = app.state.staged_versions["1"]
    staged_path = staged["path"]
    assert [result.endpoint for result in results] == ["http://test"]
    assert [result.operation for result in results] == ["stage_weights"]
    assert all(result.ok for result in results)
    assert app.state.stage_uploads == {}
    assert staged["owned"] is True
    assert staged["mode"] == "delta"
    assert staged_path.parent == app.state.staging_dir
    assert staged_path.read_bytes() == b"chunked-delta-upload"


def test_client_stage_chunk_uploads_delta_file_to_multiple_endpoints(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"multi-endpoint-delta")
    app_dirs = [tmp_path / "a", tmp_path / "b"]
    for app_dir in app_dirs:
        app_dir.mkdir()
    apps = [make_app(app_dir) for app_dir in app_dirs]

    async def run() -> None:
        clients = [
            httpx.AsyncClient(transport=httpx.ASGITransport(app=apps[0]), base_url="http://worker-a"),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=apps[1]), base_url="http://worker-b"),
        ]
        async with clients[0], clients[1]:
            return await stage_weights(
                clients,
                delta_dir,
                version="1",
                mode="delta",
                base_version="base",
                upload=True,
                upload_method="chunked",
                chunk_size_bytes=4,
            )

    results = asyncio.run(run())

    assert sorted(result.endpoint for result in results) == ["http://worker-a", "http://worker-b"]
    assert all(result.ok for result in results)
    for app in apps:
        staged = app.state.staged_versions["1"]
        assert staged["owned"] is True
        assert staged["path"].read_bytes() == b"multi-endpoint-delta"


@pytest.mark.parametrize(
    ("upload_method", "filename"),
    [("chunked", "delta.safetensors"), ("streaming", "delta.stream")],
)
def test_client_stage_upload_uses_multiple_streams(tmp_path, upload_method, filename) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    content = b"four-concurrent-upload-chunks"
    (delta_dir / filename).write_bytes(content)
    app = make_app(tmp_path, stage_chunk_delay_s=0.01)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await stage_weights(
                [client],
                delta_dir,
                version="1",
                mode="delta",
                base_version="base",
                upload=True,
                upload_method=upload_method,
                chunk_size_bytes=4,
                num_streams=4,
            )

    asyncio.run(run())

    assert app.state.max_active_stage_chunk_requests == 4
    assert app.state.staged_versions["1"]["path"].read_bytes() == content


@pytest.mark.parametrize(
    ("upload_method", "filename", "retry_paths"),
    [
        ("chunked", "delta.safetensors", {"/stage_chunk", "/stage_finalize"}),
        (
            "streaming",
            "delta.stream",
            {"/stage_stream_init", "/stage_stream_chunk", "/stage_stream_finalize"},
        ),
    ],
)
def test_client_stage_upload_retries_lost_success_responses(tmp_path, upload_method, filename, retry_paths) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    content = b"retry-safe-delta"
    (delta_dir / filename).write_bytes(content)
    app = make_app(tmp_path)
    app_transport = httpx.ASGITransport(app=app)
    attempts: dict[str, int] = {}

    async def drop_first_success(request: httpx.Request) -> httpx.Response:
        response = await app_transport.handle_async_request(request)
        await response.aread()
        path = request.url.path
        attempts[path] = attempts.get(path, 0) + 1
        if path in retry_paths and attempts[path] == 1:
            return httpx.Response(503, request=request)
        return response

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(drop_first_success), base_url="http://test"
        ) as client:
            await stage_weights(
                [client],
                delta_dir,
                version="1",
                mode="delta",
                base_version="base",
                upload=True,
                upload_method=upload_method,
                chunk_size_bytes=len(content),
                chunk_retries=1,
                retry_base_delay_s=0,
            )

    asyncio.run(run())

    assert {path: attempts[path] for path in retry_paths} == dict.fromkeys(retry_paths, 2)
    assert app.state.stage_uploads == {}
    assert app.state.staged_versions["1"]["path"].read_bytes() == content


@pytest.mark.parametrize(
    ("upload_method", "filename", "finalize_path"),
    [
        ("chunked", "delta.safetensors", "/stage_finalize"),
        ("streaming", "delta.stream", "/stage_stream_finalize"),
    ],
)
def test_client_retries_complete_stage_after_lost_finalize_response(
    tmp_path, upload_method, filename, finalize_path
) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    content = b"stage-retry-safe-delta"
    (delta_dir / filename).write_bytes(content)
    app = make_app(tmp_path)
    app_transport = httpx.ASGITransport(app=app)
    finalize_attempts = 0

    async def drop_first_finalize_success(request: httpx.Request) -> httpx.Response:
        nonlocal finalize_attempts
        response = await app_transport.handle_async_request(request)
        await response.aread()
        if request.url.path == finalize_path:
            finalize_attempts += 1
            if finalize_attempts == 1:
                return httpx.Response(503, request=request)
        return response

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(drop_first_finalize_success), base_url="http://test"
        ) as client:
            await stage_weights(
                [client],
                delta_dir,
                version="1",
                mode="delta",
                base_version="base",
                upload=True,
                upload_method=upload_method,
                chunk_size_bytes=len(content),
                chunk_retries=0,
                retry_base_delay_s=0,
                stage_retries=1,
                stage_retry_base_delay_s=0,
            )

    asyncio.run(run())

    assert finalize_attempts == 2
    assert app.state.stage_uploads == {}
    assert app.state.staged_versions["1"]["path"].read_bytes() == content


@pytest.mark.parametrize(
    ("upload_method", "filename", "chunk_path"),
    [
        ("chunked", "delta.safetensors", "/stage_chunk"),
        ("streaming", "delta.stream", "/stage_stream_chunk"),
    ],
)
def test_complete_stage_retry_waits_for_failed_upload_workers(tmp_path, upload_method, filename, chunk_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    content = b"abcdefgh"
    (delta_dir / filename).write_bytes(content)
    app = make_app(tmp_path)
    app_transport = httpx.ASGITransport(app=app)
    chunk_attempts = 0
    cancelled = False

    async def run() -> None:
        slow_chunk_started = asyncio.Event()

        async def fail_first_attempt(request: httpx.Request) -> httpx.Response:
            nonlocal chunk_attempts, cancelled
            if request.url.path == chunk_path:
                chunk_attempts += 1
                if chunk_attempts == 1:
                    await slow_chunk_started.wait()
                    return httpx.Response(503, request=request)
                if chunk_attempts == 2:
                    slow_chunk_started.set()
                    try:
                        await asyncio.Future()
                    finally:
                        cancelled = True
                assert cancelled, "previous upload workers must finish before retrying the stage"
            return await app_transport.handle_async_request(request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(fail_first_attempt), base_url="http://test"
        ) as client:
            await asyncio.wait_for(
                stage_weights(
                    [client],
                    delta_dir,
                    version="1",
                    mode="delta",
                    base_version="base",
                    upload=True,
                    upload_method=upload_method,
                    chunk_size_bytes=4,
                    num_streams=2,
                    chunk_retries=0,
                    stage_retries=1,
                    stage_retry_base_delay_s=0,
                ),
                timeout=5,
            )

    asyncio.run(run())

    assert cancelled
    assert chunk_attempts == 4
    assert app.state.stage_uploads == {}
    assert app.state.staged_versions["1"]["path"].read_bytes() == content


def test_relay_stage_multipart_uploads_delta_to_peer(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"relay-delta")
    peer = make_app(tmp_path / "peer")
    seed = make_app(
        tmp_path / "seed",
        relay_peers=["http://peer"],
        relay_transports={"http://peer": httpx.ASGITransport(app=peer)},
    )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=seed), base_url="http://seed") as client:
            return await stage_weights(
                [client],
                delta_dir,
                version="1",
                mode="delta",
                base_version="base",
                upload=True,
                upload_method="multipart",
            )

    results = asyncio.run(run())

    assert all(result.ok for result in results)
    assert seed.state.staged_versions["1"]["path"].read_bytes() == b"relay-delta"
    assert peer.state.staged_versions["1"]["path"].read_bytes() == b"relay-delta"


def test_relay_stage_chunk_uploads_delta_to_peer(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"relay-chunked-delta")
    peer = make_app(tmp_path / "peer")
    seed = make_app(
        tmp_path / "seed",
        relay_peers=["http://peer"],
        relay_transports={"http://peer": httpx.ASGITransport(app=peer)},
    )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=seed), base_url="http://seed") as client:
            return await stage_weights(
                [client],
                delta_dir,
                version="1",
                mode="delta",
                base_version="base",
                upload=True,
                upload_method="chunked",
                chunk_size_bytes=5,
            )

    results = asyncio.run(run())

    assert all(result.ok for result in results)
    assert seed.state.stage_uploads == {}
    assert peer.state.stage_uploads == {}
    assert seed.state.staged_versions["1"]["path"].read_bytes() == b"relay-chunked-delta"
    assert peer.state.staged_versions["1"]["path"].read_bytes() == b"relay-chunked-delta"


def test_relay_false_prevents_recursive_stage_fanout(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"one-hop-only")
    leaf = make_app(tmp_path / "leaf")
    peer = make_app(
        tmp_path / "peer",
        relay_peers=["http://leaf"],
        relay_transports={"http://leaf": httpx.ASGITransport(app=leaf)},
    )
    seed = make_app(
        tmp_path / "seed",
        relay_peers=["http://peer"],
        relay_transports={"http://peer": httpx.ASGITransport(app=peer)},
    )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=seed), base_url="http://seed") as client:
            return await stage_weights(
                [client],
                delta_dir,
                version="1",
                mode="delta",
                base_version="base",
                upload=True,
                upload_method="multipart",
            )

    results = asyncio.run(run())

    assert all(result.ok for result in results)
    assert seed.state.staged_versions["1"]["path"].read_bytes() == b"one-hop-only"
    assert peer.state.staged_versions["1"]["path"].read_bytes() == b"one-hop-only"
    assert leaf.state.staged_versions == {}


def test_relay_fail_on_peer_error_returns_502(tmp_path) -> None:
    seed = make_app(
        tmp_path / "seed",
        relay_peers=["http://peer"],
        relay_transports={"http://peer": httpx.MockTransport(lambda _request: httpx.Response(500))},
        relay_fail_on_peer_error=True,
    )

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=seed), base_url="http://seed") as client:
            return await client.post(
                "/stage",
                data={"version": "1", "mode": "delta", "base_version": "base"},
                files={"file": ("delta.safetensors", b"strict-relay", "application/octet-stream")},
            )

    response = asyncio.run(run())

    assert response.status_code == 502
    assert response.json()["failed_peers"] == ["http://peer"]
    assert seed.state.staged_versions["1"]["path"].read_bytes() == b"strict-relay"


def test_client_stage_streams_growing_delta_file(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.stream"
    stable_file = delta_dir / "STABLE"
    app = make_app(tmp_path)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            stage_task = asyncio.create_task(
                stage_weights(
                    [client],
                    delta_dir,
                    version="1",
                    mode="delta",
                    base_version="base",
                    upload=True,
                    upload_method="streaming",
                    chunk_size_bytes=4,
                    done_path=stable_file,
                    poll_interval_s=0.01,
                )
            )

            await asyncio.sleep(0.02)
            delta_file.write_bytes(b"stream-")
            await asyncio.sleep(0.02)
            with delta_file.open("ab") as f:
                f.write(b"delta-upload")
            stable_file.touch()
            return await stage_task

    results = asyncio.run(run())

    staged = app.state.staged_versions["1"]
    assert [result.endpoint for result in results] == ["http://test"]
    assert all(result.ok for result in results)
    assert staged["owned"] is True
    assert staged["mode"] == "delta"
    assert staged["path"].read_bytes() == b"stream-delta-upload"


def test_relay_reload_weights_to_peer(tmp_path) -> None:
    peer = make_app(tmp_path / "peer")
    seed = make_app(
        tmp_path / "seed",
        relay_peers=["http://peer"],
        relay_transports={"http://peer": httpx.ASGITransport(app=peer)},
    )
    for app, name in ((seed, "seed"), (peer, "peer")):
        staged_file = app.state.staging_dir / "1_delta_delta.safetensors"
        staged_file.write_bytes(name.encode())
        app.state.staged_versions = {"1": {"path": staged_file, "mode": "delta", "owned": True}}
        app.state.active_version = "1"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=seed), base_url="http://seed") as client:
            response = await client.post("/reload_weights")
            assert response.status_code == 200

    asyncio.run(run())

    for app in (seed, peer):
        assert app.state.active_version == "base"
        assert app.state.staged_versions == {}
        assert app.state.engine_client.calls == [("reload_weights", ())]


def test_multi_relay_streaming_stage_and_commit_to_region_peers(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"multi-region-relay-delta")
    peer_a = make_app(tmp_path / "peer-a")
    peer_b = make_app(tmp_path / "peer-b")
    seed_a = make_app(
        tmp_path / "seed-a",
        relay_peers=["http://peer-a"],
        relay_transports={"http://peer-a": httpx.ASGITransport(app=peer_a)},
    )
    seed_b = make_app(
        tmp_path / "seed-b",
        relay_peers=["http://peer-b"],
        relay_transports={"http://peer-b": httpx.ASGITransport(app=peer_b)},
    )

    async def run() -> None:
        clients = [
            httpx.AsyncClient(transport=httpx.ASGITransport(app=seed_a), base_url="http://seed-a"),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=seed_b), base_url="http://seed-b"),
        ]
        async with clients[0], clients[1]:
            stage_results = await stage_weights(
                clients,
                delta_dir,
                version="1",
                mode="delta",
                base_version="base",
                upload=True,
                upload_method="streaming",
                chunk_size_bytes=6,
            )
            commit_results = await commit_weights(clients, version="1", mode="delta")
            return stage_results, commit_results

    stage_results, commit_results = asyncio.run(run())

    assert all(result.ok for result in stage_results)
    assert all(result.ok for result in commit_results)
    for app in (seed_a, seed_b, peer_a, peer_b):
        assert app.state.active_version == "1"
        assert app.state.staged_versions["1"]["path"].read_bytes() == b"multi-region-relay-delta"
        assert app.state.engine_client.calls == [
            ("update_weights_from_delta_path", (app.state.staged_versions["1"]["path"].as_posix(),))
        ]


def test_stage_stream_finalize_accepts_complete_upload(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"streaming-delta"
    digest = hashlib.sha256(content).hexdigest()

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            init_response = await client.post(
                "/stage_stream_init",
                json={"version": "1", "mode": "delta", "base_version": "base", "filename": "delta.safetensors"},
            )
            assert init_response.status_code == 200
            upload_id = init_response.json()["upload_id"]

            first_response = await client.post(
                "/stage_stream_chunk",
                params={"upload_id": upload_id, "offset": "0"},
                content=content[:6],
            )
            assert first_response.status_code == 200
            second_response = await client.post(
                "/stage_stream_chunk",
                params={"upload_id": upload_id, "offset": "6"},
                content=content[6:],
            )
            assert second_response.status_code == 200

            finalize_response = await client.post(
                "/stage_stream_finalize",
                data={"upload_id": upload_id, "final_size": str(len(content)), "sha256": digest},
            )
            assert finalize_response.status_code == 200

    asyncio.run(run())

    staged = app.state.staged_versions["1"]
    assert staged["owned"] is True
    assert staged["path"].read_bytes() == content
    assert app.state.stage_uploads == {}


def test_stage_stream_finalize_rejects_missing_range(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"incomplete"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            init_response = await client.post(
                "/stage_stream_init",
                json={"version": "1", "mode": "delta", "base_version": "base", "filename": "delta.safetensors"},
            )
            assert init_response.status_code == 200
            upload_id = init_response.json()["upload_id"]

            chunk_response = await client.post(
                "/stage_stream_chunk",
                params={"upload_id": upload_id, "offset": "0"},
                content=content[:5],
            )
            assert chunk_response.status_code == 200
            finalize_response = await client.post(
                "/stage_stream_finalize",
                data={"upload_id": upload_id, "final_size": str(len(content))},
            )
            assert finalize_response.status_code == 409

    asyncio.run(run())

    assert app.state.staged_versions == {}
    assert app.state.stage_uploads


def test_stage_stream_finalize_rejects_hash_mismatch(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"corrupted"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            init_response = await client.post(
                "/stage_stream_init",
                json={"version": "1", "mode": "delta", "base_version": "base", "filename": "delta.safetensors"},
            )
            assert init_response.status_code == 200
            upload_id = init_response.json()["upload_id"]

            chunk_response = await client.post(
                "/stage_stream_chunk",
                params={"upload_id": upload_id, "offset": "0"},
                content=content,
            )
            assert chunk_response.status_code == 200
            finalize_response = await client.post(
                "/stage_stream_finalize",
                data={"upload_id": upload_id, "final_size": str(len(content)), "sha256": "0" * 64},
            )
            assert finalize_response.status_code == 409

    asyncio.run(run())

    assert app.state.staged_versions == {}
    assert app.state.stage_uploads == {}


def test_stage_chunk_finalize_accepts_out_of_order_chunks(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"first-second"
    digest = hashlib.sha256(content).hexdigest()

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            common_data = {
                "version": "1",
                "mode": "delta",
                "base_version": "base",
                "filename": "delta.safetensors",
                "total_size": str(len(content)),
                "sha256": digest,
            }
            second_response = await client.post(
                "/stage_chunk",
                data={**common_data, "offset": "6"},
                files={"file": ("delta.safetensors", content[6:], "application/octet-stream")},
            )
            assert second_response.status_code == 200
            first_response = await client.post(
                "/stage_chunk",
                data={**common_data, "offset": "0"},
                files={"file": ("delta.safetensors", content[:6], "application/octet-stream")},
            )
            assert first_response.status_code == 200
            finalize_response = await client.post("/stage_finalize", data=common_data)
            assert finalize_response.status_code == 200

    asyncio.run(run())

    staged_path = app.state.staged_versions["1"]["path"]
    assert staged_path.read_bytes() == content


def test_stage_chunk_finalize_rejects_missing_chunk(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"incomplete"
    digest = hashlib.sha256(content).hexdigest()

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            common_data = {
                "version": "1",
                "mode": "delta",
                "base_version": "base",
                "filename": "delta.safetensors",
                "total_size": str(len(content)),
                "sha256": digest,
            }
            chunk_response = await client.post(
                "/stage_chunk",
                data={**common_data, "offset": "0"},
                files={"file": ("delta.safetensors", content[:5], "application/octet-stream")},
            )
            assert chunk_response.status_code == 200
            finalize_response = await client.post("/stage_finalize", data=common_data)
            assert finalize_response.status_code == 409

    asyncio.run(run())

    assert app.state.staged_versions == {}
    assert app.state.stage_uploads


def test_stage_chunk_finalize_rejects_hash_mismatch(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"corrupted"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            common_data = {
                "version": "1",
                "mode": "delta",
                "base_version": "base",
                "filename": "delta.safetensors",
                "total_size": str(len(content)),
                "sha256": "0" * 64,
            }
            chunk_response = await client.post(
                "/stage_chunk",
                data={**common_data, "offset": "0"},
                files={"file": ("delta.safetensors", content, "application/octet-stream")},
            )
            assert chunk_response.status_code == 200
            finalize_response = await client.post("/stage_finalize", data=common_data)
            assert finalize_response.status_code == 409

    asyncio.run(run())

    assert app.state.staged_versions == {}
    assert app.state.stage_uploads == {}


def test_client_stage_commit_reload_helpers(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/stage":
            form = parse_qs(request.content.decode())
            assert form["version"] == ["1"]
            assert form["mode"] == ["delta"]
            assert form["path"] == [delta_dir.as_posix()]
        if request.url.path == "/commit":
            form = parse_qs(request.content.decode())
            assert form["version"] == ["1"]
            assert form["mode"] == ["delta"]
        return httpx.Response(200, json={"status": "ok"})

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await stage_weights([client], delta_dir, version="1", mode="delta")
            await commit_weights([client], version="1", mode="delta")
            await reload_weights([client])

    asyncio.run(run())

    assert [request.url.path for request in requests] == [
        "/stage",
        "/pause",
        "/commit",
        "/resume",
        "/pause",
        "/reload_weights",
        "/resume",
    ]
