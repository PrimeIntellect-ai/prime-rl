import asyncio
import hashlib
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from prime_rl.inference.vllm.delta_sync import router as delta_sync_router
from prime_rl.inference.vllm.server import router as server_router
from prime_rl.orchestrator.delta_sync import (
    DeltaEndpointPool,
    commit_weights,
    reload_weights,
    stage_weights,
)


class FakeEngineClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    async def collective_rpc(self, method: str, args: tuple = ()) -> None:
        self.calls.append((method, args))
        await asyncio.sleep(0)

    async def pause_generation(self, mode: str = "keep", clear_cache: bool = False) -> None:
        pass

    async def resume_generation(self) -> None:
        pass

    async def check_health(self) -> None:
        pass


def make_app(
    tmp_path,
    *,
    relay_peers: list[str] | None = None,
    relay_transports: dict[str, httpx.AsyncBaseTransport] | None = None,
    relay_fail_on_peer_error: bool = False,
):
    app = FastAPI()
    app.include_router(delta_sync_router)

    @app.post("/pause")
    async def pause():
        return {"status": "paused"}

    @app.post("/resume")
    async def resume():
        if getattr(app.state, "weights_dirty", False):
            return JSONResponse({"error": "weights require reload after a failed update"}, status_code=409)
        return {"status": "resumed"}

    app.state.engine_client = FakeEngineClient()
    app.state.staging_dir = tmp_path / "staging"
    app.state.staging_dir.mkdir(parents=True)
    app.state.staged_versions = {}
    app.state.stage_uploads = {}
    app.state.active_version = "base"
    app.state.api_server_count = 1
    app.state.relay_enabled = bool(relay_peers)
    app.state.relay_peers = relay_peers or []
    app.state.relay_transports = relay_transports or {}
    app.state.relay_fail_on_peer_error = relay_fail_on_peer_error
    app.state.relay_stage_timeout_s = 30.0
    app.state.relay_commit_timeout_s = 30.0
    app.state.relay_reload_timeout_s = 30.0
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
            await pool._recover_eligible()
            assert app.state.staged_versions == {}
            await pool.commit("1")
            assert app.state.active_version == "1"
            assert [method for method, _ in app.state.engine_client.calls] == [
                "reload_weights",
                "update_weights_from_delta_path",
            ]

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
