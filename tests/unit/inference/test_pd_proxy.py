import asyncio
import json

import httpx
import pytest

from prime_rl.inference.vllm.pd_proxy import (
    PDProxyConfig,
    build_prefill_payload,
    build_request_id,
    create_app,
)


def test_build_prefill_payload_forces_single_token():
    payload = {"model": "dummy", "max_tokens": 32, "max_completion_tokens": 64}
    prefill_payload = build_prefill_payload(payload)
    assert prefill_payload["max_tokens"] == 1
    assert prefill_payload["max_completion_tokens"] == 1
    assert payload["max_tokens"] == 32
    assert payload["max_completion_tokens"] == 64


def test_build_request_id_contains_prefill_and_decode_addresses():
    request_id = build_request_id("127.0.0.1:14579", "127.0.0.1:14580")
    assert "___prefill_addr_127.0.0.1:14579___decode_addr_127.0.0.1:14580_" in request_id


def test_proxy_forwards_prefill_then_decode():
    async def run_test():
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode()) if request.content else None
            calls.append(
                {
                    "host": request.url.host,
                    "path": request.url.path,
                    "body": body,
                    "request_id": request.headers.get("X-Request-Id"),
                }
            )
            if request.url.host == "prefill":
                return httpx.Response(200, json={"id": "prefill"})
            if request.url.host == "decode":
                return httpx.Response(200, json={"id": "decode"})
            raise AssertionError(f"Unexpected host: {request.url.host}")

        upstream_transport = httpx.MockTransport(handler)
        app = create_app(
            PDProxyConfig(
                host="127.0.0.1",
                port=9000,
                prefill_urls=("http://prefill:8100",),
                decode_urls=("http://decode:8200",),
                prefill_kv_addrs=("127.0.0.1:14579",),
                decode_kv_addrs=("127.0.0.1:14580",),
                timeout=10.0,
            ),
            transport=upstream_transport,
        )

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                response = await client.post(
                    "/v1/completions",
                    json={"model": "dummy", "prompt": "hello", "max_tokens": 12},
                )

        assert response.status_code == 200
        assert response.json()["id"] == "decode"
        assert len(calls) == 2
        assert calls[0]["host"] == "prefill"
        assert calls[1]["host"] == "decode"
        assert calls[0]["body"]["max_tokens"] == 1
        assert calls[1]["body"]["max_tokens"] == 12
        assert calls[0]["request_id"] == calls[1]["request_id"]
        assert "___prefill_addr_127.0.0.1:14579___decode_addr_127.0.0.1:14580_" in calls[0]["request_id"]

    asyncio.run(run_test())


def test_proxy_forwards_chat_completions_tokens_routes():
    async def run_test():
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode()) if request.content else None
            calls.append(
                {
                    "host": request.url.host,
                    "path": request.url.path,
                    "body": body,
                }
            )
            return httpx.Response(200, json={"id": request.url.host})

        upstream_transport = httpx.MockTransport(handler)
        app = create_app(
            PDProxyConfig(
                host="127.0.0.1",
                port=9000,
                prefill_urls=("http://prefill:8100",),
                decode_urls=("http://decode:8200",),
                prefill_kv_addrs=("127.0.0.1:14579",),
                decode_kv_addrs=("127.0.0.1:14580",),
                timeout=10.0,
            ),
            transport=upstream_transport,
        )

        payload = {
            "model": "dummy",
            "messages": [{"role": "user", "content": "hello"}],
            "tokens": [1, 2, 3],
            "max_tokens": 12,
        }
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                response_v1 = await client.post("/v1/chat/completions/tokens", json=payload)
                response_root = await client.post("/chat/completions/tokens", json=payload)

        assert response_v1.status_code == 200
        assert response_root.status_code == 200
        assert response_v1.json()["id"] == "decode"
        assert response_root.json()["id"] == "decode"
        assert len(calls) == 4
        assert [call["host"] for call in calls] == ["prefill", "decode", "prefill", "decode"]
        assert all(call["path"] == "/v1/chat/completions/tokens" for call in calls)
        assert calls[0]["body"]["max_tokens"] == 1
        assert calls[1]["body"]["max_tokens"] == 12
        assert calls[2]["body"]["max_tokens"] == 1
        assert calls[3]["body"]["max_tokens"] == 12

    asyncio.run(run_test())


def test_proxy_round_robins_across_prefill_and_decode_backends():
    async def run_test():
        calls: list[dict] = []
        prefill_addr_map = {
            "prefill0": "127.0.0.1:14579",
            "prefill1": "127.0.0.1:14580",
        }
        decode_addr_map = {
            "decode0": "127.0.0.1:14679",
            "decode1": "127.0.0.1:14680",
            "decode2": "127.0.0.1:14681",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode()) if request.content else None
            calls.append(
                {
                    "host": request.url.host,
                    "path": request.url.path,
                    "body": body,
                    "request_id": request.headers.get("X-Request-Id"),
                }
            )
            return httpx.Response(200, json={"id": request.url.host})

        upstream_transport = httpx.MockTransport(handler)
        app = create_app(
            PDProxyConfig(
                host="127.0.0.1",
                port=9000,
                prefill_urls=("http://prefill0:8100", "http://prefill1:8101"),
                decode_urls=("http://decode0:8200", "http://decode1:8201", "http://decode2:8202"),
                prefill_kv_addrs=("127.0.0.1:14579", "127.0.0.1:14580"),
                decode_kv_addrs=("127.0.0.1:14679", "127.0.0.1:14680", "127.0.0.1:14681"),
                timeout=10.0,
            ),
            transport=upstream_transport,
        )

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                for _ in range(5):
                    response = await client.post(
                        "/v1/completions",
                        json={"model": "dummy", "prompt": "hello", "max_tokens": 12},
                    )
                    assert response.status_code == 200

        prefill_calls = calls[0::2]
        decode_calls = calls[1::2]
        assert [call["host"] for call in prefill_calls] == ["prefill0", "prefill1", "prefill0", "prefill1", "prefill0"]
        assert [call["host"] for call in decode_calls] == ["decode0", "decode1", "decode2", "decode0", "decode1"]

        for prefill_call, decode_call in zip(prefill_calls, decode_calls, strict=True):
            assert prefill_call["request_id"] == decode_call["request_id"]
            assert (
                f"___prefill_addr_{prefill_addr_map[prefill_call['host']]}"
                f"___decode_addr_{decode_addr_map[decode_call['host']]}_"
            ) in prefill_call["request_id"]

    asyncio.run(run_test())


def test_proxy_fanouts_update_weights_to_all_pd_backends():
    async def run_test():
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.url.host or "", request.url.path))
            return httpx.Response(200, json={"status": "ok"})

        upstream_transport = httpx.MockTransport(handler)
        app = create_app(
            PDProxyConfig(
                host="127.0.0.1",
                port=9000,
                prefill_urls=("http://prefill0:8100", "http://prefill1:8101"),
                decode_urls=("http://decode0:8200", "http://decode1:8201"),
                prefill_kv_addrs=("127.0.0.1:14579", "127.0.0.1:14580"),
                decode_kv_addrs=("127.0.0.1:14679", "127.0.0.1:14680"),
                timeout=10.0,
            ),
            transport=upstream_transport,
        )

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                response = await client.post("/update_weights", json={"weight_dir": "/tmp/step_1"})

        assert response.status_code == 200
        assert ("prefill0", "/update_weights") in calls
        assert ("prefill1", "/update_weights") in calls
        assert ("decode0", "/update_weights") in calls
        assert ("decode1", "/update_weights") in calls
        assert len([path for _, path in calls if path == "/update_weights"]) == 4

    asyncio.run(run_test())


def test_proxy_fanouts_init_broadcaster_with_unique_server_ranks():
    async def run_test():
        calls: list[tuple[str, str, dict | None]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode()) if request.content else None
            calls.append((request.url.host or "", request.url.path, body))
            return httpx.Response(200, json={"status": "ok"})

        upstream_transport = httpx.MockTransport(handler)
        app = create_app(
            PDProxyConfig(
                host="127.0.0.1",
                port=9000,
                prefill_urls=("http://prefill0:8100", "http://prefill1:8101"),
                decode_urls=("http://decode0:8200", "http://decode1:8201"),
                prefill_kv_addrs=("127.0.0.1:14579", "127.0.0.1:14580"),
                decode_kv_addrs=("127.0.0.1:14679", "127.0.0.1:14680"),
                timeout=10.0,
            ),
            transport=upstream_transport,
        )

        payload = {"host": "127.0.0.1", "port": 29501, "timeout": 30, "server_rank": 99, "num_inference_server": 999}
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                response = await client.post("/init_broadcaster", json=payload)

        assert response.status_code == 200
        init_calls = [call for call in calls if call[1] == "/init_broadcaster"]
        assert len(init_calls) == 4
        assert {host for host, _, _ in init_calls} == {"prefill0", "prefill1", "decode0", "decode1"}

        per_rank_calls = sorted(init_calls, key=lambda call: call[2]["server_rank"])
        assert [call[2]["server_rank"] for call in per_rank_calls] == [0, 1, 2, 3]
        assert all(call[2]["num_inference_server"] == 4 for call in per_rank_calls)
        assert all(call[2]["host"] == "127.0.0.1" for call in per_rank_calls)
        assert all(call[2]["port"] == 29501 for call in per_rank_calls)
        assert all(call[2]["timeout"] == 30 for call in per_rank_calls)

    asyncio.run(run_test())


def test_proxy_requires_matching_url_and_kv_lengths():
    with pytest.raises(ValueError, match="must have the same length"):
        create_app(
            PDProxyConfig(
                host="127.0.0.1",
                port=9000,
                prefill_urls=("http://prefill0:8100",),
                decode_urls=("http://decode0:8200",),
                prefill_kv_addrs=("127.0.0.1:14579", "127.0.0.1:14580"),
                decode_kv_addrs=("127.0.0.1:14679",),
                timeout=10.0,
            )
        )


def test_proxy_retries_on_decode_failure_and_succeeds_with_alternate_backend():
    async def run_test():
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.url.host or "", request.url.path))
            if request.url.host == "prefill0":
                return httpx.Response(200, json={"id": "prefill0"})
            if request.url.host == "decode0":
                return httpx.Response(503, json={"error": "decode0 unavailable"})
            if request.url.host == "decode1":
                return httpx.Response(200, json={"id": "decode1"})
            raise AssertionError(f"Unexpected host: {request.url.host}")

        upstream_transport = httpx.MockTransport(handler)
        app = create_app(
            PDProxyConfig(
                host="127.0.0.1",
                port=9000,
                prefill_urls=("http://prefill0:8100",),
                decode_urls=("http://decode0:8200", "http://decode1:8201"),
                prefill_kv_addrs=("127.0.0.1:14579",),
                decode_kv_addrs=("127.0.0.1:14679", "127.0.0.1:14680"),
                timeout=10.0,
                max_generate_attempts=2,
                health_check_interval=3600.0,
            ),
            transport=upstream_transport,
        )

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                response = await client.post(
                    "/v1/completions",
                    json={"model": "dummy", "prompt": "hello", "max_tokens": 12},
                )

        assert response.status_code == 200
        assert response.json()["id"] == "decode1"
        assert [host for host, path in calls if path == "/v1/completions"] == [
            "prefill0",
            "decode0",
            "prefill0",
            "decode1",
        ]

    asyncio.run(run_test())


def test_proxy_marks_failed_backend_dead_and_skips_it():
    async def run_test():
        decode0_calls = 0
        decode1_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal decode0_calls, decode1_calls
            if request.url.host == "prefill0":
                return httpx.Response(200, json={"id": "prefill0"})
            if request.url.host == "decode0":
                decode0_calls += 1
                return httpx.Response(503, json={"error": "decode0 unavailable"})
            if request.url.host == "decode1":
                decode1_calls += 1
                return httpx.Response(200, json={"id": "decode1"})
            raise AssertionError(f"Unexpected host: {request.url.host}")

        upstream_transport = httpx.MockTransport(handler)
        app = create_app(
            PDProxyConfig(
                host="127.0.0.1",
                port=9000,
                prefill_urls=("http://prefill0:8100",),
                decode_urls=("http://decode0:8200", "http://decode1:8201"),
                prefill_kv_addrs=("127.0.0.1:14579",),
                decode_kv_addrs=("127.0.0.1:14679", "127.0.0.1:14680"),
                timeout=10.0,
                max_generate_attempts=3,
                health_check_failure_threshold=1,
                health_check_interval=3600.0,
            ),
            transport=upstream_transport,
        )

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                first = await client.post(
                    "/v1/completions",
                    json={"model": "dummy", "prompt": "hello", "max_tokens": 12},
                )
                second = await client.post(
                    "/v1/completions",
                    json={"model": "dummy", "prompt": "again", "max_tokens": 12},
                )

        assert first.status_code == 200
        assert second.status_code == 200
        assert decode0_calls == 1
        assert decode1_calls == 2

    asyncio.run(run_test())
