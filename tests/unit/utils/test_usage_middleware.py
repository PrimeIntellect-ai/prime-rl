import json

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from prime_rl.inference.vllm.usage_middleware import UsageSigningMiddleware
from prime_rl.utils.usage_signing import verify_usage

SECRET = "test-middleware-secret"
RUN_ID = "test-run-123"


async def mock_chat_completions(request: Request) -> JSONResponse:
    return JSONResponse({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
    })


async def mock_health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def make_app():
    app = Starlette(routes=[
        Route("/v1/chat/completions", mock_chat_completions, methods=["POST"]),
        Route("/health", mock_health, methods=["GET"]),
    ])
    app.add_middleware(UsageSigningMiddleware, secret=SECRET)
    return app


@pytest.fixture
def client():
    return TestClient(make_app())


def test_rejects_missing_run_id(client):
    resp = client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400
    assert "X-Run-Id" in resp.json()["error"]


def test_signs_usage_in_response(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"x-run-id": RUN_ID},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "usage_signature" in data

    sig = data["usage_signature"]
    assert verify_usage(SECRET, RUN_ID, 50, 30, sig) is True


def test_signature_invalid_for_wrong_tokens(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"x-run-id": RUN_ID},
    )
    data = resp.json()
    sig = data["usage_signature"]
    assert verify_usage(SECRET, RUN_ID, 999, 999, sig) is False


def test_non_chat_routes_unaffected(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "usage_signature" not in resp.json()
