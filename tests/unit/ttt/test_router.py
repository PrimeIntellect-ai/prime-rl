from __future__ import annotations

from typing import Any

import httpx
from fastapi.testclient import TestClient

from prime_rl.ttt.router import _shard_index, create_app


class FakeAsyncClient:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url: str, json: dict[str, Any]):
        self.calls.append(("POST", url, json))
        return httpx.Response(200, json={"ok": True, "url": url})

    async def get(self, url: str):
        self.calls.append(("GET", url, None))
        return httpx.Response(200, json={"status": "ok", "url": url})


def test_router_routes_session_endpoints_by_stable_session_hash(monkeypatch):
    FakeAsyncClient.calls = []
    monkeypatch.setattr("prime_rl.ttt.router.httpx.AsyncClient", FakeAsyncClient)
    learners = ["http://learner-0", "http://learner-1", "http://learner-2"]
    session_id = "trajectory-123"
    expected_idx = _shard_index(session_id, len(learners))

    client = TestClient(create_app(learners, request_timeout_s=1.0))
    response = client.post("/prepare_turn", json={"session_id": session_id, "new_token_ids": [1, 2]})

    assert response.status_code == 200
    assert FakeAsyncClient.calls == [
        ("POST", f"{learners[expected_idx]}/prepare_turn", {"session_id": session_id, "new_token_ids": [1, 2]})
    ]


def test_router_broadcasts_weight_updates_to_all_shards(monkeypatch):
    FakeAsyncClient.calls = []
    monkeypatch.setattr("prime_rl.ttt.router.httpx.AsyncClient", FakeAsyncClient)
    learners = ["http://learner-0", "http://learner-1"]

    client = TestClient(create_app(learners, request_timeout_s=1.0))
    response = client.post("/update_base_weights", json={"weight_dir": "/tmp/weights", "step": 3})

    assert response.status_code == 200
    assert FakeAsyncClient.calls == [
        ("POST", "http://learner-0/update_base_weights", {"weight_dir": "/tmp/weights", "step": 3}),
        ("POST", "http://learner-1/update_base_weights", {"weight_dir": "/tmp/weights", "step": 3}),
    ]


def test_router_health_checks_all_shards(monkeypatch):
    FakeAsyncClient.calls = []
    monkeypatch.setattr("prime_rl.ttt.router.httpx.AsyncClient", FakeAsyncClient)
    learners = ["http://learner-0", "http://learner-1"]

    client = TestClient(create_app(learners, request_timeout_s=1.0))
    response = client.get("/health")

    assert response.status_code == 200
    assert FakeAsyncClient.calls == [
        ("GET", "http://learner-0/health", None),
        ("GET", "http://learner-1/health", None),
    ]
    assert response.json()["num_shards"] == 2
