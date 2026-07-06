"""The TTT service HTTP surface, against a fake trainer (no torch, no GPU): update/release
routing, adapter (re)loading on the inference servers, error mapping, health."""

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from prime_rl.configs.ttt import TTTServiceConfig  # noqa: E402
from prime_rl.ttt.server import build_app  # noqa: E402

pytestmark = pytest.mark.asyncio


class FakeTrainer:
    def __init__(self):
        self.adapters: dict[str, object] = {}
        self.updates: list[tuple] = []
        self.released: list[str] = []
        self.fail_with: Exception | None = None

    def update(self, rollout_id, adapter_name, token_ids, loss_mask, seq_no, qa_pairs=None, train_rollout=True):
        if self.fail_with is not None:
            raise self.fail_with
        self.updates.append((rollout_id, adapter_name, token_ids, loss_mask, seq_no, qa_pairs, train_rollout))
        self.adapters[rollout_id] = type("S", (), {"version": seq_no})()
        return {
            "version": seq_no,
            "loss": 0.5,
            "ckpt_path": f"/ckpts/{rollout_id}/v{seq_no}",
            "num_loss_tokens": sum(loss_mask),
        }

    def release(self, rollout_id):
        self.released.append(rollout_id)
        return self.adapters.pop(rollout_id, None)


class FakeEngine:
    """Captures the vLLM admin calls the service makes."""

    def __init__(self):
        self.loads: list[dict] = []
        self.unloads: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        if request.url.path == "/load_lora_adapter":
            self.loads.append(payload)
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/unload_lora_adapter":
            self.unloads.append(payload)
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)


@pytest.fixture
def service():
    """(client, fake_trainer, fake_engine) — app served in-process over ASGITransport,
    engine admin calls intercepted by a MockTransport."""
    from contextlib import asynccontextmanager

    config = TTTServiceConfig(inference_admin_urls=["http://engine"])
    trainer = FakeTrainer()
    engine = FakeEngine()
    app = build_app(config, trainer=trainer)

    @asynccontextmanager
    async def _ctx():
        async with app.router.lifespan_context(app):
            # Swap the outbound engine client for the mock AFTER lifespan built the real one.
            await app.state.http.aclose()
            app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(engine.handler))
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://ttt") as client:
                yield client, trainer, engine

    return _ctx


UPDATE = {
    "rollout_id": "r1",
    "adapter_name": "ttt-r1",
    "token_ids": [1, 2, 3],
    "loss_mask": [True, True, True],
    "seq_no": 1,
}


async def test_update_trains_and_loads_adapter(service):
    async with service() as (client, trainer, engine):
        response = await client.post("/update", json=UPDATE)
        assert response.status_code == 200
        body = response.json()
        assert body["version"] == 1
        assert body["loss"] == 0.5
        assert trainer.updates[0][0] == "r1"
        (load,) = engine.loads
        assert load == {"lora_name": "ttt-r1", "lora_path": "/ckpts/r1/v1"}


async def test_trainer_error_maps_to_409(service):
    async with service() as (client, trainer, engine):
        trainer.fail_with = ValueError("out-of-order update")
        response = await client.post("/update", json=UPDATE)
        assert response.status_code == 409
        assert "out-of-order" in response.json()["detail"]
        assert engine.loads == []


async def test_release_unloads_updated_adapter(service):
    async with service() as (client, trainer, engine):
        await client.post("/update", json=UPDATE)
        response = await client.post("/release", json={"rollout_id": "r1", "adapter_name": "ttt-r1"})
        assert response.json() == {"released": True}
        assert trainer.released == ["r1"]
        (unload,) = engine.unloads
        assert unload == {"lora_name": "ttt-r1"}

        # Releasing an unknown rollout is a no-op (idempotent), no engine call.
        response = await client.post("/release", json={"rollout_id": "nope", "adapter_name": "ttt-nope"})
        assert response.json() == {"released": False}
        assert len(engine.unloads) == 1


async def test_health(service):
    async with service() as (client, trainer, engine):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "adapters": 0}


async def test_update_forwards_qa_fields(service):
    async with service() as (client, trainer, engine):
        response = await client.post(
            "/update",
            json={
                **UPDATE,
                "qa_pairs": [{"question": "q1", "answer": "a1"}],
                "train_rollout": False,
            },
        )
        assert response.status_code == 200
        (*_, qa_pairs, train_rollout) = trainer.updates[0]
        assert qa_pairs == [{"question": "q1", "answer": "a1"}]
        assert train_rollout is False
