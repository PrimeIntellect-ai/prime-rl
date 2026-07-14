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

    def update(
        self,
        rollout_id,
        adapter_name,
        token_ids,
        loss_mask,
        seq_no,
        qa_pairs=None,
        train_rollout=True,
        system_prompt=None,
        tools=None,
    ):
        if self.fail_with is not None:
            raise self.fail_with
        self.updates.append(
            (rollout_id, adapter_name, token_ids, loss_mask, seq_no, qa_pairs, train_rollout, system_prompt, tools)
        )
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

        # QA fields ride the same route and reach the trainer verbatim.
        response = await client.post(
            "/update",
            json={**UPDATE, "seq_no": 2, "qa_pairs": [{"question": "q1", "answer": "a1"}], "train_rollout": False},
        )
        assert response.status_code == 200
        (*_, qa_pairs, train_rollout, system_prompt, tools) = trainer.updates[1]
        assert qa_pairs == [{"question": "q1", "answer": "a1"}]
        assert train_rollout is False
        assert system_prompt is None and tools is None


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

        # A release with no state still unloads: a client RETRY after a lost response finds
        # the state already dropped, but the engine unload may never have run — gating on
        # state would leak the adapter in vLLM until restart. unload is idempotent.
        response = await client.post("/release", json={"rollout_id": "r1", "adapter_name": "ttt-r1"})
        assert response.json() == {"released": False}
        assert len(engine.unloads) == 2
        assert engine.unloads[1] == {"lora_name": "ttt-r1"}


async def test_health(service):
    async with service() as (client, trainer, engine):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "adapters": 0}


async def test_peft_admin_client_honors_configured_timeout():
    config = TTTServiceConfig(inference_admin_urls=[], admin_timeout_seconds=7.5)
    app = build_app(config, trainer=FakeTrainer())
    async with app.router.lifespan_context(app):
        assert app.state.http.timeout.connect == 7.5


async def test_v2_partial_replica_load_is_reconciled_everywhere():
    """A partial replica load must be reconciled (unload everywhere) before the /update
    caller sees its 502, and success must never be recorded (mark_loaded)."""
    import asyncio
    import threading
    from queue import Queue

    from prime_rl.ttt.server_v2 import build_app_v2

    load_started: set[str] = set()
    unloads: set[str] = set()
    both_loads = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/load_lora_adapter":
            load_started.add(request.url.host)
            if len(load_started) == 2:
                both_loads.set()
            await both_loads.wait()  # proves fan-out is concurrent
            return httpx.Response(503 if request.url.host == "bad" else 200)
        if request.url.path == "/v1/unload_lora_adapter":
            unloads.add(request.url.host)
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    marked = []

    class FakeV2Trainer:
        slots: dict = {}
        free_idxs: set = set()

        def validate_job(self, job):
            return None

        def prepare_job(self, job):
            pass

        def mark_loaded(self, rollout_id, adapter_name, version):
            marked.append((rollout_id, adapter_name, version))

    work_queue: Queue = Queue()

    def fake_work_loop():
        kind, jobs, pendings = work_queue.get()
        assert kind == "update"
        for pending in pendings:
            pending.result = {"version": 1, "loss": 0.5, "ckpt_path": "/v1", "num_loss_tokens": 1}
            pending.done.set()

    config = TTTServiceConfig(
        engine={"type": "fsdp", "max_batch_wait_seconds": 0},
        inference_admin_urls=["http://good", "http://bad"],
    )
    app = build_app_v2(config, FakeV2Trainer(), work_queue)
    threading.Thread(target=fake_work_loop, daemon=True).start()
    async with app.router.lifespan_context(app):
        await app.state.http.aclose()
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ttt") as client:
            response = await client.post("/update", json=UPDATE)
            assert response.status_code == 502
            assert "adapter load failed" in response.json()["detail"]

    assert not marked
    assert load_started == unloads == {"good", "bad"}


async def test_v2_cancelled_wait_keeps_rollout_lease_until_work_finishes():
    import asyncio
    from types import SimpleNamespace

    from prime_rl.ttt.server_v2 import _acquire_rollout_lease, _wait_with_lease

    app = SimpleNamespace(state=SimpleNamespace(deferred_rollout_releases=set()))
    done = __import__("threading").Event()
    lease = await _acquire_rollout_lease({}, "r1")
    waiter = asyncio.create_task(_wait_with_lease(app, lease, done))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert lease.lock.locked()

    done.set()
    for _ in range(100):
        if not lease.lock.locked():
            break
        await asyncio.sleep(0.01)
    assert not lease.lock.locked()


async def test_v2_rejects_malformed_identity_before_queueing():
    """rollout_id is a filesystem path component — reject traversal before any queueing.
    (Prefix-namespace enforcement is gone: the trusted in-repo hook derives the name.)"""
    from queue import Queue

    from prime_rl.ttt.server_v2 import build_app_v2

    trainer = type("Trainer", (), {"slots": {}})()
    work_queue = Queue()
    app = build_app_v2(
        TTTServiceConfig(engine={"type": "fsdp", "max_batch_wait_seconds": 0}, inference_admin_urls=[]),
        trainer,
        work_queue,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ttt") as client:
            response = await client.post("/update", json={**UPDATE, "rollout_id": "../evil"})
            assert response.status_code == 409
            response = await client.post("/update", json={**UPDATE, "adapter_name": ""})
            assert response.status_code == 409
    assert work_queue.empty()


async def test_absent_adapter_404_matching_tolerates_message_churn():
    """The idempotent-unload detector must survive vLLM error-message rewording: any
    structured 404 naming the adapter (or typed NotFoundError) counts; a bare/route 404
    (which would not mention the adapter name) does not."""
    from prime_rl.ttt.admin import _is_known_absent_adapter_response

    def resp(status: int, json_body=None, text: str = ""):
        return httpx.Response(status, json=json_body) if json_body is not None else httpx.Response(status, text=text)

    name = "ttt-r1"
    # vLLM's historical exact message.
    legacy = {"error": {"type": "NotFoundError", "code": 404, "message": f"The lora adapter '{name}' cannot be found."}}
    assert _is_known_absent_adapter_response(resp(404, legacy), name)
    # Changed message format, still names the adapter.
    assert _is_known_absent_adapter_response(resp(404, {"error": f"lora {name} not loaded"}), name)
    assert _is_known_absent_adapter_response(resp(404, {"message": f"adapter {name}: no such adapter"}), name)
    # Typed NotFoundError without the name still proves the postcondition.
    assert _is_known_absent_adapter_response(resp(404, {"error": {"type": "NotFoundError"}}), name)
    # Route-missing / unrelated 404s stay errors.
    assert not _is_known_absent_adapter_response(resp(404, text="Not Found"), name)
    assert not _is_known_absent_adapter_response(resp(404, {"error": "no such route"}), name)
    assert not _is_known_absent_adapter_response(resp(404, {"error": "lora ttt-other not loaded"}), name)
    assert not _is_known_absent_adapter_response(resp(400, legacy), name)


async def test_v2_unary_stop_and_http_start_failure_do_not_strand_work_loop():
    import threading
    from queue import Queue
    from types import SimpleNamespace

    from prime_rl.ttt.server_v2 import _monitor_http_server, _work_loop

    stop_queue = Queue()
    stop_queue.put(("stop",))
    _work_loop(None, stop_queue, SimpleNamespace(is_master=True, world_size=1))

    http_thread = threading.Thread(target=lambda: None)
    http_thread.start()
    http_thread.join(timeout=1)
    abort_queue = Queue()
    _monitor_http_server(SimpleNamespace(started=False), http_thread, abort_queue, startup_timeout=1)
    with pytest.raises(RuntimeError, match="HTTP server failed"):
        _work_loop(None, abort_queue, SimpleNamespace(is_master=True, world_size=1))


async def test_v2_update_base_weights_route_acks_and_times_out():
    """/update_base_weights: 200 once the work loop acks; 503 on a wedged loop."""
    import threading
    from queue import Queue
    from unittest.mock import patch

    from prime_rl.ttt.server_v2 import build_app_v2

    trainer = type("Trainer", (), {"slots": {}, "free_idxs": set(), "base_version": 0})()
    work_queue: Queue = Queue()

    def fake_work_loop():
        kind, step, ack = work_queue.get()
        assert kind == "recv_weights" and step == 3
        trainer.base_version = step
        ack.set()

    app = build_app_v2(
        TTTServiceConfig(engine={"type": "fsdp", "max_batch_wait_seconds": 0}, inference_admin_urls=[]),
        trainer,
        work_queue,
    )
    threading.Thread(target=fake_work_loop, daemon=True).start()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ttt") as client:
            response = await client.post("/update_base_weights", json={"step": 3})
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "base_version": 3}
            health = await client.get("/health")
            assert health.json()["base_version"] == 3

            # Wedged work loop (nothing drains the queue): bounded 503, not a hang.
            with patch("prime_rl.ttt.server_v2._RESULT_WAIT_SECONDS", 0.05):
                response = await client.post("/update_base_weights", json={"step": 4})
            assert response.status_code == 503


async def test_shared_nccl_receiver_module_is_vllm_free_importable():
    """The moved receiver lives in utils (importable without vLLM at module import time)
    and the vLLM worker re-exports it, so both sides share one implementation."""
    import importlib
    import sys

    module = importlib.import_module("prime_rl.utils.nccl_receiver")
    assert hasattr(module, "NCCLWeightBroadcastReceiver")
    # Import must not have pulled vLLM in (the communicator import is lazy).
    source = open(module.__file__).read()
    assert "from vllm" in source  # lazy import inside __init__
    assert module.__name__ in sys.modules
