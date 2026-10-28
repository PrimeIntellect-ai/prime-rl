"""The vLLM server's guard against resuming generation on a partially updated model."""

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from prime_rl.inference.vllm import server


class Engine:
    """The engine core that every API server process of one server talks to."""

    def __init__(self) -> None:
        self.fail = False
        self.resumed = False

    async def collective_rpc(self, method: str, args=(), kwargs=None) -> None:
        if self.fail:
            raise RuntimeError("the transfer failed")

    async def update_weight_version(self, version: str) -> None:
        pass

    async def resume_generation(self) -> None:
        self.resumed = True


class Request:
    def __init__(self, state: SimpleNamespace, body: dict | None = None) -> None:
        self.app = SimpleNamespace(state=state)
        self.body = body

    async def json(self) -> dict | None:
        return self.body


def test_a_failed_update_blocks_resume_on_every_api_process():
    """/update_weights and /resume can reach different API server processes of one server: none of
    them resumes before a complete update."""
    instance_id = uuid.uuid4().hex
    marker = server.server_file(instance_id, "invalid")

    async def run():
        engine = Engine()
        first, second = (
            SimpleNamespace(engine_client=engine, weight_update_lock=asyncio.Lock(), invalid_file=marker)
            for _ in range(2)
        )
        update = {"weight_dir": "/w", "weight_version": 1}
        engine.fail = True
        with pytest.raises(RuntimeError):
            await server.update_weights(Request(first, update))
        response = await server.resume(Request(second))
        assert response.status_code == 409 and not engine.resumed
        engine.fail = False
        await server.update_weights(Request(second, update))
        assert await server.resume(Request(first)) == {"status": "resumed"} and engine.resumed

    try:
        asyncio.run(run())
    finally:
        marker.unlink(missing_ok=True)
