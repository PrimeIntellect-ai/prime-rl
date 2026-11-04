"""State that the API server processes of one vLLM server share."""

import asyncio
import threading
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
    marker, policy = (server.server_file(instance_id, kind) for kind in ("invalid", "policy"))

    async def run():
        engine = Engine()
        first, second = (
            SimpleNamespace(
                engine_client=engine, weight_update_lock=asyncio.Lock(), invalid_file=marker, policy_file=policy
            )
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
        policy.unlink(missing_ok=True)


def test_a_reader_never_sees_a_partly_written_server_file(tmp_path):
    """Another API server process reads readiness files at any moment: a torn required policy would read
    as none required and take a server without a policy into service."""
    path = tmp_path / "server.required"
    server.write_server_file(path, 0)
    done = threading.Event()
    torn = []

    def read() -> None:
        while not done.is_set():
            if server.read_policy(path) == -1:
                torn.append(path)

    reader = threading.Thread(target=read)
    reader.start()
    for version in range(1, 5000):
        server.write_server_file(path, version)
    done.set()
    reader.join()
    assert not torn and server.read_policy(path) == 4999
