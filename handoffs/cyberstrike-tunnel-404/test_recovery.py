import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
import pytest
from verifiers.v1.interception import pool as pool_module
from verifiers.v1.interception import server as server_module
from verifiers.v1.interception.pool import ElasticInterceptionPool
from verifiers.v1.interception.server import InterceptionServer

class LocalTunnel:
    bind_host = "127.0.0.1"
    bind_port = 0
    tunnel_id = "synthetic-local"
    def __init__(self):
        self.closed = False
    @asynccontextmanager
    async def expose(self, port):
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            self.closed = True

class Session:
    def __init__(self, name):
        self.ctx = SimpleNamespace(client=object())
        self.trace = SimpleNamespace(id=name)
        self.released = False
    def release(self):
        self.released = True

@pytest.mark.asyncio
@pytest.mark.parametrize("active_count", [1, 32])
async def test_connected_broken_member_is_replaced_without_killing_active(monkeypatch, active_count):
    created = []
    checks = asyncio.Queue()
    completed = asyncio.Queue()
    class Server(InterceptionServer):
        def __init__(self, config, requires_tunnel, state_service_secrets):
            super().__init__(config, False, state_service_secrets)
            self.tunnel = LocalTunnel()
            self.started = False
            created.append(self)
        def _client(self, config):
            return object()
        async def _probe_health(self, scope, url):
            if not self.started:
                if scope == "public": self.started = True
                return True
            if self is created[0] and scope == "public":
                result = await checks.get()
                completed.put_nowait(None)
                return result
            return True
    monkeypatch.setattr(pool_module, "InterceptionServer", Server)
    monkeypatch.setattr(server_module, "HEALTH_CHECK_INTERVAL_SECONDS", 0)
    active_sessions = [Session(f"first-{i}") for i in range(active_count)]
    second_session = Session("second")
    pool = ElasticInterceptionPool(requires_tunnel=True)
    async with pool:
        async with AsyncExitStack() as active:
            slots = [await active.enter_async_context(pool.acquire(session)) for session in active_sessions]
            old_slot = slots[0]
            old_server = created[0]
            assert old_server.load == active_count
            for _ in range(2):
                checks.put_nowait(False)
                await completed.get()
            assert not old_server.healthy
            assert all(not session.released for session in active_sessions)
            assert not old_server.tunnel.closed
            async with pool.acquire(second_session) as new_slot:
                assert new_slot[0] != old_slot[0]
                assert len(created) == 2
                assert all(not session.released for session in active_sessions)
                assert old_server in pool.servers
            assert old_server in pool.servers
        assert all(session.released for session in active_sessions)
        assert old_server.tunnel.closed
        assert old_server not in pool.servers
    assert all(server.tunnel.closed for server in created)
