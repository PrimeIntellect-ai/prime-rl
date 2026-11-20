"""An env spreads its runs over its server's replicas, least loaded first."""

import asyncio
from types import SimpleNamespace

from prime_rl.env.types import RestoreRef, SaveResponse
from prime_rl.orchestrator.envs import Env


class FakeClient:
    def __init__(self):
        self.runs = 0
        self.release = asyncio.Event()

    async def run(self, **kwargs):
        self.runs += 1
        await self.release.wait()


def test_runs_go_to_the_least_loaded_replica():
    async def run():
        env = Env(None, None, [])
        env._env_clients = [FakeClient(), FakeClient()]
        env._in_flight = [0, 0]
        client = None
        tasks = [asyncio.create_task(env.run(client, "m", env.sampling, None, {})) for _ in range(4)]
        await asyncio.sleep(0)
        assert [c.runs for c in env._env_clients] == [2, 2]
        env._env_clients[0].release.set()
        await asyncio.sleep(0)
        assert env._in_flight == [0, 2]
        tasks.append(asyncio.create_task(env.run(client, "m", env.sampling, None, {})))
        await asyncio.sleep(0)
        assert env._env_clients[0].runs == 3
        env._env_clients[1].release.set()
        await asyncio.gather(*tasks)
        assert env._in_flight == [0, 0]

    asyncio.run(run())


class SaveClient:
    def __init__(self, request_id: str, *, fails: bool = False, waits: bool = False):
        self.restore = RestoreRef(trace_id=f"t-{request_id}", path=f"/ckpt/t-{request_id}.restore", snapshot={})
        self.request_id = request_id
        self.fails = fails
        self.release = asyncio.Event()
        if not waits:
            self.release.set()

    async def save(self, directory: str) -> SaveResponse:
        await self.release.wait()
        if self.fails:
            raise RuntimeError("worker failed")
        return SaveResponse(restores={self.request_id: self.restore})


def test_save_merges_the_replicas_that_answer_in_time():
    async def run():
        env = Env(SimpleNamespace(resolved_name="test"), None, [])
        fast, failing, slow = SaveClient("r1"), SaveClient("r2", fails=True), SaveClient("r3", waits=True)
        env._env_clients = [fast, failing, slow]
        restores = await env.save("/ckpt", timeout=0.05)
        assert restores == {"r1": fast.restore}
        assert env.late_restores == []
        # What a slow replica saved after all is kept apart, for its owner to delete.
        slow.release.set()
        for _ in range(3):
            await asyncio.sleep(0)
        assert env.late_restores == [slow.restore]

    asyncio.run(run())
