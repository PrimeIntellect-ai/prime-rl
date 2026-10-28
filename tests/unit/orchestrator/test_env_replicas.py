"""An env spreads its runs over its server's replicas, least loaded first."""

import asyncio

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
        tasks = [asyncio.create_task(env.run(client, "m", None, {})) for _ in range(4)]
        await asyncio.sleep(0)
        assert [c.runs for c in env._env_clients] == [2, 2]
        env._env_clients[0].release.set()
        await asyncio.sleep(0)
        assert env._in_flight == [0, 2]
        tasks.append(asyncio.create_task(env.run(client, "m", None, {})))
        await asyncio.sleep(0)
        assert env._env_clients[0].runs == 3
        env._env_clients[1].release.set()
        await asyncio.gather(*tasks)
        assert env._in_flight == [0, 0]

    asyncio.run(run())
