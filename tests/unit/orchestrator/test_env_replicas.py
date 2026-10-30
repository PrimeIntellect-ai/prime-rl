"""An env spreads its runs over its server's replicas, least loaded first."""

import asyncio
from types import SimpleNamespace

from prime_rl.orchestrator import dispatcher
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.envs import Env


class FakeClient:
    def __init__(self, healthy: bool = True):
        self.address = "tcp://replica"
        self.healthy = healthy
        self.runs = 0
        self.release = asyncio.Event()

    async def health(self, timeout):
        return self.healthy

    async def run(self, **kwargs):
        self.runs += 1
        await self.release.wait()


def test_runs_go_to_the_least_loaded_replica():
    async def run():
        env = Env(None, None, [])
        env._env_clients = [FakeClient(), FakeClient()]
        env._running = [set(), set()]
        client = None
        tasks = [asyncio.create_task(env.run(client, "m", env.sampling, None, {})) for _ in range(4)]
        await asyncio.sleep(0)
        assert [c.runs for c in env._env_clients] == [2, 2]
        env._env_clients[0].release.set()
        await asyncio.sleep(0)
        assert [len(running) for running in env._running] == [0, 2]
        tasks.append(asyncio.create_task(env.run(client, "m", env.sampling, None, {})))
        await asyncio.sleep(0)
        assert env._env_clients[0].runs == 3
        env._env_clients[1].release.set()
        await asyncio.gather(*tasks)
        assert env._running == [set(), set()]

    asyncio.run(run())


def test_health_monitor_fails_and_avoids_a_dead_replica(monkeypatch):
    monkeypatch.setattr(dispatcher, "ENV_HEALTH_INTERVAL_S", 0)

    async def run():
        env = Env(SimpleNamespace(resolved_name="e"), None, [])
        env._env_clients = [FakeClient(), FakeClient(healthy=False)]
        env._running = [set(), set()]
        tasks = [asyncio.create_task(env.run(None, "m", env.sampling, None, {})) for _ in range(2)]
        await asyncio.sleep(0)
        d = Dispatcher.__new__(Dispatcher)
        d.train_envs, d.eval_envs, d.breakers = None, [env], {}
        d.inflight, d.aborted = dict.fromkeys(tasks), {}
        monitor = asyncio.create_task(d.monitor_env_health())
        while not env.down:
            await asyncio.sleep(0)
        monitor.cancel()
        await asyncio.sleep(0)
        assert env.down == {1}
        assert [task.cancelled() for task in tasks] == [False, True]
        # The dead replica is the least loaded, but new runs go to the live one.
        tasks.append(asyncio.create_task(env.run(None, "m", env.sampling, None, {})))
        await asyncio.sleep(0)
        assert [c.runs for c in env._env_clients] == [2, 1]
        env._env_clients[0].release.set()
        await asyncio.gather(tasks[0], tasks[2])

    asyncio.run(run())
