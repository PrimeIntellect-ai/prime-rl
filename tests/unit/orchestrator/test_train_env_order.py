"""Train sources that share a shuffled selection each walk their own order."""

import asyncio
from types import SimpleNamespace

from prime_rl.orchestrator.envs import Env, TrainEnv


def walk(monkeypatch, name: str, shuffle: bool = True) -> list[int]:
    async def start(self):
        self.tasks = iter(range(100))
        self.num_tasks = 100

    monkeypatch.setattr(Env, "start", start)
    env = TrainEnv.__new__(TrainEnv)
    env.config = SimpleNamespace(resolved_name=name, select=SimpleNamespace(shuffle=shuffle, seed=0))
    asyncio.run(env.start())
    return list(env.tasks)


def test_sources_sharing_a_selection_walk_distinct_orders(monkeypatch):
    low, high = walk(monkeypatch, "repoforge-low"), walk(monkeypatch, "repoforge-high")
    assert sorted(low) == sorted(high) == list(range(100))
    assert low != high
    assert walk(monkeypatch, "repoforge-low") == low


def test_unshuffled_selection_keeps_its_order(monkeypatch):
    assert walk(monkeypatch, "repoforge-low", shuffle=False) == list(range(100))
