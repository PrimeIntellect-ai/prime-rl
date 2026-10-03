"""Prefix replay: groups that continue recent train episodes from inside them.

``PrefixBuffer`` keeps recent fresh-start episodes of one train env. Each prefix group
picks one episode and one cut, and all its rollouts replay the episode's first ``cut``
model calls (``vf.Prefix``) before sampling."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import PrefixSourceConfig
from prime_rl.orchestrator.utils import train_work


@dataclass
class PrefixEntry:
    task: vf.Task
    calls: list[vf.PrefixCall]
    step: int
    reward: float
    episode_id: str
    uses_left: int


class PrefixBuffer:
    def __init__(self, config: PrefixSourceConfig) -> None:
        self.config = config
        self.entries: deque[PrefixEntry] = deque(maxlen=config.buffer_size)
        self.rng = random.Random()

    def admit(self, task: vf.Task, group: list[vf.Episode]) -> None:
        """Keep the group's clean single-trace episodes that ``rollouts`` selects and
        that have at least two model calls to cut between."""
        episodes = [episode for episode in group if episode.ok and len(episode.traces) == 1 and episode.traces[0].ok]
        passed = [episode.traces[0].reward >= self.config.pass_threshold for episode in episodes]
        rollouts = self.config.rollouts
        if rollouts == "mixed" and len(set(passed)) < 2:
            return
        for episode, ok in zip(episodes, passed):
            if (rollouts == "failed" and ok) or (rollouts == "passed" and not ok):
                continue
            trace = episode.traces[0]
            calls = vf.Prefix.from_trace(trace).calls
            if len(calls) < 2:
                continue
            entry = PrefixEntry(task, calls, train_work(episode).step, trace.reward, str(episode.id), self.config.uses)
            self.entries.append(entry)

    def available(self, step: int) -> bool:
        """Drop episodes older than ``max_age`` and report whether any is left."""
        while self.entries and step - self.entries[0].step > self.config.max_age:
            self.entries.popleft()
        return bool(self.entries)

    def sample(self, name: str) -> tuple[vf.Task, vf.Prefix]:
        """Pick an episode and a cut; call after :meth:`available` returned True."""
        entry = self.rng.choice(self.entries)
        entry.uses_left -= 1
        if not entry.uses_left:
            self.entries.remove(entry)
        n = len(entry.calls)
        lo, hi = self.config.depth
        cut = min(max(round(self.rng.uniform(lo, hi) * n), 1), n - 1)
        source = {"name": name, "episode": entry.episode_id, "step": entry.step, "reward": entry.reward, "calls": n}
        return entry.task, vf.Prefix(calls=entry.calls[:cut], source=source)

    def metrics(self, step: int) -> dict[str, float]:
        ages = [step - entry.step for entry in self.entries]
        return {"buffer_size": float(len(ages)), "age": sum(ages) / len(ages) if ages else 0.0}
