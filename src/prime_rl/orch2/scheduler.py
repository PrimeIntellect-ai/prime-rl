from dataclasses import dataclass
from typing import Iterator, Literal, TypeAlias

import verifiers as vf

from prime_rl.configs.orchestrator import TrainEnvConfig

Kind: TypeAlias = Literal["train", "eval"]


@dataclass
class Task:
    """What the engine needs to run a rollout. No dataset — that's the scheduler's job."""

    id: str
    env: vf.Environment
    sampling_args: dict
    kind: Kind


class Scheduler:
    """Round-robin over env configs. Owns the datasets; hands (task, example) to the engine."""

    def __init__(self, env_cfgs: list[TrainEnvConfig], kind: Kind = "train"):
        assert env_cfgs, "Scheduler requires at least one env config"
        self.tasks: list[Task] = []
        self._datasets: list[Iterator[dict]] = []
        for cfg in env_cfgs:
            env = vf.load_environment(cfg.stripped_id, **cfg.args)
            self.tasks.append(
                Task(
                    id=cfg.resolved_name,
                    env=env,
                    sampling_args=cfg.sampling.to_sampling_args(),
                    kind=kind,
                )
            )
            self._datasets.append(_cycle_forever(env.get_dataset()))
        self._idx = 0

    def next_task(self) -> tuple[Task, dict] | None:
        for _ in range(len(self.tasks)):
            i = self._idx
            self._idx = (self._idx + 1) % len(self.tasks)
            try:
                return self.tasks[i], next(self._datasets[i])
            except StopIteration:
                continue
        return None

    async def on_new_version(self, step: int) -> None:
        # Placeholder: eval cadence, ratio rebalancing, etc.
        del step


def _cycle_forever(ds) -> Iterator[dict]:
    i = 0
    while True:
        for row in ds:
            r = dict(row)
            r["example_id"] = i
            i += 1
            yield r
