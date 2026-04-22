import asyncio
from dataclasses import dataclass

import verifiers as vf

from prime_rl.orch2.scheduler import Kind, Scheduler, Task


@dataclass
class Group:
    example: dict
    env_id: str
    kind: Kind
    rollouts: list[vf.RolloutOutput]
    policy_version: int


@dataclass
class Inflight:
    version: int
    gather: asyncio.Future


class RolloutEngine:
    def __init__(
        self,
        scheduler: Scheduler,
        out_q: asyncio.Queue[Group],
        client: vf.ClientConfig,
        model: str,
        rollouts_per_group: int,
        max_off_policy: int,
        concurrency: int,
    ):
        self.scheduler = scheduler
        self.out_q = out_q
        self.client = client
        self.model = model
        self.rollouts_per_group = rollouts_per_group
        self.max_off_policy = max_off_policy
        self.concurrency = concurrency
        self.policy_version = 0
        self._inflight: list[Inflight] = []

    async def run(self) -> None:
        sem = asyncio.Semaphore(self.concurrency)
        while True:
            await sem.acquire()
            got = self.scheduler.next_task()
            if got is None:
                sem.release()
                return
            task, example = got
            asyncio.create_task(self._run_group(task, example, sem))

    async def _run_group(self, task: Task, example: dict, sem: asyncio.Semaphore) -> None:
        try:
            version = self.policy_version  # snapshot; current version may advance during await
            gather = asyncio.gather(*(self._rollout(task, example) for _ in range(self.rollouts_per_group)))
            inflight = Inflight(version=version, gather=gather)
            self._inflight.append(inflight)

            try:
                rollouts = await gather
            except asyncio.CancelledError:
                return
            finally:
                self._inflight.remove(inflight)

            # correctness guard: group may have finished just as a new version arrived
            if self.policy_version - version > self.max_off_policy:
                return

            await self.out_q.put(
                Group(
                    example=example,
                    env_id=task.id,
                    kind=task.kind,
                    rollouts=list(rollouts),
                    policy_version=version,
                )
            )
        finally:
            sem.release()

    async def _rollout(self, task: Task, example: dict) -> vf.RolloutOutput:
        return await task.env.run_rollout(
            vf.RolloutInput(**example),
            client=self.client,
            model=self.model,
            sampling_args=task.sampling_args,
            state_columns=["trajectory", "sampling_args"],
        )

    async def on_new_version(self, step: int) -> None:
        self.policy_version = step
        for inflight in list(self._inflight):
            if step - inflight.version > self.max_off_policy:
                inflight.gather.cancel()

    def max_off_policy_level(self) -> int:
        """Max lag across currently in-flight groups."""
        if not self._inflight:
            return 0
        return max(self.policy_version - i.version for i in self._inflight)
