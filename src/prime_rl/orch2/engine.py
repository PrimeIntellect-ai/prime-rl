import asyncio
from dataclasses import dataclass
from typing import Iterator

import verifiers as vf

# ---------- data ----------


@dataclass
class Pool:
    id: str
    dataset: Iterator[dict]
    env: "EnvWorker"
    concurrency: int
    rollouts_per_group: int
    max_off_policy: int


@dataclass
class Group:
    example: dict
    pool_id: str
    rollouts: list[vf.RolloutOutput]
    policy_version: int


@dataclass
class Inflight:
    version: int
    max_off_policy: int
    gather: asyncio.Future


# ---------- env worker ----------


class EnvWorker:
    def __init__(
        self,
        env: vf.Environment,
        client: vf.ClientConfig,
        model: str,
        sampling_args: dict,
    ):
        self.env = env
        self.client = client
        self.model = model
        self.sampling_args = sampling_args

    async def rollout(self, example: dict, policy_version: int) -> vf.RolloutOutput:
        del policy_version  # tracked on the Group, not the rollout
        return await self.env.run_rollout(
            vf.RolloutInput(**example),
            client=self.client,
            model=self.model,
            sampling_args=self.sampling_args,
            state_columns=["trajectory", "sampling_args"],
        )


# ---------- engine ----------


class RolloutEngine:
    def __init__(self, pools: list[Pool], out_q: asyncio.Queue[Group]):
        self.pools = pools
        self.out_q = out_q
        self.policy_version = 0
        self._inflight: list[Inflight] = []

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for pool in self.pools:
                tg.create_task(self._pump(pool))

    async def _pump(self, pool: Pool) -> None:
        sem = asyncio.Semaphore(pool.concurrency)
        while True:
            await sem.acquire()
            try:
                example = next(pool.dataset)
            except StopIteration:
                sem.release()
                return
            asyncio.create_task(self._run_group(pool, example, sem))

    async def _run_group(self, pool: Pool, example: dict, sem: asyncio.Semaphore) -> None:
        try:
            version = self.policy_version  # snapshot; current version may advance during await
            gather = asyncio.gather(*(pool.env.rollout(example, version) for _ in range(pool.rollouts_per_group)))
            inflight = Inflight(version=version, max_off_policy=pool.max_off_policy, gather=gather)
            self._inflight.append(inflight)

            try:
                rollouts = await gather
            except asyncio.CancelledError:
                return
            finally:
                self._inflight.remove(inflight)

            # correctness guard for the async window. Main off policy logic is handle in on_new_version.
            if self.policy_version - version > pool.max_off_policy:
                return

            await self.out_q.put(
                Group(example=example, pool_id=pool.id, rollouts=list(rollouts), policy_version=version)
            )
        finally:
            sem.release()

    def on_new_version(self, version: int) -> None:
        self.policy_version = version
        # when new version is arrived, we go over all the in-flight rollouts and cancel them if they are too old.
        for inflight in list(self._inflight):
            if version - inflight.version > inflight.max_off_policy:
                inflight.gather.cancel()
