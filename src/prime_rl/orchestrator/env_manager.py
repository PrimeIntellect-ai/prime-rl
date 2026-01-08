import asyncio

from prime_rl.orchestrator.env_worker import EnvWorker


class EnvWorkerGroup:
    """Manages a group of env workers."""

    def __init__(self, env_names: list[str], env_workers: list[EnvWorker]):
        assert len(env_names) == len(env_workers)
        self.env_names = env_names
        self.env_workers = dict(zip(env_names, env_workers))
        self.response_collectors: dict[str, asyncio.Task] = {}

    async def start(self):
        for env_worker in self.env_workers.values():
            env_worker.start()
            asyncio.create_task(env_worker.collect_responses())

    async def run_group(self, env_name: str, example_id: int, rollouts_per_example: int, model_name: str):
        return await self.env_workers[env_name].run_group(
            example_id=example_id,
            rollouts_per_example=rollouts_per_example,
            model_name=model_name,
        )

    async def stop(self):
        for task in self.response_collectors.values():
            task.cancel()
        for env_worker in self.env_workers.values():
            env_worker.stop()
