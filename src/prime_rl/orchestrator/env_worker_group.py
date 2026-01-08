from prime_rl.orchestrator.config import EvalEnvGroupConfig, TrainEnvGroupConfig
from prime_rl.orchestrator.env_worker import EnvWorker


class EnvWorkerGroup:
    """Manages a group of env workers."""

    def __init__(self, env_worker_group_config: TrainEnvGroupConfig | EvalEnvGroupConfig):
        self.env_worker_group_config = env_worker_group_config
        self.env_workers: dict[str, EnvWorker] = {}
        for env_worker_config in self.env_worker_group_config.envs:
            env_worker = EnvWorker(env_worker_config)
            self.env_workers[env_worker.name] = env_worker

    @property
    def env_names(self) -> list[str]:
        return list(self.env_workers.keys())

    def start(self):
        for env_worker in self.env_workers.values():
            env_worker.start()
        return self

    async def run_group(self, env_name: str, example_id: int, model_name: str):
        return await self.env_workers[env_name].run_group(
            example_id=example_id,
            model_name=model_name,
        )

    def get_dataset_size(self, env_name: str) -> int:
        return self.env_workers[env_name].get_dataset_size()

    async def stop(self):
        for env_worker in self.env_workers.values():
            await env_worker.stop()
