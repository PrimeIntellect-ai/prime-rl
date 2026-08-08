"""Env wrappers over a v1 env server.

Each ``Env`` is an ``EnvClient`` onto its source's env server. Each server's address
is derived from the source's position in the config
(``OrchestratorConfig.env_addresses``); the launcher runs the servers at
exactly those addresses, and the orchestrator connects. The
orchestrator never *runs* an environment — the agents and their runtimes live only
in the server — but it does own the *taskset*: a v1 env's tasks are loaded here,
once, and each dispatched episode ships its task's data on the request
(``task_data``); the server pydantic-validates it into the taskset's declared
``TaskData`` type and runs it. That keeps the server (and every worker in its
pool) stateless about data — no per-worker dataset loads, no idx-addressed task
cache — and gives the orchestrator real tasks to cycle, shuffle, and filter.

The server answers one ``Episode`` per run request; the dispatcher validates its traces into
``Trace[WireTaskData]`` — real ``vf.Trace``\\ s (never loose dicts) whose task
keeps the env's task-specific fields as extras (``WireTaskData`` allows them).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar

import verifiers.v1 as vf
from verifiers.v1.serve import EnvClient

from prime_rl.configs.orchestrator import EnvConfig, EvalSourceConfig, TrainSourceConfig
from prime_rl.orchestrator.algo import Algorithm, build_algorithm
from prime_rl.orchestrator.sampler import Sampler
from prime_rl.utils.logger import get_logger

# Max wait for the env server to answer health. The launcher spawns servers
# concurrently with the orchestrator.
ENV_SERVER_STARTUP_TIMEOUT = 600.0


class Env:
    """Client onto a v1 env server. The orchestrator owns the taskset (loaded once,
    client-side); the server owns agent/harness execution."""

    def __init__(self, config: EnvConfig, address: str):
        self.config = config
        self.address = address
        self.sampling_args: dict = {}
        self.num_tasks: int | None = 0
        """Task count; ``None`` means the taskset is infinite."""
        self.tasks: Iterator[vf.TaskData] = iter(())
        """The env's task data, client-side. A finite taskset is materialized at ``start()``
        (``num_tasks`` is its count) and iterated from there; an infinite one streams
        off its generator. Consumed once — by ``TrainSource`` (train) or
        ``EvalEnv.start`` (eval)."""
        self._env_client: EnvClient | None = None

    @property
    def name(self) -> str:
        return self.config.resolved_name

    @property
    def env_client(self) -> EnvClient:
        if self._env_client is None:
            raise RuntimeError(f"Env {self.name} not started — call start() first.")
        return self._env_client

    async def start(self) -> None:
        """Connect to the env server and load the taskset client-side."""
        get_logger().debug(f"Connecting {self.name} to env server {self.address}")
        self._env_client = EnvClient(address=self.address)
        # The server may still be coming up (the launcher spawns it concurrently with
        # the orchestrator), so poll until it answers.
        await self.env_client.wait_for_server_startup(timeout=ENV_SERVER_STARTUP_TIMEOUT)
        taskset = self.load_taskset()
        if taskset.INFINITE:
            self.tasks = (task.data for task in taskset)
            self.num_tasks = None
        else:
            # Materialize off the event loop — taskset iteration may pull a dataset.
            tasks = await asyncio.to_thread(lambda: list(taskset))
            self.tasks = iter(task.data for task in tasks)
            self.num_tasks = len(tasks)
        num_tasks = self.num_tasks if self.num_tasks is not None else "infinite"
        get_logger().info(f"Env {self.name} ready: num_tasks={num_tasks}")

    def load_taskset(self) -> vf.Taskset:
        return vf.load_taskset(self.config.env.taskset)

    async def close(self) -> None:
        if self._env_client is not None:
            await self._env_client.close()
            self._env_client = None

    def _sampling(self, cache_salt: str | None) -> vf.SamplingConfig:
        sampling = {**self.sampling_args}
        if cache_salt is not None:
            sampling["extra_body"] = {**sampling.get("extra_body", {}), "cache_salt": cache_salt}
        return vf.SamplingConfig(**sampling)

    async def run(
        self,
        client: vf.ClientConfig,
        model_name: str,
        cache_salt: str | None,
        task_data: vf.TaskData,
    ) -> vf.WireEpisode:
        """Run one v1 episode and preserve its episode-level standing."""
        return await self.env_client.run(
            task_data=task_data.model_dump(mode="json"),
            client=client,
            model=model_name,
            sampling=self._sampling(cache_salt),
        )


class TrainEnv(Env):
    config: TrainSourceConfig

    def __init__(self, config: TrainSourceConfig, address: str, sampler: Sampler, algorithm: Algorithm):
        super().__init__(config, address)
        self.sampler = sampler
        self.algorithm = algorithm
        self.sampling_args = sampler.sampling_args(config.sampling.to_sampling_args())


class EvalEnv(Env):
    config: EvalSourceConfig

    def __init__(self, config: EvalSourceConfig, address: str):
        super().__init__(config, address)
        self.sampling_args = config.sampling.to_sampling_args()
        self.eval_tasks: list[vf.TaskData] = []

    def load_taskset(self) -> vf.Taskset:
        taskset = super().load_taskset()
        n = self.config.num_tasks
        if n < 0:
            if taskset.INFINITE:
                raise ValueError(f"Eval env {self.name} has an infinite taskset — set num_tasks to bound it")
            return taskset
        return taskset.head(n)

    async def start(self) -> None:
        await super().start()
        # A fixed eval set, pulled off the bounded taskset view and reused every epoch.
        self.eval_tasks = list(self.tasks)
        if not self.eval_tasks:
            raise ValueError(f"Eval env {self.name} has no tasks to evaluate")


EnvT = TypeVar("EnvT", bound=Env)


class Envs(Generic[EnvT]):
    """Base container for a set of Env instances."""

    _envs: dict[str, EnvT]

    @property
    def names(self) -> list[str]:
        return list(self._envs.keys())

    @property
    def configs(self) -> list[EnvConfig]:
        return [env.config for env in self._envs.values()]

    def get(self, name: str) -> EnvT:
        return self._envs[name]

    def __iter__(self) -> Iterator[EnvT]:
        return iter(self._envs.values())

    def __len__(self) -> int:
        return len(self._envs)

    async def start(self) -> None:
        """Connect to all env servers in parallel — every address is known up front,
        so there's nothing to serialize on."""
        await asyncio.gather(*(env.start() for env in self))

    async def stop(self) -> None:
        await asyncio.gather(*(env.close() for env in self))


class TrainEnvs(Envs[TrainEnv]):
    """Collection of training environments, each paired with its rollout
    :class:`Sampler` and runtime :class:`Algorithm`, built from the env's
    resolved algorithm config."""

    def __init__(
        self,
        configs: Sequence[TrainSourceConfig],
        addresses: dict[tuple[str, str], str],
        *,
        policy_pool,
        renderer_config=None,
    ):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            assert config.algo is not None, "TrainSourceConfig.algo must be resolved before env construction"
            env = TrainEnv(
                config,
                addresses[("train", config.resolved_name)],
                Sampler(config.algo.sampling, policy_pool, renderer_config),
                build_algorithm(config.algo, policy_pool),
            )
            self._envs[env.name] = env


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(self, configs: Sequence[EvalSourceConfig], addresses: dict[tuple[str, str], str]):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            env = EvalEnv(config, addresses[("eval", config.resolved_name)])
            self._envs[env.name] = env
