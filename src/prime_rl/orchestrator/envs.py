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
cache — and gives the orchestrator real tasks to sample.

The server answers one ``Episode`` per run request, whose traces we validate into
``Trace[WireTaskData]`` — real ``vf.Trace``\\ s (never loose dicts) whose task
keeps the env's task-specific fields as extras (``WireTaskData`` allows them).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import Generic, TypeVar

import verifiers.v1 as vf
from verifiers.v1.serve import EnvClient

from prime_rl.configs.orchestrator import EnvConfig, EvalSourceConfig, TrainSourceConfig
from prime_rl.orchestrator.algo import Algorithm, build_algorithm
from prime_rl.orchestrator.generation_source import GenerationSource
from prime_rl.orchestrator.reward_shaping import RewardShaper, build_reward_shapers
from prime_rl.utils.logger import format_time, get_logger

# Max wait for the env server to answer health. Generous because the launcher spawns
# servers concurrently with the orchestrator, and a server imports its env package
# before serving.
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
        self.tasks: Iterator[vf.Task] | None = None
        """The env's tasks, client-side, set at ``start()``. A finite taskset is
        materialized (``num_tasks`` is its count) and iterated from there; an infinite
        one streams off its generator. Consumed once — by ``TrainSource`` (train) or
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
        t0 = time.perf_counter()
        self._env_client = EnvClient(address=self.address)
        # The server may still be coming up (the launcher spawns it concurrently with
        # the orchestrator), so poll until it answers.
        await self.env_client.wait_for_server_startup(timeout=ENV_SERVER_STARTUP_TIMEOUT)
        taskset = vf.load_taskset(self.config.env.taskset)
        if type(taskset).INFINITE:
            self.tasks = iter(taskset)
            self.num_tasks = None
        else:
            # Materialize off the event loop — iterating may pull a dataset.
            materialized = await asyncio.to_thread(lambda: list(taskset))
            self.tasks = iter(materialized)
            self.num_tasks = len(materialized)
        num_tasks = self.num_tasks if self.num_tasks is not None else "infinite"
        get_logger().info(f"Env {self.name} ready in {format_time(time.perf_counter() - t0)} (num_tasks={num_tasks})")

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
        task_data: dict,
    ) -> vf.WireEpisode:
        """Run and return one typed episode. A failed multi-trace episode marks
        its otherwise-clean traces failed so partial episodes never train."""
        episode = await self.env_client.run(
            task_data=task_data,
            client=client,
            model=model_name,
            sampling=self._sampling(cache_salt),
        )
        for trace in episode.traces:
            if not episode.ok and trace.ok:
                error = episode.last_error or vf.Error(
                    type="EpisodeFailed", message="A sibling trace in this episode failed"
                )
                trace.errors = [*trace.errors, error]
                trace.ok = False
        return episode


class TrainEnv(Env):
    config: TrainSourceConfig

    def __init__(
        self,
        config: TrainSourceConfig,
        address: str,
        generation_source: GenerationSource,
        algorithm: Algorithm,
        reward_shapers: list[RewardShaper],
    ):
        super().__init__(config, address)
        self.generation_source = generation_source
        self.algorithm = algorithm
        self.reward_shapers = reward_shapers
        self.sampling_args = generation_source.sampling_args(config.sampling.to_sampling_args())


class EvalEnv(Env):
    config: EvalSourceConfig

    def __init__(self, config: EvalSourceConfig, address: str):
        super().__init__(config, address)
        self.sampling_args = config.sampling.to_sampling_args()
        self.examples: list[vf.Task] = []

    async def start(self) -> None:
        await super().start()
        n = self.config.num_examples
        if self.num_tasks is None and n < 0:
            raise ValueError(f"Eval env {self.name} has an infinite taskset — set num_examples to bound it")
        # A fixed eval set, pulled off the tasks once and reused every epoch.
        tasks = list(self.tasks) if n < 0 else list(islice(self.tasks, n))
        self.examples = tasks


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


class TrainEnvs(Envs[TrainEnv]):
    """Collection of training environments, each paired with its
    :class:`GenerationSource` and runtime :class:`Algorithm`, built from the env's
    resolved algorithm config."""

    def __init__(
        self,
        configs: Sequence[TrainSourceConfig],
        addresses: dict[tuple[str, str], str],
        *,
        clients,
        renderer_config=None,
    ):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            assert config.algo is not None, "TrainSourceConfig.algo must be resolved before env construction"
            assert config.reward_shaping is not None, (
                "TrainSourceConfig.reward_shaping must be resolved before env construction"
            )
            get_logger().info(f"Initializing {config.algo.type} algorithm for {config.resolved_name}")
            env = TrainEnv(
                config,
                addresses[("train", config.resolved_name)],
                GenerationSource(config.algo.sampling, clients, renderer_config),
                build_algorithm(config.algo, clients),
                build_reward_shapers(config.reward_shaping),
            )
            self._envs[env.name] = env


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(self, configs: Sequence[EvalSourceConfig], addresses: dict[tuple[str, str], str]):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            env = EvalEnv(config, addresses[("eval", config.resolved_name)])
            self._envs[env.name] = env
