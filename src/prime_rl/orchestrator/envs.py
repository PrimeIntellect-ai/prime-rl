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
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import Generic, TypeVar

import verifiers.v1 as vf
from verifiers.v1.serve import EnvClient

from prime_rl.configs.orchestrator import EnvConfig, EvalSourceConfig, TrainSourceConfig
from prime_rl.orchestrator.algo import Algorithm, build_algorithm
from prime_rl.orchestrator.sampler import Sampler
from prime_rl.orchestrator.types import Rollout
from prime_rl.utils.logger import get_logger

# Every wire trace validates into this type. WireTaskData (extra="allow") keeps the env's task
# fields without importing the env package — the orchestrator never reads them typed (only
# task.idx + task.model_dump).
ROLLOUT_TYPE = Rollout[vf.WireTaskData]

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
        self._env_client = EnvClient(address=self.address)
        # The server may still be coming up (the launcher spawns it concurrently with
        # the orchestrator), so poll until it answers.
        await self.env_client.wait_for_server_startup(timeout=ENV_SERVER_STARTUP_TIMEOUT)
        taskset = vf.load_taskset(self.config.env.taskset)
        if type(taskset).INFINITE:
            self.tasks = iter(taskset.load())
            self.num_tasks = None
        else:
            # Materialize off the event loop — load() may pull a dataset.
            materialized = await asyncio.to_thread(lambda: list(taskset.load()))
            self.tasks = iter(materialized)
            self.num_tasks = len(materialized)
        num_tasks = self.num_tasks if self.num_tasks is not None else "infinite"
        get_logger().info(f"Env {self.name} ready: num_tasks={num_tasks}")

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
    ) -> list[Rollout]:
        """Run one episode; return its typed Traces. A zero-trace episode raises (the
        dispatcher synthesizes the error marker); a not-``ok`` episode marks its clean
        traces failed so partial episodes never train."""
        episode = await self.env_client.run(
            task_data=task_data,
            client=client,
            model=model_name,
            sampling=self._sampling(cache_salt),
        )
        if not episode.traces:
            error = episode.last_error
            detail = f"{error.type}: {error.message}" if error is not None else "no traces and no error recorded"
            raise RuntimeError(f"episode failed before any trace was produced — {detail}")
        rollouts = [ROLLOUT_TYPE.model_construct(**dict(wire)) for wire in episode.traces]
        for rollout in rollouts:
            rollout.episode_id = episode.id
            if not episode.ok and rollout.ok:
                error = episode.last_error or vf.Error(
                    type="EpisodeFailed", message="A sibling trace in this episode failed"
                )
                rollout.errors = [*rollout.errors, error]
                rollout.ok = False
        return rollouts


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
        self.examples: list[dict] = []

    async def start(self) -> None:
        await super().start()
        n = self.config.num_examples
        if self.num_tasks is None and n < 0:
            raise ValueError(f"Eval env {self.name} has an infinite taskset — set num_examples to bound it")
        # A fixed eval set, pulled off the tasks once and reused every epoch.
        tasks = list(self.tasks) if n < 0 else list(islice(self.tasks, n))
        self.examples = [{"task": task} for task in tasks]


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
