"""Env wrappers over a v1 env server.

Each ``Env`` holds an ``EnvClient`` onto each replica of its source's env server
(``OrchestratorConfig.env_server_replicas``; multi-node runs start one per inference
node) and routes each run to the least loaded one. A source's own ``serve.address``
names a single external server (``OrchestratorConfig.env_addresses``); otherwise each
launcher-managed replica publishes the address it bound, and the orchestrator connects. The
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
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Generic, TypeVar

import verifiers.v1 as vf
from renderers import RendererConfig
from verifiers.v1.serve import EnvClient

from prime_rl.configs.algorithm import FrozenModelConfig
from prime_rl.configs.orchestrator import EnvConfig, EvalSourceConfig, TrainSourceConfig
from prime_rl.orchestrator.algo import Algorithm, build_algorithm
from prime_rl.orchestrator.clients import InferenceClient, connect_frozen_client
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import env_address_file

# Max wait for the env server to answer health. Generous because the launcher spawns
# servers concurrently with the orchestrator, and a server imports its env package
# before serving.
ENV_SERVER_STARTUP_TIMEOUT = 600.0


async def wait_for_address(path: Path, timeout: float) -> str:
    """The address a launcher-managed env server published, polling for the file the
    server writes once it has bound (it starts concurrently with this process)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if address := path.read_text().strip():
                return address
        except FileNotFoundError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"env server never published its address to {path} within {timeout}s")
        await asyncio.sleep(0.5)


class Env:
    """Client onto a v1 env server. The orchestrator owns the taskset (loaded once,
    client-side); the server owns agent/harness execution."""

    def __init__(self, config: EnvConfig, address: str | None, address_files: list[Path]):
        self.config = config
        self.address = address
        self.address_files = address_files
        """Where the replicas of a launcher-managed server publish their addresses; read
        when ``address`` is None."""
        self.sampling = vf.SamplingConfig()
        self.num_tasks: int | None = 0
        """Task count; ``None`` means the selected tasks never end."""
        self.tasks: Iterator[vf.Task] | None = None
        """The env's selected tasks (``select``), client-side, set at
        ``start()``. A bounded selection is materialized (``num_tasks`` is its count)
        and iterated from there; an unbounded one streams off the taskset. Consumed once — by ``TrainSource`` (train) or
        ``EvalEnv.start`` (eval)."""
        self._env_clients: list[EnvClient] = []
        self._in_flight: list[int] = []
        """Runs in flight per replica, to route each run to the least loaded one."""

    @property
    def name(self) -> str:
        return self.config.resolved_name

    @property
    def env_clients(self) -> list[EnvClient]:
        if not self._env_clients:
            raise RuntimeError(f"Env {self.name} not started — call start() first.")
        return self._env_clients

    async def start(self) -> None:
        """Connect to the env server's replicas and load the taskset client-side."""
        t0 = time.perf_counter()
        if self.address is not None:
            addresses = [self.address]
        else:
            addresses = await asyncio.gather(
                *(wait_for_address(path, timeout=ENV_SERVER_STARTUP_TIMEOUT) for path in self.address_files)
            )
        get_logger().debug(f"Connecting {self.name} to env servers {addresses}")
        self._env_clients = [EnvClient(address=address) for address in addresses]
        self._in_flight = [0] * len(addresses)
        # The servers may still be coming up (the launcher spawns them concurrently with
        # the orchestrator), so poll until they answer.
        await asyncio.gather(
            *(client.wait_for_server_startup(timeout=ENV_SERVER_STARTUP_TIMEOUT) for client in self.env_clients)
        )
        taskset = vf.load_taskset(self.config.env.taskset).select(self.config.select)
        if taskset.bounded:
            # Materialize off the event loop — iterating may pull a dataset.
            materialized = await asyncio.to_thread(lambda: list(taskset))
            self.tasks = iter(materialized)
            self.num_tasks = len(materialized)
        else:
            self.tasks = iter(taskset)
            self.num_tasks = None
        num_tasks = self.num_tasks if self.num_tasks is not None else "infinite"
        get_logger().info(f"Env {self.name} ready in {format_time(time.perf_counter() - t0)} (num_tasks={num_tasks})")

    def _sampling(self, cache_salt: str | None) -> vf.SamplingConfig:
        sampling = self.sampling
        if cache_salt is not None:
            sampling = sampling.model_copy(update={"cache_salt": cache_salt})
        return sampling

    async def run(
        self,
        client: vf.ClientConfig,
        model_name: str,
        cache_salt: str | None,
        task_data: dict,
        on_delta: Callable[[dict], None] | None = None,
    ) -> vf.WireEpisode:
        """Run and return the native typed episode.
        ``on_delta`` sees each delta of the env server's stream — a turn or a phase
        change of one of the episode's traces — as it lands. The run goes to the replica
        with the fewest runs in flight."""
        replica = min(range(len(self.env_clients)), key=self._in_flight.__getitem__)
        self._in_flight[replica] += 1
        try:
            return await self.env_clients[replica].run(
                task_data=task_data,
                client=client,
                model=model_name,
                sampling=self._sampling(cache_salt),
                on_delta=on_delta,
            )
        finally:
            self._in_flight[replica] -= 1


class TrainEnv(Env):
    config: TrainSourceConfig

    def __init__(
        self,
        config: TrainSourceConfig,
        address: str | None,
        address_files: list[Path],
        clients: InferenceClient,
        renderer_config: RendererConfig | None,
        algorithm: Algorithm,
    ):
        super().__init__(config, address, address_files)
        # Train rollouts are generated from `clients`: the policy, or the frozen
        # `sampling.source` connected in setup() with the renderer (token-in/out) client.
        self.clients = clients
        self.renderer_config = renderer_config
        self.connected: InferenceClient | None = None
        self.algorithm = algorithm
        self.uses_live_policy = config.algo.sampling.source == "policy"
        self.sampling = vf.SamplingConfig(
            **config.sampling.model_dump(exclude_none=True, exclude={"logprobs"}),
            # Only policy rollouts need sampling logprobs for importance ratios.
            logprobs=(config.sampling.logprobs or True) if self.uses_live_policy else None,
        )
        # Truncated policy sampling must ship the sampling masks the trainer replays.
        self.requires_sampling_masks = config.sampling.truncates_distribution() and self.uses_live_policy
        self.requires_top_logprobs = config.sampling.logprobs is not None and self.uses_live_policy

    async def setup(self) -> None:
        async def connect_source() -> None:
            source = self.config.algo.sampling.source
            if isinstance(source, FrozenModelConfig):
                self.connected = await connect_frozen_client(source, renderer_config=self.renderer_config)
                self.clients = self.connected

        await asyncio.gather(connect_source(), self.algorithm.setup())

    async def aclose(self) -> None:
        for clients in (self.connected, self.algorithm.connected):
            if clients is not None:
                await clients.aclose()


class EvalEnv(Env):
    config: EvalSourceConfig

    def __init__(self, config: EvalSourceConfig, address: str | None, address_files: list[Path]):
        super().__init__(config, address, address_files)
        self.sampling = config.sampling
        self.examples: list[vf.Task] = []

    async def start(self) -> None:
        await super().start()
        if self.num_tasks is None:
            raise ValueError(f"Eval env {self.name} has an infinite taskset — set select.limit to bound it")
        # A fixed eval set, pulled off the tasks once and reused every epoch.
        self.examples = list(self.tasks)


EnvT = TypeVar("EnvT", bound=Env)


def replica_address_files(config_dir: Path, split: str, name: str, replicas: int) -> list[Path]:
    if replicas == 1:
        return [env_address_file(config_dir, split, name)]
    return [env_address_file(config_dir, split, name, replica) for replica in range(replicas)]


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
        # When several env.start()s load_dataset() concurrently, datasets' parallel arrow read
        # (tqdm thread_map) races in ensure_lock's `del tqdm_class._lock` and crashes with
        # `AttributeError: type object 'tqdm' has no attribute '_lock'`. Pre-seed the lock so it
        # isn't created-and-deleted per call.
        from datasets.utils import tqdm as hf_tqdm

        hf_tqdm.set_lock(hf_tqdm.get_lock())
        await asyncio.gather(*(env.start() for env in self))


class TrainEnvs(Envs[TrainEnv]):
    """Collection of training environments, each with its runtime
    :class:`Algorithm`, built from the env's resolved algorithm config."""

    def __init__(
        self,
        configs: Sequence[TrainSourceConfig],
        addresses: dict[tuple[str, str], str | None],
        config_dir: Path,
        *,
        clients,
        renderer_config=None,
        replicas: int = 1,
    ):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            get_logger().info(f"Initializing {config.algo.type} algorithm for {config.resolved_name}")
            env = TrainEnv(
                config,
                addresses[("train", config.resolved_name)],
                replica_address_files(config_dir, "train", config.resolved_name, replicas),
                clients,
                renderer_config,
                build_algorithm(config.algo, clients),
            )
            self._envs[env.name] = env


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(
        self,
        configs: Sequence[EvalSourceConfig],
        addresses: dict[tuple[str, str], str | None],
        config_dir: Path,
        replicas: int = 1,
    ):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            name = config.resolved_name
            files = replica_address_files(config_dir, "eval", name, replicas)
            env = EvalEnv(config, addresses[("eval", name)], files)
            self._envs[env.name] = env
