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
from collections.abc import Callable, Iterator, Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import verifiers.v1 as vf
from verifiers.v1.serve import EnvClient

from prime_rl.configs.algorithm import FrozenModelConfig
from prime_rl.configs.orchestrator import EnvConfig, EvalSourceConfig, TrainSourceConfig
from prime_rl.orchestrator.algo import Algorithm, build_algorithm
from prime_rl.orchestrator.clients import InferenceClient, connect_frozen_client, setup_admin_clients
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import env_address_file

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

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

    def __init__(self, config: EnvConfig, address: str | None, address_file: Path):
        self.config = config
        self.address = address
        self.address_file = address_file
        """Where a launcher-managed server publishes its address; read when ``address``
        is None."""
        self.sampling_args: dict = {}
        self.num_tasks: int | None = 0
        """Task count; ``None`` means the selected tasks never end."""
        self.tasks: Iterator[vf.Task] | None = None
        """The env's selected tasks (``select``), client-side, set at
        ``start()``. A bounded selection is materialized (``num_tasks`` is its count)
        and iterated from there; an unbounded one streams off the taskset. Consumed once — by ``TrainSource`` (train) or
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
        t0 = time.perf_counter()
        if self.address is None:
            self.address = await wait_for_address(self.address_file, timeout=ENV_SERVER_STARTUP_TIMEOUT)
        get_logger().debug(f"Connecting {self.name} to env server {self.address}")
        self._env_client = EnvClient(address=self.address)
        # The server may still be coming up (the launcher spawns it concurrently with
        # the orchestrator), so poll until it answers.
        await self.env_client.wait_for_server_startup(timeout=ENV_SERVER_STARTUP_TIMEOUT)
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
        on_delta: Callable[[dict], None] | None = None,
    ) -> vf.WireEpisode:
        """Run and return one typed episode. A failed multi-trace episode marks
        its otherwise-clean traces failed so partial episodes never train.
        ``on_delta`` sees each delta of the env server's stream — a turn or a phase
        change of one of the episode's traces — as it lands."""
        episode = await self.env_client.run(
            task_data=task_data,
            client=client,
            model=model_name,
            sampling=self._sampling(cache_salt),
            on_delta=on_delta,
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
        address: str | None,
        address_file: Path,
        clients: InferenceClient,
        algorithm: Algorithm,
    ):
        super().__init__(config, address, address_file)
        # Train rollouts are generated from `clients`: the policy, or the frozen
        # `sampling.source` connected in setup().
        self.clients = clients
        self.connected: InferenceClient | None = None
        self.algorithm = algorithm
        self.uses_live_policy = config.algo.sampling.source == "policy"
        self.sampling_args = config.sampling.to_sampling_args()
        if not self.uses_live_policy:
            # Logprobs only feed importance ratios on policy-sampled tokens; frozen endpoints may reject the knob.
            self.sampling_args.pop("logprobs", None)
            self.sampling_args.pop("top_logprobs", None)
        # Truncated policy sampling must ship the sampling masks the trainer replays.
        self.requires_sampling_masks = config.sampling.truncates_distribution() and self.uses_live_policy

    async def setup(self, tokenizer: PreTrainedTokenizer) -> None:
        async def connect_source() -> None:
            source = self.config.algo.sampling.source
            if isinstance(source, FrozenModelConfig):
                clients = await connect_frozen_client(source)
                try:
                    async with AsyncExitStack() as stack:
                        admin_clients = [
                            await stack.enter_async_context(client) for client in setup_admin_clients(source)
                        ]
                        # Frozen samples train their native IDs directly, so every ID must retain its meaning.
                        policy_vocab = tokenizer.get_vocab()
                        for client in admin_clients:
                            response = await client.get("/v1/tokenizer", timeout=source.wait_for_ready_timeout)
                            if response.status_code != 200:
                                raise ValueError(
                                    f"Frozen generation requires GET /v1/tokenizer; {client.base_url} returned "
                                    f"HTTP {response.status_code}. If the router does not forward this endpoint, "
                                    "set admin_base_url to the inference engine URLs."
                                )
                            vocab = response.json()
                            if (
                                not isinstance(vocab, dict)
                                or not vocab
                                or any(type(index) is not int for index in vocab.values())
                            ):
                                raise ValueError(
                                    f"Frozen generation endpoint {client.base_url} returned an invalid vocabulary"
                                )
                            if vocab != policy_vocab:
                                raise ValueError(
                                    f"Frozen generation endpoint {client.base_url} must use the same token-to-ID mapping "
                                    "as the policy tokenizer; training across different tokenizers is not supported."
                                )
                except BaseException:
                    await clients.aclose()
                    raise
                self.clients = self.connected = clients

        await asyncio.gather(connect_source(), self.algorithm.setup())

    async def aclose(self) -> None:
        for clients in (self.connected, self.algorithm.connected):
            if clients is not None:
                await clients.aclose()


class EvalEnv(Env):
    config: EvalSourceConfig

    def __init__(self, config: EvalSourceConfig, address: str | None, address_file: Path):
        super().__init__(config, address, address_file)
        self.sampling_args = config.sampling.to_sampling_args()
        self.examples: list[vf.Task] = []

    async def start(self) -> None:
        await super().start()
        if self.num_tasks is None:
            raise ValueError(f"Eval env {self.name} has an infinite taskset — set select.limit to bound it")
        # A fixed eval set, pulled off the tasks once and reused every epoch.
        self.examples = list(self.tasks)


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
    ):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            get_logger().info(f"Initializing {config.algo.type} algorithm for {config.resolved_name}")
            env = TrainEnv(
                config,
                addresses[("train", config.resolved_name)],
                env_address_file(config_dir, "train", config.resolved_name),
                clients,
                build_algorithm(config.algo, clients),
            )
            self._envs[env.name] = env


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(
        self, configs: Sequence[EvalSourceConfig], addresses: dict[tuple[str, str], str | None], config_dir: Path
    ):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            name = config.resolved_name
            env = EvalEnv(config, addresses[("eval", name)], env_address_file(config_dir, "eval", name))
            self._envs[env.name] = env
