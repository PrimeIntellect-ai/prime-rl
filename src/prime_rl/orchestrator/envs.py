"""Env wrappers over a v1 env server.

Each ``Env`` owns a v1 ``EnvServer`` (spawned as a child process, or an
external one given by ``config.address``) and an ``EnvClient`` to drive it. The
orchestrator never *runs* an environment — the agents and their runtimes live only
in the server — but it does own the *taskset*: a v1 env's tasks are loaded here,
once, and each dispatched env-rollout ships its task's data on the request
(``task_data``); the server pydantic-validates it into the taskset's declared
``TaskData`` type and runs it. That keeps the server (and every worker in its
pool) stateless about data — no per-worker dataset loads, no idx-addressed task
cache — and gives the orchestrator real tasks to cycle, shuffle, and filter. Only
the legacy (v0) bridge, whose dataset genuinely lives server-side, is still driven
by ``task_idx`` (its count comes from ``info``).

The server answers one ``Episode`` per env-rollout, whose traces we validate into
``Trace[WireTaskData]`` — real ``vf.Trace``\\ s (never loose dicts) whose task
keeps the env's task-specific fields as extras (``WireTaskData`` allows them).
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import os
import queue
import sys
from collections.abc import Iterator, Sequence
from itertools import islice
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Generic, TypeVar

import verifiers.v1 as vf
from verifiers.v1.serve import EnvClient, env_config_data

from prime_rl.configs.orchestrator import EnvConfig, EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.algo import Algorithm, build_algorithm
from prime_rl.orchestrator.sampler import Sampler
from prime_rl.orchestrator.types import Rollout
from prime_rl.utils.logger import get_logger

# Every wire trace validates into this type. WireTaskData (extra="allow") keeps the env's task
# fields without importing the env package — the orchestrator never reads them typed (only
# task.idx + task.model_dump).
ROLLOUT_TYPE = Rollout[vf.WireTaskData]

# Max wait for a spawned env server to bind and report its address. A legacy
# child loads its dataset before reporting, so this is generous.
ENV_SERVER_SPAWN_TIMEOUT = 600.0


def _run_env_server(
    *,
    log_file: str,
    log_level: str,
    json_logging: bool,
    legacy: bool = False,
    **kwargs,
) -> None:
    """Spawned-process entry point: redirect this process's output to ``log_file`` (the
    server's logging + any subprocess-runtime output), then serve via ``serve_env``. The
    worker-pool sizing arrives in ``kwargs`` (``max_workers`` / ``multiplex`` / ``elastic``
    from the env's ``pool``). ``serve_env`` applies ``log_setup`` here and in every spawned
    worker; a worker inherits this process's redirected stdout/stderr, so its per-rollout
    logs reach ``log_file`` too. Top-level so it stays picklable for the ``spawn`` start
    method. ``legacy`` picks the v0 bridge."""
    from functools import partial

    from verifiers.v1.serve import serve_env

    from prime_rl.orchestrator.utils import setup_env_server_logging

    fh = open(log_file, "w", buffering=1)
    os.dup2(fh.fileno(), sys.stdout.fileno())
    os.dup2(fh.fileno(), sys.stderr.fileno())
    serve_env(
        legacy=legacy,
        log_setup=partial(setup_env_server_logging, log_level, json_logging),
        **kwargs,
    )


class Env:
    """Wraps a v1 env server + client. The orchestrator owns the taskset (loaded once,
    client-side); the server owns agent/harness execution."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.sampling_args: dict = {}
        self.num_tasks: int | None = 0
        """Task count; ``None`` means the taskset is infinite."""
        self.requires_group_scoring: bool = False
        self.tasks: list[vf.Task] | None = None
        """A finite v1 taskset's tasks, materialized at ``start()``. None for legacy
        (dataset lives on the server) and for infinite tasksets (see ``task_iter``)."""
        self.task_iter: Iterator[vf.Task] | None = None
        """An infinite v1 taskset's generator — pulled per example, never materialized."""
        self._env_client: EnvClient | None = None
        self._env_server_process: BaseProcess | None = None

    @property
    def name(self) -> str:
        return self.config.resolved_name

    @property
    def env_client(self) -> EnvClient:
        if self._env_client is None:
            raise RuntimeError(f"Env {self.name} not started — call start() first.")
        return self._env_client

    async def start(self, log_dir: Path, log_level: str | None = None, json_logging: bool = False) -> None:
        """Spawn the env server (if needed), connect, and load the taskset client-side
        (legacy instead asks the server for ``info`` — its dataset is server-side)."""
        external = self.config.address is not None
        address = self.config.address or await self._spawn(log_dir, log_level or "INFO", json_logging)
        get_logger().debug(f"Connecting {self.name} to env server {address}")
        self._env_client = EnvClient(address=address)
        # A spawned server already reported its address *after* binding, so it's up. An
        # external server has no such handshake, so poll until it answers.
        if external:
            await self.env_client.wait_for_server_startup()
        if self.config.is_legacy:
            info = await self.env_client.info()
            self.num_tasks = info.num_tasks
            self.requires_group_scoring = info.requires_group_scoring
        else:
            taskset = vf.load_taskset(self.config.env.taskset)
            if type(taskset).INFINITE:
                self.task_iter = iter(taskset.load())
                self.num_tasks = None
            else:
                # Materialize off the event loop — load() may pull a dataset.
                self.tasks = await asyncio.to_thread(lambda: list(taskset.load()))
                self.num_tasks = len(self.tasks)
        num_tasks = self.num_tasks if self.num_tasks is not None else "infinite"
        get_logger().info(f"Env {self.name} ready: num_tasks={num_tasks} group_scoring={self.requires_group_scoring}")

    async def _spawn(self, log_dir: Path, log_level: str, json_logging: bool) -> str:
        """Spawn a v1 EnvServer child process (it runs the agents; the tasks come from
        us). The server binds an OS-assigned port (``:0``) and reports the concrete
        address back over a queue — no free-port guess, no TOCTOU race. Its output
        goes to ``<log_dir>/<name>.log`` (``log_dir`` is already the train/eval-split
        ``.../logs/envs/{train,eval}`` the orchestrator passes in)."""
        ctx = mp.get_context("spawn")
        address_queue: mp.Queue = ctx.Queue()
        log_file = log_dir / f"{self.name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        get_logger().debug(f"Spawning env server {self.name} (id={self.config.env_id}, log={log_file})")
        server_kwargs = (
            dict(
                legacy=True,
                env_id=self.config.env_id,
                env_args=self.config.args,
                extra_env_kwargs=self.config.extra_env_kwargs,
            )
            if self.config.is_legacy
            # Picklable dict — the narrowed config class doesn't survive the spawn.
            else dict(legacy=False, config_data=env_config_data(self.config.env))
        )
        process = ctx.Process(
            target=_run_env_server,
            kwargs=dict(
                log_file=str(log_file),
                log_level=log_level,
                json_logging=json_logging,
                **vf.pool_serve_kwargs(self.config.pool),
                address="tcp://127.0.0.1:0",
                address_queue=address_queue,
                **server_kwargs,
            ),
            daemon=False,
        )
        process.start()
        self._env_server_process = process
        try:
            address = await asyncio.to_thread(address_queue.get, timeout=ENV_SERVER_SPAWN_TIMEOUT)
        except queue.Empty:
            raise RuntimeError(f"Env server {self.name} did not report its address within {ENV_SERVER_SPAWN_TIMEOUT}s")
        finally:
            address_queue.close()
            address_queue.join_thread()
        get_logger().debug(f"Env server {self.name} bound at {address}")
        return address

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
        task_data: dict | None = None,
        task_idx: int | None = None,
    ) -> list[Rollout]:
        """Run one episode; return its typed Traces. A v1 env takes the task itself
        (``task_data``); the legacy bridge is addressed by dataset row (``task_idx``).
        A zero-trace episode raises (the dispatcher synthesizes the error marker); a
        not-``ok`` episode marks its clean traces failed so partial episodes never
        train."""
        episode = await self.env_client.run(
            task_data=task_data,
            task_idx=task_idx,
            client=client,
            model=model_name,
            sampling=self._sampling(cache_salt),
        )
        if not episode.traces:
            error = episode.error
            detail = f"{error.type}: {error.message}" if error is not None else "no traces and no error recorded"
            raise RuntimeError(f"env-rollout failed before any trace was minted — {detail}")
        rollouts = [ROLLOUT_TYPE.model_construct(**dict(wire)) for wire in episode.traces]
        for rollout in rollouts:
            rollout.episode_id = episode.id
            if not episode.ok and rollout.ok:
                error = episode.error or vf.Error(
                    type="EpisodeFailed", message="A sibling trace in this episode failed"
                )
                rollout.errors = [*rollout.errors, error]
                rollout.ok = False
        return rollouts

    async def run_group(
        self, client: vf.ClientConfig, task_idx: int, model_name: str, group_size: int, cache_salt: str | None
    ) -> list[Rollout]:
        """Run a group of rollouts for ``task_idx`` (group-scoring envs); return typed Traces."""
        wires = await self.env_client.run_group(
            task_idx=task_idx,
            n=group_size,
            client=client,
            model=model_name,
            sampling=self._sampling(cache_salt),
        )
        return [ROLLOUT_TYPE.model_construct(**dict(wire)) for wire in wires]

    def shutdown(self) -> None:
        if self._env_server_process is None:
            return
        self._env_server_process.terminate()
        self._env_server_process = None


class TrainEnv(Env):
    config: TrainEnvConfig

    def __init__(self, config: TrainEnvConfig, sampler: Sampler, algorithm: Algorithm):
        super().__init__(config)
        self.sampler = sampler
        self.algorithm = algorithm
        self.sampling_args = sampler.sampling_args(config.sampling.to_sampling_args())


class EvalEnv(Env):
    config: EvalEnvConfig

    def __init__(self, config: EvalEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()
        self.examples: list[dict] = []

    async def start(self, log_dir: Path, log_level: str | None = None, json_logging: bool = False) -> None:
        await super().start(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        n = self.config.num_examples
        if self.num_tasks is None:
            if n < 0:
                raise ValueError(f"Eval env {self.name} has an infinite taskset — set num_examples to bound it")
            assert self.task_iter is not None
            # A fixed eval set off the generator, pulled once and reused every epoch.
            tasks = list(islice(self.task_iter, n))
        elif self.tasks is not None:
            tasks = self.tasks if n < 0 else self.tasks[:n]
        else:  # legacy: the dataset lives on the server — address it by row
            count = self.num_tasks if n < 0 else min(n, self.num_tasks)
            self.examples = [{"task_idx": i} for i in range(count)]
            return
        self.examples = [{"task_idx": task.data.idx, "task": task} for task in tasks]


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

    async def start(self, log_dir: Path, log_level: str | None = None, json_logging: bool = False) -> None:
        """Spawn env servers (where needed) and connect, one at a time. Each server
        binds an OS-assigned port and reports it back, so there's no port race."""
        for env in self:
            await env.start(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Terminate all spawned env server processes."""
        processes = [env._env_server_process for env in self if env._env_server_process is not None]
        if not processes:
            return
        logger = get_logger()
        logger.debug(f"Shutting down {len(processes)} env server(s)")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=25)
            if p.is_alive():
                logger.warning(f"Env server {p.pid} did not exit after 25s, force killing")
                p.kill()
                p.join(timeout=5)
        for env in self:
            env._env_server_process = None


class TrainEnvs(Envs[TrainEnv]):
    """Collection of training environments, each paired with its rollout
    :class:`Sampler` and runtime :class:`Algorithm`, built from the env's
    resolved algorithm config."""

    def __init__(self, configs: Sequence[TrainEnvConfig], *, policy_pool, renderer_config=None):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            assert config.algo is not None, "TrainEnvConfig.algo must be resolved before env construction"
            env = TrainEnv(
                config,
                Sampler(config.algo.sampling, policy_pool, renderer_config),
                build_algorithm(config.algo, policy_pool),
            )
            self._envs[env.name] = env


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(self, configs: Sequence[EvalEnvConfig]):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            env = EvalEnv(config)
            self._envs[env.name] = env
