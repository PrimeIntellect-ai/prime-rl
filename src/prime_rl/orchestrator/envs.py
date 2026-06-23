"""Env wrappers over a v1 env server.

Each ``Env`` owns a v1 ``EnvServer`` (spawned as a child process, or an
external one given by ``config.address``) and an ``EnvClient`` to drive it. The
orchestrator never *runs* an environment: it asks the server for ``info``
(``num_tasks`` + whether group scoring is needed), then runs rollouts purely by
**task index**. The server returns a ``Trace`` (a plain ``model_dump`` — derived values are
properties, not serialized) which we validate into a ``Trace[WireTask]`` — a real ``vf.Trace``
(never a loose dict) whose task keeps the env's
task-specific fields as extras (``WireTask`` allows them). The orchestrator never imports the
env package: the env's *type* and *runtime* both live only in the server, and the orchestrator
drives it purely by task index. (Nothing here reads typed env task fields — only ``task.idx``
and a full ``task.model_dump``, both of which ``WireTask`` preserves.)
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import os
import queue
import sys
from collections.abc import Iterator, Sequence
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Generic, TypeVar

import verifiers.v1 as vf
from verifiers.v1.serve import EnvClient

from prime_rl.configs.orchestrator import EnvConfig, EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.advantage import AdvantageFn, setup_advantage_fn
from prime_rl.orchestrator.types import Rollout
from prime_rl.utils.logger import get_logger

# Every wire trace validates into this type. WireTask (extra="allow") keeps the env's task
# fields without importing the env package — the orchestrator never reads them typed (only
# task.idx + task.model_dump).
ROLLOUT_TYPE = Rollout[vf.WireTask]

# Max wait for a spawned env server to bind and report its address. The child
# loads the taskset (possibly downloading a dataset) before reporting, so this
# is generous.
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
    """Wraps a v1 env server + client. The orchestrator never loads the env."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.sampling_args: dict = {}
        self.num_tasks: int = 0
        self.requires_group_scoring: bool = False
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
        """Spawn the env server (if needed), connect, and cache its ``info``."""
        external = self.config.address is not None
        address = self.config.address or await self._spawn(log_dir, log_level or "INFO", json_logging)
        get_logger().debug(f"Connecting {self.name} to env server {address}")
        self._env_client = EnvClient(address=address)
        # A spawned server already reported its address *after* binding + loading,
        # so it's up — the untimed ``info`` below is enough. An external server has
        # no such handshake, so poll until it answers before we block on ``info``.
        if external:
            await self.env_client.wait_for_server_startup()
        info = await self.env_client.info()
        self.num_tasks = info.num_tasks
        self.requires_group_scoring = info.requires_group_scoring
        get_logger().info(
            f"Env {self.name} ready: num_tasks={self.num_tasks} group_scoring={self.requires_group_scoring}"
        )

    async def _spawn(self, log_dir: Path, log_level: str, json_logging: bool) -> str:
        """Spawn a v1 EnvServer child process (it loads the env; we never do).
        The server binds an OS-assigned port (``:0``) and reports the concrete
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
            else dict(legacy=False, config=self.config)
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

    async def run_rollout(
        self, client: vf.ClientConfig, task_idx: int, model_name: str, cache_salt: str | None
    ) -> Rollout:
        """Run a single rollout for ``task_idx``; return a typed Trace."""
        wire = await self.env_client.run_rollout(
            task_idx=task_idx,
            client=client,
            model=model_name,
            sampling=self._sampling(cache_salt),
        )
        return ROLLOUT_TYPE.model_construct(**dict(wire))

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

    def __init__(self, config: TrainEnvConfig, max_seq_len: int):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()
        # Built once — custom advantage funcs do an ``import_object`` we don't want to pay per group.
        self.advantage_fn: AdvantageFn | None = (
            setup_advantage_fn(config.advantage, max_seq_len=max_seq_len) if config.advantage is not None else None
        )


class EvalEnv(Env):
    config: EvalEnvConfig

    def __init__(self, config: EvalEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()
        self.examples: list[dict] = []

    async def start(self, log_dir: Path, log_level: str | None = None, json_logging: bool = False) -> None:
        await super().start(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        n = self.num_tasks if self.config.num_examples < 0 else min(self.config.num_examples, self.num_tasks)
        self.examples = [{"task_idx": i} for i in range(n)]


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
    """Collection of training environments."""

    def __init__(self, configs: Sequence[TrainEnvConfig], max_seq_len: int):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            env = TrainEnv(config, max_seq_len=max_seq_len)
            self._envs[env.name] = env


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(self, configs: Sequence[EvalEnvConfig]):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            env = EvalEnv(config)
            self._envs[env.name] = env
