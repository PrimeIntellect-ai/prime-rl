"""Env wrappers over a vf-nano env server.

Each ``Env`` owns a vf-nano ``EnvServer`` (spawned as a child process, or an
external one given by ``config.address``) and an ``EnvClient`` to drive it. The
orchestrator never loads an environment: it asks the server for ``info``
(``num_tasks`` + whether group scoring is needed), then runs rollouts purely by
**task index**. The server returns a ``Trace`` dict, which we adapt into the
RolloutOutput-shaped dict the rest of the orchestrator consumes.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import socket
from collections.abc import Iterator, Sequence
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Generic, TypeVar

import verifiers.nano as vf
from verifiers.nano.serve import EnvClient, EnvServer

from prime_rl.configs.orchestrator import EnvConfig, EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.trajectories import trace_to_output
from prime_rl.utils.logger import get_logger


def _get_free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class Env:
    """Wraps a vf-nano env server + client. The orchestrator never loads the env."""

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
        address = self.config.address or self._spawn()
        get_logger().debug(f"Connecting {self.name} to env server {address}")
        self._env_client = EnvClient(address=address, name=self.name)
        await self.env_client.wait_for_server_startup()
        info = await self.env_client.info()
        self.num_tasks = info["num_tasks"]
        self.requires_group_scoring = info["requires_group_scoring"]
        get_logger().info(
            f"Env {self.name} ready: num_tasks={self.num_tasks} group_scoring={self.requires_group_scoring}"
        )

    def _spawn(self) -> str:
        """Spawn a vf-nano EnvServer child process (it loads the env; we never do)."""
        address = f"tcp://127.0.0.1:{_get_free_port()}"
        get_logger().debug(f"Spawning env server {self.name} ({address=}, id={self.config.stripped_id})")
        process = mp.get_context("spawn").Process(
            target=EnvServer.run_server,
            kwargs=dict(
                env_id=self.config.stripped_id,
                taskset_args=self.config.args,
                agent_config=self.config.agent,
                agent_timeout=self.config.timeout,
                max_turns=self.config.max_turns,
                address=address,
            ),
            daemon=False,
        )
        process.start()
        self._env_server_process = process
        return address

    def _sampling(self, cache_salt: str | None) -> dict:
        sampling = {**self.sampling_args}
        if cache_salt is not None:
            sampling["extra_body"] = {**sampling.get("extra_body", {}), "cache_salt": cache_salt}
        return sampling

    async def run_rollout(
        self, client: vf.ClientConfig, example: dict, model_name: str, cache_salt: str | None
    ) -> dict:
        """Run a single rollout for ``example`` (by task index); return an adapted Trace dict."""
        task_idx = example["example_id"]
        trace = await self.env_client.run_rollout(
            task_idx=task_idx,
            client_config=client.model_dump(),
            model=model_name,
            sampling=self._sampling(cache_salt),
            timeout=self.config.timeout,
        )
        return trace_to_output(trace, task_idx)

    async def run_group(
        self, client: vf.ClientConfig, example: dict, model_name: str, group_size: int, cache_salt: str | None
    ) -> list[dict]:
        """Run a group of rollouts for ``example`` (group-scoring envs)."""
        task_idx = example["example_id"]
        traces = await self.env_client.run_group(
            task_idx=task_idx,
            n=group_size,
            client_config=client.model_dump(),
            model=model_name,
            sampling=self._sampling(cache_salt),
            timeout=self.config.timeout,
        )
        return [trace_to_output(trace, task_idx) for trace in traces]

    def shutdown(self) -> None:
        if self._env_server_process is None:
            return
        self._env_server_process.terminate()
        self._env_server_process = None


class TrainEnv(Env):
    config: TrainEnvConfig

    def __init__(self, config: TrainEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()


class EvalEnv(Env):
    config: EvalEnvConfig

    def __init__(self, config: EvalEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()
        self.examples: list[dict] = []

    async def start(self, log_dir: Path, log_level: str | None = None, json_logging: bool = False) -> None:
        await super().start(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        n = self.num_tasks if self.config.num_examples < 0 else min(self.config.num_examples, self.num_tasks)
        self.examples = [{"example_id": i} for i in range(n)]


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
        """Spawn env servers (where needed) and connect, one at a time.

        Serialized to avoid a TOCTOU port race: ``_get_free_port`` only holds the
        port until it returns, so parallel spawns could hand the same port out twice.
        """
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

    def __init__(self, configs: Sequence[TrainEnvConfig]):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            env = TrainEnv(config)
            self._envs[env.name] = env


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(self, configs: Sequence[EvalEnvConfig]):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            env = EvalEnv(config)
            self._envs[env.name] = env
