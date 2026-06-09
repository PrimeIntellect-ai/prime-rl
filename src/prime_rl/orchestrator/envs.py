from __future__ import annotations

import atexit
import multiprocessing as mp
from collections.abc import Iterator, Sequence
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Generic, TypeVar

import verifiers as vf
from verifiers.serve import ZMQEnvClient, ZMQEnvServer
from verifiers.utils.serve_utils import get_free_port

from prime_rl.configs.orchestrator import EnvConfig, EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.algorithms import Algorithm, ModelRegistry
from prime_rl.utils.logger import get_logger

REQUIRED_STATE_COLUMNS = ["trajectory"]


class Env:
    """Wraps a vf.Environment - only exposes features used in PRIME-RL."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.sampling_args: dict = {}

        get_logger().debug(f"Initializing {config.resolved_name} ({config})")
        self._env: vf.Environment = vf.load_environment(config.stripped_id, **config.args)
        self._env_client: ZMQEnvClient | None = None
        self._env_server_process: BaseProcess | None = None

    @property
    def name(self) -> str:
        return self.config.resolved_name

    @property
    def env(self) -> vf.Environment:
        return self._env

    @property
    def env_client(self) -> ZMQEnvClient:
        if not self._env_client:
            raise RuntimeError(
                f"Env {self.name} has no env client connected. Call connect() first to connect to an env server."
            )
        return self._env_client

    @property
    def requires_group_scoring(self) -> bool:
        return any(self.env.rubric._is_group_func(func) for func in self.env.rubric._get_reward_funcs())

    async def start(
        self,
        log_dir: Path,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> None:
        """Spawn an env server (if needed) and connect to it."""
        if self.config.address is None:
            address = self._spawn(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        else:
            address = self.config.address
        get_logger().debug(f"Connecting {self.name} to env server {address}")
        self._env_client = ZMQEnvClient(address=address, name=self.name)
        await self.env_client.wait_for_server_startup()

    def _spawn(
        self,
        log_dir: Path,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> str:
        assert isinstance(self.config.num_workers, int), (
            f"num_workers must be resolved before spawn, got {self.config.num_workers!r}"
        )
        num_workers = self.config.num_workers
        address = f"tcp://127.0.0.1:{get_free_port()}"
        get_logger().debug(f"Spawning env server {self.name} ({address=}, {num_workers=})")
        process = mp.get_context("spawn").Process(
            target=ZMQEnvServer.run_server,
            args=(
                self.config.stripped_id,
                self.config.args,
                self.config.extra_env_kwargs,
                log_level,
                (log_dir / self.name).as_posix(),
            ),
            kwargs=dict(
                address=address,
                json_logging=json_logging,
                console_logging=False,
                num_workers=num_workers,
            ),
            daemon=False,
        )
        process.start()
        self._env_server_process = process
        return address

    def _sampling_args_with_salt(self, cache_salt: str | None) -> dict:
        sampling_args = {**self.sampling_args}
        if cache_salt is None:
            return sampling_args
        extra_body = {**sampling_args.get("extra_body", {}), "cache_salt": cache_salt}
        sampling_args["extra_body"] = extra_body
        return sampling_args

    @property
    def state_columns(self) -> list[str]:
        """Required columns plus any extras configured on the env, deduped (required first)."""
        merged: list[str] = []
        for col in (*REQUIRED_STATE_COLUMNS, *self.config.state_columns):
            if col not in merged:
                merged.append(col)
        return merged

    async def run_rollout(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        cache_salt: str | None,
    ) -> vf.RolloutOutput:
        """Run a single rollout for an example."""
        return await self.env.run_rollout(
            vf.RolloutInput(**example),
            client=client,
            model=model_name,
            sampling_args=self._sampling_args_with_salt(cache_salt),
            max_retries=self.config.max_retries,
            state_columns=self.state_columns,
            env_client=self.env_client,
        )

    async def run_group(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        group_size: int,
        cache_salt: str | None,
    ) -> list[vf.RolloutOutput]:
        """Run a group of rollouts for an example. Required for group-scoring envs."""
        return await self.env.run_group(
            [vf.RolloutInput(**example) for _ in range(group_size)],
            client=client,
            model=model_name,
            sampling_args=self._sampling_args_with_salt(cache_salt),
            max_retries=self.config.max_retries,
            state_columns=self.state_columns,
            env_client=self.env_client,
        )

    def shutdown(self) -> None:
        if self._env_server_process is None:
            return
        self._env_server_process.terminate()
        self._env_server_process = None


class TrainEnv(Env):
    config: TrainEnvConfig

    def __init__(self, config: TrainEnvConfig, algorithm: Algorithm):
        super().__init__(config)
        self.algorithm = algorithm
        self.sampling_args = algorithm.sampling_args(config.sampling.to_sampling_args())

    def get_dataset(self, seed: int | None = None):
        return self.env.get_dataset(seed=seed)


class EvalEnv(Env):
    config: EvalEnvConfig

    def __init__(self, config: EvalEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()
        self.examples = self.env.get_eval_dataset(n=config.num_examples).to_list()


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

    async def start(
        self,
        log_dir: Path,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> None:
        """Spawn env servers (where needed) and connect env clients one at a time.

        Serialized to avoid a TOCTOU port race: get_free_port() only holds the port
        until it returns, so parallel spawns can hand the same port to two children.
        """
        for env in self:
            await env.start(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Terminate all spawned env server processes in parallel."""
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
    """Collection of training environments, each paired with its runtime
    :class:`Algorithm` built from the env's resolved algorithm config."""

    def __init__(self, configs: Sequence[TrainEnvConfig], *, registry: ModelRegistry, tokenizer):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            assert config.algorithm is not None, "TrainEnvConfig.algorithm must be resolved before env construction"
            env = TrainEnv(config, Algorithm(config.algorithm, registry, tokenizer))
            self._envs[env.name] = env


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(self, configs: Sequence[EvalEnvConfig]):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            env = EvalEnv(config)
            self._envs[env.name] = env
