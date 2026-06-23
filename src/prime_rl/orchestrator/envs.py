"""Env wrappers over a v1 env server.

Each ``Env`` attaches to an external v1 ``EnvServer`` (given by ``config.address``,
assigned by the launcher or set manually) and drives it through an ``EnvClient``. The
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

from collections.abc import Iterator, Sequence
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


class Env:
    """Wraps a v1 env server + client. The orchestrator never loads the env."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.sampling_args: dict = {}
        self.num_tasks: int = 0
        self.requires_group_scoring: bool = False
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
        """Connect to the env server at ``config.address`` and cache its ``info``."""
        if self.config.address is None:
            raise ValueError(
                f"Env {self.name} has no address. Set `address` in the env config (e.g. "
                "tcp://127.0.0.1:5000) pointing at a running `env-server`, or launch via "
                "`uv run rl`, which assigns one automatically."
            )
        get_logger().debug(f"Connecting {self.name} to env server {self.config.address}")
        self._env_client = EnvClient(address=self.config.address)
        await self.env_client.wait_for_server_startup()
        info = await self.env_client.info()
        self.num_tasks = info.num_tasks
        self.requires_group_scoring = info.requires_group_scoring
        get_logger().info(
            f"Env {self.name} ready: num_tasks={self.num_tasks} group_scoring={self.requires_group_scoring}"
        )

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

    async def start(self) -> None:
        await super().start()
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

    async def start(self) -> None:
        """Connect to each env server (assigned by the launcher or set manually)."""
        for env in self:
            await env.start()


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
