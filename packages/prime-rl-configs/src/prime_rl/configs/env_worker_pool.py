from pathlib import Path
from typing import Annotated, Literal

import verifiers.v1 as vf
from pydantic import Field, SerializeAsAny, model_validator

from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig, default_output_dir


class StaticPoolConfig(BaseConfig):
    """Fixed env worker pool: pre-spawn `num_workers` workers up front."""

    type: Literal["static"] = "static"
    num_workers: int = Field(4, ge=1)
    """Worker processes to pre-spawn."""


class ElasticPoolConfig(BaseConfig):
    """Elastic env worker pool: start at one worker and scale up on demand."""

    type: Literal["elastic"] = "elastic"
    max_workers: int | None = Field(None, ge=1)
    """Upper bound on workers (None = unbounded)."""
    multiplex: int = Field(128, ge=1)
    """Episodes per worker for the scale-up trigger: add a worker once in-flight episodes
    reach 90% of `workers * multiplex`."""


# Discriminated on `type` so the CLI selects with `--serve.pool.type static|elastic`.
PoolConfig = Annotated[StaticPoolConfig | ElasticPoolConfig, Field(discriminator="type")]


class ServeConfig(BaseConfig):
    """The `[serve]` block: the worker pool, the ZMQ address, and each worker's
    episode bound."""

    pool: PoolConfig = Field(default_factory=ElasticPoolConfig)
    """Worker-pool sizing. `elastic` (default) starts at one worker and scales up on
    demand; `static` pre-spawns a fixed `num_workers`."""
    address: str | None = None
    """ZMQ address the pool's broker binds (and clients connect to). None binds an
    OS-assigned loopback port and publishes it through the pool's address file."""
    max_concurrent: int | None = Field(None, ge=1)
    """Episodes in flight per worker (None = unbounded). How many agent runs one episode
    carries is the env's own (`--env.max-concurrent-agents`, one at a time by default)."""


class EnvWorkerPoolConfig(BaseConfig):
    """``uv run env-worker-pool``: one source's env and its broker/worker pool replica.
    Launchers write one config per managed train/eval source. Each replica's broker
    publishes its bound address to ``address_file``; its workers execute episodes."""

    env: SerializeAsAny[vf.EnvConfig] = vf.SingleAgentEnvConfig()
    """The environment — which env, its seed taskset, each agent, its knobs. Narrowed to the selected env's config class by the env id, else the taskset id."""

    serve: ServeConfig = ServeConfig()
    """How it's served: the worker pool, the bind address, each worker's episode bound."""

    address_file: Path | None = None
    """Publish the bound address to this file once serving. Set by the launchers for the
    pools they spawn, so the address never has to be agreed on up front (``serve.address``
    unset binds an OS-assigned port)."""

    store_dir: Path | None = None
    """Keep finished episodes' training arrays as files in a directory of this pool's
    under ``store_dir``, served by one rollout store for the whole pool, instead of in
    each worker's memory. The directory is local to the node and deleted when the pool
    exits; a pool that died without deleting it is cleaned up by the next pool that starts
    on the node. Set by the ``rl`` launcher for ``rollout_transport.retain``."""

    log: LogConfig = LogConfig()

    output_dir: Path = Field(default_factory=default_output_dir)
    """Directory to write outputs to — logs and any generated artifacts are written as subdirectories. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        """Narrow ``env`` to the selected env's config class."""
        return vf.resolve_env_field(data, vf.narrowed_env_annotation(cls))

    @property
    def env_id(self) -> str:
        return self.env.env_id or ""

    @model_validator(mode="after")
    def validate_env(self):
        if not self.env_id:
            raise ValueError('no env configured — set env = { taskset = { id = "<id>" } }')
        return self
