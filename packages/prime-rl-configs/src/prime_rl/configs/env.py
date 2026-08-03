from pathlib import Path

import verifiers.v1 as vf
from pydantic import SerializeAsAny, model_validator

from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig


class EnvConfig(BaseConfig):
    """``uv run env``: what to serve (``[env]``, or ``[legacy]`` for a classic v0
    env) and how it's hosted (``[serve]``). The ``rl`` launcher writes one of these per
    train/eval source, with ``serve.address`` set to the source's derived address."""

    env: SerializeAsAny[vf.EnvConfig] = vf.SingleAgentEnvConfig()
    """The environment — which env, its seed taskset, each agent, its knobs. Narrowed to the selected env's config class by the env id, else the taskset id."""

    serve: vf.ServeConfig = vf.ServeConfig()
    """How it's served: the worker pool, the bind address, each worker's episode bound."""

    legacy: vf.LegacyEnvConfig = vf.LegacyEnvConfig()
    """A classic (v0) environment to serve through the bridge instead of ``env``."""

    log: LogConfig = LogConfig()

    output_dir: Path = Path("outputs")
    """Directory to write outputs to — logs and any generated artifacts are written as subdirectories."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        """Narrow ``env`` to the selected env's config class."""
        return vf.resolve_env_field(data, vf.narrowed_env_annotation(cls))

    @property
    def is_legacy(self) -> bool:
        """Whether this serves the v0 bridge: a legacy id and no v1 taskset."""
        return self.legacy.id is not None and not self.env.taskset.id

    @property
    def env_id(self) -> str:
        """The served env's identifier: the v1 env's, else the v0 env id."""
        return self.env.env_id or self.legacy.id or ""
