from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field, SerializeAsAny, model_validator

from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig, default_output_dir


class EnvServerConfig(BaseConfig):
    """``uv run env-server``: what to serve (``[env]``) and how it's hosted (``[serve]``).
    The ``rl`` launcher writes one of these per train/eval source, with ``serve.address``
    set to the source's derived address."""

    env: SerializeAsAny[vf.EnvConfig] = vf.SingleAgentEnvConfig()
    """The environment — which env, its seed taskset, each agent, its knobs. Narrowed to the selected env's config class by the env id, else the taskset id."""

    serve: vf.ServeConfig = vf.ServeConfig()
    """How it's served: the worker pool, the bind address, each worker's episode bound."""

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
