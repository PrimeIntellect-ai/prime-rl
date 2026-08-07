from pathlib import Path

import verifiers.v1 as vf
from pydantic import SerializeAsAny, model_validator

from prime_rl.configs.shared import LogConfig, MultimodalConfig
from prime_rl.utils.config import BaseConfig


class EnvServerConfig(BaseConfig):
    """``uv run env-server``: what to serve (``[env]``, or ``[legacy]`` for a classic v0
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

    multimodal: MultimodalConfig = MultimodalConfig()
    """Raw multimodal image offload settings; with ``output_dir``, resolves the run image-asset dir this server offloads into."""

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

    @model_validator(mode="after")
    def validate_env(self):
        # A v0 id next to any v1 env identity leaves one of the two going nowhere, and
        # which one depends on `is_legacy`: a taskset makes it False, so the v0 env never
        # loads; a bare `env.id` leaves it True, so the v0 env runs under the v1 name.
        if self.legacy.id is not None and self.env.env_id:
            if self.env.taskset.id:
                raise ValueError(
                    f"legacy.id {self.legacy.id!r} is a classic (v0) env and can't combine with "
                    f"the v1 taskset {self.env.taskset.id!r}. Pairing a reusable env with a taskset "
                    f"is env.id = {self.legacy.id!r}; to run the v0 env instead, drop the taskset."
                )
            raise ValueError(
                f"legacy.id {self.legacy.id!r} is a classic (v0) env and can't combine with the "
                f"v1 env.id {self.env.id!r}: the v0 env is what would run, stamped with the v1 "
                "env's name. Keep whichever one you meant to run."
            )
        if not self.env_id:
            raise ValueError(
                'no env configured — set env = { taskset = { id = "<id>" } } (v1) or legacy = { id = "<id>" } (v0)'
            )
        return self
