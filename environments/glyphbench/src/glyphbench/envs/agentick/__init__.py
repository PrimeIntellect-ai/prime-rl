"""AgenticK task adapters."""

from __future__ import annotations

from glyphbench.core.registry import register_env
from glyphbench.envs.agentick.env import AGENTICK_ENV_CLASSES

for _cls in AGENTICK_ENV_CLASSES:
    register_env(
        f"glyphbench/agentick-{_cls.task_slug}-{_cls.difficulty}-v0",
        _cls,
        default_use_memory=False,
    )

__all__ = ["AGENTICK_ENV_CLASSES"]
