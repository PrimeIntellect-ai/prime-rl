"""Core contracts shared by every env and the verifiers integration."""

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import (
    REGISTRY,
    all_glyphbench_env_ids,
    all_task_memory_defaults,
    default_use_memory,
    make_env,
    register_env,
)
from glyphbench.core.task_selection import list_task_ids

__all__ = [
    "ActionSpec",
    "BaseGlyphEnv",
    "GridObservation",
    "REGISTRY",
    "all_glyphbench_env_ids",
    "all_task_memory_defaults",
    "default_use_memory",
    "list_task_ids",
    "make_env",
    "register_env",
]
