"""Plain-Python class-object registry for glyphbench environments."""

from __future__ import annotations

from typing import Any

from glyphbench.core.base_env import BaseGlyphEnv

REGISTRY: dict[str, type[BaseGlyphEnv]] = {}
DEFAULT_USE_MEMORY: dict[str, bool] = {}


def register_env(
    env_id: str,
    cls: type[BaseGlyphEnv],
    *,
    default_use_memory: bool | None = None,
) -> None:
    """Register a class under an env id.

    Idempotent for the same (id, class) pair; raises ``ValueError`` on
    conflicting registrations and ``TypeError`` if ``cls`` is not a
    ``BaseGlyphEnv`` subclass.

    ``default_use_memory`` is task-level rollout metadata for the verifiers
    integration. ``None`` falls back to ``cls.default_use_memory``.
    """
    if not isinstance(cls, type) or not issubclass(cls, BaseGlyphEnv):
        raise TypeError(
            f"register_env expected a BaseGlyphEnv subclass, got {cls!r}"
        )
    resolved_default = (
        bool(getattr(cls, "default_use_memory", False))
        if default_use_memory is None
        else bool(default_use_memory)
    )
    existing = REGISTRY.get(env_id)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"env_id {env_id!r} already registered to {existing.__name__}; "
            f"refusing to overwrite with {cls.__name__}"
        )
    existing_default = DEFAULT_USE_MEMORY.get(env_id)
    if (
        existing is cls
        and existing_default is not None
        and existing_default != resolved_default
    ):
        raise ValueError(
            f"env_id {env_id!r} already registered with default_use_memory="
            f"{existing_default}; refusing to overwrite with {resolved_default}"
        )
    REGISTRY[env_id] = cls
    DEFAULT_USE_MEMORY[env_id] = resolved_default


def make_env(env_id: str, **kwargs: Any) -> BaseGlyphEnv:
    """Instantiate the class registered under ``env_id``.

    Extra kwargs are forwarded to the class constructor.
    """
    cls = REGISTRY.get(env_id)
    if cls is None:
        raise KeyError(
            f"unknown env_id {env_id!r}; known ids: {sorted(REGISTRY)[:5]}…"
        )
    return cls(**kwargs)


def all_glyphbench_env_ids() -> list[str]:
    """Return every registered id as a sorted list."""
    return sorted(REGISTRY)


def default_use_memory(env_id: str) -> bool:
    """Return the registered memory-scaffold default for ``env_id``."""
    if env_id not in REGISTRY:
        raise KeyError(
            f"unknown env_id {env_id!r}; known ids: {sorted(REGISTRY)[:5]}…"
        )
    return DEFAULT_USE_MEMORY[env_id]


def all_task_memory_defaults() -> dict[str, bool]:
    """Return a sorted copy of task id -> default memory-mode setting."""
    return {env_id: DEFAULT_USE_MEMORY[env_id] for env_id in sorted(REGISTRY)}
