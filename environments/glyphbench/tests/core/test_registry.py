"""Tests for the plain-Python class-object registry (no gym)."""

from __future__ import annotations

import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import (
    DEFAULT_USE_MEMORY,
    REGISTRY,
    all_glyphbench_env_ids,
    default_use_memory,
    make_env,
    register_env,
)


class _E(BaseGlyphEnv):
    action_spec = ActionSpec(names=("A",), descriptions=("a",))

    def _reset(self, seed: int) -> GridObservation:
        return GridObservation(grid=".", legend="", hud="", message="")
    def _step(self, action: int):
        return GridObservation(grid=".", legend="", hud="", message=""), 0.0, True, False, {}
    def _render_current_observation(self) -> GridObservation:
        return GridObservation(grid=".", legend="", hud="", message="")
    def system_prompt(self) -> str:
        return ""
    def env_id(self) -> str:
        return "test/e-v0"


class _F(BaseGlyphEnv):
    action_spec = ActionSpec(names=("A",), descriptions=("a",))

    def _reset(self, seed: int) -> GridObservation:
        return GridObservation(grid=".", legend="", hud="", message="")
    def _step(self, action: int):
        return GridObservation(grid=".", legend="", hud="", message=""), 0.0, True, False, {}
    def _render_current_observation(self) -> GridObservation:
        return GridObservation(grid=".", legend="", hud="", message="")
    def system_prompt(self) -> str:
        return ""
    def env_id(self) -> str:
        return "test/f-v0"


@pytest.fixture(autouse=True)
def _save_restore_registry():
    snapshot = dict(REGISTRY)
    memory_snapshot = dict(DEFAULT_USE_MEMORY)
    yield
    REGISTRY.clear()
    REGISTRY.update(snapshot)
    DEFAULT_USE_MEMORY.clear()
    DEFAULT_USE_MEMORY.update(memory_snapshot)


def test_register_and_make():
    register_env("test/e-v0", _E)
    env = make_env("test/e-v0")
    assert isinstance(env, _E)
    assert default_use_memory("test/e-v0") is False


def test_register_idempotent_same_class():
    register_env("test/e-v0", _E, default_use_memory=True)
    register_env("test/e-v0", _E, default_use_memory=True)  # idempotent, no error
    assert default_use_memory("test/e-v0") is True


def test_register_rejects_conflicting_memory_default():
    register_env("test/e-v0", _E, default_use_memory=True)
    with pytest.raises(ValueError, match="default_use_memory"):
        register_env("test/e-v0", _E, default_use_memory=False)


def test_register_rejects_duplicate_with_different_class():
    register_env("test/e-v0", _E)
    with pytest.raises(ValueError, match="already registered"):
        register_env("test/e-v0", _F)


def test_make_unknown_raises():
    with pytest.raises(KeyError, match="unknown env_id"):
        make_env("test/not-there-v0")


def test_make_forwards_kwargs():
    register_env("test/e-v0", _E)
    env = make_env("test/e-v0", max_turns=7)
    assert env.max_turns == 7


def test_all_env_ids_sorted():
    register_env("test/b-v0", _E)
    register_env("test/a-v0", _F)
    ids = all_glyphbench_env_ids()
    assert ids == sorted(ids)


def test_register_rejects_non_baseclass():
    class NotAnEnv:
        pass
    with pytest.raises(TypeError):
        register_env("test/bad-v0", NotAnEnv)  # type: ignore[arg-type]
