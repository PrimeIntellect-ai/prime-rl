from __future__ import annotations

import pytest

import glyphbench  # noqa: F401
from glyphbench.core.registry import (
    DEFAULT_USE_MEMORY,
    REGISTRY,
    all_task_memory_defaults,
    default_use_memory,
)
from glyphbench.verifiers_integration.env import load_environment


def test_every_registered_task_has_memory_default() -> None:
    assert set(DEFAULT_USE_MEMORY) == set(REGISTRY)
    assert all(isinstance(v, bool) for v in DEFAULT_USE_MEMORY.values())
    assert all_task_memory_defaults() == {
        env_id: DEFAULT_USE_MEMORY[env_id] for env_id in sorted(REGISTRY)
    }


@pytest.mark.parametrize(
    ("env_id", "expected"),
    [
        ("glyphbench/classics-snake-medium-v0", False),
        ("glyphbench/classics-memorymatch-easy-v0", True),
        ("glyphbench/minigrid-empty-5x5-v0", False),
        ("glyphbench/minigrid-memory-s7-v0", True),
        ("glyphbench/minihack-river-lava-v0", False),
        ("glyphbench/minihack-mazewalk-9x9-v0", True),
        ("glyphbench/minihack-memento-f4-v0", True),
        ("glyphbench/craftax-dungeon-v0", True),
        ("glyphbench/craftax-craftchain-v0", True),
        ("glyphbench/craftax-fightzombie-v0", False),
        ("glyphbench/craftax-wave-defense-v0", False),
        ("glyphbench/procgen-heist-v0", True),
        ("glyphbench/procgen-maze-v0", True),
        ("glyphbench/procgen-miner-v0", False),
        ("glyphbench/procgen-bossfight-v0", False),
        ("glyphbench/miniatari-spaceinvaders-v0", False),
    ],
)
def test_representative_task_memory_defaults(env_id: str, expected: bool) -> None:
    assert default_use_memory(env_id) is expected


def test_load_environment_keeps_task_default_as_override_none() -> None:
    env = load_environment(
        task_id=[
            "glyphbench/classics-snake-medium-v0",
            "glyphbench/craftax-dungeon-v0",
        ],
        num_episodes=1,
    )
    assert env._use_memory_override is None  # type: ignore[attr-defined]
    assert env._resolve_use_memory("glyphbench/classics-snake-medium-v0") is False  # type: ignore[attr-defined]
    assert env._resolve_use_memory("glyphbench/craftax-dungeon-v0") is True  # type: ignore[attr-defined]


@pytest.mark.parametrize("override", [True, False])
def test_load_environment_memory_override_forces_all_tasks(override: bool) -> None:
    env = load_environment(
        task_id="glyphbench/classics-snake-medium-v0",
        num_episodes=1,
        use_memory=override,
    )
    assert env._resolve_use_memory("glyphbench/craftax-dungeon-v0") is override  # type: ignore[attr-defined]
