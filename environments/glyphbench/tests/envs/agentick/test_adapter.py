from __future__ import annotations

import pytest

from glyphbench.core.registry import REGISTRY, make_env
from glyphbench.envs import _import_all_suites
from glyphbench.envs.agentick.env import _ensure_agentick_importable
from glyphbench.verifiers_integration.env import (
    AGENTICK_MARKOV_REASONER_HARNESS,
    load_environment,
)
from glyphbench.verifiers_integration.prompting import build_system_prompt

try:
    _ensure_agentick_importable()
except ModuleNotFoundError:
    pytest.skip("the optional AgenticK checkout is not installed", allow_module_level=True)

AGENTICK_TASK_IDS = (
    "glyphbench/agentick-instruction-following-hard-v0",
    "glyphbench/agentick-sokoban-push-hard-v0",
    "glyphbench/agentick-precise-navigation-hard-v0",
    "glyphbench/agentick-packing-puzzle-hard-v0",
    "glyphbench/agentick-graph-coloring-hard-v0",
    "glyphbench/agentick-herding-hard-v0",
)


def test_agentick_hard_tasks_are_registered() -> None:
    _import_all_suites()

    assert set(AGENTICK_TASK_IDS) <= set(REGISTRY)


def test_agentick_adapter_resets_and_steps_each_requested_task() -> None:
    _import_all_suites()

    for env_id in AGENTICK_TASK_IDS:
        env = make_env(env_id, max_turns=3)
        try:
            obs, info = env.reset(seed=42)
            assert info["env_id"] == env_id
            assert "[Grid]" in obs
            assert "[HUD]" in obs
            assert "AgenticK task:" in obs
            assert "Valid AgenticK actions this turn:" in obs

            next_obs, reward, terminated, truncated, step_info = env.step(
                env.action_spec.index_of("NOOP")
            )
            assert "[Grid]" in next_obs
            assert -1.0 <= reward <= 1.0
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert step_info["agentick_action_index"] == 0
        finally:
            env.close()


def test_agentick_action_spec_accepts_numeric_agentick_ids() -> None:
    env = make_env("glyphbench/agentick-sokoban-push-hard-v0")
    try:
        assert env.action_spec.index_of("0") == env.action_spec.index_of("NOOP")
        assert env.action_spec.index_of("4") == env.action_spec.index_of("MOVE_RIGHT")
        assert env.action_spec.index_of("right") == env.action_spec.index_of("MOVE_RIGHT")
    finally:
        env.close()


def test_agentick_markov_reasoner_harness_uses_reasoning_format() -> None:
    env = make_env("glyphbench/agentick-instruction-following-hard-v0", max_turns=1)
    try:
        env.reset(seed=42)
        prompt = build_system_prompt(
            env,
            512,
            harness=AGENTICK_MARKOV_REASONER_HARNESS,
        )
    finally:
        env.close()

    assert "AgenticK markovian_reasoner setup" in prompt
    assert "Reason before acting" in prompt
    assert "Reply with exactly one XML action tag and no other text" not in prompt


def test_load_environment_accepts_agentick_markov_reasoner_harness() -> None:
    env = load_environment(
        task_id="glyphbench/agentick-sokoban-push-hard-v0",
        num_episodes=1,
        max_turns=1,
        harness="markovian_reasoner",
    )

    assert env._harness == AGENTICK_MARKOV_REASONER_HARNESS  # type: ignore[attr-defined]
    assert env._resolve_use_memory("glyphbench/agentick-sokoban-push-hard-v0") is False  # type: ignore[attr-defined]
