from __future__ import annotations

import importlib.util

import pytest

import glyphbench  # noqa: F401
from glyphbench.core.registry import REGISTRY, make_env

if "glyphbench/craftaxfull-v0" not in REGISTRY:
    pytest.skip("the optional pinned Craftax fork is not installed", allow_module_level=True)

craftax_fork = pytest.importorskip(
    "glyphbench.envs.craftaxfull.fork",
    reason="the optional pinned Craftax fork is not installed",
    exc_type=ImportError,
)
CRAFTAX_FORK_ACTION_SPEC = craftax_fork.CRAFTAX_FORK_ACTION_SPEC
CRAFTAX_GLYPHBENCH_API = craftax_fork.CRAFTAX_GLYPHBENCH_API


def test_craftaxfull_registry_uses_submodule_wrapper() -> None:
    assert "glyphbench/craftaxfull-v0" in REGISTRY


def test_craftaxfull_reports_raw_upstream_returns() -> None:
    env = make_env("glyphbench/craftaxfull-v0")
    assert env.clamp_episode_return is False


def test_action_spec_and_renderer_tables_are_aligned() -> None:
    assert len(CRAFTAX_FORK_ACTION_SPEC.names) == len(
        CRAFTAX_GLYPHBENCH_API.UPSTREAM_ACTION_NAMES
    )
    assert (
        len(CRAFTAX_GLYPHBENCH_API.UNICODE_CHAR_TABLE)
        == len(CRAFTAX_GLYPHBENCH_API.CHAR_ID_MEANING)
        == 59
    )
    assert all(len(ch) == 1 for ch in CRAFTAX_GLYPHBENCH_API.UNICODE_CHAR_TABLE)
    assert CRAFTAX_FORK_ACTION_SPEC.index_of("LEFT") == CRAFTAX_FORK_ACTION_SPEC.index_of(
        "MOVE_LEFT"
    )
    assert CRAFTAX_FORK_ACTION_SPEC.index_of(
        "MAKE_IRON_ARMOUR"
    ) == CRAFTAX_FORK_ACTION_SPEC.index_of("MAKE_IRON_ARMOR")
    assert CRAFTAX_FORK_ACTION_SPEC.index_of(
        "ENCHANT_WEAPON"
    ) == CRAFTAX_FORK_ACTION_SPEC.index_of("ENCHANT_SWORD")


def test_craftaxfull_wrapper_is_local_view_only() -> None:
    with pytest.raises(TypeError):
        make_env("glyphbench/craftaxfull-v0", view_mode="full")


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="Craftax/JAX optional extra is not installed",
)
def test_craftaxfull_observation_contract_smoke() -> None:
    env = make_env("glyphbench/craftaxfull-v0", max_turns=3)
    try:
        obs_text, info = env.reset(seed=0)
        assert info["env_id"] == "glyphbench/craftaxfull-v0"
        assert "[Legend]" in obs_text
        assert "[HUD]" in obs_text
        assert "[Grid]" in obs_text

        obs = env.get_observation()
        rows = obs.grid.splitlines()
        assert len(rows) == 9
        assert {len(row) for row in rows} == {11}

        action = env.action_spec.index_of("MOVE_RIGHT")
        _next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["action_name"] == "MOVE_RIGHT"
        assert step_info["upstream_action_name"] == "RIGHT"
    finally:
        env.close()


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="Craftax/JAX optional extra is not installed",
)
def test_craftaxfull_sleep_fast_forwards_until_awake() -> None:
    env = make_env("glyphbench/craftaxfull-v0", max_turns=10000)
    try:
        env.reset(seed=0)
        # Put the upstream state one recovery tick away from full energy so the
        # test exercises the fast-forward loop without spending many JAX steps.
        env._state = env._state.replace(player_energy=8, player_fatigue=-11.0)
        sleep = env.action_spec.index_of("SLEEP")
        obs_text, _reward, terminated, truncated, info = env.step(sleep)

        assert not terminated
        assert not truncated
        assert info["fast_forward_ticks"] >= 1
        assert "Sleeping: no" in obs_text
        assert "Resting: no" in obs_text
    finally:
        env.close()
