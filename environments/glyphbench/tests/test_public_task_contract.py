from __future__ import annotations

import pytest

import glyphbench  # noqa: F401
from glyphbench.core import make_env
from glyphbench.core.task_selection import list_task_ids
from tests._reward_helpers import _random_rollout_return

NON_ARCHIVAL_ENV_IDS = list_task_ids(
    exclude_suites=["atari", "craftaxfull", "nethack", "agentick"]
)
REWARD_EPS = 1e-6


def test_non_archival_public_envs_have_short_horizons() -> None:
    for env_id in NON_ARCHIVAL_ENV_IDS:
        env = make_env(env_id)
        try:
            assert env.max_turns < 512, (
                f"{env_id} has max_turns={env.max_turns}; non-archival "
                "public tasks must keep natural episode lengths < 512."
            )
        finally:
            env.close()


@pytest.mark.parametrize("env_id", NON_ARCHIVAL_ENV_IDS)
def test_non_archival_public_env_contract(env_id: str) -> None:
    env = make_env(env_id)
    try:
        obs_text, info = env.reset(0)
        obs = env.get_observation()

        assert info["env_id"] == env_id
        assert "[Grid]" in obs_text
        assert "[Legend]" in obs_text
        assert env.system_prompt().strip()
        assert env.action_spec.n == len(env.action_spec.names)
        assert env.action_spec.n == len(env.action_spec.descriptions)
        assert len(set(env.action_spec.names)) == len(env.action_spec.names)
        assert all(name and name.strip() == name for name in env.action_spec.names)

        rows = obs.grid.split("\n")
        assert rows
        assert "" not in rows
        assert len({len(row) for row in rows}) == 1
    finally:
        env.close()


@pytest.mark.parametrize("env_id", NON_ARCHIVAL_ENV_IDS)
def test_non_archival_public_env_random_rollout_bounds(env_id: str) -> None:
    for seed in range(3):
        total, length = _random_rollout_return(env_id, seed=seed, max_steps=512)
        assert length <= 512
        assert -1.0 - REWARD_EPS <= total <= 1.0 + REWARD_EPS, (
            f"{env_id} seed={seed}: cumulative return {total:.4f} after {length} steps "
            "is outside [-1, 1]."
        )
