from __future__ import annotations

from glyphbench.envs.craftax import _REGISTRATIONS
from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC, craftax_prompt_contract
from glyphbench.envs.craftax.docs import compose
from glyphbench.envs.craftax.scenario import CraftaxScenarioEnv
from glyphbench.envs.craftaxfull.full import _MOB_STATS, CraftaxFullEnv


def test_craftax_prompt_contract_is_single_universe() -> None:
    prompt = craftax_prompt_contract(focused_subtask=True)

    assert "CRAFTAX MECHANICS" in prompt
    assert "universal Craftax action menu" in prompt
    assert "FOCUSED SUBTASK CONTRACT" in prompt
    assert "classic" not in prompt.lower()
    assert "legacy" not in prompt.lower()


def test_registered_craftax_tasks_share_action_space_and_prompt_contract() -> None:
    assert len(_REGISTRATIONS) == 60

    for env_id, env_cls in sorted(_REGISTRATIONS.items()):
        assert issubclass(env_cls, (CraftaxScenarioEnv, CraftaxFullEnv)), env_id

        default_env = env_cls()
        assert default_env.max_turns <= 512, env_id

        env = env_cls(max_turns=3)
        assert env.action_spec.names == CRAFTAX_FULL_ACTION_SPEC.names, env_id

        prompt = env.system_prompt()
        assert "CRAFTAX MECHANICS" in prompt, env_id
        assert "universal Craftax action menu" in prompt, env_id
        assert "classic" not in prompt.lower(), env_id
        assert "legacy" not in prompt.lower(), env_id

        obs_text, info = env.reset(seed=0)
        assert info["env_id"] == env_id
        assert "[Grid]" in obs_text
        assert "[Legend]" in obs_text

        _obs, reward, terminated, truncated, _info = env.step(
            env.action_spec.index_of("NOOP")
        )
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


def test_craftax_docs_and_reference_constants_are_available() -> None:
    overview = compose(["overview"])

    assert "Craftax is a 9-floor survival crafting game" in overview
    assert "universal Craftax action menu" in overview
    assert "classic" not in overview.lower()
    assert "legacy" not in overview.lower()
    assert _MOB_STATS["zombie"] == {"hp": 5, "damage": 2}
    assert _MOB_STATS["skeleton"] == {"hp": 3, "damage": 2}
