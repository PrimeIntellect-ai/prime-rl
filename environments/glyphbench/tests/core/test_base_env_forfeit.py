"""Tests for BaseGlyphEnv.forfeit_turn (introduced for parse-fail semantics)."""

from __future__ import annotations

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation


class _StubEnv(BaseGlyphEnv):
    action_spec = ActionSpec(names=("NOOP", "GO"), descriptions=("n", "g"))

    def __init__(self, max_turns: int = 5) -> None:
        super().__init__(max_turns=max_turns)
        self._calls = 0
        self._obs_text = "[Grid]\nA\n\n[HUD]\nStep: 0 / 5"

    def _reset(self, seed: int) -> GridObservation:
        return GridObservation(grid="A", legend="A=agent", hud="Step: 0 / 5", message="")

    def _step(self, action: int):
        self._calls += 1
        return (
            GridObservation(grid="A", legend="A=agent", hud=f"Step: {self._turn} / 5", message=""),
            1.0, False, False, {}
        )

    def _render_current_observation(self) -> GridObservation:
        return GridObservation(
            grid="A", legend="A=agent", hud=f"Step: {self._turn} / 5", message=""
        )

    def system_prompt(self) -> str:
        return "stub env"

    def env_id(self) -> str:
        return "stub/forfeit-v0"


def test_forfeit_turn_advances_counter_only():
    env = _StubEnv(max_turns=5)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.forfeit_turn()
    assert env.turn == 1
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert env._calls == 0  # _step was NOT invoked


def test_forfeit_turn_does_not_change_observation_content():
    env = _StubEnv(max_turns=5)
    obs0, _ = env.reset(seed=0)
    obs1, *_ = env.forfeit_turn()
    # Same grid+legend, only HUD step changed.
    assert "A" in obs1
    assert "A=agent" in obs1
    assert "Step: 1 / 5" in obs1


def test_forfeit_turn_truncates_when_max_turns_hit():
    env = _StubEnv(max_turns=2)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.forfeit_turn()  # turn=1
    assert truncated is False
    obs, reward, terminated, truncated, info = env.forfeit_turn()  # turn=2
    assert truncated is True
    assert info.get("truncation_reason") == "max_turns"


def test_forfeit_turn_preserves_info_metadata():
    env = _StubEnv(max_turns=5)
    env.reset(seed=42)
    _, _, _, _, info = env.forfeit_turn()
    assert info["env_id"] == "stub/forfeit-v0"
    assert info["turn"] == 1
