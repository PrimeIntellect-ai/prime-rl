from __future__ import annotations

from glyphbench.core.registry import make_env
from glyphbench.envs import _import_all_suites
from glyphbench.verifiers_integration.prompting import build_system_prompt, render_user_turn


def test_uncapped_prompt_describes_native_budget_without_truncation_threat() -> None:
    _import_all_suites()
    game = make_env("glyphbench/classics-snake-easy-v0")
    try:
        prompt = build_system_prompt(game, None, use_memory=True, memory_update_max_tokens=None)
    finally:
        game.close()
    assert "No explicit output-token cap is imposed" in prompt
    assert "No explicit output-token cap is imposed on memory updates" in prompt
    assert "exceeds None tokens" not in prompt


def test_capped_prompt_keeps_explicit_closing_tag_budget() -> None:
    _import_all_suites()
    game = make_env("glyphbench/classics-snake-easy-v0")
    try:
        prompt = build_system_prompt(game, 123, use_memory=False)
    finally:
        game.close()
    assert "output budget is 123 tokens" in prompt
    assert "closing </action> tag" in prompt


def test_reasoning_without_memory_has_consistent_turn_instructions() -> None:
    game = make_env("glyphbench/classics-snake-easy-v0")
    try:
        observation, _ = game.reset(42)
        system = build_system_prompt(game, 4096, use_memory=False, enable_reasoning=True)
        user = render_user_turn(
            game, (), observation, 0, 4096, enable_reasoning=True
        )
    finally:
        game.close()
    assert "Reason before acting" in system
    assert "Now reason as much as useful" in user
    assert "no other text" not in user
