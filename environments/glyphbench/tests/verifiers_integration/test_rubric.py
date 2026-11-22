from __future__ import annotations

import pytest

from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric


def _rubric() -> EpisodicReturnRubric:
    return EpisodicReturnRubric(parser=GlyphbenchXMLParser())


@pytest.mark.asyncio
async def test_core_episode_metrics() -> None:
    rubric = _rubric()
    state = {
        "episode_return": 1.5,
        "trajectory": [{"reward": 0.5}, {"reward": 0.5}, {"reward": 0.5}],
        "num_action_turns": 4,
        "forfeit_count": 1,
        "terminated": True,
        "truncated": False,
    }

    assert await rubric.episodic_return(state=state) == pytest.approx(1.5)
    assert await rubric.episode_length(state=state) == 3.0
    assert await rubric.forfeit_rate(state=state) == pytest.approx(0.25)
    assert await rubric.episode_terminated_rate(state=state) == 1.0
    assert await rubric.episode_truncated_max_turns_rate(state=state) == 0.0
    assert await rubric.context_limit_hit(
        state={"stop_condition": "prompt_too_long"}
    ) == 1.0
    assert await rubric.context_limit_hit(
        state={"stop_condition": "is_done"}
    ) == 0.0


@pytest.mark.asyncio
async def test_completion_and_memory_rates() -> None:
    rubric = _rubric()

    assert await rubric.action_completion_truncation_rate(
        state={"num_action_turns": 5, "action_completion_truncations": 2}
    ) == pytest.approx(0.4)
    assert await rubric.memory_completion_truncation_rate(
        state={"num_memory_turns": 4, "memory_completion_truncations": 1}
    ) == pytest.approx(0.25)
    assert await rubric.memory_parse_failure_rate(
        state={"num_memory_turns": 4, "memory_parse_failures": 3}
    ) == pytest.approx(0.75)
    assert await rubric.parse_recovery_rate(
        state={"num_action_turns": 5, "parse_recoveries": 2}
    ) == pytest.approx(0.4)


def test_required_metric_names_are_registered() -> None:
    metric_names = {m.__name__ for m in _rubric().funcs}

    assert {
        "episodic_return",
        "forfeit_rate",
        "parse_recovery_rate",
        "episode_terminated_rate",
        "episode_truncated_max_turns_rate",
        "context_limit_hit",
        "action_completion_truncation_rate",
        "memory_completion_truncation_rate",
        "memory_parse_failure_rate",
    } <= metric_names
    assert "parse_failure_rate" not in metric_names
