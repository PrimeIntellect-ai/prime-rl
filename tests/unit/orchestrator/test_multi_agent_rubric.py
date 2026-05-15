"""Contract + error-boundary tests for MultiAgentRubric base class."""

from __future__ import annotations

import asyncio

import pytest
import verifiers as vf
from verifiers.errors import KernelProtocolError
from verifiers.rubrics.multi_agent_rubric import MultiAgentRubric
from verifiers.types import MARScore, MemberScore


def _run(coro):
    return asyncio.run(coro)


class _StructuredRubric(MultiAgentRubric):
    """Minimal subclass that writes mar_score with all members covered."""

    def __init__(self, members: list[str]):
        super().__init__()
        self.members = members

    async def build_marscore(self, state) -> MARScore:
        return MARScore(
            members=[MemberScore(member_id=m, reward=1.0) for m in self.members],
            episode_scalar=1.0,
        )


class _RaisingRubric(MultiAgentRubric):
    """Raises KernelProtocolError on the configured call-indexes."""

    def __init__(self, members: list[str], fail_on: set[int]):
        super().__init__()
        self.members = members
        self.fail_on = fail_on
        self._i = -1

    async def build_marscore(self, state) -> MARScore:
        self._i += 1
        if self._i in self.fail_on:
            raise KernelProtocolError(f"boom-{self._i}")
        return MARScore(
            members=[MemberScore(member_id=m, reward=1.0) for m in self.members],
            episode_scalar=1.0,
        )


def _rewards_by_member(state) -> dict[str, float]:
    mar = state["mar_score"]
    return {m.member_id: m.reward for m in mar.members}


def test_multi_agent_rubric_contract_writes_mar_score():
    rubric = _StructuredRubric(members=["alice", "bob"])
    state: dict = {}
    _run(rubric.score_rollout(state))
    assert _rewards_by_member(state) == {"alice": 1.0, "bob": 1.0}


def test_score_rollout_short_circuits_prompt_too_long():
    class _ShouldNotRunRubric(MultiAgentRubric):
        members = ["a"]

        async def build_marscore(self, state) -> MARScore:
            raise AssertionError("build_marscore should not run")

    rubric = _ShouldNotRunRubric()
    state = {"prompt_too_long": True}

    _run(rubric.score_rollout(state))

    assert _rewards_by_member(state) == {"a": 0.0}
    assert state["mar_score"].episode_metrics == {"errored_rollout": 1.0}
    assert state["mar_score"].episode_error == {
        "error_type": "prompt_too_long",
        "error_phase": "rollout",
    }


def test_score_rollout_short_circuits_existing_state_error():
    class _ShouldNotRunRubric(MultiAgentRubric):
        members = ["a"]

        async def build_marscore(self, state) -> MARScore:
            raise AssertionError("build_marscore should not run")

    class _SimulatedEnvError(Exception):
        pass

    rubric = _ShouldNotRunRubric()
    error = _SimulatedEnvError("boom")
    state = {"error": error}

    _run(rubric.score_rollout(state))

    assert state["error"] is error
    assert _rewards_by_member(state) == {"a": 0.0}
    assert state["mar_score"].episode_metrics == {"errored_rollout": 1.0}
    assert state["mar_score"].episode_error == {
        "error_type": "_SimulatedEnvError",
        "error_phase": "rollout",
    }


def test_score_group_error_boundary_isolates_failures():
    """One rollout raises; other rollouts still score + zero-reward mar_score for failed."""
    rubric = _RaisingRubric(members=["alice", "bob"], fail_on={1})
    states: list[dict] = [{}, {}, {}]
    _run(rubric.score_group(states))

    failed = [s for s in states if s.get("error") is not None]
    succeeded = [s for s in states if s.get("error") is None]
    assert len(failed) == 1
    assert len(succeeded) == 2

    f = failed[0]
    assert isinstance(f["error"], KernelProtocolError)
    assert _rewards_by_member(f) == {"alice": 0.0, "bob": 0.0}
    assert f["mar_score"].episode_metrics["errored_rollout"] == 1.0
    assert f["mar_score"].episode_error == {
        "error_type": "KernelProtocolError",
        "error_phase": "scoring",
    }

    for s in succeeded:
        assert _rewards_by_member(s) == {"alice": 1.0, "bob": 1.0}


def test_score_group_non_vf_error_propagates():
    """Non-vf.Error must NOT be swallowed — programming bugs escape loud."""

    class _BuggyRubric(MultiAgentRubric):
        members = ["a"]

        async def build_marscore(self, state) -> MARScore:
            raise AttributeError("programming bug")

    rubric = _BuggyRubric()
    with pytest.raises(AttributeError, match="programming bug"):
        _run(rubric.score_group([{}]))


def test_score_group_all_succeed():
    rubric = _StructuredRubric(members=["alice", "bob"])
    states = [{}, {}]
    _run(rubric.score_group(states))
    for s in states:
        assert "error" not in s
        assert _rewards_by_member(s) == {"alice": 1.0, "bob": 1.0}


def test_multi_agent_rubric_is_subclass_of_rubric():
    """Type invariant: stays a Rubric so existing orchestrator code keeps working."""
    assert issubclass(MultiAgentRubric, vf.Rubric)


def test_score_rollout_overwrites_partial_mar_score_on_vf_error():
    """A scoring-time vf.Error must end in the canonical zero-reward MARScore."""

    class _PartialFailRubric(MultiAgentRubric):
        members = ["a", "b"]

        async def build_marscore(self, state) -> MARScore:
            state["mar_score"] = MARScore(
                members=[
                    MemberScore(member_id="a", reward=0.7),
                    MemberScore(member_id="b", reward=0.3),
                ],
                episode_scalar=0.5,
            )
            raise KernelProtocolError("late raise after partial mar_score write")

    rubric = _PartialFailRubric()
    state: dict = {}
    _run(rubric.score_rollout(state))

    assert isinstance(state["error"], KernelProtocolError)
    assert _rewards_by_member(state) == {"a": 0.0, "b": 0.0}
    assert state["mar_score"].episode_metrics == {"errored_rollout": 1.0}
    assert state["mar_score"].episode_error == {
        "error_type": "KernelProtocolError",
        "error_phase": "scoring",
    }
