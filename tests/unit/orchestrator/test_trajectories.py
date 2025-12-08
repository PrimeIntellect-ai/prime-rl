from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.orchestrator.trajectories import (
    _interleave_rollout_legacy,
    branch_rollout,
    interleave_rollout,
)


@pytest.fixture
def single_step_trajectory_state():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            )
        ],
    )
    return state


@pytest.fixture
def multi_step_trajectory_state():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                    prompt_logprobs=[-0.01, -0.02, -0.1, -0.2, 0.0, 0.0],
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )
    return state


def test_branching_rollout_single_step(single_step_trajectory_state):
    rollouts = branch_rollout(single_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["completion_ids"] == [3, 4]
    assert rollout["completion_logprobs"] == [-0.1, -0.2]


def test_branching_rollout_multi_step(multi_step_trajectory_state):
    rollouts = branch_rollout(multi_step_trajectory_state)
    assert len(rollouts) == 2
    assert rollouts[0]["prompt_ids"] == [1, 2]
    assert rollouts[0]["completion_ids"] == [3, 4]
    assert rollouts[1]["prompt_ids"] == [1, 2, 3, 4, 5, 6]
    assert rollouts[1]["completion_ids"] == [7, 8]


def test_interleave_single_step(single_step_trajectory_state):
    rollouts = interleave_rollout(single_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["completion_ids"] == [3, 4]
    assert rollout["completion_logprobs"] == [-0.1, -0.2]


def test_interleave_multi_step_with_prompt_logprobs(multi_step_trajectory_state):
    rollouts = interleave_rollout(multi_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["completion_ids"] == [3, 4, 5, 6, 7, 8]
    assert rollout["completion_logprobs"] == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
    assert rollout["completion_mask"] == [1, 1, 0, 0, 1, 1]


def test_interleave_legacy_fallback_without_prompt_logprobs():
    state = vf.State(
        example_id="test_legacy",
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )

    rollouts = interleave_rollout(state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["completion_ids"] == [3, 4, 5, 6, 7, 8]
    assert rollout["completion_logprobs"] == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
    assert rollout["completion_mask"] == [1, 1, 0, 0, 1, 1]


def test_compare_legacy_vs_new_with_token_mismatch():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 99, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                    prompt_logprobs=[-0.05, -0.15, -0.55, -0.18, -0.25, -0.35],
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )

    legacy_rollout = _interleave_rollout_legacy(state)[0]
    new_rollout = interleave_rollout(state)[0]

    assert legacy_rollout["completion_ids"][0] == 3
    assert new_rollout["completion_ids"][0] == 99

    assert legacy_rollout["completion_logprobs"][0] == -0.1
    assert new_rollout["completion_logprobs"][0] == -0.55


def test_interleave_three_turn():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[10, 11],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 10, 11, 20, 21],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[30, 31],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                    {"role": "assistant", "content": "A2"},
                    {"role": "tool", "tool_call_id": "tc1", "content": "TR"},
                ],
                completion=[{"role": "assistant", "content": "A3"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 10, 11, 20, 21, 30, 31, 40, 41],
                    prompt_mask=[0] * 10,
                    completion_ids=[50, 51],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.5, -0.6],
                    overlong_prompt=False,
                    is_truncated=False,
                    prompt_logprobs=[-0.01, -0.02, -0.11, -0.12, -0.21, -0.22, -0.31, -0.32, -0.41, -0.42],
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )

    rollouts = interleave_rollout(state)
    rollout = rollouts[0]

    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["completion_ids"] == [10, 11, 20, 21, 30, 31, 40, 41, 50, 51]
    assert rollout["completion_logprobs"] == [-0.11, -0.12, -0.21, -0.22, -0.31, -0.32, -0.41, -0.42, -0.5, -0.6]
    assert rollout["completion_mask"] == [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
