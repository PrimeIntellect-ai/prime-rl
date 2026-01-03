from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.orchestrator.trajectories import branch_rollout, grouped_interleave_rollout


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
                is_truncated=False,
                trajectory_id="main",
                extras={},
            )
        ],
        error=None,
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
                is_truncated=False,
                trajectory_id="main",
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
                is_truncated=False,
                trajectory_id="main",
                extras={},
            ),
        ],
        error=None,
    )
    return state


@pytest.fixture
def multi_step_trajectory_with_tool_calls():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1 + TC1"}],
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
                is_truncated=False,
                trajectory_id="main",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1 + TC1"},
                    {"role": "tool", "tool_call_id": "TR1", "content": "TR1"},
                ],
                completion=[{"role": "assistant", "content": "A2 + TC2"}],
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
                is_truncated=False,
                trajectory_id="main",
                extras={},
            ),
        ],
        reward=1.0,
        advantage=None,
        stop_condition=None,
        metrics={"has_error": 0.0, "tool_calls": 1.0},
        error=None,
    )
    return state


@pytest.fixture
def multi_trajectory_state():
    """State with multiple trajectory_ids (e.g., RLM with sub-LLM calls)."""
    state = vf.State(
        trajectory=[
            # Sub-LLM trajectory (single turn)
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Sub prompt"}],
                completion=[{"role": "assistant", "content": "Sub response"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[100, 101],
                    prompt_mask=[0, 0],
                    completion_ids=[102, 103],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.5, -0.6],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="sub_batch1_req1",
                extras={"is_sub_llm_call": True},
            ),
            # Main trajectory step 1
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
                is_truncated=False,
                trajectory_id="main",
                extras={},
            ),
            # Main trajectory step 2
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
                is_truncated=False,
                trajectory_id="main",
                extras={},
            ),
        ],
        error=None,
    )
    return state


def test_branching_rollout_single_step_trajectory(single_step_trajectory_state):
    rollouts = branch_rollout(single_step_trajectory_state)

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]


def test_branching_rollout_multi_step_trajectory(multi_step_trajectory_state):
    rollouts = branch_rollout(multi_step_trajectory_state)
    assert len(rollouts) == 2

    # first step
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]

    # second step
    rollout = rollouts[1]
    assert rollout.prompt_ids == [1, 2, 3, 4, 5, 6]
    assert rollout.prompt_mask == [False, False, False, False, False, False]
    assert rollout.completion_ids == [7, 8]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.3, -0.4]


def test_branching_rollout_multi_step_trajectory_with_tool_calls(multi_step_trajectory_with_tool_calls):
    rollouts = branch_rollout(multi_step_trajectory_with_tool_calls)
    assert len(rollouts) == 2

    # first step
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]

    # second step
    rollout = rollouts[1]
    assert rollout.prompt_ids == [1, 2, 3, 4, 5, 6]
    assert rollout.prompt_mask == [False, False, False, False, False, False]
    assert rollout.completion_ids == [7, 8]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.3, -0.4]


def test_grouped_interleave_rollout_single_step_trajectory(single_step_trajectory_state):
    rollouts = grouped_interleave_rollout(single_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]


def test_grouped_interleave_rollout_multi_step_trajectory(multi_step_trajectory_state):
    rollouts = grouped_interleave_rollout(multi_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollout.completion_mask == [True, True, False, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4]


def test_grouped_interleave_rollout_multi_step_trajectory_with_tool_calls(multi_step_trajectory_with_tool_calls):
    rollouts = grouped_interleave_rollout(multi_step_trajectory_with_tool_calls)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollout.completion_mask == [True, True, False, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4]


def test_grouped_interleave_rollout_multiple_trajectory_ids(multi_trajectory_state):
    """Test that steps with different trajectory_ids become separate training samples."""
    rollouts = grouped_interleave_rollout(multi_trajectory_state)

    # Should have 2 rollouts: one for sub-LLM, one for main trajectory
    assert len(rollouts) == 2

    # Find the sub-LLM rollout (single step, prompt_ids=[100, 101])
    sub_rollout = next(r for r in rollouts if r.prompt_ids == [100, 101])
    assert sub_rollout.completion_ids == [102, 103]
    assert sub_rollout.completion_mask == [True, True]
    assert sub_rollout.completion_logprobs == [-0.5, -0.6]

    # Find the main rollout (interleaved, prompt_ids=[1, 2])
    main_rollout = next(r for r in rollouts if r.prompt_ids == [1, 2])
    # Main trajectory should be interleaved: step1 completion + step2 prompt extension + step2 completion
    assert main_rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert main_rollout.completion_mask == [True, True, False, False, True, True]
    assert main_rollout.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4]
