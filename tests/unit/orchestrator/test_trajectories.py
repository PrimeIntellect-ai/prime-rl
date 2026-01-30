from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.orchestrator.trajectories import interleave_rollout


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
                temperature=1.0,
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
                extras={},
                temperature=1.0,
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
                temperature=1.0,
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
                extras={},
                temperature=1.0,
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
                extras={},
                temperature=1.0,
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
def multi_step_trajectory_extension_never_holds():
    """
    2-step trajectory where extension NEVER holds (step 2 has completely different tokens).
    This simulates e.g. a chat template that re-renders the entire conversation differently.
    """
    state = vf.State(
        input=vf.RolloutInput(messages=[], example_id="test-no-extension"),
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
                temperature=1.0,
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
                    # Different tokens - extension breaks (e.g. thinking was stripped)
                    prompt_ids=[10, 20, 30, 40, 50, 60],
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
                temperature=1.0,
            ),
        ],
        error=None,
    )
    return state


@pytest.fixture
def multi_step_trajectory_with_tool_calls_extension_never_holds():
    """2-step trajectory with tool calls where extension NEVER holds."""
    state = vf.State(
        input=vf.RolloutInput(messages=[], example_id="test-no-extension-tools"),
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
                extras={},
                temperature=1.0,
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
                    # Different tokens - extension breaks
                    prompt_ids=[10, 20, 30, 40, 50, 60],
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
                temperature=1.0,
            ),
        ],
        reward=1.0,
        advantage=None,
        stop_condition=None,
        metrics={"has_error": 0.0, "tool_calls": 1.0},
        error=None,
    )
    return state


def test_branching_equivalent_single_step_trajectory(single_step_trajectory_state):
    """Single step: interleave produces same output as branching would have."""
    rollouts = interleave_rollout(single_step_trajectory_state)

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]
    assert rollout.completion_temperatures == [1.0, 1.0]


def test_branching_equivalent_multi_step_trajectory(multi_step_trajectory_extension_never_holds):
    """When extension never holds, each step becomes its own sample (same as old branching)."""
    rollouts = interleave_rollout(multi_step_trajectory_extension_never_holds)
    assert len(rollouts) == 2

    # first step
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]
    assert rollout.completion_temperatures == [1.0, 1.0]

    # second step
    rollout = rollouts[1]
    assert rollout.prompt_ids == [10, 20, 30, 40, 50, 60]
    assert rollout.prompt_mask == [False, False, False, False, False, False]
    assert rollout.completion_ids == [7, 8]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.3, -0.4]
    assert rollout.completion_temperatures == [1.0, 1.0]


def test_branching_equivalent_multi_step_trajectory_with_tool_calls(
    multi_step_trajectory_with_tool_calls_extension_never_holds,
):
    """When extension never holds (with tool calls), same as old branching."""
    rollouts = interleave_rollout(multi_step_trajectory_with_tool_calls_extension_never_holds)
    assert len(rollouts) == 2

    # first step
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]
    assert rollout.completion_temperatures == [1.0, 1.0]

    # second step
    rollout = rollouts[1]
    assert rollout.prompt_ids == [10, 20, 30, 40, 50, 60]
    assert rollout.prompt_mask == [False, False, False, False, False, False]
    assert rollout.completion_ids == [7, 8]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.3, -0.4]
    assert rollout.completion_temperatures == [1.0, 1.0]


def test_interleave_rollout_single_step_trajectory(single_step_trajectory_state):
    rollouts = interleave_rollout(single_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]
    assert rollout.completion_temperatures == [1.0, 1.0]


def test_interleave_rollout_multi_step_trajectory(multi_step_trajectory_state):
    rollouts = interleave_rollout(multi_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollout.completion_mask == [True, True, False, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4]
    # Temperatures: 2 completion tokens at temp 1.0, then 2 prompt tokens at temp 1.0, then 2 completion tokens at temp 1.0
    assert rollout.completion_temperatures == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_interleave_rollout_multi_step_trajectory_with_tool_calls(multi_step_trajectory_with_tool_calls):
    rollouts = interleave_rollout(multi_step_trajectory_with_tool_calls)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollout.completion_mask == [True, True, False, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4]


@pytest.fixture
def five_step_trajectory_with_extension_break():
    """
    5-step trajectory where extension property breaks at step 4.

    Steps 1-3: extension holds (tokens grow by appending)
    Step 4: extension breaks (completely different prefix, e.g. context compaction)
    Steps 4-5: extension holds again

    Expected: 2 samples (steps 1-3 merged, steps 4-5 merged)
    """
    state = vf.State(
        input=vf.RolloutInput(messages=[], example_id="test-extension-break"),
        trajectory=[
            # Step 1: initial prompt and completion
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
                temperature=1.0,
            ),
            # Step 2: extends step 1 (prefix [1,2,3,4] matches)
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
                temperature=1.0,
            ),
            # Step 3: extends step 2 (prefix [1,2,3,4,5,6,7,8] matches)
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                    {"role": "assistant", "content": "A2"},
                    {"role": "user", "content": "U3"},
                ],
                completion=[{"role": "assistant", "content": "A3"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    prompt_mask=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    completion_ids=[11, 12],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.5, -0.6],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
                temperature=1.0,
            ),
            # Step 4: EXTENSION BREAKS - different prefix (e.g. thinking stripped, context compacted)
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},  # thinking stripped
                    {"role": "user", "content": "U2"},
                    {"role": "assistant", "content": "A2"},
                    {"role": "user", "content": "U3"},
                    {"role": "assistant", "content": "A3"},
                    {"role": "user", "content": "U4"},
                ],
                completion=[{"role": "assistant", "content": "A4"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[100, 101, 102, 103],  # completely different tokens (re-rendered)
                    prompt_mask=[0, 0, 0, 0],
                    completion_ids=[104, 105],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.7, -0.8],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
                temperature=1.0,
            ),
            # Step 5: extends step 4 (prefix [100,101,102,103,104,105] matches)
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                    {"role": "assistant", "content": "A2"},
                    {"role": "user", "content": "U3"},
                    {"role": "assistant", "content": "A3"},
                    {"role": "user", "content": "U4"},
                    {"role": "assistant", "content": "A4"},
                    {"role": "user", "content": "U5"},
                ],
                completion=[{"role": "assistant", "content": "A5"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[100, 101, 102, 103, 104, 105, 106, 107],
                    prompt_mask=[0, 0, 0, 0, 0, 0, 0, 0],
                    completion_ids=[108, 109],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.9, -1.0],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
                temperature=1.0,
            ),
        ],
        error=None,
    )
    return state


def test_interleave_rollout_extension_break_creates_multiple_samples(five_step_trajectory_with_extension_break):
    """
    When extension property breaks mid-trajectory, interleave_rollout should:
    - Merge steps 1-3 into first sample (extension held)
    - Start new sample at step 4 (extension broke)
    - Merge steps 4-5 into second sample (extension held again)
    """
    rollouts = interleave_rollout(five_step_trajectory_with_extension_break)

    assert len(rollouts) == 2, "Should produce 2 samples when extension breaks at step 4"

    # First sample: steps 1-3 merged
    sample1 = rollouts[0]
    assert sample1.prompt_ids == [1, 2]
    assert sample1.prompt_mask == [False, False]
    # completion_ids: step1 completion [3,4] + step2 new prompt [5,6] + step2 completion [7,8]
    #                 + step3 new prompt [9,10] + step3 completion [11,12]
    assert sample1.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # completion_mask: step1 [T,T] + step2 prompt [F,F] + step2 completion [T,T]
    #                  + step3 prompt [F,F] + step3 completion [T,T]
    assert sample1.completion_mask == [True, True, False, False, True, True, False, False, True, True]
    assert sample1.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4, 0, 0, -0.5, -0.6]

    # Second sample: steps 4-5 merged (fresh start after extension break)
    sample2 = rollouts[1]
    assert sample2.prompt_ids == [100, 101, 102, 103]
    assert sample2.prompt_mask == [False, False, False, False]
    # completion_ids: step4 completion [104,105] + step5 new prompt [106,107] + step5 completion [108,109]
    assert sample2.completion_ids == [104, 105, 106, 107, 108, 109]
    # completion_mask: step4 [T,T] + step5 prompt [F,F] + step5 completion [T,T]
    assert sample2.completion_mask == [True, True, False, False, True, True]
    assert sample2.completion_logprobs == [-0.7, -0.8, 0, 0, -0.9, -1.0]


@pytest.fixture
def interleaved_agents_trajectory():
    """
    Trajectory with interleaved agents: agent1 steps, then agent2 step, then agent1 continues.
    This tests multi-prefix tracking where agent1-step3 should merge back with agent1 sample.

    agent1-step1: prompt=[1,2], completion=[3,4]
    agent1-step2: prompt=[1,2,3,4,5,6], completion=[7,8]  (extends agent1-step1)
    agent2-step1: prompt=[100,101], completion=[102,103]  (different prefix, new sample)
    agent1-step3: prompt=[1,2,3,4,5,6,7,8,9,10], completion=[11,12]  (extends agent1-step2!)
    """
    state = vf.State(
        example_id=1,
        task="test",
        trajectory=[
            # agent1-step1
            vf.TrajectoryStep(
                prompt="agent1 turn 1",
                completion="response 1",
                response=None,
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
                trajectory_id="traj1",
                extras={},
                temperature=1.0,
            ),
            # agent1-step2 (extends agent1-step1)
            vf.TrajectoryStep(
                prompt="agent1 turn 2",
                completion="response 2",
                response=None,
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
                trajectory_id="traj1",
                extras={},
                temperature=1.0,
            ),
            # agent2-step1 (different prefix, starts new sample)
            vf.TrajectoryStep(
                prompt="agent2 turn 1",
                completion="agent2 response",
                response=None,
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
                trajectory_id="traj2",
                extras={},
                temperature=1.0,
            ),
            # agent1-step3 (extends agent1-step2, should merge back!)
            vf.TrajectoryStep(
                prompt="agent1 turn 3",
                completion="response 3",
                response=None,
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    prompt_mask=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    completion_ids=[11, 12],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.7, -0.8],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="traj1",
                extras={},
                temperature=1.0,
            ),
        ],
        error=None,
    )
    return state


def test_interleave_rollout_interleaved_agents(interleaved_agents_trajectory):
    """
    When agents are interleaved (agent1, agent1, agent2, agent1), the multi-prefix
    tracking should merge agent1-step3 back into the agent1 sample, not start a new one.
    """
    rollouts = interleave_rollout(interleaved_agents_trajectory)

    assert len(rollouts) == 2, "Should produce 2 samples (agent1 merged, agent2 separate)"

    # First sample: agent1 steps 1, 2, 3 merged
    agent1_sample = rollouts[0]
    assert agent1_sample.prompt_ids == [1, 2]
    assert agent1_sample.prompt_mask == [False, False]
    # completion_ids: step1 [3,4] + step2 new prompt [5,6] + step2 completion [7,8]
    #                 + step3 new prompt [9,10] + step3 completion [11,12]
    assert agent1_sample.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert agent1_sample.completion_mask == [True, True, False, False, True, True, False, False, True, True]
    assert agent1_sample.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4, 0, 0, -0.7, -0.8]

    # Second sample: agent2 step 1 only
    agent2_sample = rollouts[1]
    assert agent2_sample.prompt_ids == [100, 101]
    assert agent2_sample.prompt_mask == [False, False]
    assert agent2_sample.completion_ids == [102, 103]
    assert agent2_sample.completion_mask == [True, True]
    assert agent2_sample.completion_logprobs == [-0.5, -0.6]
