import base64
from io import BytesIO
from unittest.mock import MagicMock

import pytest
import verifiers as vf
from PIL import Image

from prime_rl.orchestrator.trajectories import (
    VLMImageCache,
    _extract_images_from_examples,
    _extract_images_from_messages,
    build_vlm_image_cache,
    interleave_rollout,
)


@pytest.fixture
def single_step_trajectory_output():
    output = vf.RolloutOutput(
        example_id=0,
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
                trajectory_id="1",
                extras={},
            )
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )
    return output


@pytest.fixture
def multi_step_trajectory_output():
    output = vf.RolloutOutput(
        example_id=0,
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
                trajectory_id="1",
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
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )
    return output


@pytest.fixture
def multi_step_trajectory_with_tool_calls_output():
    output = vf.RolloutOutput(
        example_id=0,
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
                trajectory_id="1",
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
                trajectory_id="1",
                extras={},
            ),
        ],
        reward=1.0,
        advantage=None,
        stop_condition=None,
        metrics={"has_error": 0.0, "tool_calls": 1.0},
        sampling_args={"temperature": 1.0},
        error=None,
    )
    return output


@pytest.fixture
def multi_step_trajectory_extension_never_holds():
    """
    2-step trajectory where extension NEVER holds (step 2 has completely different tokens).
    This simulates e.g. a chat template that re-renders the entire conversation differently.
    """
    output = vf.RolloutOutput(
        example_id=0,
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
                trajectory_id="1",
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
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )
    return output


@pytest.fixture
def multi_step_trajectory_with_tool_calls_extension_never_holds():
    """2-step trajectory with tool calls where extension NEVER holds."""
    output = vf.RolloutOutput(
        example_id=0,
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
                is_truncated=False,
                trajectory_id="1",
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
                is_truncated=False,
                trajectory_id="1",
            ),
        ],
        reward=1.0,
        advantage=None,
        stop_condition=None,
        sampling_args={"temperature": 1.0},
        metrics={"has_error": 0.0, "tool_calls": 1.0},
        error=None,
    )
    return output


def test_branching_equivalent_multi_step_trajectory(multi_step_trajectory_extension_never_holds):
    """When extension never holds, each step becomes its own sample (same as old branching)."""
    rollouts = interleave_rollout(multi_step_trajectory_extension_never_holds)
    assert rollouts is not None
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
    assert rollouts is not None
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


def test_interleave_rollout_single_step_trajectory(single_step_trajectory_output):
    rollouts = interleave_rollout(single_step_trajectory_output)
    assert rollouts is not None
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2]
    assert rollout.completion_temperatures == [1.0, 1.0]


def test_interleave_rollout_multi_step_trajectory(multi_step_trajectory_output):
    rollouts = interleave_rollout(multi_step_trajectory_output)
    assert rollouts is not None
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollout.completion_mask == [True, True, False, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4]
    # Temperatures: 2 completion tokens at temp 1.0, then 2 prompt tokens at temp 1.0, then 2 completion tokens at temp 1.0
    assert rollout.completion_temperatures == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_interleave_rollout_multi_step_trajectory_with_tool_calls(multi_step_trajectory_with_tool_calls_output):
    rollouts = interleave_rollout(multi_step_trajectory_with_tool_calls_output)
    assert rollouts is not None
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.prompt_ids == [1, 2]
    assert rollout.prompt_mask == [False, False]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollout.completion_mask == [True, True, False, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0, 0, -0.3, -0.4]
    # Temperatures: 2 completion tokens at temp 1.0, then 2 prompt tokens at temp 1.0, then 2 completion tokens at temp 1.0
    assert rollout.completion_temperatures == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


@pytest.fixture
def five_step_trajectory_with_extension_break():
    """
    5-step trajectory where extension property breaks at step 4.

    Steps 1-3: extension holds (tokens grow by appending)
    Step 4: extension breaks (completely different prefix, e.g. context compaction)
    Steps 4-5: extension holds again

    Expected: 2 samples (steps 1-3 merged, steps 4-5 merged)
    """
    output = vf.RolloutOutput(
        example_id=0,
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
                is_truncated=False,
                trajectory_id="1",
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
                is_truncated=False,
                trajectory_id="1",
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
                is_truncated=False,
                trajectory_id="1",
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
                is_truncated=False,
                trajectory_id="1",
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
                is_truncated=False,
                trajectory_id="1",
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )
    return output


def test_interleave_rollout_extension_break_creates_multiple_samples(five_step_trajectory_with_extension_break):
    """
    When extension property breaks mid-trajectory, interleave_rollout should:
    - Merge steps 1-3 into first sample (extension held)
    - Start new sample at step 4 (extension broke)
    - Merge steps 4-5 into second sample (extension held again)
    """
    rollouts = interleave_rollout(five_step_trajectory_with_extension_break)

    assert rollouts is not None
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
    output = vf.RolloutOutput(
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
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )
    return output


def test_interleave_rollout_interleaved_agents(interleaved_agents_trajectory):
    """
    When agents are interleaved (agent1, agent1, agent2, agent1), the multi-prefix
    tracking should merge agent1-step3 back into the agent1 sample, not start a new one.
    """
    rollouts = interleave_rollout(interleaved_agents_trajectory)

    assert rollouts is not None
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


# =============================================================================
# VLM Multi-Turn Tests
# =============================================================================


def _create_test_image(color: str = "red") -> str:
    """Create a small test image and return its base64 data URL."""
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
    img = Image.new("RGB", (10, 10), colors.get(color, (255, 255, 255)))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _create_image_message(image_url: str, text: str = "What is this?") -> dict:
    """Create an OpenAI-style user message with an image."""
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text},
        ],
    }


def test_extract_images_from_messages_no_images():
    messages = [{"role": "user", "content": "Hello"}]
    images = _extract_images_from_messages(messages)
    assert images == []


def test_extract_images_from_messages_single_image():
    image_url = _create_test_image("red")
    messages = [_create_image_message(image_url)]
    images = _extract_images_from_messages(messages)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_extract_images_from_messages_multiple_images():
    messages = [
        _create_image_message(_create_test_image("red")),
        {"role": "assistant", "content": "I see a red image"},
        _create_image_message(_create_test_image("green")),
    ]
    images = _extract_images_from_messages(messages)
    assert len(images) == 2


def test_extract_images_from_examples_single_turn():
    image_url = _create_test_image("red")
    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[_create_image_message(image_url)],
                completion=[{"role": "assistant", "content": "A red square"}],
                response=MagicMock(),
                tokens=MagicMock(),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    all_images, images_per_step = _extract_images_from_examples([(1, output)])

    assert len(all_images) == 1
    assert images_per_step == {1: [1]}  # 1 image after step 0


def test_extract_images_from_examples_multi_turn_new_image_each_turn():
    """Test that new images in later turns are correctly extracted."""
    red_url = _create_test_image("red")
    green_url = _create_test_image("green")

    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            # Turn 1: just the red image
            vf.TrajectoryStep(
                prompt=[_create_image_message(red_url, "What color is this?")],
                completion=[{"role": "assistant", "content": "Red"}],
                response=MagicMock(),
                tokens=MagicMock(),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
            # Turn 2: cumulative prompt with red image + green image
            vf.TrajectoryStep(
                prompt=[
                    _create_image_message(red_url, "What color is this?"),
                    {"role": "assistant", "content": "Red"},
                    _create_image_message(green_url, "And this one?"),
                ],
                completion=[{"role": "assistant", "content": "Green"}],
                response=MagicMock(),
                tokens=MagicMock(),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    all_images, images_per_step = _extract_images_from_examples([(1, output)])

    assert len(all_images) == 2  # 2 unique images total
    assert images_per_step == {1: [1, 2]}  # 1 after step 0, 2 after step 1


def test_extract_images_from_examples_multi_turn_no_new_images():
    """Test turns where no new images are added."""
    red_url = _create_test_image("red")

    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[_create_image_message(red_url)],
                completion=[{"role": "assistant", "content": "Red"}],
                response=MagicMock(),
                tokens=MagicMock(),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
            # Turn 2: same image, no new ones
            vf.TrajectoryStep(
                prompt=[
                    _create_image_message(red_url),
                    {"role": "assistant", "content": "Red"},
                    {"role": "user", "content": "Are you sure?"},  # text only
                ],
                completion=[{"role": "assistant", "content": "Yes"}],
                response=MagicMock(),
                tokens=MagicMock(),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    all_images, images_per_step = _extract_images_from_examples([(1, output)])

    assert len(all_images) == 1  # Only 1 unique image
    assert images_per_step == {1: [1, 1]}  # 1 after step 0, still 1 after step 1


def test_vlm_image_cache_get_for_step():
    cache_data = {
        1: [
            ([[1.0, 2.0]], [[1, 2, 3]]),  # Step 0: 1 image
            ([[1.0, 2.0], [3.0, 4.0]], [[1, 2, 3], [1, 4, 4]]),  # Step 1: 2 images cumulative
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    # Step 0 should have 1 image
    pv, grid = cache.get_for_step(1, 0)
    assert pv == [[1.0, 2.0]]
    assert grid == [[1, 2, 3]]

    # Step 1 should have 2 images
    pv, grid = cache.get_for_step(1, 1)
    assert pv == [[1.0, 2.0], [3.0, 4.0]]
    assert grid == [[1, 2, 3], [1, 4, 4]]


def test_vlm_image_cache_get_all():
    cache_data = {
        1: [
            ([[1.0]], [[1, 2, 3]]),
            ([[1.0], [2.0]], [[1, 2, 3], [1, 4, 4]]),
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    # get_all should return the last step's data
    pv, grid = cache.get_all(1)
    assert pv == [[1.0], [2.0]]
    assert grid == [[1, 2, 3], [1, 4, 4]]


def test_vlm_image_cache_step_out_of_range():
    cache_data = {
        1: [
            ([[1.0]], [[1, 2, 3]]),
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    pv, grid = cache.get_for_step(1, 2)
    assert pv is None
    assert grid is None


def test_vlm_image_cache_missing_example():
    cache = VLMImageCache({}, num_unique_examples=0, extract_time=0.0, preprocess_time=0.0)

    pv, grid = cache.get_for_step(999, 0)
    assert pv is None
    assert grid is None

    pv, grid = cache.get_all(999)
    assert pv is None
    assert grid is None


def test_interleave_rollout_with_vlm_cache():
    """Test that interleave_rollout correctly uses per-step images from VLM cache."""
    cache_data = {
        1: [
            ([[1.0]], [[1, 2, 3]]),  # Step 0
            ([[1.0], [2.0]], [[1, 2, 3], [1, 4, 4]]),  # Step 1
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Turn 1"}],
                completion=[{"role": "assistant", "content": "Response 1"}],
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
                trajectory_id="1",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Turn 2"}],
                completion=[{"role": "assistant", "content": "Response 2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5],
                    prompt_mask=[0, 0, 0, 0, 0],
                    completion_ids=[6, 7],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    rollouts = interleave_rollout(output, vlm_cache=cache)

    # Extension holds (step 1 prompt [1,2,3,4,5] extends prefix [1,2,3,4])
    # so both steps merge into a single sample with cumulative images from step 1
    assert rollouts is not None
    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.completion_ids == [3, 4, 5, 6, 7]
    assert rollout.completion_mask == [True, True, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0.0, -0.3, -0.4]
    # Images: cumulative from last merged step (step 1 has 2 images)
    assert rollout.pixel_values == [[1.0], [2.0]]
    assert rollout.image_grid_thw == [[1, 2, 3], [1, 4, 4]]


def test_interleave_rollout_uses_cache_key_override():
    cache_data = {
        7: [
            ([[9.0]], [[1, 2, 3]]),
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    output = vf.RolloutOutput(
        example_id=123,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Turn 1"}],
                completion=[{"role": "assistant", "content": "Response 1"}],
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
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    rollouts = interleave_rollout(output, vlm_cache=cache, cache_key=7)

    assert rollouts is not None
    assert len(rollouts) == 1
    assert rollouts[0].pixel_values == [[9.0]]
    assert rollouts[0].image_grid_thw == [[1, 2, 3]]


def test_interleave_rollout_vlm_image_then_text_turns():
    """
    VLM 3-step trajectory: image in step 0, text-only in steps 1 and 2.
    Extension holds throughout so all steps merge into 1 sample carrying
    step 0's pixel_values (no new images added in later steps).
    """
    cache_data = {
        1: [
            ([[1.0, 2.0]], [[1, 3, 3]]),  # Step 0: 1 image
            ([[1.0, 2.0]], [[1, 3, 3]]),  # Step 1: same 1 image (no new)
            ([[1.0, 2.0]], [[1, 3, 3]]),  # Step 2: same 1 image (no new)
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            # Step 0: user sends image
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Describe"}],
                completion=[{"role": "assistant", "content": "A cat"}],
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
                trajectory_id="1",
                extras={},
            ),
            # Step 1: text-only follow-up (extension holds)
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "More detail"}],
                completion=[{"role": "assistant", "content": "Fluffy"}],
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
                trajectory_id="1",
                extras={},
            ),
            # Step 2: another text-only follow-up (extension holds)
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Color?"}],
                completion=[{"role": "assistant", "content": "Orange"}],
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
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    rollouts = interleave_rollout(output, vlm_cache=cache)

    # All 3 steps merge into 1 sample (extension always holds)
    assert rollouts is not None
    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    # completion: step0 [3,4] + step1 new prompt [5,6] + step1 completion [7,8]
    #             + step2 new prompt [9,10] + step2 completion [11,12]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert rollout.completion_mask == [True, True, False, False, True, True, False, False, True, True]
    # pixel_values from step 2 (cumulative = same 1 image throughout)
    assert rollout.pixel_values == [[1.0, 2.0]]
    assert rollout.image_grid_thw == [[1, 3, 3]]


def test_interleave_rollout_vlm_new_image_mid_conversation():
    """
    VLM 3-step trajectory: image in step 0, text in step 1, NEW image in step 2.
    Extension holds throughout, so 1 merged sample with cumulative images from step 2.
    """
    cache_data = {
        1: [
            ([[1.0]], [[1, 2, 3]]),  # Step 0: 1 image
            ([[1.0]], [[1, 2, 3]]),  # Step 1: still 1 image
            ([[1.0], [2.0]], [[1, 2, 3], [1, 4, 4]]),  # Step 2: 2 images
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Image 1"}],
                completion=[{"role": "assistant", "content": "A"}],
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
                trajectory_id="1",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Text only"}],
                completion=[{"role": "assistant", "content": "B"}],
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
                trajectory_id="1",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Image 2"}],
                completion=[{"role": "assistant", "content": "C"}],
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
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    rollouts = interleave_rollout(output, vlm_cache=cache)

    assert rollouts is not None
    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Cumulative images from last merged step (step 2): both images
    assert rollout.pixel_values == [[1.0], [2.0]]
    assert rollout.image_grid_thw == [[1, 2, 3], [1, 4, 4]]


def test_interleave_rollout_vlm_extension_break():
    """
    VLM 3-step trajectory where extension breaks at step 2.
    Step 0 has image, step 1 extends (text-only), step 2 breaks (different prefix).
    Should produce 2 samples, each with their own cumulative images.
    """
    cache_data = {
        1: [
            ([[1.0]], [[1, 2, 3]]),  # Step 0: 1 image
            ([[1.0]], [[1, 2, 3]]),  # Step 1: still 1 image
            ([[1.0], [2.0]], [[1, 2, 3], [1, 4, 4]]),  # Step 2: 2 images (new image added)
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Image 1"}],
                completion=[{"role": "assistant", "content": "A"}],
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
                trajectory_id="1",
                extras={},
            ),
            # Step 1: extends step 0
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Follow-up"}],
                completion=[{"role": "assistant", "content": "B"}],
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
                trajectory_id="1",
                extras={},
            ),
            # Step 2: extension breaks (different prefix, e.g. context compaction)
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Image 2"}],
                completion=[{"role": "assistant", "content": "C"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[100, 101, 102, 103],
                    prompt_mask=[0, 0, 0, 0],
                    completion_ids=[104, 105],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.5, -0.6],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    rollouts = interleave_rollout(output, vlm_cache=cache)

    assert rollouts is not None
    assert len(rollouts) == 2

    # Sample 1: steps 0-1 merged, images from step 1 (still 1 image)
    assert rollouts[0].prompt_ids == [1, 2]
    assert rollouts[0].completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollouts[0].pixel_values == [[1.0]]
    assert rollouts[0].image_grid_thw == [[1, 2, 3]]

    # Sample 2: step 2 alone (extension broke), images from step 2 (2 images)
    assert rollouts[1].prompt_ids == [100, 101, 102, 103]
    assert rollouts[1].completion_ids == [104, 105]
    assert rollouts[1].pixel_values == [[1.0], [2.0]]
    assert rollouts[1].image_grid_thw == [[1, 2, 3], [1, 4, 4]]


def test_interleave_rollout_vlm_image_appears_late():
    """
    VLM 3-step trajectory: text-only in steps 0 and 1, first image in step 2.
    Extension holds throughout so all steps merge into 1 sample.
    The sample should have pixel_values=None until step 2 sets them.
    """
    cache_data = {
        1: [
            (None, None),  # Step 0: no images
            (None, None),  # Step 1: no images
            ([[5.0, 6.0]], [[1, 3, 3]]),  # Step 2: first image appears
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            # Step 0: text-only
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Hello"}],
                completion=[{"role": "assistant", "content": "Hi"}],
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
                trajectory_id="1",
                extras={},
            ),
            # Step 1: text-only (extension holds)
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Question"}],
                completion=[{"role": "assistant", "content": "Answer"}],
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
                trajectory_id="1",
                extras={},
            ),
            # Step 2: user sends image (extension holds)
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Describe this"}],
                completion=[{"role": "assistant", "content": "A photo"}],
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
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    rollouts = interleave_rollout(output, vlm_cache=cache)

    assert rollouts is not None
    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.prompt_ids == [1, 2]
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert rollout.completion_mask == [True, True, False, False, True, True, False, False, True, True]
    # pixel_values from step 2 (the first step with an image)
    assert rollout.pixel_values == [[5.0, 6.0]]
    assert rollout.image_grid_thw == [[1, 3, 3]]


def test_interleave_rollout_empty_trajectory():
    """Empty trajectory returns None."""
    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[],
        error=None,
    )
    assert interleave_rollout(output) is None


def test_interleave_rollout_error_masks_all_false():
    """
    When rollout output has an error, all completion_mask values should be False
    across both make_sample (step 0) and extend_sample (step 1).
    """
    output = vf.RolloutOutput(
        example_id=1,
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
                trajectory_id="1",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U2"}],
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
                trajectory_id="1",
                extras={},
            ),
        ],
        error="timeout: environment exceeded time limit",
        sampling_args={"temperature": 0.8},
    )

    rollouts = interleave_rollout(output)

    assert rollouts is not None
    assert len(rollouts) == 1
    rollout = rollouts[0]
    # Extension holds so tokens merge, but ALL completion_mask should be False
    assert rollout.completion_ids == [3, 4, 5, 6, 7, 8]
    assert rollout.completion_mask == [False, False, False, False, False, False]
    # Logprobs and temperatures still present
    assert rollout.completion_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
    assert rollout.completion_temperatures == [0.8] * 6


def test_interleave_rollout_vlm_interleaved_agents():
    """
    VLM + interleaved agents: agent1 and agent2 interleaved, each with images.
    agent1 gets cumulative images from its own steps, agent2 from its step.

    Steps (0-indexed):
      0: agent1-step1 (image A)
      1: agent1-step2 (extends step 0, image A still)
      2: agent2-step1 (different prefix, image B)
      3: agent1-step3 (extends step 0+1, image A + new image C)

    Expected: 2 samples
      - agent1: merged steps 0,1,3 → pixel_values from step 3 (images A+C)
      - agent2: step 2 alone → pixel_values from step 2 (image B)
    """
    cache_data = {
        1: [
            ([[1.0]], [[1, 2, 2]]),  # Step 0: image A
            ([[1.0]], [[1, 2, 2]]),  # Step 1: still image A
            ([[9.0]], [[1, 5, 5]]),  # Step 2: image B (agent2)
            ([[1.0], [3.0]], [[1, 2, 2], [1, 3, 3]]),  # Step 3: images A+C (agent1)
        ],
    }
    cache = VLMImageCache(cache_data, num_unique_examples=1, extract_time=0.0, preprocess_time=0.0)

    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            # Step 0: agent1-step1
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Image A"}],
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
                trajectory_id="1",
                extras={},
            ),
            # Step 1: agent1-step2 (extends step 0)
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Follow-up"}],
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
                trajectory_id="1",
                extras={},
            ),
            # Step 2: agent2-step1 (different prefix)
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Image B"}],
                completion=[{"role": "assistant", "content": "B1"}],
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
                trajectory_id="1",
                extras={},
            ),
            # Step 3: agent1-step3 (extends agent1, merges back)
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Image C added"}],
                completion=[{"role": "assistant", "content": "A3"}],
                response=MagicMock(),
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
                trajectory_id="1",
                extras={},
            ),
        ],
        error=None,
        sampling_args={"temperature": 1.0},
    )

    rollouts = interleave_rollout(output, vlm_cache=cache)

    assert rollouts is not None
    assert len(rollouts) == 2

    # Agent1: steps 0,1,3 merged → images from step 3 (A+C)
    agent1 = rollouts[0]
    assert agent1.prompt_ids == [1, 2]
    assert agent1.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert agent1.completion_mask == [True, True, False, False, True, True, False, False, True, True]
    assert agent1.pixel_values == [[1.0], [3.0]]
    assert agent1.image_grid_thw == [[1, 2, 2], [1, 3, 3]]

    # Agent2: step 2 alone → images from step 2 (B)
    agent2 = rollouts[1]
    assert agent2.prompt_ids == [100, 101]
    assert agent2.completion_ids == [102, 103]
    assert agent2.completion_mask == [True, True]
    assert agent2.pixel_values == [[9.0]]
    assert agent2.image_grid_thw == [[1, 5, 5]]


def test_build_vlm_image_cache_handles_divergent_rollouts():
    """Test that build_vlm_image_cache keys images per rollout when trajectories diverge."""
    import torch

    red_url = _create_test_image("red")
    blue_url = _create_test_image("blue")
    green_url = _create_test_image("green")

    rollout_a = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[_create_image_message(red_url, "What color?")],
                completion=[{"role": "assistant", "content": "Red"}],
                response=MagicMock(),
                tokens=MagicMock(),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    rollout_b = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[_create_image_message(blue_url, "What color?")],
                completion=[{"role": "assistant", "content": "Blue"}],
                response=MagicMock(),
                tokens=MagicMock(),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    _create_image_message(blue_url, "What color?"),
                    {"role": "assistant", "content": "Blue"},
                    _create_image_message(green_url, "And this one?"),
                ],
                completion=[{"role": "assistant", "content": "Green"}],
                response=MagicMock(),
                tokens=MagicMock(),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    # Mock processor that returns predictable tensors
    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock(
        side_effect=lambda images, return_tensors: {
            "pixel_values": torch.arange(len(images), dtype=torch.float32).view(-1, 1),
            "image_grid_thw": torch.tensor([[1, 1, 1]] * len(images)),
        }
    )

    rollouts = [rollout_a, rollout_b]
    cache = build_vlm_image_cache(rollouts, mock_processor)

    assert cache.num_unique_examples == 1

    pv, grid = cache.get_for_step(0, 0)
    assert pv == [[0.0]]
    assert grid == [[1, 1, 1]]

    pv, grid = cache.get_for_step(1, 0)
    assert pv == [[1.0]]
    assert grid == [[1, 1, 1]]

    pv, grid = cache.get_for_step(1, 1)
    assert pv == [[1.0], [2.0]]
    assert grid == [[1, 1, 1], [1, 1, 1]]


def test_build_vlm_image_cache_no_images():
    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "Hello"}],
                completion=[{"role": "assistant", "content": "Hi"}],
                response=MagicMock(),
                tokens=MagicMock(),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            )
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    cache = build_vlm_image_cache([output], MagicMock())

    pv, grid = cache.get_for_step(0, 0)
    assert pv is None
    assert grid is None
