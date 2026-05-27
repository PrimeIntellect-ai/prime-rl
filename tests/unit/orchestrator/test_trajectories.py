from unittest.mock import MagicMock

import numpy as np
import pybase64
import pytest
import verifiers as vf

from prime_rl.orchestrator.trajectories import (
    _deserialize_tool_calls,
    align_routed_experts,
    interleave_rollout,
)

_interleave_rollout = interleave_rollout


def interleave_rollout(output, *args, **kwargs):
    output.setdefault("env_name", "test-env")
    return _interleave_rollout(output, *args, **kwargs)


def _decode_mm_pixels(sample) -> list:
    """Decode ``sample.mm_kwargs['pixel_values']`` to a nested list."""
    p = sample.mm_kwargs["pixel_values"]
    return np.frombuffer(p.data, dtype=np.dtype(p.dtype)).reshape(p.shape).tolist()


def _decode_mm_thw(sample) -> list:
    """Decode ``sample.mm_kwargs['image_grid_thw']`` to a nested list."""
    g = sample.mm_kwargs["image_grid_thw"]
    return np.frombuffer(g.data, dtype=np.dtype(g.dtype)).reshape(g.shape).tolist()


def _routed_experts_payload(data) -> dict:
    arr = np.asarray(data, dtype=np.uint8)
    return {
        "data": pybase64.b64encode(memoryview(np.ascontiguousarray(arr))).decode("ascii"),
        "shape": list(arr.shape),
    }


def _sample_routed_experts(sample) -> np.ndarray:
    assert sample.routed_experts is not None
    return np.frombuffer(sample.routed_experts.data, dtype=np.dtype(sample.routed_experts.dtype)).reshape(
        sample.routed_experts.shape
    )


def test_deserialize_tool_calls_does_not_inject_missing_key():
    messages = [{"role": "assistant", "content": "hello"}]

    deserialized = _deserialize_tool_calls(messages)

    assert "tool_calls" not in deserialized[0]


def test_deserialize_tool_calls_parses_arguments_when_present():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"x": 1}'},
                }
            ],
        }
    ]

    deserialized = _deserialize_tool_calls(messages)

    assert deserialized[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}


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
    single_step_trajectory_output["env_name"] = "test-env"
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
    assert rollout.env_name == "test-env"


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


def test_align_routed_experts_none():
    assert align_routed_experts(None, 10) is None


def test_align_routed_experts_empty():
    experts = np.empty((0, 2, 2), dtype=np.uint8)
    result = align_routed_experts(experts, 10)
    assert result is not None
    assert result.shape == (10, 2, 2)
    assert np.all(result == 0)


def test_align_routed_experts_no_deficit():
    # 3 tokens, 2 layers, topk=2
    experts = np.asarray([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 2], [1, 3]]], dtype=np.uint8)
    result = align_routed_experts(experts, expected_len=3)
    np.testing.assert_array_equal(result, experts)


def test_align_routed_experts_with_deficit():
    # 2 tokens but expected 4 (deficit of 2)
    experts = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 0]]], dtype=np.uint8)
    result = align_routed_experts(experts, expected_len=4)
    assert result is not None
    assert result.shape == (4, 2, 2)
    np.testing.assert_array_equal(result[:2], experts)
    # Padded entries should be zero-filled with same shape [layers=2, topk=2]
    np.testing.assert_array_equal(result[2], [[0, 0], [0, 0]])
    np.testing.assert_array_equal(result[3], [[0, 0], [0, 0]])


def test_align_routed_experts_excess_length():
    experts = np.asarray([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.uint8)
    result = align_routed_experts(experts, expected_len=2)
    np.testing.assert_array_equal(result, experts[:2])


def test_interleave_rollout_single_step_with_routed_experts():
    """Routed experts are aligned and passed through for a single-step trajectory."""
    # prompt_ids=[1,2], completion_ids=[3,4] -> total 4 tokens
    # vLLM returns num_tokens-1 = 3 routed expert entries
    routed_experts_from_vllm = np.asarray([[[0, 1]], [[2, 3]], [[4, 5]]], dtype=np.uint8)
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
                    routed_experts=_routed_experts_payload(routed_experts_from_vllm),
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

    rollouts = interleave_rollout(output)
    assert rollouts is not None
    assert len(rollouts) == 1
    sample = rollouts[0]

    # Should be aligned to 4 tokens (2 prompt + 2 completion)
    assert sample.routed_experts is not None
    routed_experts = _sample_routed_experts(sample)
    assert routed_experts.shape == (4, 1, 2)
    # First 3 are original, last one is zero-padded
    np.testing.assert_array_equal(routed_experts[:3], routed_experts_from_vllm)
    np.testing.assert_array_equal(routed_experts[3], [[0, 0]])


def test_interleave_rollout_multi_step_with_routed_experts():
    """Routed experts are extended and aligned across multi-step trajectories."""
    # Step 1: prompt=[1,2], completion=[3,4] -> 4 tokens, vLLM returns 3
    step1_experts = np.asarray([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.uint8)
    # Step 2: prompt=[1,2,3,4,5,6], completion=[7,8] -> 8 tokens, vLLM returns 7
    step2_experts = np.asarray([[[1, 0]], [[2, 0]], [[3, 0]], [[4, 0]], [[5, 0]], [[6, 0]], [[7, 0]]], dtype=np.uint8)

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
                    routed_experts=_routed_experts_payload(step1_experts),
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
                    routed_experts=_routed_experts_payload(step2_experts),
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

    rollouts = interleave_rollout(output)
    assert rollouts is not None
    assert len(rollouts) == 1
    sample = rollouts[0]

    # Merged sample: prompt=[1,2], completion=[3,4,5,6,7,8] -> 8 tokens total
    assert len(sample.prompt_ids) + len(sample.completion_ids) == 8
    assert sample.routed_experts is not None
    assert _sample_routed_experts(sample).shape == (8, 1, 2)


def test_interleave_rollout_none_routed_experts_stays_none():
    """When routed_experts is None, sample.routed_experts remains None."""
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
                    routed_experts=None,
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

    rollouts = interleave_rollout(output)
    assert rollouts is not None
    assert rollouts[0].routed_experts is None


# =============================================================================
# Renderer-emitted multimodal data
# =============================================================================


def test_interleave_rollout_packs_pixels_from_renderer_mm_data():
    """``interleave_rollout`` packs renderer-emitted ``multi_modal_data``
    (pixel_values / image_grid_thw / mm_token_type_ids) onto the
    TrainingSample.

    verifiers' ``_delta_intermediate_mm_data`` ships per-step *delta*
    mm_data (each step contains only items not present in the prior
    step's cumulative set). Prime-rl unions across the sample's step
    range to recover the cumulative set in image-placeholder order.
    """
    import torch as _torch
    from renderers.base import MultiModalData, PlaceholderRange

    # Two synthetic single-image items — values are arbitrary, what
    # matters is that the packer concatenates them correctly.
    item1_pv = _torch.tensor([[1.0, 2.0]], dtype=_torch.float32)
    item2_pv = _torch.tensor([[3.0, 4.0]], dtype=_torch.float32)
    item1_thw = _torch.tensor([[1, 2, 3]], dtype=_torch.int64)
    item2_thw = _torch.tensor([[1, 4, 4]], dtype=_torch.int64)

    # Step 0: image h1 (first time it's seen, included in delta).
    mm_step_0 = MultiModalData(
        mm_hashes={"image": ["h1"]},
        mm_placeholders={"image": [PlaceholderRange(offset=1, length=1)]},
        mm_items={"image": [{"pixel_values": item1_pv, "image_grid_thw": item1_thw}]},
    )
    # Step 1: post-delta — only h2 (h1 was dropped because it was in
    # the prior step's cumulative set). Renderer's bridge would have
    # produced cumulative [h1, h2] before verifiers' delta rewrite.
    mm_step_1 = MultiModalData(
        mm_hashes={"image": ["h2"]},
        mm_placeholders={"image": [PlaceholderRange(offset=4, length=1)]},
        mm_items={"image": [{"pixel_values": item2_pv, "image_grid_thw": item2_thw}]},
    )

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
                    multi_modal_data=mm_step_0,
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
                    multi_modal_data=mm_step_1,
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

    # Token 2 is the image placeholder, token 5 is the video placeholder.
    mm_mapping = {2: 1, 5: 2}
    rollouts = interleave_rollout(output, mm_token_type_ids_mapping=mm_mapping)

    assert rollouts is not None and len(rollouts) == 1
    sample = rollouts[0]
    # Extension holds; both steps merge into one sample. mm_data is
    # the union of step 0's delta ([h1]) and step 1's delta ([h2]).
    assert sample.prompt_ids == [1, 2]
    assert sample.completion_ids == [3, 4, 5, 6, 7]
    # Pixel values packed by concatenating step 0's item then step 1's.
    assert _decode_mm_pixels(sample) == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    assert _decode_mm_thw(sample) == [[1, 2, 3], [1, 4, 4]]
    # mm_token_type_ids: image at token 2, video at token 5, rest 0.
    assert sample.mm_token_type_ids == [0, 1, 0, 0, 2, 0, 0]
