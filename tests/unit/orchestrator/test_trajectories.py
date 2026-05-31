from unittest.mock import MagicMock

import numpy as np
import pybase64
import pytest
import verifiers as vf

from pydantic import ValidationError

from prime_rl.configs.orchestrator import (
    AssistantRoleEchoConfig,
    EchoConfig,
    EchoFilterConfig,
    SystemRoleEchoConfig,
    ToolRoleEchoConfig,
    UserRoleEchoConfig,
)
from prime_rl.orchestrator.trajectories import (
    _deserialize_tool_calls,
    _step_echo_alpha,
    align_routed_experts,
    apply_echo_filter,
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


def _routed_experts_payload(data, start: int = 0) -> dict:
    arr = np.asarray(data, dtype=np.uint8)
    return {
        "data": pybase64.b64encode(memoryview(np.ascontiguousarray(arr))).decode("ascii"),
        "shape": list(arr.shape),
        "start": start,
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
    assert rollout.completion_temperatures == []

    # second step
    rollout = rollouts[1]
    assert rollout.prompt_ids == [10, 20, 30, 40, 50, 60]
    assert rollout.prompt_mask == [False, False, False, False, False, False]
    assert rollout.completion_ids == [7, 8]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.3, -0.4]
    assert rollout.completion_temperatures == []


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
    assert rollout.completion_temperatures == []

    # second step
    rollout = rollouts[1]
    assert rollout.prompt_ids == [10, 20, 30, 40, 50, 60]
    assert rollout.prompt_mask == [False, False, False, False, False, False]
    assert rollout.completion_ids == [7, 8]
    assert rollout.completion_mask == [True, True]
    assert rollout.completion_logprobs == [-0.3, -0.4]
    assert rollout.completion_temperatures == []


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
    assert rollout.completion_temperatures == []
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
    # ``completion_temperatures`` is filled by the orchestrator post-interleave; empty here.
    assert rollout.completion_temperatures == []


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
    # ``completion_temperatures`` is filled by the orchestrator post-interleave; empty here.
    assert rollout.completion_temperatures == []


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


@pytest.fixture
def prefix_of_prefix_trajectory():
    """
    Trajectory where one active sample's prefix is a strict prefix of another's.

    Construction:
    - step 0: prompt=[1,2], completion=[3,4]                  -> sample A, P_A=[1,2,3,4]
    - step 1: extends A. prompt=[1,2,3,4,5], completion=[6]   -> P_A=[1,2,3,4,5,6]
    - step 2: rollback/regenerate. prompt=[1,2] (shorter than P_A so no match),
              completion=[3,4,5,6,7]                          -> sample B, P_B=[1,2,3,4,5,6,7]
              P_B starts with P_A.
    - step 3: extends B. prompt=[1,2,3,4,5,6,7,8], completion=[9]
              Both P_A and P_B are token-prefixes of the step's prompt.

    The correct match is the longer P_B. First-match-wins picks P_A and silently
    folds B's generated tokens into A as user-input tokens (mask=False).
    """
    output = vf.RolloutOutput(
        example_id=2,
        task="test",
        trajectory=[
            vf.TrajectoryStep(
                prompt="step 0",
                completion="completion 0",
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
                trajectory_id="traj_A",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt="step 1",
                completion="completion 1",
                response=None,
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5],
                    prompt_mask=[0, 0, 0, 0, 0],
                    completion_ids=[6],
                    completion_mask=[1],
                    completion_logprobs=[-0.3],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="traj_A",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt="step 2 (rollback)",
                completion="completion 2",
                response=None,
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4, 5, 6, 7],
                    completion_mask=[1, 1, 1, 1, 1],
                    completion_logprobs=[-0.4, -0.5, -0.6, -0.7, -0.8],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="traj_B",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt="step 3 (extends B)",
                completion="completion 3",
                response=None,
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                    prompt_mask=[0, 0, 0, 0, 0, 0, 0, 0],
                    completion_ids=[9],
                    completion_mask=[1],
                    completion_logprobs=[-0.9],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="traj_B",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )
    return output


def test_interleave_rollout_picks_longest_matching_prefix(prefix_of_prefix_trajectory):
    """
    When two active samples both match (one's prefix is a strict prefix of the
    other's), the longer prefix is the correct extension. Previously the first-
    match-wins loop folded the longer sample's generated tokens into the shorter
    sample as user input (mask=False) and left the longer sample stale.
    """
    rollouts = interleave_rollout(prefix_of_prefix_trajectory)

    assert rollouts is not None
    assert len(rollouts) == 2

    # Sample A: steps 0 and 1 only. Step 3 must NOT have been folded in here.
    sample_a = rollouts[0]
    assert sample_a.prompt_ids == [1, 2]
    # step 0 completion [3,4] + step 1 new prompt [5] + step 1 completion [6]
    assert sample_a.completion_ids == [3, 4, 5, 6]
    assert sample_a.completion_mask == [True, True, False, True]
    assert sample_a.completion_logprobs == [-0.1, -0.2, 0.0, -0.3]

    # Sample B: steps 2 and 3 merged. The token 7 (from step 2's completion)
    # must remain masked as a generated token, not silently re-classified.
    sample_b = rollouts[1]
    assert sample_b.prompt_ids == [1, 2]
    # step 2 completion [3,4,5,6,7] + step 3 new prompt [8] + step 3 completion [9]
    assert sample_b.completion_ids == [3, 4, 5, 6, 7, 8, 9]
    assert sample_b.completion_mask == [True, True, True, True, True, False, True]
    assert sample_b.completion_logprobs == [-0.4, -0.5, -0.6, -0.7, -0.8, 0.0, -0.9]


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
    # Logprobs preserved; ``completion_temperatures`` is filled by the orchestrator post-interleave.
    assert rollout.completion_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
    assert rollout.completion_temperatures == []


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
    # Step 2: prompt=[1,2,3,4,5,6], completion=[7,8], bridged from prefix len 4.
    # vLLM returns routed experts starting at row 3: boundary token 4, then 5, 6, 7.
    step2_experts = np.asarray([[[40, 41]], [[50, 51]], [[60, 61]], [[70, 71]]], dtype=np.uint8)

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
                    routed_experts=_routed_experts_payload(step2_experts, start=3),
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
    routed_experts = _sample_routed_experts(sample)
    assert routed_experts.shape == (8, 1, 2)
    np.testing.assert_array_equal(
        routed_experts,
        np.asarray(
            [
                [[1, 2]],
                [[3, 4]],
                [[5, 6]],
                [[40, 41]],
                [[50, 51]],
                [[60, 61]],
                [[70, 71]],
                [[0, 0]],
            ],
            dtype=np.uint8,
        ),
    )


def test_interleave_rollout_branch_delta_uses_prior_routed_prefix():
    step1_experts = np.asarray([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.uint8)
    step2_experts = np.asarray([[[40, 41]], [[50, 51]], [[60, 61]], [[70, 71]]], dtype=np.uint8)
    step3_experts = np.asarray([[[80, 81]], [[90, 91]], [[100, 101]], [[110, 111]]], dtype=np.uint8)

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
                    routed_experts=_routed_experts_payload(step2_experts, start=3),
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
                    {"role": "user", "content": "branch"},
                ],
                completion=[{"role": "assistant", "content": "A3"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 9, 10],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[11, 12],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.5, -0.6],
                    overlong_prompt=False,
                    is_truncated=False,
                    routed_experts=_routed_experts_payload(step3_experts, start=3),
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
    assert len(rollouts) == 2
    branched = rollouts[1]
    assert branched.prompt_ids == [1, 2, 3, 4, 9, 10]
    assert branched.completion_ids == [11, 12]
    routed_experts = _sample_routed_experts(branched)
    assert routed_experts.shape == (8, 1, 2)
    np.testing.assert_array_equal(
        routed_experts,
        np.asarray(
            [
                [[1, 2]],
                [[3, 4]],
                [[5, 6]],
                [[80, 81]],
                [[90, 91]],
                [[100, 101]],
                [[110, 111]],
                [[0, 0]],
            ],
            dtype=np.uint8,
        ),
    )


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


# ---------------------------------------------------------------------------
# Per-role echo_alpha construction
# ---------------------------------------------------------------------------


def _attribution(
    message_indices: list[int],
    is_content: list[bool],
    message_roles: list[str] | None = None,
    message_tool_names: list[str | None] | None = None,
) -> dict:
    """Minimal stand-in for the serialised ``renderers.base.RenderedTokens``
    dict that the verifiers env-server hands to ``_step_echo_alpha`` — only
    the keys the helper subscripts are populated."""
    out: dict = {"message_indices": message_indices, "is_content": is_content}
    if message_roles is not None:
        out["message_roles"] = message_roles
    if message_tool_names is not None:
        out["message_tool_names"] = message_tool_names
    return out


def test_step_echo_alpha_returns_all_none_when_echo_config_is_none():
    """No echo config → all-None array. The helper bails before any
    attribution-shape work."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            message_indices=[0, 0, 1, 1],
            is_content=[False, True, False, True],
            message_roles=["user", "tool"],
            message_tool_names=[None, "lookup"],
        ),
        prompt_len=4,
        completion_len=2,
        echo_config=None,
    )
    assert alpha == [None] * 6


def test_echo_config_requires_at_least_one_role():
    """``EchoConfig`` with every role set to None is meaningless — the
    validator rejects it. The caller should omit ``[echo]`` entirely to
    disable echo for the env."""
    with pytest.raises(ValidationError, match="at least one role"):
        EchoConfig()
    with pytest.raises(ValidationError, match="at least one role"):
        EchoConfig(system=None, user=None, assistant=None, tool=None)


def test_step_echo_alpha_no_prompt_attribution_still_marks_assistant_completion():
    """Non-renderer client rollouts have no attribution → can't mark
    prompt-side. But completion-side assistant echo is independent of
    attribution (the completion is always assistant-role by construction)."""
    alpha = _step_echo_alpha(
        prompt_attribution=None,
        prompt_len=4,
        completion_len=2,
        echo_config=EchoConfig(assistant=AssistantRoleEchoConfig(alpha=0.3)),
    )
    # Prompt-side stays None (no attribution to drive masking), completion
    # bears assistant alpha throughout.
    assert alpha == [None, None, None, None, 0.3, 0.3]


def test_step_echo_alpha_no_prompt_attribution_no_completion_echo():
    """With no attribution AND no assistant-role echo, the result is uniformly
    None — prompt-side requires attribution to resolve roles, completion-side
    only fires when assistant echo is enabled."""
    alpha = _step_echo_alpha(
        prompt_attribution=None,
        prompt_len=4,
        completion_len=2,
        # Tool echo enabled but no attribution to resolve roles against.
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
    )
    assert alpha == [None] * 6


def test_step_echo_alpha_returns_all_none_without_message_roles():
    """Attribution present but no ``message_roles`` sub-field → no per-role
    resolution possible; only completion-side assistant echo can fire."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            message_indices=[0, 0],
            is_content=[False, True],
            message_roles=None,
            message_tool_names=["lookup"],
        ),
        prompt_len=2,
        completion_len=0,
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
    )
    assert alpha == [None, None]


def test_step_echo_alpha_tool_role_default_all_tools():
    """``ToolRoleEchoConfig`` with ``tool_names=None`` means echo every tool's
    body. Scaffold (``is_content=False``) and non-tool messages stay None;
    completion stays None (no assistant-role echo configured)."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            # 5 prompt tokens, 2 completion tokens.
            # Token 0 = user-role scaffold; 1 = user body; 2 = tool wrap;
            # 3-4 = tool body; completion = [None, None].
            message_indices=[0, 0, 1, 1, 1],
            is_content=[False, True, False, True, True],
            message_roles=["user", "tool"],
            message_tool_names=[None, "lookup"],
        ),
        prompt_len=5,
        completion_len=2,
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.7, tool_names=None)),
    )
    # Tool body tokens (indices 3, 4) get alpha; everything else None.
    assert alpha == [None, None, None, 0.7, 0.7, None, None]


def test_step_echo_alpha_tool_role_name_filter():
    """``ToolRoleEchoConfig.tool_names=['lookup']`` masks only the body of
    ``lookup`` tool messages; ``calc`` tool stays None."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            # 6 tokens: msg 0 = user, msg 1 = calc tool, msg 2 = lookup tool.
            message_indices=[0, 0, 1, 1, 2, 2],
            is_content=[False, True, False, True, False, True],
            message_roles=["user", "tool", "tool"],
            message_tool_names=[None, "calc", "lookup"],
        ),
        prompt_len=6,
        completion_len=0,
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5, tool_names=["lookup"])),
    )
    # Only the lookup body (idx 5) gets alpha; calc body (idx 3) stays None.
    assert alpha == [None, None, None, None, None, 0.5]


def test_step_echo_alpha_skips_non_content_tokens():
    """Scaffold tokens inside a tool message (``is_content=False``) stay
    None even when the message is in the tool allowlist. The body/scaffold
    cut is the load-bearing invariant."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            # All four tokens belong to a tool message; only token 2 is body.
            message_indices=[0, 0, 0, 0],
            is_content=[False, False, True, False],
            message_roles=["tool"],
            message_tool_names=["lookup"],
        ),
        prompt_len=4,
        completion_len=0,
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.4)),
    )
    assert alpha == [None, None, 0.4, None]


def test_step_echo_alpha_user_role():
    """``UserRoleEchoConfig`` marks the body of user-role messages.
    System/tool/assistant roles stay None unless their own role configs are
    set."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            # 4 tokens: msg 0 = user (wrap + body), msg 1 = tool (wrap + body).
            message_indices=[0, 0, 1, 1],
            is_content=[False, True, False, True],
            message_roles=["user", "tool"],
            message_tool_names=[None, "lookup"],
        ),
        prompt_len=4,
        completion_len=0,
        echo_config=EchoConfig(user=UserRoleEchoConfig(alpha=0.2), tool=None),
    )
    # User body (idx 1) gets alpha; tool body (idx 3) stays None (tool=None).
    assert alpha == [None, 0.2, None, None]


def test_step_echo_alpha_system_role():
    """``SystemRoleEchoConfig`` marks the body of system-role messages —
    e.g. for progressive-compression curricula where the model has to
    internalize the system prompt over training."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            # 3 tokens: msg 0 = system (wrap + body + body).
            message_indices=[0, 0, 0],
            is_content=[False, True, True],
            message_roles=["system"],
        ),
        prompt_len=3,
        completion_len=0,
        echo_config=EchoConfig(system=SystemRoleEchoConfig(alpha=0.1), tool=None),
    )
    assert alpha == [None, 0.1, 0.1]


def test_step_echo_alpha_assistant_role_prompt_and_completion():
    """``AssistantRoleEchoConfig`` marks BOTH prompt-side assistant messages
    (prior turns in multi-turn rollouts) AND the current step's completion.
    Completion-side marking is independent of ``is_content`` (which is
    prompt-side-only)."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            # 4 tokens: msg 0 = user (wrap + body), msg 1 = assistant (wrap + body).
            message_indices=[0, 0, 1, 1],
            is_content=[False, True, False, True],
            message_roles=["user", "assistant"],
        ),
        prompt_len=4,
        completion_len=3,
        echo_config=EchoConfig(assistant=AssistantRoleEchoConfig(alpha=0.8), tool=None),
    )
    # Prompt-side assistant body (idx 3) gets alpha; all 3 completion tokens
    # get alpha (current step's assistant emission).
    assert alpha == [None, None, None, 0.8, 0.8, 0.8, 0.8]


def test_step_echo_alpha_assistant_zero_kills_rl():
    """The canonical alpha=0 use case: explicit ``AssistantRoleEchoConfig(alpha=0.0)``
    overrides the RL gradient on assistant tokens by setting per-token alpha
    to 0 (advantage=0, loss_mask=True in prepare_sample → zero gradient).
    Distinct from "not echoed" because the position still gets the overlay
    treatment in the loss function."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            message_indices=[0],
            is_content=[True],
            message_roles=["user"],
        ),
        prompt_len=1,
        completion_len=2,
        echo_config=EchoConfig(assistant=AssistantRoleEchoConfig(alpha=0.0), tool=None),
    )
    # Completion positions carry alpha=0.0 — distinct from None.
    assert alpha == [None, 0.0, 0.0]


def test_step_echo_alpha_per_role_alphas_differ():
    """Each role's per-role config carries its own ``alpha`` — different
    roles can carry different weights in the same rollout."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            # msg 0 = user, msg 1 = tool, msg 2 = system.
            message_indices=[0, 1, 2],
            is_content=[True, True, True],
            message_roles=["user", "tool", "system"],
            message_tool_names=[None, "lookup", None],
        ),
        prompt_len=3,
        completion_len=2,
        echo_config=EchoConfig(
            user=UserRoleEchoConfig(alpha=0.1),
            tool=ToolRoleEchoConfig(alpha=0.5),
            system=SystemRoleEchoConfig(alpha=0.05),
            assistant=AssistantRoleEchoConfig(alpha=0.9),
        ),
    )
    # Prompt: user=0.1, tool=0.5, system=0.05. Completion: assistant=0.9.
    assert alpha == [0.1, 0.5, 0.05, 0.9, 0.9]


# ---------------------------------------------------------------------------
# EchoFilterConfig pydantic
# ---------------------------------------------------------------------------


def test_echo_filter_config_requires_import_path():
    """``import_path`` is mandatory — no implicit default."""
    with pytest.raises(ValidationError, match="import_path"):
        EchoFilterConfig()  # type: ignore[call-arg]


def test_echo_filter_config_kwargs_default_empty():
    """``kwargs`` defaults to an empty dict (not None) — the filter
    invocation does ``**filter_kwargs`` which would fail on None."""
    cfg = EchoFilterConfig(import_path="my_module.my_filter")
    assert cfg.kwargs == {}


def test_echo_filter_config_kwargs_explicit():
    """Caller-supplied kwargs survive validation as a plain dict."""
    cfg = EchoFilterConfig(
        import_path="my_module.my_filter",
        kwargs={"pattern": "warn", "min_lines": 3},
    )
    assert cfg.kwargs == {"pattern": "warn", "min_lines": 3}


def test_echo_config_filter_nests_under_echo():
    """``EchoConfig.filter`` is the documented attachment point — a
    populated ``EchoFilterConfig`` must coexist with at least one role
    (the at-least-one-role validator still applies; the filter is NOT
    a role by itself)."""
    cfg = EchoConfig(
        tool=ToolRoleEchoConfig(alpha=0.05),
        filter=EchoFilterConfig(import_path="my_module.my_filter"),
    )
    assert cfg.filter is not None
    assert cfg.filter.import_path == "my_module.my_filter"


def test_echo_config_filter_without_role_still_rejected():
    """The at-least-one-role validator does NOT treat ``filter`` as a
    role — setting only ``filter`` with no role enabled still raises.
    The filter is a narrowing overlay; it can't enable echo by itself."""
    with pytest.raises(ValidationError, match="at least one role"):
        EchoConfig(filter=EchoFilterConfig(import_path="my_module.my_filter"))


# ---------------------------------------------------------------------------
# _step_echo_alpha — filter_mask composition
# ---------------------------------------------------------------------------


def _tool_only_attribution(prompt_len: int) -> dict:
    """All prompt tokens marked as content of a single tool message named
    ``"lookup"`` — minimal setup for filter-narrowing tests where every
    prompt position has a baseline echo alpha to be narrowed."""
    return _attribution(
        message_indices=[0] * prompt_len,
        is_content=[True] * prompt_len,
        message_roles=["tool"],
        message_tool_names=["lookup"],
    )


def test_step_echo_alpha_filter_none_is_no_op():
    """``filter_mask=None`` (or omitted) preserves the role baseline
    exactly — backwards-compatible default."""
    baseline = _step_echo_alpha(
        prompt_attribution=_tool_only_attribution(3),
        prompt_len=3,
        completion_len=2,
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
    )
    filtered = _step_echo_alpha(
        prompt_attribution=_tool_only_attribution(3),
        prompt_len=3,
        completion_len=2,
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
        filter_mask=None,
    )
    assert filtered == baseline == [0.5, 0.5, 0.5, None, None]


def test_step_echo_alpha_filter_narrows_baseline():
    """Filter ``False`` at a position drops the role-level echo alpha to
    ``None`` (drops to RL gradient). Filter ``True`` preserves the role
    baseline as-is."""
    alpha = _step_echo_alpha(
        prompt_attribution=_tool_only_attribution(4),
        prompt_len=4,
        completion_len=2,
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
        filter_mask=[True, False, True, False, False, False],
    )
    # Positions 0 + 2: filter True, role baseline alpha = 0.5 preserved.
    # Positions 1 + 3: filter False, dropped to None (RL applies).
    # Completion (4, 5): no role baseline (assistant disabled) AND filter
    # False — stays None.
    assert alpha == [0.5, None, 0.5, None, None, None]


def test_step_echo_alpha_filter_cannot_add_echo():
    """Filter ``True`` at a position that had no role baseline keeps that
    position at ``None`` — the filter narrows the baseline, it cannot
    expand it. Critical invariant: a permissive filter cannot accidentally
    turn on echo for roles the user never enabled."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            # msg 0 = user (NOT enabled in echo config), msg 1 = tool.
            message_indices=[0, 0, 1, 1],
            is_content=[True, True, True, True],
            message_roles=["user", "tool"],
            message_tool_names=[None, "lookup"],
        ),
        prompt_len=4,
        completion_len=0,
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
        # All True — filter approves every position.
        filter_mask=[True, True, True, True],
    )
    # User positions (0, 1) stay None — role disabled, filter cannot add.
    # Tool positions (2, 3) carry alpha — role enabled, filter approved.
    assert alpha == [None, None, 0.5, 0.5]


def test_step_echo_alpha_filter_all_true_preserves_baseline():
    """``filter_mask = [True] * N`` is identical to ``filter_mask=None``
    semantically — the all-approve case is a no-op narrowing."""
    args = dict(
        prompt_attribution=_tool_only_attribution(3),
        prompt_len=3,
        completion_len=2,
        echo_config=EchoConfig(
            tool=ToolRoleEchoConfig(alpha=0.5),
            assistant=AssistantRoleEchoConfig(alpha=0.8),
        ),
    )
    without_filter = _step_echo_alpha(**args)
    with_all_true = _step_echo_alpha(**args, filter_mask=[True] * 5)
    assert without_filter == with_all_true == [0.5, 0.5, 0.5, 0.8, 0.8]


def test_step_echo_alpha_filter_all_false_zeros_everything():
    """``filter_mask = [False] * N`` drops every position to ``None``,
    regardless of role baseline — the "kill all echo for this step" case."""
    alpha = _step_echo_alpha(
        prompt_attribution=_tool_only_attribution(3),
        prompt_len=3,
        completion_len=2,
        echo_config=EchoConfig(
            tool=ToolRoleEchoConfig(alpha=0.5),
            assistant=AssistantRoleEchoConfig(alpha=0.8),
        ),
        filter_mask=[False] * 5,
    )
    assert alpha == [None, None, None, None, None]


def test_step_echo_alpha_filter_narrows_assistant_completion():
    """Filter applies to completion-side positions too — assistant
    completion echo can be narrowed by the filter just like prompt-side
    role echoes. Critical for the alpha=0 kill-RL use case (filter lets
    users selectively keep or drop individual completion tokens from
    the override)."""
    alpha = _step_echo_alpha(
        prompt_attribution=None,  # no prompt-side; only completion matters
        prompt_len=2,
        completion_len=4,
        echo_config=EchoConfig(assistant=AssistantRoleEchoConfig(alpha=0.3)),
        # Approve only the first two completion tokens.
        filter_mask=[True, True, True, True, False, False],
    )
    # Prompt-side: assistant role doesn't trigger without attribution,
    # so baseline is None there; filter True/False doesn't matter.
    # Completion-side: alpha=0.3 baseline; first two kept, last two dropped.
    assert alpha == [None, None, 0.3, 0.3, None, None]


def test_step_echo_alpha_filter_length_mismatch_too_short_raises():
    """Filter mask shorter than ``prompt_len + completion_len`` → ValueError
    with both lengths in the message (load-bearing for debugging)."""
    with pytest.raises(ValueError, match="filter_mask length 3.*does not match.*5"):
        _step_echo_alpha(
            prompt_attribution=_tool_only_attribution(3),
            prompt_len=3,
            completion_len=2,
            echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
            filter_mask=[True, True, True],
        )


def test_step_echo_alpha_filter_length_mismatch_too_long_raises():
    """Filter mask longer than ``prompt_len + completion_len`` → ValueError."""
    with pytest.raises(ValueError, match="filter_mask length 6.*does not match.*5"):
        _step_echo_alpha(
            prompt_attribution=_tool_only_attribution(3),
            prompt_len=3,
            completion_len=2,
            echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
            filter_mask=[True] * 6,
        )


def test_step_echo_alpha_filter_validates_before_building_baseline():
    """Length validation runs before any baseline work — so even when
    ``echo_config=None`` (baseline would be all-None anyway), passing a
    wrong-length filter raises. Makes the contract uniformly enforced."""
    with pytest.raises(ValueError, match="filter_mask length"):
        _step_echo_alpha(
            prompt_attribution=None,
            prompt_len=3,
            completion_len=2,
            echo_config=None,
            filter_mask=[True, True],  # length 2 ≠ 5
        )


def test_step_echo_alpha_filter_mixed_roles():
    """Filter composes with a fully-loaded EchoConfig (all four roles
    enabled). Each role's baseline is narrowed independently per-token."""
    alpha = _step_echo_alpha(
        prompt_attribution=_attribution(
            # msg 0 = system, msg 1 = user, msg 2 = tool.
            message_indices=[0, 1, 2],
            is_content=[True, True, True],
            message_roles=["system", "user", "tool"],
            message_tool_names=[None, None, "lookup"],
        ),
        prompt_len=3,
        completion_len=2,
        echo_config=EchoConfig(
            system=SystemRoleEchoConfig(alpha=0.05),
            user=UserRoleEchoConfig(alpha=0.1),
            tool=ToolRoleEchoConfig(alpha=0.5),
            assistant=AssistantRoleEchoConfig(alpha=0.9),
        ),
        # Approve system + tool + first completion token; drop user +
        # second completion token.
        filter_mask=[True, False, True, True, False],
    )
    assert alpha == [0.05, None, 0.5, 0.9, None]


# ---------------------------------------------------------------------------
# apply_echo_filter — shape/type validation + invocation contract
# ---------------------------------------------------------------------------


def _step_with_tokens(prompt_len: int, completion_len: int) -> vf.TrajectoryStep:
    """Build a minimal TrajectoryStep with controllable token lengths.
    Used to exercise ``apply_echo_filter``'s per-step length checks
    without depending on a renderer-produced attribution."""
    return vf.TrajectoryStep(
        prompt=[{"role": "user", "content": "U"}],
        completion=[{"role": "assistant", "content": "A"}],
        response=MagicMock(),
        tokens=vf.TrajectoryStepTokens(
            prompt_ids=list(range(prompt_len)),
            prompt_mask=[0] * prompt_len,
            completion_ids=list(range(prompt_len, prompt_len + completion_len)),
            completion_mask=[1] * completion_len,
            completion_logprobs=[-0.1] * completion_len,
            overlong_prompt=False,
            is_truncated=False,
        ),
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="t",
        extras={},
    )


def _rollout_with_steps(*step_dims: tuple[int, int]) -> vf.RolloutOutput:
    """Build a RolloutOutput from a list of ``(prompt_len, completion_len)``
    tuples. Each tuple yields one trajectory step with those lengths."""
    return vf.RolloutOutput(
        example_id=0,
        trajectory=[_step_with_tokens(p, c) for p, c in step_dims],
        sampling_args={"temperature": 1.0},
        error=None,
    )


def test_apply_echo_filter_valid_returns_masks():
    """A well-behaved filter that returns the right shapes passes through.
    Verifies the happy path as a regression anchor."""
    rollout = _rollout_with_steps((3, 2), (4, 1))

    def filter_fn(rollout):
        return [
            [True, False, True, True, False],  # step 0: 3+2 = 5
            [False, False, True, True, False],  # step 1: 4+1 = 5
        ]

    result = apply_echo_filter(rollout, filter_fn, None)
    assert result == [
        [True, False, True, True, False],
        [False, False, True, True, False],
    ]


def test_apply_echo_filter_outer_length_too_short_raises():
    """Filter returns fewer per-step masks than trajectory steps → ValueError
    naming both counts."""
    rollout = _rollout_with_steps((3, 2), (4, 1))

    def filter_fn(rollout):
        return [[True] * 5]  # only 1 step mask but trajectory has 2

    with pytest.raises(ValueError, match="returned 1 per-step masks.*has 2"):
        apply_echo_filter(rollout, filter_fn, None)


def test_apply_echo_filter_outer_length_too_long_raises():
    """Filter returns more per-step masks than trajectory steps → ValueError."""
    rollout = _rollout_with_steps((3, 2))

    def filter_fn(rollout):
        return [[True] * 5, [True] * 5]  # 2 masks for 1 trajectory step

    with pytest.raises(ValueError, match="returned 2 per-step masks.*has 1"):
        apply_echo_filter(rollout, filter_fn, None)


def test_apply_echo_filter_inner_length_mismatch_raises():
    """Filter step-mask length ≠ ``prompt_len + completion_len`` →
    ValueError pinpointing the step index and both expected/actual."""
    rollout = _rollout_with_steps((3, 2), (4, 1))

    def filter_fn(rollout):
        return [
            [True] * 5,  # step 0: correct (3+2=5)
            [True] * 3,  # step 1: wrong (3 != 4+1=5)
        ]

    with pytest.raises(
        ValueError, match=r"step 1.*mask length 3.*expected 5.*prompt_len=4.*completion_len=1"
    ):
        apply_echo_filter(rollout, filter_fn, None)


def test_apply_echo_filter_non_list_return_raises():
    """Filter returning something that isn't a list at all → TypeError."""
    rollout = _rollout_with_steps((2, 1))

    def filter_fn(rollout):
        return "not a list"  # type: ignore[return-value]

    with pytest.raises(TypeError, match="must return list.*got str"):
        apply_echo_filter(rollout, filter_fn, None)


def test_apply_echo_filter_non_list_inner_raises():
    """Filter outer-list-of-non-lists → TypeError pinpointing step."""
    rollout = _rollout_with_steps((2, 1), (3, 0))

    def filter_fn(rollout):
        return [[True, True, True], "not a list"]  # type: ignore[list-item]

    with pytest.raises(TypeError, match="step 1.*mask must be a list.*str"):
        apply_echo_filter(rollout, filter_fn, None)


def test_apply_echo_filter_non_bool_element_raises_int():
    """Integer 1 is truthy but NOT bool — we reject to prevent silent
    bugs where ``1`` got past the contract. ``isinstance(1, bool)`` would
    return False, but ``type(1) is bool`` catches this explicitly even
    though the two coincide here; we use ``type(v) is bool`` for clarity."""
    rollout = _rollout_with_steps((1, 1))

    def filter_fn(rollout):
        return [[True, 1]]  # second element is int, not bool

    with pytest.raises(TypeError, match=r"step 0.*mask\[1\].*must be a plain bool.*int"):
        apply_echo_filter(rollout, filter_fn, None)


def test_apply_echo_filter_non_bool_element_raises_string():
    """String element → TypeError."""
    rollout = _rollout_with_steps((2, 0))

    def filter_fn(rollout):
        return [[True, "yes"]]  # type: ignore[list-item]

    with pytest.raises(TypeError, match=r"step 0.*mask\[1\].*must be a plain bool.*str"):
        apply_echo_filter(rollout, filter_fn, None)


def test_apply_echo_filter_propagates_user_exception():
    """When the user's filter raises, the exception propagates verbatim —
    no swallowing, no fallback to "no filter for this rollout"."""
    rollout = _rollout_with_steps((2, 1))

    class FilterCrash(RuntimeError):
        pass

    def filter_fn(rollout):
        raise FilterCrash("boom")

    with pytest.raises(FilterCrash, match="boom"):
        apply_echo_filter(rollout, filter_fn, None)


def test_apply_echo_filter_forwards_kwargs():
    """``filter_kwargs`` flow through to the filter as ``**kwargs``. The
    filter's signature can declare them positionally or accept ``**kwargs``."""
    rollout = _rollout_with_steps((2, 0))
    captured: dict = {}

    def filter_fn(rollout, *, pattern: str, threshold: int):
        captured["pattern"] = pattern
        captured["threshold"] = threshold
        return [[True, True]]

    apply_echo_filter(rollout, filter_fn, {"pattern": "warn", "threshold": 3})
    assert captured == {"pattern": "warn", "threshold": 3}


def test_apply_echo_filter_kwargs_none_means_empty():
    """``filter_kwargs=None`` is equivalent to ``filter_kwargs={}`` —
    no kwargs passed, no TypeError from ``**None``."""
    rollout = _rollout_with_steps((1, 1))
    captured: dict = {}

    def filter_fn(rollout, **kwargs):
        captured.update(kwargs)
        return [[True, True]]

    apply_echo_filter(rollout, filter_fn, None)
    assert captured == {}


def test_apply_echo_filter_empty_trajectory_returns_empty_masks():
    """Empty trajectory + filter returning ``[]`` is valid. Edge case that
    falls out of the validation logic but worth pinning down."""
    rollout = vf.RolloutOutput(
        example_id=0,
        trajectory=[],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    def filter_fn(rollout):
        return []

    assert apply_echo_filter(rollout, filter_fn, None) == []


def test_apply_echo_filter_receives_full_rollout():
    """The filter sees the full rollout dict — not a stripped-down view.
    Lets users branch on reward / error / metrics / info / example_id."""
    rollout = _rollout_with_steps((2, 1))
    seen_rollout = {}

    def filter_fn(rollout):
        seen_rollout["example_id"] = rollout["example_id"]
        seen_rollout["error"] = rollout["error"]
        seen_rollout["trajectory_len"] = len(rollout["trajectory"])
        return [[True] * 3]

    apply_echo_filter(rollout, filter_fn, None)
    assert seen_rollout == {"example_id": 0, "error": None, "trajectory_len": 1}


# ---------------------------------------------------------------------------
# interleave_rollout — end-to-end with filter_masks
# ---------------------------------------------------------------------------


def test_interleave_rollout_filter_masks_none_no_op(single_step_trajectory_output):
    """``filter_masks=None`` produces identical samples to the no-filter
    call — verifies the parameter is opt-in and doesn't change baseline
    behavior."""
    rollout = single_step_trajectory_output
    rollout["env_name"] = "test-env"

    baseline = _interleave_rollout(rollout)
    with_none = _interleave_rollout(rollout, filter_masks=None)

    assert len(baseline) == len(with_none) == 1
    assert with_none[0].echo_alpha == baseline[0].echo_alpha
    # Both should be None for this fixture (no echo_config supplied).
    assert with_none[0].echo_alpha is None


def test_interleave_rollout_filter_masks_narrows_sample_echo_alpha():
    """End-to-end: filter False at a position drops the role-level echo
    alpha to None on the resulting sample's ``echo_alpha`` array. The
    most important integration assertion — proves the filter mask flows
    from the orchestrator boundary all the way to the wire format."""
    # Single-step rollout with tool-role attribution on the prompt side.
    rollout = vf.RolloutOutput(
        example_id=0,
        env_name="test-env",
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "tool", "content": "T1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3],
                    prompt_mask=[0, 0, 0],
                    completion_ids=[4, 5],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                    prompt_attribution={
                        "message_indices": [0, 0, 0],
                        "is_content": [True, True, True],
                        "message_roles": ["tool"],
                        "message_tool_names": ["lookup"],
                    },
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="t",
                extras={},
            )
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )
    echo_config = EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5))

    # Without filter: tool body → alpha=0.5 on all 3 prompt tokens.
    samples_unfiltered = _interleave_rollout(rollout, echo_config=echo_config)
    assert samples_unfiltered[0].echo_alpha == [0.5, 0.5, 0.5, None, None]

    # With filter narrowing position 1: that position drops back to None,
    # others preserved.
    samples_filtered = _interleave_rollout(
        rollout,
        echo_config=echo_config,
        filter_masks=[[True, False, True, True, True]],
    )
    assert samples_filtered[0].echo_alpha == [0.5, None, 0.5, None, None]


def test_interleave_rollout_filter_masks_outer_length_mismatch_raises():
    """Wrong outer length → ValueError, before per-step processing starts.
    Defensive: catch shape bugs at the orchestrator/interleave boundary
    even if the validation in ``apply_echo_filter`` is somehow bypassed."""
    rollout = vf.RolloutOutput(
        example_id=0,
        env_name="test-env",
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U"}],
                completion=[{"role": "assistant", "content": "A"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1],
                    prompt_mask=[0],
                    completion_ids=[2],
                    completion_mask=[1],
                    completion_logprobs=[-0.1],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="t",
                extras={},
            )
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )
    with pytest.raises(ValueError, match="filter_masks outer length 2.*trajectory length 1"):
        _interleave_rollout(rollout, filter_masks=[[True, True], [True, True]])


def test_interleave_rollout_filter_masks_inner_length_mismatch_raises():
    """Wrong inner length per step → ValueError out of ``_step_echo_alpha``
    (propagated unchanged from inside ``interleave_rollout``'s per-step
    work)."""
    rollout = vf.RolloutOutput(
        example_id=0,
        env_name="test-env",
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U"}],
                completion=[{"role": "assistant", "content": "A"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3],
                    completion_mask=[1],
                    completion_logprobs=[-0.1],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="t",
                extras={},
            )
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )
    # Step has prompt_len=2 + completion_len=1 = 3, filter mask length 2.
    with pytest.raises(ValueError, match="filter_mask length 2.*does not match.*3"):
        _interleave_rollout(rollout, filter_masks=[[True, True]])
