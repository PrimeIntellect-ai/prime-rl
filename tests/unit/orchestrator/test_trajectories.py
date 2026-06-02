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
    kwargs.setdefault("env_name", output.get("env_name", "test-env"))
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


@pytest.mark.parametrize(
    ("attribution", "prompt_len", "completion_len", "echo_config", "expected"),
    [
        pytest.param(
            _attribution(
                message_indices=[0, 0, 1, 1],
                is_content=[False, True, False, True],
                message_roles=["user", "tool"],
                message_tool_names=[None, "lookup"],
            ),
            4,
            2,
            None,
            [None] * 6,
            id="echo_config_none",
        ),
        pytest.param(
            None,
            4,
            2,
            EchoConfig(assistant=AssistantRoleEchoConfig(alpha=0.3)),
            [None, None, None, None, 0.3, 0.3],
            id="no_attribution_marks_assistant_completion",
        ),
        pytest.param(
            None,
            4,
            2,
            EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
            [None] * 6,
            id="no_attribution_no_completion_echo",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 0],
                is_content=[False, True],
                message_roles=None,
                message_tool_names=["lookup"],
            ),
            2,
            0,
            EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
            [None, None],
            id="no_message_roles",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 0, 1, 1, 1],
                is_content=[False, True, False, True, True],
                message_roles=["user", "tool"],
                message_tool_names=[None, "lookup"],
            ),
            5,
            2,
            EchoConfig(tool=ToolRoleEchoConfig(alpha=0.7, tool_names=None)),
            [None, None, None, 0.7, 0.7, None, None],
            id="tool_default_all_tools",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 0, 1, 1, 2, 2],
                is_content=[False, True, False, True, False, True],
                message_roles=["user", "tool", "tool"],
                message_tool_names=[None, "calc", "lookup"],
            ),
            6,
            0,
            EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5, tool_names=["lookup"])),
            [None, None, None, None, None, 0.5],
            id="tool_name_filter",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 0, 0, 0],
                is_content=[False, False, True, False],
                message_roles=["tool"],
                message_tool_names=["lookup"],
            ),
            4,
            0,
            EchoConfig(tool=ToolRoleEchoConfig(alpha=0.4)),
            [None, None, 0.4, None],
            id="skips_non_content_tokens",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 0, 1, 1],
                is_content=[False, True, False, True],
                message_roles=["user", "tool"],
                message_tool_names=[None, "lookup"],
            ),
            4,
            0,
            EchoConfig(user=UserRoleEchoConfig(alpha=0.2), tool=None),
            [None, 0.2, None, None],
            id="user_role",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 0, 0],
                is_content=[False, True, True],
                message_roles=["system"],
            ),
            3,
            0,
            EchoConfig(system=SystemRoleEchoConfig(alpha=0.1), tool=None),
            [None, 0.1, 0.1],
            id="system_role",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 0, 1, 1],
                is_content=[False, True, False, True],
                message_roles=["user", "assistant"],
            ),
            4,
            3,
            EchoConfig(assistant=AssistantRoleEchoConfig(alpha=0.8), tool=None),
            [None, None, None, 0.8, 0.8, 0.8, 0.8],
            id="assistant_prompt_and_completion",
        ),
        pytest.param(
            _attribution(message_indices=[0], is_content=[True], message_roles=["user"]),
            1,
            2,
            EchoConfig(assistant=AssistantRoleEchoConfig(alpha=0.0), tool=None),
            [None, 0.0, 0.0],
            id="assistant_zero_kills_rl",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 1, 2],
                is_content=[True, True, True],
                message_roles=["user", "tool", "system"],
                message_tool_names=[None, "lookup", None],
            ),
            3,
            2,
            EchoConfig(
                user=UserRoleEchoConfig(alpha=0.1),
                tool=ToolRoleEchoConfig(alpha=0.5),
                system=SystemRoleEchoConfig(alpha=0.05),
                assistant=AssistantRoleEchoConfig(alpha=0.9),
            ),
            [0.1, 0.5, 0.05, 0.9, 0.9],
            id="per_role_alphas_differ",
        ),
    ],
)
def test_step_echo_alpha_baseline(attribution, prompt_len, completion_len, echo_config, expected):
    """``_step_echo_alpha`` builds the per-token alpha array from the renderer
    attribution + per-role config. Only content tokens of enabled roles get a
    float; scaffold/disabled-role tokens stay None. Completion-side assistant
    echo is independent of attribution. ``alpha=0`` is a real value (kill-RL),
    distinct from None (not echoed)."""
    assert (
        _step_echo_alpha(
            prompt_attribution=attribution,
            prompt_len=prompt_len,
            completion_len=completion_len,
            echo_config=echo_config,
        )
        == expected
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no_roles_default"),
        pytest.param(dict(system=None, user=None, assistant=None, tool=None), id="all_roles_none"),
        pytest.param(dict(filter=EchoFilterConfig(import_path="my_module.my_filter")), id="filter_only_no_role"),
    ],
)
def test_echo_config_rejects_without_role(kwargs):
    """``EchoConfig`` requires at least one role. The custom
    ``require_at_least_one_role`` validator rejects an empty config, an
    all-None config, and a filter-only config — the filter is a narrowing
    overlay, not a role, so it can't enable echo on its own."""
    with pytest.raises(ValidationError, match="at least one role"):
        EchoConfig(**kwargs)


# ---------------------------------------------------------------------------
# _step_echo_alpha — filter_mask composition
# ---------------------------------------------------------------------------


def _tool_only_attribution(prompt_len: int) -> dict:
    """All prompt tokens are content of a single ``lookup`` tool message —
    minimal setup where every prompt position has a baseline echo alpha to be
    narrowed by the filter."""
    return _attribution(
        message_indices=[0] * prompt_len,
        is_content=[True] * prompt_len,
        message_roles=["tool"],
        message_tool_names=["lookup"],
    )


_TOOL_AND_ASSISTANT = EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5), assistant=AssistantRoleEchoConfig(alpha=0.8))


@pytest.mark.parametrize(
    ("attribution", "prompt_len", "completion_len", "echo_config", "filter_mask", "expected"),
    [
        pytest.param(
            _tool_only_attribution(3),
            3,
            2,
            EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
            None,
            [0.5, 0.5, 0.5, None, None],
            id="filter_none_is_no_op",
        ),
        pytest.param(
            _tool_only_attribution(3),
            3,
            2,
            _TOOL_AND_ASSISTANT,
            [True] * 5,
            [0.5, 0.5, 0.5, 0.8, 0.8],
            id="all_true_preserves_baseline",
        ),
        pytest.param(
            _tool_only_attribution(3),
            3,
            2,
            _TOOL_AND_ASSISTANT,
            [False] * 5,
            [None] * 5,
            id="all_false_zeros_everything",
        ),
        pytest.param(
            _tool_only_attribution(4),
            4,
            2,
            EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
            [True, False, True, False, False, False],
            [0.5, None, 0.5, None, None, None],
            id="narrows_baseline",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 0, 1, 1],
                is_content=[True, True, True, True],
                message_roles=["user", "tool"],
                message_tool_names=[None, "lookup"],
            ),
            4,
            0,
            EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)),
            [True, True, True, True],
            [None, None, 0.5, 0.5],
            id="cannot_add_echo_to_disabled_role",
        ),
        pytest.param(
            None,
            2,
            4,
            EchoConfig(assistant=AssistantRoleEchoConfig(alpha=0.3)),
            [True, True, True, True, False, False],
            [None, None, 0.3, 0.3, None, None],
            id="narrows_assistant_completion",
        ),
        pytest.param(
            _attribution(
                message_indices=[0, 1, 2],
                is_content=[True, True, True],
                message_roles=["system", "user", "tool"],
                message_tool_names=[None, None, "lookup"],
            ),
            3,
            2,
            EchoConfig(
                system=SystemRoleEchoConfig(alpha=0.05),
                user=UserRoleEchoConfig(alpha=0.1),
                tool=ToolRoleEchoConfig(alpha=0.5),
                assistant=AssistantRoleEchoConfig(alpha=0.9),
            ),
            [True, False, True, True, False],
            [0.05, None, 0.5, 0.9, None],
            id="mixed_roles",
        ),
    ],
)
def test_step_echo_alpha_filter_composition(
    attribution, prompt_len, completion_len, echo_config, filter_mask, expected
):
    """The optional ``filter_mask`` narrows the role baseline per-token: False
    drops a position to None (RL applies), True preserves it. The filter can
    only narrow — it never adds echo where no role enabled it (``cannot_add``)
    — and it applies to completion-side assistant echo too."""
    assert (
        _step_echo_alpha(
            prompt_attribution=attribution,
            prompt_len=prompt_len,
            completion_len=completion_len,
            echo_config=echo_config,
            filter_mask=filter_mask,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("filter_mask", "echo_config"),
    [
        pytest.param([True, True, True], EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)), id="too_short"),
        pytest.param([True] * 6, EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5)), id="too_long"),
        pytest.param([True, True], None, id="validates_even_when_echo_disabled"),
    ],
)
def test_step_echo_alpha_filter_length_mismatch_raises(filter_mask, echo_config):
    """``filter_mask`` length must equal ``prompt_len + completion_len`` (5
    here). Length is validated before any baseline work, so a wrong length
    raises even when echo is disabled (``echo_config=None``)."""
    with pytest.raises(ValueError, match="filter_mask length"):
        _step_echo_alpha(
            prompt_attribution=None,
            prompt_len=3,
            completion_len=2,
            echo_config=echo_config,
            filter_mask=filter_mask,
        )


# ---------------------------------------------------------------------------
# apply_echo_filter — shape/type validation + invocation contract
# ---------------------------------------------------------------------------


def _step_with_tokens(prompt_len: int, completion_len: int, attribution: dict | None = None) -> vf.TrajectoryStep:
    """Minimal TrajectoryStep with controllable token lengths (and optional
    ``prompt_attribution``) for exercising the per-step length checks without
    a renderer."""
    tokens_kwargs: dict = dict(
        prompt_ids=list(range(prompt_len)),
        prompt_mask=[0] * prompt_len,
        completion_ids=list(range(prompt_len, prompt_len + completion_len)),
        completion_mask=[1] * completion_len,
        completion_logprobs=[-0.1] * completion_len,
        overlong_prompt=False,
        is_truncated=False,
    )
    if attribution is not None:
        tokens_kwargs["prompt_attribution"] = attribution
    return vf.TrajectoryStep(
        prompt=[{"role": "user", "content": "U"}],
        completion=[{"role": "assistant", "content": "A"}],
        response=MagicMock(),
        tokens=vf.TrajectoryStepTokens(**tokens_kwargs),
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="t",
        extras={},
    )


def _rollout_with_steps(*step_dims: tuple, env_name: str = "test-env") -> vf.RolloutOutput:
    """Build a RolloutOutput from ``(prompt_len, completion_len[, attribution])``
    tuples — one trajectory step per tuple."""
    return vf.RolloutOutput(
        example_id=0,
        env_name=env_name,
        trajectory=[_step_with_tokens(*dims) for dims in step_dims],
        sampling_args={"temperature": 1.0},
        error=None,
    )


def _const_filter(masks):
    """A filter callable that ignores the rollout and returns ``masks``."""

    def filter_fn(rollout):
        return masks

    return filter_fn


@pytest.mark.parametrize(
    ("dims", "filter_return", "exc_type", "match"),
    [
        pytest.param(
            [(3, 2), (4, 1)], [[True] * 5], ValueError, r"returned 1 per-step masks.*has 2", id="outer_too_short"
        ),
        pytest.param(
            [(3, 2)], [[True] * 5, [True] * 5], ValueError, r"returned 2 per-step masks.*has 1", id="outer_too_long"
        ),
        pytest.param(
            [(3, 2), (4, 1)],
            [[True] * 5, [True] * 3],
            ValueError,
            r"step 1.*mask length 3.*expected 5.*prompt_len=4.*completion_len=1",
            id="inner_mismatch",
        ),
        pytest.param([(2, 1)], "not a list", TypeError, r"must return list.*got str", id="non_list_return"),
        pytest.param(
            [(2, 1), (3, 0)],
            [[True, True, True], "not a list"],
            TypeError,
            r"step 1.*mask must be a list.*str",
            id="non_list_inner",
        ),
        pytest.param(
            [(1, 1)], [[True, 1]], TypeError, r"step 0.*mask\[1\].*must be a plain bool.*int", id="non_bool_int"
        ),
        pytest.param(
            [(2, 0)], [[True, "yes"]], TypeError, r"step 0.*mask\[1\].*must be a plain bool.*str", id="non_bool_string"
        ),
    ],
)
def test_apply_echo_filter_invalid_raises(dims, filter_return, exc_type, match):
    """``apply_echo_filter`` validates the filter's return loudly: outer length
    must equal the trajectory length, each inner mask must equal that step's
    ``prompt_len + completion_len``, and every element must be a plain ``bool``
    (a truthy ``1`` is rejected)."""
    rollout = _rollout_with_steps(*dims)
    with pytest.raises(exc_type, match=match):
        apply_echo_filter(rollout, _const_filter(filter_return))


def test_apply_echo_filter_valid_returns_masks():
    """Happy path: a well-shaped filter return passes through unchanged."""
    rollout = _rollout_with_steps((3, 2), (4, 1))
    masks = [[True, False, True, True, False], [False, False, True, True, False]]
    assert apply_echo_filter(rollout, _const_filter(masks)) == masks


def test_apply_echo_filter_empty_trajectory_returns_empty_masks():
    """Empty trajectory + a filter returning ``[]`` is valid."""
    rollout = vf.RolloutOutput(
        example_id=0, env_name="test-env", trajectory=[], sampling_args={"temperature": 1.0}, error=None
    )
    assert apply_echo_filter(rollout, _const_filter([])) == []


def test_apply_echo_filter_propagates_user_exception():
    """A raising filter propagates verbatim — no swallowing, no fallback."""
    rollout = _rollout_with_steps((2, 1))

    class FilterCrash(RuntimeError):
        pass

    def filter_fn(rollout):
        raise FilterCrash("boom")

    with pytest.raises(FilterCrash, match="boom"):
        apply_echo_filter(rollout, filter_fn)


def test_apply_echo_filter_receives_full_rollout():
    """The filter sees the full rollout dict (example_id / error / trajectory),
    so it can branch on reward / error / metrics / info."""
    rollout = _rollout_with_steps((2, 1))
    seen: dict = {}

    def filter_fn(rollout):
        seen.update(example_id=rollout["example_id"], error=rollout["error"], n=len(rollout["trajectory"]))
        return [[True] * 3]

    apply_echo_filter(rollout, filter_fn)
    assert seen == {"example_id": 0, "error": None, "n": 1}


# ---------------------------------------------------------------------------
# interleave_rollout — end-to-end with filter_masks
# ---------------------------------------------------------------------------


def test_interleave_rollout_filter_masks_none_no_op(single_step_trajectory_output):
    """``filter_masks=None`` matches the no-filter call exactly (opt-in)."""
    rollout = single_step_trajectory_output
    rollout["env_name"] = "test-env"
    baseline = _interleave_rollout(rollout)
    with_none = _interleave_rollout(rollout, filter_masks=None)
    assert with_none[0].echo_alpha == baseline[0].echo_alpha is None


_TOOL_ATTRIBUTION = {
    "message_indices": [0, 0, 0],
    "is_content": [True, True, True],
    "message_roles": ["tool"],
    "message_tool_names": ["lookup"],
}


def test_interleave_rollout_filter_masks_narrows_sample_echo_alpha():
    """End-to-end: a filter False drops that position's role echo alpha to None
    on the resulting sample's ``echo_alpha`` — proving the mask flows from the
    orchestrator boundary through to the wire format."""
    rollout = _rollout_with_steps((3, 2, _TOOL_ATTRIBUTION))
    echo_config = EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5))

    unfiltered = _interleave_rollout(rollout, echo_config=echo_config)
    assert unfiltered[0].echo_alpha == [0.5, 0.5, 0.5, None, None]

    filtered = _interleave_rollout(rollout, echo_config=echo_config, filter_masks=[[True, False, True, True, True]])
    assert filtered[0].echo_alpha == [0.5, None, 0.5, None, None]


@pytest.mark.parametrize(
    ("dims", "filter_masks", "match"),
    [
        pytest.param(
            [(1, 1)],
            [[True, True], [True, True]],
            r"filter_masks outer length 2.*trajectory length 1",
            id="outer_mismatch",
        ),
        pytest.param([(2, 1)], [[True, True]], r"filter_mask length 2.*does not match.*3", id="inner_mismatch"),
    ],
)
def test_interleave_rollout_filter_masks_length_mismatch_raises(dims, filter_masks, match):
    """Wrong outer length (vs trajectory) or inner length (vs step tokens) →
    ValueError at the interleave boundary."""
    with pytest.raises(ValueError, match=match):
        _interleave_rollout(_rollout_with_steps(*dims), filter_masks=filter_masks)
