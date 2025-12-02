import json
from copy import deepcopy

import pytest
import verifiers as vf
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from prime_rl.utils.vf import from_serializable_state, to_serializable_state


@pytest.fixture
def single_step_trajectory_state():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=ChatCompletion(
                    id="cm123",
                    object="chat.completion",
                    created=1716806400,
                    model="gpt-4o",
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content="A1",
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=CompletionUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
                ),
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


def test_serialize_state(single_step_trajectory_state):
    # Regular state is not JSON serializable
    with pytest.raises(Exception):
        json.dumps(single_step_trajectory_state)

    # Serialized state is JSON serializable
    serializable_state = to_serializable_state(single_step_trajectory_state)
    json.dumps(serializable_state)


def test_deserialize_state(single_step_trajectory_state):
    original_state = deepcopy(single_step_trajectory_state)
    serialized_state = to_serializable_state(single_step_trajectory_state)
    deserialized_state = from_serializable_state(serialized_state)
    assert deserialized_state == original_state
