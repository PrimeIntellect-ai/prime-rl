import json
import pytest
from collections import defaultdict

from vllm import RequestOutput, CompletionOutput

from zeroband.inference.rewards import compute_rewards


@pytest.fixture
def samples():
    with open("tests/units/example_rewards.json", "r") as f:
        return json.load(f)


@pytest.fixture
def request_outputs(samples):
    request_outputs_dict = defaultdict(dict)
    for sample in samples["samples"]:
        request_id = sample["request_idx"]
        completion_id = sample["output_idx"]
        request_outputs_dict[request_id][completion_id] = CompletionOutput(
            index=completion_id,
            token_ids=None,
            text=sample["completion"],
            cumulative_logprob=None,
            logprobs=None,
        )

    request_outputs = []
    for request_id, completion_outputs in request_outputs_dict.items():
        completion_outputs = list(completion_outputs.values())
        request_outputs.append(
            RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=None,
                prompt_logprobs=None,
                outputs=completion_outputs,
                finished=True,
            )
        )

    yield request_outputs


@pytest.fixture
def verification_infos(samples):
    verification_infos = []
    request_ids = set()
    for sample in samples["samples"]:
        request_id = sample["request_idx"]
        if request_id not in request_ids:
            verification_infos.append(json.loads(sample["verification_info"]))
            request_ids.add(request_id)
    return verification_infos


@pytest.fixture
def ground_truth_rewards(samples):
    rewards = defaultdict(list)
    for sample in samples["samples"]:
        request_id = sample["request_idx"]
        rewards[request_id].append(sample["reward"])

    yield rewards


@pytest.fixture
def ground_truth_task_rewards(samples):
    task_rewards = defaultdict(list)
    for sample in samples["samples"]:
        request_id = sample["request_idx"]
        task_rewards[request_id].append(sample["task_reward"])

    yield task_rewards


@pytest.fixture
def ground_truth_length_penalties(samples):
    penalties = defaultdict(list)
    for sample in samples["samples"]:
        request_id = sample["request_idx"]
        penalties[request_id].append(sample["length_penalty"])
    yield penalties


@pytest.fixture
def ground_truth_advantages(samples):
    advantages = defaultdict(dict)
    for sample in samples["samples"]:
        request_id = sample["request_idx"]
        completion_id = sample["output_idx"]
        advantages[request_id][completion_id] = sample["advantage"]
    advantages = {k: list(v.values()) for k, v in advantages.items()}
    yield advantages


def test_compute_rewards(
    request_outputs,
    ground_truth_rewards,
    ground_truth_task_rewards,
    ground_truth_length_penalties,
    ground_truth_advantages,
    verification_infos,
):
    # Re-compute rewards
    task_types = ["verifiable_math"] * len(request_outputs)
    rewards, task_rewards, length_penalties, advantages = compute_rewards(
        request_outputs,
        verification_infos,
        task_types,
        config=None,
    )

    # Assert type
    for reward_or_advantage in [rewards, task_rewards, length_penalties, advantages]:
        assert isinstance(reward_or_advantage, dict)
        assert all(isinstance(key, int) for key in reward_or_advantage.keys())
        assert all(isinstance(value, list) for value in reward_or_advantage.values())

    # Assert computation
    assert rewards == ground_truth_rewards
    assert task_rewards == ground_truth_task_rewards
    assert length_penalties == ground_truth_length_penalties
    assert advantages == ground_truth_advantages
