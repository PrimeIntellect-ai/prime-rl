from vllm import RequestOutput, CompletionOutput
import json
import torch
from zeroband.inference.rewards import compute_rewards, compute_advantages
from collections import defaultdict

import pytest


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
    rewards = defaultdict(dict)
    for sample in samples["samples"]:
        request_id = sample["request_idx"]
        completion_id = sample["output_idx"]
        rewards[request_id][completion_id] = {
            "reward": sample["reward"],
            "task_reward": sample["task_reward"],
            "length_penalty": sample["length_penalty"],
        }
    yield rewards


@pytest.fixture
def ground_truth_advantages(samples):
    advantages = defaultdict(dict)
    for sample in samples["samples"]:
        request_id = sample["request_idx"]
        completion_id = sample["output_idx"]
        advantages[request_id][completion_id] = sample["advantage"]
    advantages = {k: list(v.values()) for k, v in advantages.items()}
    yield advantages


def test_compute_rewards(request_outputs, ground_truth_rewards, verification_infos):
    task_types = ["verifiable_math"] * len(request_outputs)
    grouped_rewards, grouped_task_rewards, grouped_length_penalties = compute_rewards(
        request_outputs,
        verification_infos,
        task_types,
        config=None,
    )

    # Assert return type
    assert all(isinstance(rewards, torch.FloatTensor) for rewards in grouped_rewards.values())
    assert all(isinstance(task_rewards, torch.FloatTensor) for task_rewards in grouped_task_rewards.values())
    assert all(isinstance(length_penalties, torch.FloatTensor) for length_penalties in grouped_length_penalties.values())

    # Assert reward computation
    for request_id, rewards in grouped_rewards.items():
        for completion_id, reward in enumerate(rewards.tolist()):
            assert reward == ground_truth_rewards[request_id][completion_id]["reward"]
    for request_id, task_rewards in grouped_task_rewards.items():
        for completion_id, reward in enumerate(task_rewards.tolist()):
            assert reward == ground_truth_rewards[request_id][completion_id]["task_reward"]
    for request_id, length_penalties in grouped_length_penalties.items():
        for completion_id, penalty in enumerate(length_penalties.tolist()):
            assert penalty == ground_truth_rewards[request_id][completion_id]["length_penalty"]


def test_compute_advantages(request_outputs, ground_truth_advantages, verification_infos):
    target_lengths = [-1] * len(request_outputs)
    task_types = ["verifiable_math"] * len(request_outputs)
    grouped_rewards, _, _ = compute_rewards(
        request_outputs,
        verification_infos,
        task_types,
        config=None,
    )
    advantages = compute_advantages(grouped_rewards, epsilon=1e-6)
    # Assert return type
    assert all(isinstance(advantage, list) for advantage in advantages.values())
    # Assert advantage computation
    for request_id, advantage in advantages.items():
        assert advantage == ground_truth_advantages[request_id]
