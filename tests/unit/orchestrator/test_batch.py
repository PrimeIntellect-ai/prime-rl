import pytest
import torch
from transformers import AutoTokenizer

from prime_rl.orchestrator.batch import prepare_batch_packing, prepare_batch_padding, prepare_sample
from prime_rl.orchestrator.buffer import Rollout


@pytest.fixture
def rollout() -> Rollout:
    return Rollout(
        problem_id=0,
        prompt_tokens=[0, 1, 2],
        completion_tokens=[3, 4, 5],
        prompt_mask=[0, 0, 0],
        completion_mask=[1, 1, 1],
        completion_logprobs=[0.1, 0.2, 0.3],
        reward=1.0,
        advantage=1.0,
    )


@pytest.fixture
def rollouts(rollout: Rollout) -> list[Rollout]:
    return [
        rollout,
        Rollout(
            problem_id=1,
            prompt_tokens=[0, 1, 2],
            completion_tokens=[6, 7, 8],
            prompt_mask=[0, 0, 0],
            completion_mask=[1, 1, 1],
            completion_logprobs=[0.4, 0.5, 0.6],
            reward=1.0,
            advantage=0.0,
        ),
    ]


@pytest.fixture
def tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_prepare_sample(rollout: Rollout, tokenizer: AutoTokenizer):
    sample = prepare_sample(rollout, 10, tokenizer, pad=True)
    assert sample["input_ids"].dtype == torch.long
    assert sample["position_ids"].dtype == torch.long
    assert sample["loss_mask"].dtype == torch.long
    assert sample["advantages"].dtype == torch.float
    assert sample["logprobs"].dtype == torch.float
    assert sample["input_ids"].tolist() == [0, 1, 2, 3, 4, 5, 151643, 151643, 151643, 151643]
    assert sample["position_ids"].tolist() == [0, 1, 2, 3, 4, 5, 0, 0, 0, 0]
    assert sample["loss_mask"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert sample["advantages"].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert torch.allclose(sample["logprobs"], torch.tensor([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]))


def test_prepare_sample_no_pad(rollout: Rollout, tokenizer: AutoTokenizer):
    sample = prepare_sample(rollout, 10, tokenizer, pad=False)
    assert sample["input_ids"].dtype == torch.long
    assert sample["position_ids"].dtype == torch.long
    assert sample["loss_mask"].dtype == torch.long
    assert sample["advantages"].dtype == torch.float
    assert sample["logprobs"].dtype == torch.float
    assert sample["input_ids"].tolist() == [0, 1, 2, 3, 4, 5]
    assert sample["position_ids"].tolist() == [0, 1, 2, 3, 4, 5]
    assert sample["loss_mask"].tolist() == [0, 0, 0, 1, 1, 1]
    assert sample["advantages"].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert torch.allclose(sample["logprobs"], torch.tensor([0.0, 0.0, 0.0, 0.1, 0.2, 0.3]))


def test_prepare_batch_padding(rollouts: list[Rollout], tokenizer: AutoTokenizer):
    micro_batches_per_gpu = prepare_batch_padding(
        rollouts=rollouts,
        temperature=1.0,
        tokenizer=tokenizer,
        micro_batch_size=2,
        seq_len=10,
        num_train_workers=1,
    )
    assert len(micro_batches_per_gpu) == 1
    micro_batches = micro_batches_per_gpu[0]
    assert len(micro_batches) == 1
    micro_batch = micro_batches[0]

    # Check types
    assert micro_batch["input_ids"].dtype == torch.long
    assert micro_batch["position_ids"].dtype == torch.long
    assert micro_batch["loss_mask"].dtype == torch.long
    assert micro_batch["advantages"].dtype == torch.float
    assert micro_batch["logprobs"].dtype == torch.float

    # Check values
    assert micro_batch["input_ids"].tolist() == [
        [0, 1, 2, 3, 4, 5, 151643, 151643, 151643, 151643],
        [0, 1, 2, 6, 7, 8, 151643, 151643, 151643, 151643],
    ]
    assert micro_batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 0, 0, 0, 0]]
    assert micro_batch["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]
    assert micro_batch["advantages"].tolist() == [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert torch.allclose(
        micro_batch["logprobs"],
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0]]
        ),
    )


def test_prepare_batch_padding_micro_batches(rollouts: list[Rollout], tokenizer: AutoTokenizer):
    micro_batches_per_gpu = prepare_batch_padding(
        rollouts=rollouts,
        temperature=1.0,
        tokenizer=tokenizer,
        micro_batch_size=1,
        seq_len=10,
        num_train_workers=1,
    )
    assert len(micro_batches_per_gpu) == 1
    micro_batches = micro_batches_per_gpu[0]
    assert len(micro_batches) == 2

    # Check types
    assert micro_batches[0]["input_ids"].dtype == torch.long
    assert micro_batches[0]["position_ids"].dtype == torch.long
    assert micro_batches[0]["loss_mask"].dtype == torch.long
    assert micro_batches[0]["advantages"].dtype == torch.float
    assert micro_batches[0]["logprobs"].dtype == torch.float

    # Check values
    assert micro_batches[0]["input_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 151643, 151643, 151643, 151643]]
    assert micro_batches[0]["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 0, 0, 0]]
    assert micro_batches[0]["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]
    assert micro_batches[0]["advantages"].tolist() == [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert torch.allclose(
        micro_batches[0]["logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]]),
    )

    assert micro_batches[1]["input_ids"].tolist() == [
        [0, 1, 2, 6, 7, 8, 151643, 151643, 151643, 151643],
    ]
    assert micro_batches[1]["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 0, 0, 0]]
    assert micro_batches[1]["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]
    assert micro_batches[1]["advantages"].tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert torch.allclose(
        micro_batches[1]["logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0]]),
    )


def test_prepare_batch_padding_train_workers(rollouts: list[Rollout], tokenizer: AutoTokenizer):
    micro_batches_per_gpu = prepare_batch_padding(
        rollouts=rollouts,
        temperature=1.0,
        tokenizer=tokenizer,
        micro_batch_size=1,
        seq_len=10,
        num_train_workers=2,
    )
    assert len(micro_batches_per_gpu) == 2
    assert all(len(micro_batches_per_gpu[i]) == 1 for i in range(2))

    # Check first worker's micro batches
    assert micro_batches_per_gpu[0][0]["input_ids"].dtype == torch.long
    assert micro_batches_per_gpu[0][0]["position_ids"].dtype == torch.long
    assert micro_batches_per_gpu[0][0]["loss_mask"].dtype == torch.long
    assert micro_batches_per_gpu[0][0]["advantages"].dtype == torch.float
    assert micro_batches_per_gpu[0][0]["logprobs"].dtype == torch.float

    assert micro_batches_per_gpu[0][0]["input_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 151643, 151643, 151643, 151643]]
    assert micro_batches_per_gpu[0][0]["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 0, 0, 0]]
    assert micro_batches_per_gpu[0][0]["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]
    assert micro_batches_per_gpu[0][0]["advantages"].tolist() == [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert torch.allclose(
        micro_batches_per_gpu[0][0]["logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]]),
    )

    # Check second worker's micro batches
    assert micro_batches_per_gpu[1][0]["input_ids"].dtype == torch.long
    assert micro_batches_per_gpu[1][0]["position_ids"].dtype == torch.long
    assert micro_batches_per_gpu[1][0]["loss_mask"].dtype == torch.long
    assert micro_batches_per_gpu[1][0]["advantages"].dtype == torch.float
    assert micro_batches_per_gpu[1][0]["logprobs"].dtype == torch.float

    assert micro_batches_per_gpu[1][0]["input_ids"].tolist() == [
        [0, 1, 2, 6, 7, 8, 151643, 151643, 151643, 151643],
    ]
    assert micro_batches_per_gpu[1][0]["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 0, 0, 0]]
    assert micro_batches_per_gpu[1][0]["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]
    assert micro_batches_per_gpu[1][0]["advantages"].tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert torch.allclose(
        micro_batches_per_gpu[1][0]["logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0]]),
    )


def test_prepare_batch_packing(rollouts: list[Rollout], tokenizer: AutoTokenizer):
    micro_batches_per_gpu = prepare_batch_packing(
        rollouts=rollouts,
        temperature=1.0,
        tokenizer=tokenizer,
        micro_batch_size=2,
        seq_len=10,
        num_train_workers=1,
    )
    assert len(micro_batches_per_gpu) == 1
    micro_batches = micro_batches_per_gpu[0]
    assert len(micro_batches) == 1
    micro_batch = micro_batches[0]

    # Check types
    assert micro_batch["input_ids"].dtype == torch.long
    assert micro_batch["position_ids"].dtype == torch.long
    assert micro_batch["loss_mask"].dtype == torch.long
    assert micro_batch["advantages"].dtype == torch.float
    assert micro_batch["logprobs"].dtype == torch.float

    # Check values
    assert micro_batch["input_ids"].tolist() == [
        [0, 1, 2, 3, 4, 5, 0, 1, 2, 6, 7, 8],
    ]
    assert micro_batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]]
    assert micro_batch["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]]
    assert micro_batch["advantages"].tolist() == [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert torch.allclose(
        micro_batch["logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6]]),
    )


def test_prepare_batch_packing_train_workers(rollouts: list[Rollout], tokenizer: AutoTokenizer):
    micro_batches_per_gpu = prepare_batch_packing(
        rollouts=rollouts,
        temperature=1.0,
        tokenizer=tokenizer,
        micro_batch_size=2,
        seq_len=10,
        num_train_workers=2,
    )
    assert len(micro_batches_per_gpu) == 2
    assert all(len(micro_batches_per_gpu[i]) == 1 for i in range(2))

    # Check first worker's micro batches
    assert micro_batches_per_gpu[0][0]["input_ids"].dtype == torch.long
    assert micro_batches_per_gpu[0][0]["position_ids"].dtype == torch.long
    assert micro_batches_per_gpu[0][0]["loss_mask"].dtype == torch.long
    assert micro_batches_per_gpu[0][0]["advantages"].dtype == torch.float
    assert micro_batches_per_gpu[0][0]["logprobs"].dtype == torch.float

    assert micro_batches_per_gpu[0][0]["input_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 1, 2, 6, 7, 8]]
    assert micro_batches_per_gpu[0][0]["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]]
    assert micro_batches_per_gpu[0][0]["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]]
    assert micro_batches_per_gpu[0][0]["advantages"].tolist() == [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert torch.allclose(
        micro_batches_per_gpu[0][0]["logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6]]),
    )

    # Check second worker's micro batches
    assert micro_batches_per_gpu[1][0]["input_ids"].dtype == torch.long
    assert micro_batches_per_gpu[1][0]["position_ids"].dtype == torch.long
    assert micro_batches_per_gpu[1][0]["loss_mask"].dtype == torch.long
    assert micro_batches_per_gpu[1][0]["advantages"].dtype == torch.float
    assert micro_batches_per_gpu[1][0]["logprobs"].dtype == torch.float

    assert micro_batches_per_gpu[1][0]["input_ids"].tolist() == [
        [0, 1, 2, 3, 4, 5, 0, 1, 2, 6, 7, 8],
    ]
    assert micro_batches_per_gpu[1][0]["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]]
    assert micro_batches_per_gpu[1][0]["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]]
    assert micro_batches_per_gpu[1][0]["advantages"].tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert torch.allclose(
        micro_batches_per_gpu[1][0]["logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6]]),
    )
