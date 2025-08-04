import pytest
import torch
from transformers import AutoTokenizer

from prime_rl.orchestrator.batch import prepare_sample
from prime_rl.orchestrator.buffer import Rollout


@pytest.fixture
def rollout() -> Rollout:
    return Rollout(
        problem_id=1,
        prompt_tokens=[0, 1, 2],
        completion_tokens=[3, 4, 5],
        prompt_mask=[0, 0, 0],
        completion_mask=[1, 1, 1],
        completion_logprobs=[0.1, 0.2, 0.3],
        reward=1.0,
        advantage=1.0,
    )


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
