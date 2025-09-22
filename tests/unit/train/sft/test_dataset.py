from typing import cast

import pytest
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import FakeDataConfig, SFTDataConfig
from prime_rl.trainer.sft.data import FakeDataset, SFTDataset
from prime_rl.trainer.utils import print_sample


def test_init_fake_dataset():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed")
    fake_dataset = FakeDataset(tokenizer, config)
    assert fake_dataset is not None


def test_fake_dataset_state():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", num_examples=2)
    dataset = FakeDataset(tokenizer, config)
    dataiter = iter(dataset)
    assert dataset.state_dict() == {"step": 0, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 1, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 2, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 3, "epoch": 1}


def test_init_sft_dataset():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = SFTDataConfig(name="mikasenghaas/test-sft", num_examples=2)
    dataset = cast(Dataset, load_dataset("mikasenghaas/test-sft", split="train"))
    dataset = SFTDataset(dataset, tokenizer, config)
    assert dataset is not None


def test_raise_error_if_no_prompt_and_completion():
    dataset = Dataset.from_list([{"text": "Text 0"}])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = SFTDataConfig(num_examples=1)
    with pytest.raises(ValueError):
        SFTDataset(dataset, tokenizer, config)


def test_raise_error_if_wrong_format():
    dataset = Dataset.from_list([{"completion": ["Completion 0"]}])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = SFTDataConfig(num_examples=1)
    with pytest.raises(ValueError):
        SFTDataset(dataset, tokenizer, config)


def test_multiturn_loss_mask():
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "system", "content": "System 0"}, {"role": "user", "content": "Prompt 0"}],
                "completion": [
                    {"role": "assistant", "content": "Completion 0"},
                    {"role": "user", "content": "Prompt 1"},
                    {"role": "assistant", "content": "Completion 1"},
                ],
            },
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = SFTDataConfig(num_examples=1)
    dataset = SFTDataset(dataset, tokenizer, config)
    sample = next(iter(dataset))
    print_sample(sample["input_ids"], sample["loss_mask"], tokenizer)


SAMPLE_TEMPLATE = """\
<|im_start|>user
Prompt {idx}<|im_end|>
<|im_start|>assistant
<think>

</think>

Completion {idx}<|im_end|>
"""


def test_sft_dataset_state():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = SFTDataConfig(name="mikasenghaas/test-sft", num_examples=2)
    dataset = cast(Dataset, load_dataset("mikasenghaas/test-sft", split="train"))
    dataset = SFTDataset(dataset, tokenizer, config)
    dataiter = iter(dataset)
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Step 1
    micro_batch = next(dataiter)
    assert tokenizer.decode(micro_batch["input_ids"]) == SAMPLE_TEMPLATE.format(idx=0).strip()
    assert dataset.state_dict() == {"step": 1, "epoch": 0}

    # Step 2
    micro_batch = next(dataiter)
    assert tokenizer.decode(micro_batch["input_ids"]) == SAMPLE_TEMPLATE.format(idx=1).strip()
    assert dataset.state_dict() == {"step": 2, "epoch": 0}

    # Step 3 (next epoch)
    micro_batch = next(dataiter)
    assert tokenizer.decode(micro_batch["input_ids"]) == SAMPLE_TEMPLATE.format(idx=0).strip()
    assert dataset.state_dict() == {"step": 3, "epoch": 1}
