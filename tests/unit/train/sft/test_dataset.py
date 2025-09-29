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
    assert dataset.state_dict() == {"step": -1, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 0, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 1, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 2, "epoch": 1}


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
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")  # Properly handles multi-turn think
    config = SFTDataConfig(num_examples=1)
    dataset = SFTDataset(dataset, tokenizer, config)
    sample = next(iter(dataset))
    print_sample(sample["input_ids"], sample["loss_mask"], tokenizer)


def test_multiturn_loss_mask_with_tools():
    tool_example = {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What's the weather like in San Francisco and New York?"},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "San Francisco, CA"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "New York, NY"}'},
                    },
                ],
            },
            {"role": "tool", "content": '{"temperature": 65, "condition": "Sunny"}', "tool_call_id": "call_1"},
            {"role": "tool", "content": '{"temperature": 45, "condition": "Cloudy"}', "tool_call_id": "call_2"},
            {
                "role": "assistant",
                "content": "Based on the weather data:\n\n**San Francisco, CA**: It's currently 65°F and sunny - perfect weather!\n\n**New York, NY**: It's 45°F and cloudy - you might want to bring a jacket.",
            },
            {"role": "user", "content": "Should I pack an umbrella for New York?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "get_precipitation_forecast",
                            "arguments": '{"location": "New York, NY", "days": 3}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"forecast": [{"day": 1, "chance_of_rain": 20}, {"day": 2, "chance_of_rain": 60}, {"day": 3, "chance_of_rain": 40}]}',
                "tool_call_id": "call_3",
            },
            {
                "role": "assistant",
                "content": "Looking at the 3-day precipitation forecast for New York:\n- Day 1: 20% chance of rain\n- Day 2: 60% chance of rain\n- Day 3: 40% chance of rain\n\nI'd recommend packing an umbrella, especially for day 2 when there's a 60% chance of rain.",
            },
        ],
    }

    dataset = Dataset.from_list([tool_example])
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")  # Properly handles multi-turn think
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

Completion {idx}<|im_end|>\
"""


def test_sft_dataset_state():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = SFTDataConfig(name="mikasenghaas/test-sft", num_examples=2)
    dataset = cast(Dataset, load_dataset("mikasenghaas/test-sft", split="train"))
    dataset = SFTDataset(dataset, tokenizer, config)
    dataiter = iter(dataset)
    assert dataset.state_dict() == {"step": -1, "epoch": 0}

    # Step 1
    micro_batch = next(dataiter)
    assert tokenizer.decode(micro_batch["input_ids"]) == SAMPLE_TEMPLATE.format(idx=0)
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Step 2
    micro_batch = next(dataiter)
    assert tokenizer.decode(micro_batch["input_ids"]) == SAMPLE_TEMPLATE.format(idx=1)
    assert dataset.state_dict() == {"step": 1, "epoch": 0}

    # Step 3 (next epoch)
    micro_batch = next(dataiter)
    assert tokenizer.decode(micro_batch["input_ids"]) == SAMPLE_TEMPLATE.format(idx=0)
    assert dataset.state_dict() == {"step": 2, "epoch": 1}
