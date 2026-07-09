from collections import Counter

import pytest
from datasets import Dataset, interleave_datasets
from renderers import create_renderer
from renderers.base import RenderedTokens
from transformers import AutoTokenizer

from prime_rl.trainer.sft.data import SFTDataset
from prime_rl.trainer.utils import print_sample

_BOS_TOKEN_ID = 0
_STOP_TOKEN_ID = 1


def _sample_token_ids(value: str) -> list[int]:
    return [ord(char) + 2 for char in value]


class _DummyRenderer:
    def render(self, messages, **kwargs):
        content_ids = _sample_token_ids(messages[-1]["content"])
        token_ids = [_BOS_TOKEN_ID, *content_ids, _STOP_TOKEN_ID]
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=[-1, *([len(messages) - 1] * (len(content_ids) + 1))],
            sampled_mask=[False, *([True] * (len(content_ids) + 1))],
        )

    def get_stop_token_ids(self):
        return [_STOP_TOKEN_ID]


@pytest.fixture(scope="module")
def build_dummy_dataset():
    return lambda letter, num_examples: Dataset.from_list(
        [{"messages": [{"role": "assistant", "content": f"{letter}{i}"}]} for i in range(num_examples)]
    )


@pytest.fixture
def dummy_renderer():
    return _DummyRenderer()


def test_init_sft_dataset(build_dummy_dataset, dummy_renderer):
    """Tests basic initialization."""
    dataset = build_dummy_dataset("a", 1)
    sft_dataset = SFTDataset(dataset, dummy_renderer)
    assert sft_dataset is not None


def test_raise_error_if_no_prompt_and_completion(build_dummy_dataset):
    """Tests that an error is raised if no supported SFT message fields are provided."""
    dataset = Dataset.from_list([{"text": "a0"}])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    sft_dataset = SFTDataset(dataset, create_renderer(tokenizer))
    with pytest.raises(ValueError):
        next(iter(sft_dataset))


@pytest.mark.parametrize("max_epochs", [1, 2, 4])
def test_sft_first_exhausted(build_dummy_dataset, dummy_renderer, max_epochs: int):
    a = build_dummy_dataset("a", 1)
    b = build_dummy_dataset("b", 2)
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="first_exhausted")
    dataset = SFTDataset(dataset, dummy_renderer, shuffle=False, max_epochs=max_epochs)
    num_samples = 0
    sampling_order = []
    for x in dataset:
        sampling_order.append(x["target_ids"][:-1])
        num_samples += 1
    assert num_samples == max_epochs * min([len(d) for d in ds]) * len(ds)
    assert sampling_order == [_sample_token_ids("a0"), _sample_token_ids("b0")] * max_epochs


@pytest.mark.parametrize("max_epochs", [1, 2, 4])
def test_sft_all_exhausted(build_dummy_dataset, dummy_renderer, max_epochs: int):
    a = build_dummy_dataset("a", 1)
    b = build_dummy_dataset("b", 2)
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="all_exhausted")
    dataset = SFTDataset(dataset, dummy_renderer, shuffle=False, max_epochs=max_epochs)
    num_samples = 0
    sampling_order = []
    for x in dataset:
        sampling_order.append(x["target_ids"][:-1])
        num_samples += 1
    assert num_samples == max_epochs * max([len(d) for d in ds]) * len(ds)
    print(sampling_order)
    assert (
        sampling_order
        == [
            _sample_token_ids("a0"),
            _sample_token_ids("b0"),
            _sample_token_ids("a0"),
            _sample_token_ids("b1"),
        ]
        * max_epochs
    )


@pytest.mark.parametrize(
    "probs",
    [
        pytest.param((0.5, 0.5), id="equal_probs"),
        pytest.param((1 / 10, 9 / 10), id="low_high_probs"),
        pytest.param((9 / 10, 1 / 10), id="high_low_probs"),
    ],
)
def test_sft_all_exhausted_with_probs(build_dummy_dataset, dummy_renderer, probs: list[float]):
    """Tests that the ratio of samples from different datasets is as specified, in expectation."""
    a = build_dummy_dataset("a", int(1e3))
    b = build_dummy_dataset("b", int(10e3))
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="all_exhausted", probabilities=probs)
    dataset = SFTDataset(dataset, dummy_renderer, shuffle=False, max_epochs=1)
    num_samples = 0
    sampling_freq = []
    for x in dataset:
        sampling_freq.append(x["target_ids"][0])
        num_samples += 1
    sampling_freq = Counter(sampling_freq)
    ratio_a = sampling_freq[ord("a") + 2] / num_samples
    ratio_b = sampling_freq[ord("b") + 2] / num_samples
    assert ratio_a > probs[0] * 0.8 and ratio_a < probs[0] * 1.2, (
        f"Expected frequency of samples from a to be between {probs[0] * 0.8} and {probs[0] * 1.2}, but got {ratio_a}"
    )
    assert ratio_b > probs[1] * 0.8 and ratio_b < probs[1] * 1.2, (
        f"Exepcted frequency of samples from b to be between {probs[1] * 0.8} and {probs[1] * 1.2}, but got {ratio_b}"
    )


def test_sft_dataset_state(build_dummy_dataset, dummy_renderer):
    """Tests the state of the dataset within and across epochs."""
    dataset = build_dummy_dataset("", 4)
    dataset = SFTDataset(dataset, dummy_renderer, shuffle=False, max_epochs=2)
    dataiter = iter(dataset)

    # Initial state
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Epoch 1
    for i in range(4):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 0, "step": i + 1}

    # Epoch 2
    for i in range(4):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 1, "step": 4 + i + 1}

    with pytest.raises(StopIteration):
        next(dataiter)


def test_sft_dataset_state_resume(build_dummy_dataset, dummy_renderer):
    """Tests resuming the dataset from checkpoint in between epochs."""
    dataset = SFTDataset(
        build_dummy_dataset("", 4),
        dummy_renderer,
        shuffle=False,
        max_epochs=2,
    )
    dataiter = iter(dataset)

    # Initial state
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Epoch 1
    for i in range(4):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 0, "step": i + 1}

    # Resuming from checkpoint cross epoch
    state_dict = dataset.state_dict()
    del dataset
    dataset = SFTDataset(
        build_dummy_dataset("", 4),
        dummy_renderer,
        shuffle=False,
        max_epochs=2,
    )
    dataset.load_state_dict(state_dict)
    dataiter = iter(dataset)

    # Epoch 2.1
    for i in range(2):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 1, "step": 4 + i + 1}

    # Resuming from checkpoint mid epoch
    state_dict = dataset.state_dict()
    del dataset
    dataset = SFTDataset(
        build_dummy_dataset("", 4),
        dummy_renderer,
        shuffle=False,
        max_epochs=2,
    )
    dataset.load_state_dict(state_dict)
    dataiter = iter(dataset)

    # Epoch 2.2
    for i in range(2, 4):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 1, "step": 4 + i + 1}

    with pytest.raises(StopIteration):
        next(dataiter)


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
    dataset = SFTDataset(dataset, create_renderer(tokenizer), max_examples=1)
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
    dataset = SFTDataset(dataset, create_renderer(tokenizer), max_examples=1)
    sample = next(iter(dataset))
    print_sample(sample["input_ids"], sample["loss_mask"], tokenizer)


def test_messages_rows_are_equivalent_to_empty_prompt_completion():
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What's the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "San Francisco, CA"}'},
                }
            ],
        },
        {"role": "tool", "content": '{"temperature": 65, "condition": "Sunny"}', "tool_call_id": "call_1"},
        {"role": "assistant", "content": "It is 65F and sunny in San Francisco."},
    ]

    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    messages_dataset = SFTDataset(
        Dataset.from_list([{"messages": messages}]),
        create_renderer(tokenizer),
        max_examples=1,
    )
    split_dataset = SFTDataset(
        Dataset.from_list([{"prompt": [], "completion": messages}]),
        create_renderer(tokenizer),
        max_examples=1,
    )

    assert next(iter(messages_dataset)) == next(iter(split_dataset))


def test_messages_take_precedence_over_prompt_and_completion():
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    row = {
        "messages": [
            {"role": "system", "content": "System from messages"},
            {"role": "user", "content": "Prompt from messages"},
            {"role": "assistant", "content": "Completion from messages"},
        ],
        "prompt": [{"role": "user", "content": "Ignored prompt"}],
        "completion": [{"role": "assistant", "content": "Ignored completion"}],
    }

    messages_dataset = SFTDataset(
        Dataset.from_list([row]),
        create_renderer(tokenizer),
        max_examples=1,
    )
    expected_dataset = SFTDataset(
        Dataset.from_list([{"prompt": [], "completion": row["messages"]}]),
        create_renderer(tokenizer),
        max_examples=1,
    )

    assert next(iter(messages_dataset)) == next(iter(expected_dataset))


def test_null_messages_falls_back_to_prompt_and_completion():
    # Arrow schema union adds `messages: None` to prompt/completion rows when
    # other rows in the file have a `messages` column
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "4"}]

    mixed_row_dataset = SFTDataset(
        Dataset.from_list([{"messages": None, "prompt": prompt, "completion": completion}]),
        create_renderer(tokenizer),
        max_examples=1,
    )
    expected_dataset = SFTDataset(
        Dataset.from_list([{"prompt": prompt, "completion": completion}]),
        create_renderer(tokenizer),
        max_examples=1,
    )

    assert next(iter(mixed_row_dataset)) == next(iter(expected_dataset))
