from collections import Counter

import pytest
import torch
from datasets import Dataset, interleave_datasets
from renderers import create_renderer
from renderers.base import MultiModalData, PlaceholderRange, RenderedTrainingSample
from transformers import AutoTokenizer

import prime_rl.trainer.sft.data as sft_data
from prime_rl.trainer.sft.data import CatDataset, SFTDataset, StatefulIterableDataset, _drop_null_fields
from prime_rl.trainer.utils import print_sample

_BOS_TOKEN_ID = 0
_STOP_TOKEN_ID = 1


def _sample_token_ids(value: str) -> list[int]:
    return [ord(char) + 2 for char in value]


@pytest.fixture(scope="module")
def build_dummy_dataset():
    return lambda letter, num_examples: Dataset.from_list(
        [{"messages": [{"role": "assistant", "content": f"{letter}{i}"}]} for i in range(num_examples)]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param('{"reasoning_effort": null}', id="json-string"),
        pytest.param({"reasoning_effort": None}, id="dict"),
    ],
)
def test_drop_null_fields_preserves_tool_call_arguments(arguments):
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Calling a tool", "image_url": None}],
        "tool_calls": [{"function": {"name": "listReasoningModels", "arguments": arguments}}],
        "metadata": {"arguments": {"unrelated_null": None}},
    }

    cleaned = _drop_null_fields(message)

    assert cleaned["tool_calls"][0]["function"]["arguments"] == arguments
    assert cleaned["content"] == [{"type": "text", "text": "Calling a tool"}]
    assert cleaned["metadata"] == {"arguments": {}}


def test_init_sft_dataset(build_dummy_dataset, dummy_renderer):
    """Tests basic initialization."""
    dataset = build_dummy_dataset("a", 1)
    sft_dataset = SFTDataset(dataset, lambda _: dummy_renderer)
    assert sft_dataset is not None


def test_raise_error_if_no_prompt_and_completion(build_dummy_dataset):
    """Tests that an error is raised if no supported SFT message fields are provided."""
    dataset = Dataset.from_list([{"text": "a0"}])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    sft_dataset = SFTDataset(dataset, lambda _: create_renderer(tokenizer))
    with pytest.raises(ValueError):
        next(iter(sft_dataset))


@pytest.mark.parametrize("max_epochs", [1, 2, 4])
def test_sft_first_exhausted(build_dummy_dataset, dummy_renderer, max_epochs: int):
    a = build_dummy_dataset("a", 1)
    b = build_dummy_dataset("b", 2)
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="first_exhausted")
    dataset = SFTDataset(dataset, lambda _: dummy_renderer, shuffle=False, max_epochs=max_epochs)
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
    dataset = SFTDataset(dataset, lambda _: dummy_renderer, shuffle=False, max_epochs=max_epochs)
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
    dataset = SFTDataset(dataset, lambda _: dummy_renderer, shuffle=False, max_epochs=1)
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
    dataset = SFTDataset(dataset, lambda _: dummy_renderer, shuffle=False, max_epochs=2)
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
        lambda _: dummy_renderer,
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
        lambda _: dummy_renderer,
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
        lambda _: dummy_renderer,
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
    dataset = SFTDataset(dataset, lambda _: create_renderer(tokenizer), max_examples=1)
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
    dataset = SFTDataset(dataset, lambda _: create_renderer(tokenizer), max_examples=1)
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
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )
    split_dataset = SFTDataset(
        Dataset.from_list([{"prompt": [], "completion": messages}]),
        lambda _: create_renderer(tokenizer),
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
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )
    expected_dataset = SFTDataset(
        Dataset.from_list([{"prompt": [], "completion": row["messages"]}]),
        lambda _: create_renderer(tokenizer),
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
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )
    expected_dataset = SFTDataset(
        Dataset.from_list([{"prompt": prompt, "completion": completion}]),
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )

    assert next(iter(mixed_row_dataset)) == next(iter(expected_dataset))


def test_vlm_truncation_does_not_append_trainable_eos(monkeypatch, dummy_renderer):
    mm = MultiModalData(
        mm_placeholders={"image": [PlaceholderRange(offset=1, length=1)]},
        mm_items={"image": [{"pixel_values": torch.ones(1, 1), "image_grid_thw": torch.tensor([[1, 1, 1]])}]},
    )

    def fake_build_training_sample(*args, **kwargs):
        return RenderedTrainingSample(
            token_ids=[10, 11, 12, _STOP_TOKEN_ID],
            loss_mask=[False, False, False, True],
            multi_modal_data=mm,
            mm_token_type_ids=[0, 1, 0, 0],
        )

    monkeypatch.setattr(sft_data, "build_training_sample", fake_build_training_sample)
    dataset = SFTDataset(Dataset.from_list([]), lambda _: dummy_renderer, seq_len=2, multimodal=True)

    assert dataset._process({"messages": [{"role": "assistant", "content": "ignored"}]}) is None


def _sft_sample(
    input_ids: list[int],
    *,
    mm_kwargs: dict[str, torch.Tensor] | None = None,
    mm_token_type_ids: list[int] | None = None,
) -> dict:
    return {
        "input_ids": input_ids,
        "position_ids": list(range(len(input_ids))),
        "loss_mask": [True] * len(input_ids),
        "target_ids": [x + 1 for x in input_ids],
        "seq_lens": [len(input_ids)],
        "mm_kwargs": mm_kwargs,
        "mm_token_type_ids": mm_token_type_ids,
    }


class _ListDataset(StatefulIterableDataset):
    def __init__(self, samples: list[dict]):
        super().__init__()
        self.samples = samples

    def sample_at(self, step: int) -> dict:
        return self.samples[step - 1]

    def __iter__(self):
        while self.step < len(self.samples):
            self.step += 1
            yield self.samples[self.step - 1]


def test_cat_dataset_packs_multimodal_samples():
    dataset = CatDataset(
        _ListDataset(
            [
                _sft_sample(
                    [1, 2],
                    mm_kwargs={
                        "pixel_values": torch.ones(2, 3),
                        "image_grid_thw": torch.tensor([[1, 1, 2]]),
                    },
                    mm_token_type_ids=[0, 1],
                ),
                _sft_sample(
                    [3, 4, 5],
                    mm_kwargs={
                        "pixel_values": 2 * torch.ones(3, 3),
                        "image_grid_thw": torch.tensor([[1, 1, 3]]),
                    },
                    mm_token_type_ids=[0, 1, 1],
                ),
            ]
        ),
        seq_len=5,
    )

    packed = next(iter(dataset))

    assert packed["input_ids"] == [1, 2, 3, 4, 5]
    assert packed["seq_lens"] == [2, 3]
    assert packed["mm_token_type_ids"] == [0, 1, 0, 1, 1]
    assert packed["mm_kwargs"]["pixel_values"].shape == (5, 3)
    assert packed["mm_kwargs"]["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 3]]


def test_cat_dataset_packs_text_and_multimodal_samples_together():
    dataset = CatDataset(
        _ListDataset(
            [
                _sft_sample([1]),
                _sft_sample(
                    [2, 3],
                    mm_kwargs={
                        "pixel_values": torch.ones(2, 3),
                        "image_grid_thw": torch.tensor([[1, 1, 2]]),
                    },
                    mm_token_type_ids=[0, 1],
                ),
                _sft_sample([4]),
                _sft_sample([5, 6]),
            ]
        ),
        seq_len=5,
    )

    dataiter = iter(dataset)
    packed = next(dataiter)
    text_pack = next(dataiter)

    assert packed["input_ids"] == [1, 2, 3, 4]
    assert packed["loss_mask"] == [True, True, True, True]
    assert packed["seq_lens"] == [1, 2, 1]
    assert packed["mm_kwargs"] is not None
    assert packed["mm_token_type_ids"] == [0, 0, 1, 0]
    assert text_pack["input_ids"] == [5, 6]
    assert text_pack["loss_mask"] == [True, True]
    assert text_pack["seq_lens"] == [2]
    assert text_pack["mm_kwargs"] is None
    assert text_pack["mm_token_type_ids"] is None


def test_cat_dataset_pads_rows_to_a_multiple_of_cp():
    dataset = CatDataset(_ListDataset([_sft_sample([1, 2, 3, 4, 5]), _sft_sample([6, 7])]), seq_len=8, cp_world_size=4)

    (row,) = list(dataset)

    # 7 tokens round up to 8 for 4 CP ranks; the pad token joins the last sample, loss-masked.
    assert row["input_ids"] == [1, 2, 3, 4, 5, 6, 7, 0]
    assert row["loss_mask"] == [True] * 7 + [False]
    assert row["seq_lens"] == [5, 3]


def test_cat_dataset_lookahead_fills_rows_first_fit_decreasing():
    lengths = [6, 7, 3, 2, 4, 1]
    samples = [_sft_sample(list(range(10 * i, 10 * i + length))) for i, length in enumerate(lengths)]

    greedy = [row["seq_lens"] for row in CatDataset(_ListDataset(samples), seq_len=10)]
    lookahead = [row["seq_lens"] for row in CatDataset(_ListDataset(samples), seq_len=10, lookahead=4)]

    assert greedy == [[6], [7, 3], [2, 4, 1]]
    # Each row starts with the oldest of the 4 buffered samples and fills the gap largest-first.
    assert lookahead == [[6, 3], [7, 2, 1], [4]]


def test_cat_dataset_lookahead_resumes_with_the_same_rows():
    lengths = [5, 9, 2, 7, 3, 3, 8, 1, 6, 4, 2, 5]
    samples = [_sft_sample(list(range(10 * i, 10 * i + length))) for i, length in enumerate(lengths)]

    dataset = CatDataset(_ListDataset(samples), seq_len=10, lookahead=3)
    dataiter = iter(dataset)
    next(dataiter), next(dataiter)
    state_dict = dataset.state_dict()
    expected = [row["input_ids"] for row in dataiter]

    resumed = CatDataset(_ListDataset(samples), seq_len=10, lookahead=3)
    resumed.load_state_dict(state_dict)
    assert [row["input_ids"] for row in resumed] == expected
