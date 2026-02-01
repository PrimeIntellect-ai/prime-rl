from collections import Counter

import pytest
from datasets import Dataset, interleave_datasets
from transformers import AutoTokenizer

from prime_rl.trainer.sft.data import FakeDataset, SFTDataset, SingleSampleDataset
from prime_rl.trainer.utils import print_sample


@pytest.fixture(scope="module")
def build_dummy_dataset():
    return lambda letter, num_examples: Dataset.from_list([{"text": f"{letter}{i}"} for i in range(num_examples)])


def test_init_sft_dataset(build_dummy_dataset):
    """Tests basic initialization."""
    dataset = build_dummy_dataset("a", 1)
    sft_dataset = SFTDataset(dataset, tokenizer=None)
    assert sft_dataset is not None


def test_raise_error_if_no_prompt_and_completion(build_dummy_dataset):
    """Tests that an error is raised if no prompt and completion are provided but a tokenizer is provided."""
    dataset = build_dummy_dataset("a", 1)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    sft_dataset = SFTDataset(dataset, tokenizer=tokenizer)
    with pytest.raises(ValueError):
        next(iter(sft_dataset))


@pytest.mark.parametrize("max_epochs", [1, 2, 4])
def test_sft_first_exhausted(build_dummy_dataset, max_epochs: int):
    a = build_dummy_dataset("a", 1)
    b = build_dummy_dataset("b", 2)
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="first_exhausted")
    dataset = SFTDataset(dataset, tokenizer=None, shuffle=False, max_epochs=max_epochs)
    num_samples = 0
    sampling_order = []
    for x in dataset:
        sampling_order.append(x["text"])
        num_samples += 1
    assert num_samples == max_epochs * min([len(d) for d in ds]) * len(ds)
    assert sampling_order == ["a0", "b0"] * max_epochs


@pytest.mark.parametrize("max_epochs", [1, 2, 4])
def test_sft_all_exhausted(build_dummy_dataset, max_epochs: int):
    a = build_dummy_dataset("a", 1)
    b = build_dummy_dataset("b", 2)
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="all_exhausted")
    dataset = SFTDataset(dataset, tokenizer=None, shuffle=False, max_epochs=max_epochs)
    num_samples = 0
    sampling_order = []
    for x in dataset:
        sampling_order.append(x["text"])
        num_samples += 1
    assert num_samples == max_epochs * max([len(d) for d in ds]) * len(ds)
    print(sampling_order)
    assert sampling_order == ["a0", "b0", "a0", "b1"] * max_epochs


@pytest.mark.parametrize(
    "probs",
    [
        pytest.param((0.5, 0.5), id="equal_probs"),
        pytest.param((1 / 10, 9 / 10), id="low_high_probs"),
        pytest.param((9 / 10, 1 / 10), id="high_low_probs"),
    ],
)
def test_sft_all_exhausted_with_probs(build_dummy_dataset, probs: list[float]):
    """Tests that the ratio of samples from different datasets is as specified, in expectation."""
    a = build_dummy_dataset("a", int(1e3))
    b = build_dummy_dataset("b", int(10e3))
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="all_exhausted", probabilities=probs)
    dataset = SFTDataset(dataset, tokenizer=None, shuffle=False, max_epochs=1)
    num_samples = 0
    sampling_freq = []
    for x in dataset:
        sampling_freq.append(x["text"][0])
        num_samples += 1
    sampling_freq = Counter(sampling_freq)
    ratio_a = sampling_freq["a"] / num_samples
    ratio_b = sampling_freq["b"] / num_samples
    assert ratio_a > probs[0] * 0.8 and ratio_a < probs[0] * 1.2, (
        f"Expected frequency of samples from a to be between {probs[0] * 0.8} and {probs[0] * 1.2}, but got {ratio_a}"
    )
    assert ratio_b > probs[1] * 0.8 and ratio_b < probs[1] * 1.2, (
        f"Exepcted frequency of samples from b to be between {probs[1] * 0.8} and {probs[1] * 1.2}, but got {ratio_b}"
    )


def test_sft_dataset_state(build_dummy_dataset):
    """Tests the state of the dataset within and across epochs."""
    dataset = build_dummy_dataset("", 4)
    dataset = SFTDataset(dataset, tokenizer=None, shuffle=False, max_epochs=2)
    dataiter = iter(dataset)

    # Initial state
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Epoch 1
    for i in range(4):
        sample = next(dataiter)
        assert sample["text"] == str(i)
        assert dataset.state_dict() == {"epoch": 0, "step": i + 1}

    # Epoch 2
    for i in range(4):
        sample = next(dataiter)
        assert sample["text"] == str(i)
        assert dataset.state_dict() == {"epoch": 1, "step": 4 + i + 1}

    with pytest.raises(StopIteration):
        next(dataiter)


def test_sft_dataset_state_resume(build_dummy_dataset):
    """Tests resuming the dataset from checkpoint in between epochs."""
    dataset = SFTDataset(
        build_dummy_dataset("", 4),
        tokenizer=None,
        shuffle=False,
        max_epochs=2,
    )
    dataiter = iter(dataset)

    # Initial state
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Epoch 1
    for i in range(4):
        sample = next(dataiter)
        print(sample["text"])
        assert sample["text"] == str(i)
        assert dataset.state_dict() == {"epoch": 0, "step": i + 1}

    # Resuming from checkpoint cross epoch
    state_dict = dataset.state_dict()
    del dataset
    dataset = SFTDataset(
        build_dummy_dataset("", 4),
        tokenizer=None,
        shuffle=False,
        max_epochs=2,
    )
    dataset.load_state_dict(state_dict)
    dataiter = iter(dataset)

    # Epoch 2.1
    for i in range(2):
        sample = next(dataiter)
        print(sample["text"])
        assert sample["text"] == str(i)
        assert dataset.state_dict() == {"epoch": 1, "step": 4 + i + 1}

    # Resuming from checkpoint mid epoch
    state_dict = dataset.state_dict()
    del dataset
    dataset = SFTDataset(
        build_dummy_dataset("", 4),
        tokenizer=None,
        shuffle=False,
        max_epochs=2,
    )
    dataset.load_state_dict(state_dict)
    dataiter = iter(dataset)

    # Epoch 2.2
    for i in range(2, 4):
        sample = next(dataiter)
        print(sample["text"])
        assert sample["text"] == str(i)
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
    dataset = SFTDataset(dataset, tokenizer=tokenizer, max_examples=1)
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
    dataset = SFTDataset(dataset, tokenizer=tokenizer, max_examples=1)
    sample = next(iter(dataset))
    print_sample(sample["input_ids"], sample["loss_mask"], tokenizer)


def test_single_sample_dataset_padding():
    """Tests that SingleSampleDataset pads samples shorter than seq_len."""
    seq_len = 128
    fake_dataset = FakeDataset(vocab_size=10000, seq_len=32, length="fixed")
    single_dataset = SingleSampleDataset(fake_dataset, seq_len=seq_len)
    dataiter = iter(single_dataset)

    sample = next(dataiter)
    assert len(sample["input_ids"]) == seq_len
    assert len(sample["target_ids"]) == seq_len
    assert len(sample["position_ids"]) == seq_len
    assert len(sample["loss_mask"]) == seq_len

    # Verify padding: loss_mask should be False for padded positions
    assert sample["loss_mask"][:32] == [True] * 32
    assert sample["loss_mask"][32:] == [False] * (seq_len - 32)

    # Verify padding values are 0 for input_ids, target_ids, position_ids
    assert sample["input_ids"][32:] == [0] * (seq_len - 32)
    assert sample["target_ids"][32:] == [0] * (seq_len - 32)
    assert sample["position_ids"][32:] == [0] * (seq_len - 32)


def test_single_sample_dataset_truncation():
    """Tests that SingleSampleDataset truncates samples longer than seq_len."""
    seq_len = 64
    fake_dataset = FakeDataset(vocab_size=10000, seq_len=128, length="fixed")
    single_dataset = SingleSampleDataset(fake_dataset, seq_len=seq_len)
    dataiter = iter(single_dataset)

    sample = next(dataiter)
    assert len(sample["input_ids"]) == seq_len
    assert len(sample["target_ids"]) == seq_len
    assert len(sample["position_ids"]) == seq_len
    assert len(sample["loss_mask"]) == seq_len


def test_single_sample_dataset_exact_length():
    """Tests that SingleSampleDataset handles samples with exact seq_len."""
    seq_len = 128
    fake_dataset = FakeDataset(vocab_size=10000, seq_len=seq_len, length="fixed")
    single_dataset = SingleSampleDataset(fake_dataset, seq_len=seq_len)
    dataiter = iter(single_dataset)

    sample = next(dataiter)
    assert len(sample["input_ids"]) == seq_len
    assert len(sample["target_ids"]) == seq_len
    assert len(sample["position_ids"]) == seq_len
    assert len(sample["loss_mask"]) == seq_len


def test_single_sample_dataset_state():
    """Tests that SingleSampleDataset correctly delegates state management."""
    fake_dataset = FakeDataset(vocab_size=10000, seq_len=32, length="fixed")
    single_dataset = SingleSampleDataset(fake_dataset, seq_len=128)

    # Initial state
    state = single_dataset.state_dict()
    assert state == {"dataset": {"step": 0, "epoch": 0}}

    # Iterate and check state
    dataiter = iter(single_dataset)
    next(dataiter)
    state = single_dataset.state_dict()
    assert state == {"dataset": {"step": 1, "epoch": 0}}

    next(dataiter)
    state = single_dataset.state_dict()
    assert state == {"dataset": {"step": 2, "epoch": 0}}


def test_single_sample_dataset_resume():
    """Tests that SingleSampleDataset can resume from checkpoint."""
    fake_dataset = FakeDataset(vocab_size=10000, seq_len=32, length="fixed", input_ids="increasing")
    single_dataset = SingleSampleDataset(fake_dataset, seq_len=128)
    dataiter = iter(single_dataset)

    # Get first sample
    sample1 = next(dataiter)
    sample2 = next(dataiter)
    state_dict = single_dataset.state_dict()

    # Create new dataset and resume
    fake_dataset2 = FakeDataset(vocab_size=10000, seq_len=32, length="fixed", input_ids="increasing")
    single_dataset2 = SingleSampleDataset(fake_dataset2, seq_len=128)
    single_dataset2.load_state_dict(state_dict)
    dataiter2 = iter(single_dataset2)

    # Should continue from where we left off
    sample3 = next(dataiter2)
    assert sample3["input_ids"][:32] != sample1["input_ids"][:32]
    assert sample3["input_ids"][:32] != sample2["input_ids"][:32]
