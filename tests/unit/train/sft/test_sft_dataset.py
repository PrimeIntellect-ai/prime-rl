import json
from collections import Counter

import numpy as np
import pytest
import torch
from datasets import Dataset
from renderers import create_renderer
from renderers.base import MultiModalData, PlaceholderRange, RenderedTrainingSample
from transformers import AutoTokenizer

import prime_rl.trainer.sft.data as sft_data
from prime_rl.trainer.data import CatDataset
from prime_rl.trainer.sft.data import HFSource, SFTDataset, _drop_null_fields
from prime_rl.trainer.utils import print_sample

_BOS_TOKEN_ID = 0
_STOP_TOKEN_ID = 1


def _sample_token_ids(value: str) -> list[int]:
    return [ord(char) + 2 for char in value]


def _as_lists(sample: dict) -> dict:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in sample.items()}


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
    source = HFSource(dataset, lambda _: dummy_renderer)
    assert SFTDataset([source]) is not None


def test_raise_error_if_no_prompt_and_completion(build_dummy_dataset):
    """Tests that an error is raised if there is no prompt and completion."""
    dataset = Dataset.from_list([{"text": "a"}])
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    source = HFSource(dataset, lambda _: create_renderer(tokenizer))
    with pytest.raises(ValueError):
        next(iter(SFTDataset([source])))


def position(state: dict) -> dict:
    return {key: state[key] for key in ("step", "epoch", "draws")}


def _content(sample: dict) -> str:
    return "".join(chr(token - 2) for token in sample["target_ids"][:-1].tolist())


@pytest.fixture
def source(build_dummy_dataset, dummy_renderer):
    def make(letter: str, num_examples: int, **kwargs) -> HFSource:
        dataset = build_dummy_dataset(letter, num_examples)
        return HFSource(dataset, lambda _: dummy_renderer, name=letter, shuffle=kwargs.pop("shuffle", False), **kwargs)

    return make


@pytest.mark.parametrize("max_epochs", [1, 2])
def test_natural_shares_read_every_row_once_per_epoch(source, max_epochs: int):
    dataset = SFTDataset([source("a", 2), source("b", 4)], max_epochs=max_epochs)
    order = [_content(sample) for sample in dataset]
    # Due times (draws + 1) / rows, ties to the first source
    assert order == ["b0", "a0", "b1", "b2", "a1", "b3"] * max_epochs
    assert position(dataset.state_dict()) == {
        "step": 6 * max_epochs,
        "epoch": max_epochs - 1,
        "draws": {"a": 2 * max_epochs, "b": 4 * max_epochs},
    }


def test_weights_set_the_share_of_every_stretch_of_draws(source):
    dataset = SFTDataset([source("a", 1), source("b", 10)], weights=[1, 3])
    dataiter = iter(dataset)
    order = [_content(next(dataiter))[0] for _ in range(400)]
    assert Counter(order) == {"a": 100, "b": 300}
    assert all(Counter(order[i : i + 4]) == {"a": 1, "b": 3} for i in range(0, 400, 4))


def test_zero_weight_source_is_never_drawn(source):
    dataiter = iter(SFTDataset([source("a", 2), source("b", 2)], weights=[0, 1]))
    assert {_content(next(dataiter))[0] for _ in range(10)} == {"b"}


@pytest.mark.parametrize("dp_size", [2, 3])
def test_ranks_split_the_same_draws(source, dp_size: int):
    def contents(dp_rank: int, dp_size: int, count: int) -> list[str]:
        dataset = SFTDataset(
            [source("a", 3, shuffle=True), source("b", 5, shuffle=True)], dp_rank=dp_rank, dp_size=dp_size
        )
        dataiter = iter(dataset)
        return [_content(next(dataiter)) for _ in range(count)]

    single = contents(0, 1, 6 * dp_size)
    per_rank = [contents(rank, dp_size, 6) for rank in range(dp_size)]
    assert [per_rank[i % dp_size][i // dp_size] for i in range(6 * dp_size)] == single


def test_sft_dataset_state(source):
    """Tests the state of the dataset within and across epochs."""
    dataset = SFTDataset([source("", 4)], max_epochs=2)
    dataiter = iter(dataset)

    assert position(dataset.state_dict()) == {"step": 0, "epoch": 0, "draws": {"": 0}}
    for epoch in range(2):
        for i in range(4):
            assert _content(next(dataiter)) == str(i)
            step = 4 * epoch + i + 1
            assert position(dataset.state_dict()) == {"step": step, "epoch": (step - 1) // 4, "draws": {"": step}}

    with pytest.raises(StopIteration):
        next(dataiter)


def test_sft_dataset_state_resume(source):
    """A resumed dataset continues the draws of an uninterrupted one, within a pass and across passes."""

    def make() -> SFTDataset:
        return SFTDataset([source("a", 3, shuffle=True), source("b", 4, shuffle=True)], max_epochs=3)

    expected = [_content(sample) for sample in make()]
    for stop in (2, 7, 12):
        dataset = make()
        dataiter = iter(dataset)
        contents = [_content(next(dataiter)) for _ in range(stop)]
        resumed = make()
        resumed.load_state_dict(dataset.state_dict())
        contents += [_content(sample) for sample in resumed]
        assert contents == expected


def test_resume_single_source_from_state_without_draws(source):
    dataset = SFTDataset([source("", 4)])
    dataset.num_samples["train"] = 3
    dataset.load_state_dict({"step": 3, "epoch": 0})
    assert dict(dataset.num_samples) == {"": 3}
    dataiter = iter(dataset)
    assert [_content(next(dataiter)) for _ in range(2)] == ["3", "0"]


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
    dataset = HFSource(dataset, lambda _: create_renderer(tokenizer), max_examples=1)
    sample = dataset.sample(0)
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
    dataset = HFSource(dataset, lambda _: create_renderer(tokenizer), max_examples=1)
    sample = dataset.sample(0)
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
    messages_dataset = HFSource(
        Dataset.from_list([{"messages": messages}]),
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )
    split_dataset = HFSource(
        Dataset.from_list([{"prompt": [], "completion": messages}]),
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )

    assert _as_lists(messages_dataset.sample(0)) == _as_lists(split_dataset.sample(0))


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

    messages_dataset = HFSource(
        Dataset.from_list([row]),
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )
    expected_dataset = HFSource(
        Dataset.from_list([{"prompt": [], "completion": row["messages"]}]),
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )

    assert _as_lists(messages_dataset.sample(0)) == _as_lists(expected_dataset.sample(0))


def test_null_messages_falls_back_to_prompt_and_completion():
    # Arrow schema union adds `messages: None` to prompt/completion rows when
    # other rows in the file have a `messages` column
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "4"}]

    mixed_row_dataset = HFSource(
        Dataset.from_list([{"messages": None, "prompt": prompt, "completion": completion}]),
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )
    expected_dataset = HFSource(
        Dataset.from_list([{"prompt": prompt, "completion": completion}]),
        lambda _: create_renderer(tokenizer),
        max_examples=1,
    )

    assert _as_lists(mixed_row_dataset.sample(0)) == _as_lists(expected_dataset.sample(0))


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
    dataset = HFSource(Dataset.from_list([]), lambda _: dummy_renderer, seq_len=2, multimodal=True)

    assert dataset.process({"messages": [{"role": "assistant", "content": "ignored"}]}) is None


def _sft_sample(
    input_ids: list[int],
    *,
    mm_kwargs: dict[str, torch.Tensor] | None = None,
    mm_token_type_ids: list[int] | None = None,
) -> dict:
    return {
        "input_ids": np.array(input_ids),
        "position_ids": np.arange(len(input_ids)),
        "loss_mask": np.ones(len(input_ids), dtype=bool),
        "target_ids": np.array(input_ids) + 1,
        "seq_lens": [len(input_ids)],
        "mm_kwargs": mm_kwargs,
        "mm_token_type_ids": np.array(mm_token_type_ids) if mm_token_type_ids is not None else None,
    }


def test_cat_dataset_packs_multimodal_samples():
    dataset = CatDataset(
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
        ],
        seq_len=5,
        pad_to_multiple_of=1,
    )

    packed = next(iter(dataset))

    assert packed["input_ids"].tolist() == [1, 2, 3, 4, 5]
    assert packed["seq_lens"] == [2, 3]
    assert packed["mm_token_type_ids"].tolist() == [0, 1, 0, 1, 1]
    assert packed["mm_kwargs"]["pixel_values"].shape == (5, 3)
    assert packed["mm_kwargs"]["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 3]]


def test_cat_dataset_packs_text_and_multimodal_samples_together():
    dataset = CatDataset(
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
            _sft_sample([5, 6, 7, 8, 9, 10, 11]),
        ],
        seq_len=9,
        pad_to_multiple_of=3,
    )

    dataiter = iter(dataset)
    packed = next(dataiter)
    text_pack = next(dataiter)

    assert packed["input_ids"].tolist() == [1, 2, 3, 4, 0, 0]
    assert packed["loss_mask"].tolist() == [True, True, True, True, False, False]
    assert packed["seq_lens"] == [1, 2, 3]
    assert packed["mm_kwargs"] is not None
    assert packed["mm_token_type_ids"].tolist() == [0, 0, 1, 0, 0, 0]
    # A text pack pads with its own tokens.
    assert text_pack["input_ids"].tolist() == [5, 6, 7, 8, 9, 10, 11, 5, 6]
    assert text_pack["position_ids"].tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 1]
    assert text_pack["loss_mask"].tolist() == [True] * 7 + [False] * 2
    assert text_pack["seq_lens"] == [9]
    assert text_pack["mm_kwargs"] is None
    assert text_pack["mm_token_type_ids"] is None


CUSTOM_RENDERER_SOURCE = """
from typing import Literal

from renderers.base import RenderedTokens
from renderers.configs import BaseRendererConfig


class EffortRendererConfig(BaseRendererConfig):
    name: Literal["effort"] = "effort"
    _template_fields = frozenset({"reasoning_effort"})

    reasoning_effort: Literal["low", "high"] = "low"


class EffortRenderer:
    config_class = EffortRendererConfig

    def __init__(self, tokenizer, config):
        self.config = config

    @property
    def is_prefix_stable(self):
        return self.config.reasoning_effort == "low"

    def render(self, messages, **kwargs):
        return RenderedTokens(token_ids=[0, 1], message_indices=[-1, 0], sampled_mask=[False, True])

    def get_stop_token_ids(self):
        return [1]
"""


@pytest.fixture
def custom_renderer_config(tmp_path):
    from renderers import CustomRendererConfig

    path = tmp_path / "effort_renderer.py"
    path.write_text(CUSTOM_RENDERER_SOURCE)
    return CustomRendererConfig(import_path=f"{path}:EffortRenderer")


def test_source_renderer_kwargs_apply_to_that_source_only(custom_renderer_config):
    from prime_rl.configs.sft import SFTDataConfig

    config = SFTDataConfig(
        source=[
            {"name": "high", "dataset": "unused", "renderer": {"reasoning_effort": "high"}},
            {"name": "default", "dataset": "unused"},
        ]
    )
    rows = Dataset.from_list([{"messages": [{"role": "assistant", "content": "a"}]}])
    dataset = sft_data.setup_sft_dataset(
        None,
        config,
        dp_rank=0,
        dp_size=1,
        raw_datasets={"high": rows, "default": rows},
        renderer_config=custom_renderer_config,
    )
    efforts = [source.renderers({}).config.reasoning_effort for source in dataset.sources]
    assert efforts == ["high", "low"]


def test_renderer_resolver_applies_renderer_columns_to_custom_renderer(custom_renderer_config):
    resolver = sft_data.RendererResolver(tokenizer=None, config=custom_renderer_config)

    warnings = []
    logger = sft_data.get_logger()
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING", format="{message}")
    try:
        default = resolver({"messages": [], "reasoning_effort": None})
        assert warnings == []
        high = resolver({"messages": [], "reasoning_effort": "high"})
        assert resolver({"reasoning_effort": "high"}) is high
        assert len(warnings) == 1
        assert "does not guarantee prefix stability" in warnings[0]
        assert "one training sample" in warnings[0]
        assert "does not expand N assistant turns into N samples" in warnings[0]
        assert "reasoning from earlier turns may be omitted" in warnings[0]
    finally:
        logger.remove(sink)

    assert default.config.reasoning_effort == "low"
    assert high.config.reasoning_effort == "high"
    assert resolver({"messages": [], "reasoning_effort": "high"}) is high
    with pytest.raises(ValueError):
        resolver({"messages": [], "reasoning_effort": "medium"})


def test_renderer_resolver_reads_mapped_columns(custom_renderer_config):
    resolver = sft_data.RendererResolver(
        tokenizer=None, config=custom_renderer_config, columns={"reasoning_effort": "effort"}
    )

    assert resolver({"effort": "high", "reasoning_effort": "low"}).config.reasoning_effort == "high"
    assert resolver({"reasoning_effort": "high"}).config.reasoning_effort == "low"


def test_renderer_resolver_rejects_renderer_columns_without_the_field():
    from renderers import PrimeQwen3RendererConfig

    resolver = sft_data.RendererResolver(tokenizer=None, config=PrimeQwen3RendererConfig())

    with pytest.raises(ValueError, match="reasoning_effort"):
        resolver({"messages": [], "reasoning_effort": "high"})


def test_sft_dataset_reads_mapped_message_columns(dummy_renderer):
    from prime_rl.configs.sft import SFTColumnsConfig

    dataset = Dataset.from_list([{"conversation": [{"role": "assistant", "content": "a0"}]}])

    sample = HFSource(dataset, lambda _: dummy_renderer, columns=SFTColumnsConfig(messages="conversation")).sample(0)
    assert sample["input_ids"].tolist() == [_BOS_TOKEN_ID, *_sample_token_ids("a0")]
    assert sample["target_ids"].tolist() == [*_sample_token_ids("a0"), _STOP_TOKEN_ID]

    with pytest.raises(ValueError, match="'messages' column"):
        HFSource(dataset, lambda _: dummy_renderer).sample(0)
    with pytest.raises(ValueError, match="data.columns.message_loss_mask"):
        HFSource(dataset, lambda _: dummy_renderer, columns=SFTColumnsConfig(message_loss_mask="selection"))
    with pytest.raises(ValueError, match="data.columns.prompt"):
        HFSource(dataset, lambda _: dummy_renderer, columns=SFTColumnsConfig(prompt="question"))


def test_sft_dataset_passes_tools_through_from_the_mapped_column(dummy_renderer):
    from prime_rl.configs.sft import SFTColumnsConfig

    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    dataset = Dataset.from_list(
        [{"messages": [{"role": "assistant", "content": "a0"}], "schemas": json.dumps(tools), "tools": []}]
    )
    seen: list = []

    class RecordingRenderer:
        def render(self, messages, tools=None, **kwargs):
            seen.append(tools)
            return dummy_renderer.render(messages)

        def get_stop_token_ids(self):
            return dummy_renderer.get_stop_token_ids()

    HFSource(dataset, lambda _: RecordingRenderer()).sample(0)
    HFSource(dataset, lambda _: RecordingRenderer(), columns=SFTColumnsConfig(tools="schemas")).sample(0)

    assert seen == [[], tools]


@pytest.mark.parametrize("model", ["zai-org/GLM-5", "Qwen/Qwen3-0.6B"])
@pytest.mark.parametrize("layout", ["messages", "prompt_completion"])
@pytest.mark.parametrize("column", ["message_loss_mask", "selection"])
@pytest.mark.parametrize("mask", [None, [True, True, True, True], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]])
def test_sft_message_selection(model, layout, column, mask):
    from prime_rl.configs.sft import SFTColumnsConfig

    tokenizer = AutoTokenizer.from_pretrained(model)
    renderer = create_renderer(tokenizer)
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Prefilled"},
        {"role": "user", "content": "Follow up"},
        {"role": "assistant", "content": "Answer"},
    ]
    row = {"messages": messages} if layout == "messages" else {"prompt": messages[:2], "completion": messages[2:]}
    baseline = HFSource(Dataset.from_list([row]), lambda _: renderer).process(row)
    row[column] = mask
    dataset = HFSource(Dataset.from_list([row]), lambda _: renderer, columns=SFTColumnsConfig(message_loss_mask=column))
    sample = dataset.process(row)
    if mask == [0, 0, 0, 0]:
        assert sample is None
        return
    assert sample["input_ids"].tolist() == baseline["input_ids"].tolist()
    assert sample["target_ids"].tolist() == baseline["target_ids"].tolist()
    if mask is None or all(mask):
        assert sample["loss_mask"].tolist() == baseline["loss_mask"].tolist()
        return
    trained = tokenizer.decode(sample["target_ids"][sample["loss_mask"]].tolist())
    if mask[1]:
        assert "Prefilled" in trained
        assert "Answer" not in trained
        if model == "zai-org/GLM-5":
            assert trained == "Prefilled<|user|>"
    else:
        assert "Prefilled" not in trained
        assert "Answer" in trained
        if model == "zai-org/GLM-5":
            assert trained == "Answer<|endoftext|>"
    assert "Question" not in trained
    assert "Follow up" not in trained


@pytest.mark.parametrize("mask", [[], [1, 0], ["0"], [2], [None]])
def test_sft_rejects_invalid_message_selection(dummy_renderer, mask):
    row = {"messages": [{"role": "assistant", "content": "Answer"}], "message_loss_mask": mask}
    dataset = HFSource(Dataset.from_list([row]), lambda _: dummy_renderer)
    with pytest.raises(ValueError, match="message_loss_mask"):
        dataset.process(row)
