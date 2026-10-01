import pytest
from datasets import Dataset
from renderers import DeepSeekV4RendererConfig, PrimeQwen3RendererConfig

import prime_rl.trainer.sft.data as sft_data
from prime_rl.configs.sft import FakeSourceConfig, SFTDataConfig
from prime_rl.trainer.sft.data import (
    RendererResolver,
    decode_json_columns,
    load_sft_dataset,
    validate_source_renderer_args,
)


def _messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text}, {"role": "assistant", "content": text.upper()}]


@pytest.fixture
def hub(monkeypatch):
    """Serves in-memory datasets to ``load_dataset`` keyed by ``(name, subset, split)``."""
    datasets: dict[tuple, Dataset] = {}

    def load_dataset(name, subset=None, split="train", revision=None):
        return datasets[(name, subset, split)]

    monkeypatch.setattr(sft_data, "load_dataset", load_dataset)
    return datasets


def test_data_name_resolves_to_one_train_source():
    config = SFTDataConfig(name="org/data", revision="abc")
    (source,) = config.resolved_sources()
    assert (source.name, source.dataset, source.revision, source.subset, source.split, source.weight) == (
        "train",
        "org/data",
        "abc",
        None,
        "train",
        None,
    )


def test_sources_inherit_data_defaults():
    config = SFTDataConfig.model_validate(
        {
            "name": "org/data",
            "revision": "abc",
            "columns": {"messages": "chat"},
            "source": [
                {"subset": "a"},
                {"dataset": "org/other", "split": "test", "columns": {"tools": "tool_list"}, "renderer": {"x": 1}},
            ],
        }
    )
    first, second = config.resolved_sources()
    assert (first.name, first.dataset, first.revision, first.columns.messages) == (
        "org/data/a/train",
        "org/data",
        "abc",
        "chat",
    )
    assert (second.name, second.dataset, second.revision) == ("org/other/test", "org/other", None)
    assert (second.columns.messages, second.columns.tools, second.renderer) == ("chat", "tool_list", {"x": 1})


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"source": []}, id="empty"),
        pytest.param({"source": [{"subset": "a", "weight": 1}, {"subset": "b"}]}, id="partial-weights"),
        pytest.param({"source": [{"subset": "a"}, {"subset": "a"}]}, id="duplicate-names"),
        pytest.param({"source": [{"type": "fake"}, {"subset": "a"}]}, id="fake-and-hf"),
        pytest.param({"source": [{"type": "fake"}, {"type": "fake"}]}, id="two-fakes"),
    ],
)
def test_invalid_sources(data):
    with pytest.raises(ValueError):
        SFTDataConfig.model_validate({"name": "org/data", **data})


def test_load_maps_source_columns_to_run_wide_names(hub):
    hub[("org/a", None, "train")] = Dataset.from_list(
        [{"chat": _messages("a0"), "messages": "ignored", "effort": "high", "extra": 1}]
    )
    hub[("org/b", None, "train")] = Dataset.from_list([{"messages": _messages("b0")}])
    config = SFTDataConfig.model_validate(
        {
            "source": [
                {"dataset": "org/a", "columns": {"messages": "chat", "renderer": {"reasoning_effort": "effort"}}},
                {"dataset": "org/b"},
            ]
        }
    )
    rows = list(load_sft_dataset(config))
    assert [row["messages"][0]["content"] for row in rows] == ["a0", "b0"]
    assert [row["reasoning_effort"] for row in rows] == ["high", None]
    assert [row["__source"] for row in rows] == ["org/a/train", "org/b/train"]
    assert "extra" not in rows[0] and "chat" not in rows[0]


def test_sources_with_different_message_columns_interleave(hub):
    hub[("org/a", None, "train")] = Dataset.from_list([{"messages": _messages("a0")}])
    hub[("org/b", None, "train")] = Dataset.from_list(
        [{"prompt": _messages("b0")[:1], "completion": _messages("b0")[1:]}]
    )
    config = SFTDataConfig.model_validate({"source": [{"dataset": "org/a"}, {"dataset": "org/b"}]})
    rows = list(load_sft_dataset(config))
    assert rows[0]["messages"][0]["content"] == "a0" and rows[0]["prompt"] is None
    assert rows[1]["messages"] is None and rows[1]["completion"][0]["content"] == "B0"


def test_configured_source_column_must_exist(hub):
    hub[("org/a", None, "train")] = Dataset.from_list([{"messages": _messages("a0")}])
    config = SFTDataConfig.model_validate({"source": [{"dataset": "org/a", "columns": {"messages": "chat"}}]})
    with pytest.raises(ValueError, match="reads messages from 'chat'"):
        load_sft_dataset(config)


def test_conflicting_column_types_interleave_as_json(hub):
    hub[("org/a", None, "train")] = Dataset.from_list([{"messages": _messages("a0"), "depth": 16}])
    hub[("org/b", None, "train")] = Dataset.from_list(
        [{"messages": [{**message, "tool_calls": None} for message in _messages("b0")], "depth": "max"}]
    )
    config = SFTDataConfig.model_validate(
        {"columns": {"renderer": {"depth": "depth"}}, "source": [{"dataset": "org/a"}, {"dataset": "org/b"}]}
    )
    rows = [decode_json_columns(row) for row in load_sft_dataset(config)]
    assert [row["depth"] for row in rows] == [16, "max"]
    assert [row["messages"][0]["content"] for row in rows] == ["a0", "b0"]


def test_source_weights_are_normalized(hub):
    hub[("org/a", None, "train")] = Dataset.from_list([{"messages": _messages(f"a{i}")} for i in range(200)])
    hub[("org/b", None, "train")] = Dataset.from_list([{"messages": _messages(f"b{i}")} for i in range(200)])
    config = SFTDataConfig.model_validate(
        {"source": [{"dataset": "org/a", "weight": 3}, {"dataset": "org/b", "weight": 1}]}
    )
    sources = [row["__source"] for row in load_sft_dataset(config)]
    assert sources.count("org/a/train") > 2 * sources.count("org/b/train")


def test_renderer_precedence_is_global_then_source_then_sample():
    config = DeepSeekV4RendererConfig(enable_thinking=True, reasoning_effort="low")
    resolver = RendererResolver(tokenizer=None, config=config, source_kwargs={"s": {"reasoning_effort": "high"}})
    assert resolver.resolve_config({"__source": "other"}).reasoning_effort == "low"
    assert resolver.resolve_config({"__source": "s"}).reasoning_effort == "high"
    assert resolver.resolve_config({"__source": "s", "reasoning_effort": None}).reasoning_effort == "high"
    assert resolver.resolve_config({"__source": "s", "reasoning_effort": "max"}).reasoning_effort == "max"


def test_validate_source_renderer_args_rejects_unknown_kwargs():
    config = SFTDataConfig.model_validate({"source": [{"dataset": "org/a", "renderer": {"depth": 16}}]})
    with pytest.raises(ValueError, match="depth"):
        validate_source_renderer_args(PrimeQwen3RendererConfig(), config.resolved_sources())
    validate_source_renderer_args(DeepSeekV4RendererConfig(), SFTDataConfig(name="org/a").resolved_sources())


def test_fake_source_is_the_only_source():
    config = SFTDataConfig.model_validate({"source": [{"type": "fake", "length": "variable"}]})
    assert isinstance(config.fake_source, FakeSourceConfig)
    assert config.fake_source.length == "variable"
    assert config.resolved_sources() == []
    assert SFTDataConfig(name="org/a").fake_source is None
